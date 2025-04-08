# server.py
import argparse
import logging
import json
import os
import queue
import uuid
import sys
import random
from concurrent import futures
from functools import partial
import time # For potential delays/debugging
import subprocess # For running ffmpeg
import threading # For reading ffmpeg output potentially

import grpc
import numpy as np
import soundfile as sf
import ffmpeg # Import ffmpeg-python

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException

# Import LlamaIndex SentenceSplitter
try:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document # SentenceSplitter now often needs Document
except ImportError:
    logging.error("llama-index-core is not installed. Please run: pip install llama-index-core")
    sys.exit(1)


# Import updated generated gRPC files
import gateway_pb2
import gateway_pb2_grpc

# --- Triton client helper classes/functions ---
# (UserData, callback, prepare_grpc_sdk_request remain the same)
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        logging.error(f"Triton inference error: {error}")
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def prepare_grpc_sdk_request(
    waveform,
    reference_text,
    target_text, # Now expects a potentially shorter chunk
    sample_rate=16000,
):
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    assert len(waveform.shape) == 1, "waveform should be 1D"
    assert sample_rate == 16000, "sample rate must be 16000"

    samples = waveform.reshape(1, -1)
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    inputs = [
        grpcclient.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        grpcclient.InferInput("reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)),
        grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
        grpcclient.InferInput("target_text", [1, 1], "BYTES"), # For the current text chunk
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    ref_text_bytes = np.array([[reference_text.encode('utf-8')]], dtype=object)
    inputs[2].set_data_from_numpy(ref_text_bytes)
    # Encode the current text chunk
    target_text_bytes = np.array([[target_text.encode('utf-8')]], dtype=object)
    inputs[3].set_data_from_numpy(target_text_bytes)
    return inputs

# --- Opus/WebM Encoding Helper ---
def start_ffmpeg_opus_encoder(sample_rate=16000):
    """Starts an FFmpeg process for encoding raw f32le PCM to Opus/WebM."""
    logging.debug("Starting FFmpeg for Opus/WebM encoding...")
    try:
        # TO-DO: optimize options
        process = (
            ffmpeg
            .input('pipe:', format='f32le', ac=1, ar=str(sample_rate))
            .output('pipe:', format='webm', acodec='libopus', vbr='on', # vbr='on' for variable bitrate
                    **{'b:a': '64k'} # Target bitrate (Opus VBR will fluctuate around this)
                    # Removed +global_header as WebM streaming usually handles headers per cluster
                   )
            .global_args('-hide_banner', '-loglevel', 'warning') # Quieter ffmpeg logs
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        logging.debug(f"FFmpeg process started (PID: {process.pid})")
        return process
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please ensure FFmpeg is installed and in PATH.")
        raise
    except Exception as e:
        logging.error(f"Failed to start FFmpeg process: {e}", exc_info=True)
        raise

def read_ffmpeg_output(ffmpeg_process, output_queue, request_id):
    """Reads stdout from FFmpeg process and puts chunks into a queue."""
    logging.debug(f"[ReqID: {request_id}] FFmpeg output reader thread started.")
    try:
        while True:
            chunk = ffmpeg_process.stdout.read(4096) # Read in chunks
            if not chunk:
                logging.debug(f"[ReqID: {request_id}] FFmpeg stdout EOF reached.")
                break
            # logging.debug(f"[ReqID: {request_id}] Read {len(chunk)} bytes from FFmpeg stdout.")
            output_queue.put(chunk)
    except Exception as e:
        logging.error(f"[ReqID: {request_id}] Error reading FFmpeg stdout: {e}", exc_info=True)
        output_queue.put(e) # Signal error
    finally:
        output_queue.put(None) # Signal completion
        logging.debug(f"[ReqID: {request_id}] FFmpeg output reader thread finished.")

# --- Gateway Servicer Implementation ---
class AudioGatewayServicer(gateway_pb2_grpc.AudioGatewayServicer):
    def __init__(self, triton_url, model_name, templates_file, verbose=False):
        self.triton_url = triton_url
        self.model_name = model_name
        self.verbose = verbose
        self.templates_path = templates_file
        self.templates = {}
        try:
            with open(self.templates_path, 'r') as f:
                self.templates = json.load(f)
            if not self.templates:
                 logging.error(f"No templates found or loaded from {self.templates_path}")
                 sys.exit(1)
            self.template_ids = list(self.templates.keys())
            logging.info(f"Loaded {len(self.template_ids)} templates from {self.templates_path}")
        except Exception as e:
            logging.error(f"Failed to load templates from {self.templates_path}: {e}", exc_info=True)
            sys.exit(1)
        # Ensure FFmpeg is available on init? Or let it fail per-request? Per-request is probably fine.

    def _synthesize_opus_webm(self, pcm_producer, request_id, context):
        """Handles encoding PCM to Opus/WebM and yielding chunks."""
        ffmpeg_process = None
        reader_thread = None
        output_queue = queue.Queue()

        try:
            ffmpeg_process = start_ffmpeg_opus_encoder()
            # Start a thread to read FFmpeg's output concurrently
            reader_thread = threading.Thread(target=read_ffmpeg_output, args=(ffmpeg_process, output_queue, request_id))
            reader_thread.start()

            # Consume PCM chunks from the producer generator and feed to FFmpeg
            pcm_chunks_fed = 0
            for pcm_chunk_bytes in pcm_producer:
                if pcm_chunk_bytes:
                    # logging.debug(f"[ReqID: {request_id}] Feeding {len(pcm_chunk_bytes)} PCM bytes to FFmpeg.")
                    try:
                        ffmpeg_process.stdin.write(pcm_chunk_bytes)
                        pcm_chunks_fed += 1
                    except (OSError, BrokenPipeError) as e:
                        logging.error(f"[ReqID: {request_id}] Error writing to FFmpeg stdin: {e}. Maybe FFmpeg exited early.")
                        # Attempt to read stderr below
                        break # Stop feeding input

                # Check the output queue periodically while feeding input
                try:
                    while True:
                        encoded_chunk = output_queue.get_nowait()
                        if isinstance(encoded_chunk, Exception):
                            raise encoded_chunk # Propagate reader error
                        if encoded_chunk is None: # Reader finished (shouldn't happen yet)
                            logging.warning(f"[ReqID: {request_id}] Reader thread ended prematurely.")
                            output_queue.put(None) # Put it back for the final loop
                            break
                        # logging.debug(f"[ReqID: {request_id}] Yielding {len(encoded_chunk)} Opus/WebM bytes.")
                        yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)
                except queue.Empty:
                    pass # No output ready yet, continue feeding input or finish

            logging.info(f"[ReqID: {request_id}] Finished feeding {pcm_chunks_fed} PCM chunks to FFmpeg.")

            # Signal FFmpeg that input is finished
            if ffmpeg_process.stdin:
                logging.debug(f"[ReqID: {request_id}] Closing FFmpeg stdin.")
                try:
                    ffmpeg_process.stdin.close()
                except OSError as e:
                    logging.warning(f"[ReqID: {request_id}] Non-critical error closing FFmpeg stdin: {e}")


            # Read remaining output from the queue
            logging.debug(f"[ReqID: {request_id}] Reading remaining encoded output...")
            while True:
                encoded_chunk = output_queue.get() # Blocking wait now
                if isinstance(encoded_chunk, Exception):
                    raise encoded_chunk # Propagate reader error
                if encoded_chunk is None: # Sentinel indicating reader thread finished
                    logging.debug(f"[ReqID: {request_id}] Reached end of encoded stream.")
                    break
                # logging.debug(f"[ReqID: {request_id}] Yielding final {len(encoded_chunk)} Opus/WebM bytes.")
                yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)

            logging.info(f"[ReqID: {request_id}] Opus/WebM stream finished.")

        except Exception as e:
             logging.error(f"[ReqID: {request_id}] Error during Opus/WebM encoding stream: {e}", exc_info=True)
             context.abort(grpc.StatusCode.INTERNAL, f"Opus/WebM encoding failed: {e}")
             # Fall through to finally block for cleanup

        finally:
            # Cleanup FFmpeg process and reader thread
            if reader_thread and reader_thread.is_alive():
                logging.debug(f"[ReqID: {request_id}] Waiting for FFmpeg reader thread to join...")
                reader_thread.join(timeout=2.0) # Add a timeout
                if reader_thread.is_alive():
                    logging.warning(f"[ReqID: {request_id}] FFmpeg reader thread did not join cleanly.")

            if ffmpeg_process:
                logging.debug(f"[ReqID: {request_id}] Cleaning up FFmpeg process...")
                try:
                    # Check if stdin/stdout are already closed
                    if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                       ffmpeg_process.stdin.close()
                    if ffmpeg_process.stdout and not ffmpeg_process.stdout.closed:
                       ffmpeg_process.stdout.close()

                    # Read stderr *after* closing pipes and waiting
                    stderr_output = ""
                    if ffmpeg_process.stderr:
                       try:
                            stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='replace')
                       except Exception as stderr_e:
                            logging.warning(f"[ReqID: {request_id}] Error reading FFmpeg stderr: {stderr_e}")
                       finally:
                           if not ffmpeg_process.stderr.closed:
                               ffmpeg_process.stderr.close()

                    return_code = ffmpeg_process.wait(timeout=5.0) # Wait with timeout
                    logging.info(f"[ReqID: {request_id}] FFmpeg process exited with code {return_code}.")
                    if return_code != 0:
                        logging.error(f"[ReqID: {request_id}] FFmpeg process failed (code {return_code}). Stderr:\n{stderr_output}")
                        # Avoid aborting context if stream already finished/aborted
                        if context.is_active():
                           context.abort(grpc.StatusCode.INTERNAL, f"FFmpeg encoding process failed (code {return_code})")
                    elif stderr_output:
                         logging.warning(f"[ReqID: {request_id}] FFmpeg stderr output:\n{stderr_output}")


                except subprocess.TimeoutExpired:
                    logging.error(f"[ReqID: {request_id}] FFmpeg process timed out on wait. Terminating.")
                    ffmpeg_process.terminate()
                    time.sleep(0.5) # Give it a moment
                    if ffmpeg_process.poll() is None: # Still running?
                        logging.warning(f"[ReqID: {request_id}] FFmpeg process did not terminate gracefully. Killing.")
                        ffmpeg_process.kill()
                    if context.is_active():
                        context.abort(grpc.StatusCode.INTERNAL, "FFmpeg process timed out during cleanup.")
                except Exception as e:
                    logging.error(f"[ReqID: {request_id}] Error during FFmpeg cleanup: {e}", exc_info=True)
                    # Don't abort here if it might already be finishing/aborted


    def _synthesize_pcm_float32(self, pcm_producer, request_id, context):
        """Handles yielding raw PCM Float32 chunks."""
        pcm_chunks_yielded = 0
        try:
            for pcm_chunk_bytes in pcm_producer:
                if pcm_chunk_bytes:
                    # logging.debug(f"[ReqID: {request_id}] Yielding {len(pcm_chunk_bytes)} raw PCM bytes.")
                    yield gateway_pb2.AudioChunk(audio_data=pcm_chunk_bytes)
                    pcm_chunks_yielded += 1
            logging.info(f"[ReqID: {request_id}] Raw PCM stream finished. Yielded {pcm_chunks_yielded} chunks.")
        except Exception as e:
             logging.error(f"[ReqID: {request_id}] Error during PCM streaming: {e}", exc_info=True)
             context.abort(grpc.StatusCode.INTERNAL, f"PCM streaming failed: {e}")


    def _process_triton_stream(self, triton_client, model_name, inputs, chunk_req_id, request_id, context):
        """Generator that yields raw PCM f32 bytes received from Triton."""
        user_data = UserData()
        stream_has_error = False
        stream_started = False
        audio_chunks_in_stream = 0

        try:
            triton_client.start_stream(callback=partial(callback, user_data))
            stream_started = True
            logging.debug(f"[ReqID: {request_id}] Triton stream started for {chunk_req_id}.")

            triton_client.async_stream_infer(
                model_name=model_name,
                inputs=inputs,
                request_id=chunk_req_id,
                outputs=[grpcclient.InferRequestedOutput("waveform")],
                enable_empty_final_response=True,
            )
            logging.debug(f"[ReqID: {request_id}] Async infer request sent to Triton for {chunk_req_id}.")

            while True:
                result_or_error = user_data._completed_requests.get() # Blocking wait

                if isinstance(result_or_error, InferenceServerException):
                    logging.error(f"[ReqID: {request_id}] Triton error during chunk stream {chunk_req_id}: {result_or_error}")
                    context.set_details(f"Triton inference failed on chunk: {result_or_error}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    stream_has_error = True
                    break

                try:
                    response = result_or_error.get_response()
                except Exception as e:
                     logging.error(f"[ReqID: {request_id}] Error getting response object for {chunk_req_id}: {e}", exc_info=True)
                     context.set_details("Error processing Triton callback response.")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     stream_has_error = True
                     break

                if response.parameters.get("triton_final_response", None) and \
                   response.parameters["triton_final_response"].bool_param:
                    logging.debug(f"[ReqID: {request_id}] Received final marker for {chunk_req_id} stream.")
                    break # Successful end of this stream

                try:
                    audio_chunk_np = result_or_error.as_numpy("waveform")
                    if audio_chunk_np is None: continue

                    if audio_chunk_np.dtype != np.float32:
                        audio_chunk_np = audio_chunk_np.astype(np.float32)
                    audio_chunk_np = audio_chunk_np.reshape(-1)
                    if audio_chunk_np.size == 0: continue

                    audio_bytes = audio_chunk_np.tobytes()
                    yield audio_bytes # Yield the raw PCM bytes
                    audio_chunks_in_stream += 1

                except InferenceServerException as e:
                     logging.error(f"[ReqID: {request_id}] Numpy conversion error for {chunk_req_id}: {e}")
                     context.set_details(f"Numpy conversion error processing chunk: {e}")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     stream_has_error = True
                     break
                except Exception as e:
                     logging.error(f"[ReqID: {request_id}] Unexpected error processing audio chunk for {chunk_req_id}: {e}", exc_info=True)
                     context.set_details("Unexpected error processing audio chunk.")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     stream_has_error = True
                     break

            logging.debug(f"[ReqID: {request_id}] Finished receiving Triton stream for {chunk_req_id}. Received {audio_chunks_in_stream} raw PCM chunks.")

        # Ensure stream is stopped even if errors occurred before/during processing loop
        finally:
            if stream_started:
                logging.debug(f"[ReqID: {request_id}] Stopping Triton stream for {chunk_req_id}...")
                triton_client.stop_stream()
                logging.debug(f"[ReqID: {request_id}] Triton stream stopped for {chunk_req_id}.")

        if stream_has_error:
             # Raise an exception to signal the outer loop about the error
             raise RuntimeError(f"Triton stream processing failed for {chunk_req_id}. See logs.")


    def SynthesizeSpeech(self, request, context):
        """
        Handles request: splits text, calls Triton sequentially, encodes if necessary, streams audio.
        """
        request_id = str(uuid.uuid4())
        output_format = request.output_format
        if output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_UNSPECIFIED:
            output_format = gateway_pb2.OutputFormat.OUTPUT_FORMAT_WAV_PCM_FLOAT32 # Default
            format_str = "WAV_PCM_FLOAT32 (Default)"
        elif output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_WAV_PCM_FLOAT32:
            format_str = "WAV_PCM_FLOAT32"
        elif output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_OPUS_WEBM:
            format_str = "OPUS_WEBM"
        else:
            logging.warning(f"[ReqID: {request_id}] Received unknown output format value: {output_format}. Defaulting to WAV_PCM_FLOAT32.")
            output_format = gateway_pb2.OutputFormat.OUTPUT_FORMAT_WAV_PCM_FLOAT32
            format_str = "WAV_PCM_FLOAT32 (Defaulted)"

        logging.info(f"[ReqID: {request_id}] Received SynthesizeSpeech request. Format: {format_str}. Target text length: {len(request.target_text)}")

        # --- 1. Select Template & Load Reference Audio ---
        # (Same logic as before)
        try:
            selected_template_id = random.choice(self.template_ids)
            template_info = self.templates[selected_template_id]
            logging.info(f"[ReqID: {request_id}] Selected template ID: '{selected_template_id}'")
            reference_text = template_info.get("reference_text")
            reference_audio_path = template_info.get("reference_audio")
            if not reference_text or not reference_audio_path: raise ValueError("Template missing ref text/audio path")
            if not os.path.isabs(reference_audio_path):
                base_dir = os.path.dirname(self.templates_path)
                reference_audio_path = os.path.normpath(os.path.join(base_dir, reference_audio_path))
            if not os.path.exists(reference_audio_path): raise FileNotFoundError(f"Ref audio not found: {reference_audio_path}")

            waveform, sr = sf.read(reference_audio_path)
            if sr != 16000: raise ValueError("Reference audio must be 16kHz.")
            if waveform.ndim > 1: waveform = waveform[:, 0]
            if waveform.dtype != np.float32: waveform = waveform.astype(np.float32)
            logging.info(f"[ReqID: {request_id}] Loaded reference audio: {reference_audio_path}")

        except Exception as e:
            logging.error(f"[ReqID: {request_id}] Error preparing reference data: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Failed to prepare reference data: {e}")
            return

        # --- 2. Split Target Text ---
        # (Same logic as before)
        target_text = request.target_text
        text_chunks = []
        try:
            splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
            document = Document(text=target_text)
            nodes = splitter.get_nodes_from_documents([document])
            raw_chunks = [node.get_content() for node in nodes]

            final_chunks = []
            for chunk in raw_chunks:
                word_count = len(chunk.split())
                if word_count > 75:
                     logging.warning(f"[ReqID: {request_id}] Splitter chunk > 75 words ({word_count}).")
                if chunk.strip():
                    final_chunks.append(chunk.strip())
            text_chunks = final_chunks

            if not text_chunks:
                 if target_text.strip():
                      logging.warning(f"[ReqID: {request_id}] Text splitting yielded zero chunks, using original text.")
                      text_chunks = [target_text.strip()]
                 else:
                      logging.info(f"[ReqID: {request_id}] Input text empty. No audio.")
                      return # Empty stream

            logging.info(f"[ReqID: {request_id}] Split text into {len(text_chunks)} chunks.")

        except Exception as e:
            logging.error(f"[ReqID: {request_id}] Error during text splitting: {e}", exc_info=True)
            text_chunks = [target_text] if target_text.strip() else []
            if text_chunks: logging.warning(f"[ReqID: {request_id}] Using original text due to splitting error.")
            else: logging.info(f"[ReqID: {request_id}] Empty text after splitting error."); return


        # --- 3. Define PCM Producer Generator ---
        def pcm_chunk_producer():
            """Generator that yields PCM f32 bytes from Triton for all text chunks."""
            triton_client = None
            try:
                triton_client = grpcclient.InferenceServerClient(url=self.triton_url, verbose=self.verbose)
                logging.info(f"[ReqID: {request_id}] Triton client created for PCM production.")

                for i, text_chunk in enumerate(text_chunks):
                    chunk_req_id = f"{request_id}-C{i+1}"
                    logging.info(f"[ReqID: {request_id}] Requesting PCM for text chunk {i+1}/{len(text_chunks)} (TritonReqID: {chunk_req_id})")

                    try:
                        inputs = prepare_grpc_sdk_request(waveform, reference_text, text_chunk)
                    except Exception as e:
                        logging.error(f"[ReqID: {request_id}] Error preparing inputs for chunk {i+1}: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to prepare inputs for chunk {i+1}: {e}") # Signal error

                    # Yield PCM bytes from this chunk's Triton stream
                    try:
                        yield from self._process_triton_stream(triton_client, self.model_name, inputs, chunk_req_id, request_id, context)
                        logging.info(f"[ReqID: {request_id}] Finished processing Triton stream for chunk {i+1}.")
                    except Exception as e: # Catch errors from _process_triton_stream
                         logging.error(f"[ReqID: {request_id}] Error processing Triton stream for chunk {i+1}: {e}")
                         # If context is still active, abort. _process_triton_stream might have already set details.
                         if context.is_active():
                             context.abort(grpc.StatusCode.INTERNAL, f"Processing failed for text chunk {i+1}: {e}")
                         raise # Re-raise to stop the producer

                logging.info(f"[ReqID: {request_id}] Successfully produced PCM for all {len(text_chunks)} text chunks.")

            except InferenceServerException as e:
                logging.error(f"[ReqID: {request_id}] Triton communication error during PCM production: {e}", exc_info=True)
                if context.is_active(): context.abort(grpc.StatusCode.UNAVAILABLE, f"Triton communication failed: {e}")
                raise # Stop producer
            except grpc.RpcError as e:
                 logging.error(f"[ReqID: {request_id}] gRPC error during Triton call: {e.code()} - {e.details()}")
                 if context.is_active(): context.abort(e.code(), f"gRPC error calling Triton: {e.details()}")
                 raise # Stop producer
            except Exception as e:
                 logging.error(f"[ReqID: {request_id}] Unexpected error during PCM production: {e}", exc_info=True)
                 if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Unexpected gateway error during PCM production: {e}")
                 raise # Stop producer
            finally:
                if triton_client:
                    try:
                        logging.debug(f"[ReqID: {request_id}] Closing Triton client connection used by producer.")
                        triton_client.close()
                        logging.info(f"[ReqID: {request_id}] Triton client (producer) connection closed.")
                    except Exception as e:
                        logging.warning(f"[ReqID: {request_id}] Error closing Triton client (producer): {e}")


        # --- 4. Process PCM Chunks based on requested format ---
        try:
            pcm_producer = pcm_chunk_producer()

            if output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_OPUS_WEBM:
                logging.info(f"[ReqID: {request_id}] Starting Opus/WebM encoding process.")
                yield from self._synthesize_opus_webm(pcm_producer, request_id, context)
            else: # Default or explicit WAV_PCM_FLOAT32
                logging.info(f"[ReqID: {request_id}] Starting raw PCM Float32 streaming.")
                yield from self._synthesize_pcm_float32(pcm_producer, request_id, context)

            logging.info(f"[ReqID: {request_id}] SynthesizeSpeech request processing completed.")

        except Exception as e:
             # Catch errors raised by the pcm_producer or the format-specific handlers (_synthesize_*)
             # Context should ideally be aborted by the function that failed.
             # Log just in case it wasn't caught cleanly.
             if context.is_active():
                logging.error(f"[ReqID: {request_id}] Unhandled exception at SynthesizeSpeech top level: {e}", exc_info=True)
                # Avoid aborting again if already done
                try: context.abort(grpc.StatusCode.INTERNAL, f"Request processing failed unexpectedly: {e}")
                except Exception as abort_e: logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")
             else:
                 logging.info(f"[ReqID: {request_id}] Request processing terminated due to earlier error.")


# --- Server main execution ---
# (serve function and if __name__ == '__main__' block remain the same)
def serve(port, triton_url, model_name, templates_file, verbose):
    # Check for FFmpeg before starting the server fully (optional but helpful)
    try:
        ffmpeg_version_proc = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        logging.info(f"FFmpeg found:\n{ffmpeg_version_proc.stdout.splitlines()[0]}") # Log first line
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error(f"FFmpeg check failed: {e}. Opus/WebM encoding will not work.")
        # Decide whether to exit or continue (allowing only PCM)
        # sys.exit("FFmpeg not found or failed check.") # Or just warn

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # Consider adjusting max_workers
    servicer = AudioGatewayServicer(triton_url, model_name, templates_file, verbose)
    if servicer.templates:
        gateway_pb2_grpc.add_AudioGatewayServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        logging.info(f"Starting gRPC Gateway Server on port {port}...")
        logging.info(f"Connecting to Triton at: {triton_url}")
        logging.info(f"Using Triton model: {model_name}")
        logging.info(f"Using templates from: {templates_file}")
        server.start()
        logging.info("Server started. Waiting for requests...")
        server.wait_for_termination()
    else:
        logging.error("Server startup failed due to template loading issues.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server with Text Splitting & Format Selection")
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    parser.add_argument("--triton-url", type=str, default="inference:8001", help="URL of the Triton Inference Server (gRPC)")
    parser.add_argument("--model-name", type=str, default="spark_tts_decoupled", help="Name of the TTS model on Triton")
    parser.add_argument("--templates-file", type=str, default="templates/templates.json", help="Path to voice templates JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for Triton client")
    args = parser.parse_args()
    serve(args.port, args.triton_url, args.model_name, args.templates_file, args.verbose)