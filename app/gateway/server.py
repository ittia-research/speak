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
import threading # For reading ffmpeg output

import grpc
import numpy as np # Required for cross-fading
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

# --- Constants ---
SAMPLE_RATE = 16000 # Expected sample rate from Triton and for output

# --- Triton client helper classes/functions ---
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        # Log error, but let the main processing loop handle context abort etc.
        logging.error(f"Triton inference callback error: {error}")
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def prepare_grpc_sdk_request(
    waveform,
    reference_text,
    target_text,
    sample_rate=SAMPLE_RATE, # Use constant
):
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    assert len(waveform.shape) == 1, "waveform should be 1D"
    assert sample_rate == SAMPLE_RATE, f"sample rate must be {SAMPLE_RATE}"

    samples = waveform.reshape(1, -1)
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    inputs = [
        grpcclient.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        grpcclient.InferInput("reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)),
        grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
        grpcclient.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    ref_text_bytes = np.array([[reference_text.encode('utf-8')]], dtype=object)
    inputs[2].set_data_from_numpy(ref_text_bytes)
    target_text_bytes = np.array([[target_text.encode('utf-8')]], dtype=object)
    inputs[3].set_data_from_numpy(target_text_bytes)
    return inputs

# --- Opus/WebM Encoding Helper ---
def start_ffmpeg_opus_encoder(sample_rate=SAMPLE_RATE):
    """Starts an FFmpeg process for encoding raw f32le PCM to Opus/WebM."""
    logging.debug("Starting FFmpeg for Opus/WebM encoding...")
    try:
        # TO-DO: dynamic parameters
        process = (
            ffmpeg
            .input('pipe:', format='f32le', ac=1, ar=str(sample_rate)) # Input is float32 Little Endian PCM
            .output('pipe:', format='webm', acodec='libopus', vbr='on', # Output WebM container with Opus codec, VBR
                    ac=1, ar=str(sample_rate), # Explicitly set output sample rate/channels
                    **{'b:a': '64k'} # Target average bitrate (VBR will fluctuate)
                   )
            .global_args('-hide_banner', '-loglevel', 'warning') # Keep FFmpeg quiet unless error
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
            chunk = ffmpeg_process.stdout.read(4096) # Read in reasonable chunks
            if not chunk:
                logging.debug(f"[ReqID: {request_id}] FFmpeg stdout EOF reached.")
                break
            output_queue.put(chunk)
    except Exception as e:
        # Log error and signal it via the queue
        logging.error(f"[ReqID: {request_id}] Error reading FFmpeg stdout: {e}", exc_info=True)
        output_queue.put(e)
    finally:
        output_queue.put(None) # Signal completion (None sentinel)
        logging.debug(f"[ReqID: {request_id}] FFmpeg output reader thread finished.")

# --- Gateway Servicer Implementation ---
class AudioGatewayServicer(gateway_pb2_grpc.AudioGatewayServicer):
    def __init__(self, triton_url, model_name, templates_file, verbose=False, crossfade_duration=0.1):
        self.triton_url = triton_url
        self.model_name = model_name
        self.verbose = verbose
        self.templates_path = templates_file
        self.crossfade_duration_sec = crossfade_duration # Duration for gateway-side crossfade (for Opus)
        self.templates = {}
        try:
            with open(self.templates_path, 'r') as f:
                self.templates = json.load(f)
            if not self.templates:
                 logging.error(f"No templates found or loaded from {self.templates_path}")
                 sys.exit(1)
            self.template_ids = list(self.templates.keys())
            logging.info(f"Loaded {len(self.template_ids)} templates from {self.templates_path}")
            logging.info(f"Gateway crossfade duration for Opus set to: {self.crossfade_duration_sec * 1000:.0f} ms")
        except Exception as e:
            logging.error(f"Failed to load templates from {self.templates_path}: {e}", exc_info=True)
            sys.exit(1)

    def _process_triton_stream_np(self, triton_client, model_name, inputs, chunk_req_id, request_id, context):
        """
        Generator that yields raw PCM f32 chunks as NumPy arrays received from Triton.
        Handles Triton communication for a single text chunk request.
        Raises exceptions on fatal errors.
        """
        user_data = UserData()
        stream_has_error = False
        stream_started = False
        audio_chunks_received = 0

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
                    logging.error(f"[ReqID: {request_id}] Triton error during {chunk_req_id}: {result_or_error}")
                    stream_has_error = True
                    break # Exit loop, error will be raised below

                try:
                    response = result_or_error.get_response()
                except Exception as e:
                     logging.error(f"[ReqID: {request_id}] Error getting response object for {chunk_req_id}: {e}", exc_info=True)
                     stream_has_error = True
                     break # Exit loop, error will be raised below

                # Check for final response marker
                if response.parameters.get("triton_final_response", None) and \
                   response.parameters["triton_final_response"].bool_param:
                    logging.debug(f"[ReqID: {request_id}] Received final marker for {chunk_req_id} stream.")
                    break # Successful end of this stream

                # Process audio chunk
                try:
                    audio_chunk_np = result_or_error.as_numpy("waveform")
                    if audio_chunk_np is None:
                        logging.debug(f"[ReqID: {request_id}] Received None numpy array for {chunk_req_id}. Skipping.")
                        continue

                    # Ensure correct dtype and shape
                    if audio_chunk_np.dtype != np.float32:
                        audio_chunk_np = audio_chunk_np.astype(np.float32)
                    audio_chunk_np = audio_chunk_np.reshape(-1) # Ensure 1D
                    if audio_chunk_np.size == 0:
                        logging.debug(f"[ReqID: {request_id}] Received empty numpy array for {chunk_req_id}. Skipping.")
                        continue

                    # >>> Yield NumPy array <<<
                    yield audio_chunk_np
                    audio_chunks_received += 1

                except InferenceServerException as e:
                     logging.error(f"[ReqID: {request_id}] Numpy conversion error for {chunk_req_id}: {e}")
                     stream_has_error = True
                     break # Exit loop, error will be raised below
                except Exception as e:
                     logging.error(f"[ReqID: {request_id}] Unexpected error processing numpy chunk for {chunk_req_id}: {e}", exc_info=True)
                     stream_has_error = True
                     break # Exit loop, error will be raised below

            logging.debug(f"[ReqID: {request_id}] Finished receiving Triton stream for {chunk_req_id}. Received {audio_chunks_received} numpy chunks.")

        finally:
            # Always try to stop the stream if it was started
            if stream_started:
                logging.debug(f"[ReqID: {request_id}] Stopping Triton stream for {chunk_req_id}...")
                triton_client.stop_stream()
                logging.debug(f"[ReqID: {request_id}] Triton stream stopped for {chunk_req_id}.")

        # If an error occurred, raise it to stop the consuming generator
        if stream_has_error:
             # The error is already logged, just signal failure
             raise RuntimeError(f"Triton stream processing failed for {chunk_req_id}. Check logs for details.")


    def pcm_numpy_producer(self, text_chunks, waveform, reference_text, request_id, context):
        """
        Generator that yields PCM f32 chunks as NumPy arrays from Triton for all text chunks.
        Manages Triton client lifecycle for the sequence of requests.
        """
        triton_client = None
        try:
            triton_client = grpcclient.InferenceServerClient(url=self.triton_url, verbose=self.verbose)
            logging.info(f"[ReqID: {request_id}] Triton client created for NumPy production.")

            for i, text_chunk in enumerate(text_chunks):
                chunk_req_id = f"{request_id}-C{i+1}"
                logging.info(f"[ReqID: {request_id}] Requesting PCM NumPy for text chunk {i+1}/{len(text_chunks)} (TritonReqID: {chunk_req_id})")

                try:
                    inputs = prepare_grpc_sdk_request(waveform, reference_text, text_chunk)
                except Exception as e:
                    logging.error(f"[ReqID: {request_id}] Error preparing inputs for chunk {i+1}: {e}", exc_info=True)
                    # Abort context and raise to stop the generator
                    if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Failed to prepare inputs for chunk {i+1}: {e}")
                    raise RuntimeError(f"Input preparation failed for chunk {i+1}") from e

                # Yield NumPy arrays from this chunk's Triton stream
                try:
                    yield from self._process_triton_stream_np(triton_client, self.model_name, inputs, chunk_req_id, request_id, context)
                    logging.info(f"[ReqID: {request_id}] Finished processing Triton stream for chunk {i+1}.")
                except Exception as e: # Catch errors raised by _process_triton_stream_np
                     logging.error(f"[ReqID: {request_id}] Error processing Triton NumPy stream for chunk {i+1}: {e}")
                     # Context should have been aborted or details set by the inner function
                     # Re-raise to stop this producer generator
                     raise

            logging.info(f"[ReqID: {request_id}] Successfully produced PCM NumPy for all {len(text_chunks)} text chunks.")

        except InferenceServerException as e:
            logging.error(f"[ReqID: {request_id}] Triton communication error during NumPy production: {e}", exc_info=True)
            if context.is_active(): context.abort(grpc.StatusCode.UNAVAILABLE, f"Triton communication failed: {e}")
            raise # Stop generator
        except grpc.RpcError as e:
             logging.error(f"[ReqID: {request_id}] gRPC error during Triton call: {e.code()} - {e.details()}")
             if context.is_active(): context.abort(e.code(), f"gRPC error calling Triton: {e.details()}")
             raise # Stop generator
        except Exception as e: # Includes potential runtime errors raised from helpers
             logging.error(f"[ReqID: {request_id}] Unexpected error during PCM NumPy production: {e}", exc_info=True)
             if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Unexpected gateway error during PCM NumPy production: {e}")
             raise # Stop generator
        finally:
            # Ensure Triton client is closed if it was created
            if triton_client:
                try:
                    logging.debug(f"[ReqID: {request_id}] Closing Triton client connection (NumPy producer).")
                    triton_client.close()
                    logging.info(f"[ReqID: {request_id}] Triton client (NumPy producer) connection closed.")
                except Exception as e:
                    logging.warning(f"[ReqID: {request_id}] Error closing Triton client (NumPy producer): {e}")


    def _crossfade_pcm_chunks(self, pcm_np_producer, overlap_duration_sec, sample_rate, request_id, context):
        """
        Generator that takes raw PCM numpy chunks, performs cross-fading,
        and yields smoothed PCM chunks as float32 bytes.
        """
        logging.info(f"[ReqID: {request_id}] Initializing PCM cross-fading (overlap: {overlap_duration_sec*1000:.0f}ms).")
        cross_fade_samples = int(overlap_duration_sec * sample_rate)

        if cross_fade_samples <= 0:
            logging.warning(f"[ReqID: {request_id}] Cross-fade duration results in <= 0 samples. Yielding raw chunks without fading.")
            chunk_count = 0
            for chunk_np in pcm_np_producer:
                if chunk_np is not None and chunk_np.size > 0:
                    yield chunk_np.tobytes()
                    chunk_count += 1
            logging.info(f"[ReqID: {request_id}] Finished yielding {chunk_count} raw chunks (no fading).")
            return

        # Pre-calculate fade curves
        fade_out_curve = np.linspace(1, 0, cross_fade_samples, dtype=np.float32)
        fade_in_curve = np.linspace(0, 1, cross_fade_samples, dtype=np.float32)

        previous_chunk_np = None
        processed_chunks = 0

        try:
            for current_chunk_np in pcm_np_producer:
                # Input producer might yield None or empty, filter them
                if current_chunk_np is None or current_chunk_np.size == 0:
                    logging.debug(f"[ReqID: {request_id}] Crossfader received empty chunk, skipping.")
                    continue

                if previous_chunk_np is None:
                    # This is the first valid chunk, store it and wait for the next one to fade with
                    previous_chunk_np = current_chunk_np
                    logging.debug(f"[ReqID: {request_id}] Crossfader stored first chunk (len: {len(previous_chunk_np)}).")
                    continue

                # --- Perform cross-fade ---
                len_prev = len(previous_chunk_np)
                len_curr = len(current_chunk_np)

                # Determine effective overlap based on actual chunk lengths
                effective_overlap = min(len_prev, len_curr, cross_fade_samples)

                if effective_overlap <= 0:
                    # Cannot overlap (one chunk is too short), yield the previous chunk entirely
                    logging.warning(f"[ReqID: {request_id}] Cannot overlap chunks (prev len {len_prev}, curr len {len_curr}, overlap {cross_fade_samples}). Yielding previous chunk directly.")
                    yield previous_chunk_np.tobytes()
                    processed_chunks += 1
                    previous_chunk_np = current_chunk_np # Current becomes the new previous
                    continue

                # Get the segments that will overlap
                prev_overlap_segment = previous_chunk_np[-effective_overlap:]
                curr_overlap_segment = current_chunk_np[:effective_overlap]

                # Adjust fade curves if the effective overlap is smaller than planned
                if effective_overlap < cross_fade_samples:
                    logging.debug(f"[ReqID: {request_id}] Reducing fade curve length to effective overlap: {effective_overlap} samples.")
                    effective_fade_out = np.linspace(1, 0, effective_overlap, dtype=np.float32)
                    effective_fade_in = np.linspace(0, 1, effective_overlap, dtype=np.float32)
                else:
                    effective_fade_out = fade_out_curve
                    effective_fade_in = fade_in_curve

                # Calculate the blended/cross-faded segment
                cross_faded_segment = (prev_overlap_segment * effective_fade_out +
                                       curr_overlap_segment * effective_fade_in)

                # Yield the part of the previous chunk *before* the overlap region
                yield previous_chunk_np[:-effective_overlap].tobytes()
                processed_chunks += 1

                # The current chunk for the *next* iteration starts with the cross-faded part,
                # followed by the rest of the actual current chunk.
                previous_chunk_np = np.concatenate([cross_faded_segment, current_chunk_np[effective_overlap:]])
                # logging.debug(f"[ReqID: {request_id}] Cross-faded. Stored new previous chunk (len: {len(previous_chunk_np)}).")
                # --- End cross-fade ---

            # After the loop finishes, yield the final remaining (processed) chunk
            if previous_chunk_np is not None and previous_chunk_np.size > 0:
                logging.debug(f"[ReqID: {request_id}] Yielding final processed chunk from crossfader (len: {len(previous_chunk_np)}).")
                yield previous_chunk_np.tobytes()
                processed_chunks += 1

            logging.info(f"[ReqID: {request_id}] Finished PCM cross-fading. Processed {processed_chunks} final chunks.")

        except Exception as e:
             logging.error(f"[ReqID: {request_id}] Error during PCM cross-fading: {e}", exc_info=True)
             # Don't abort context here, let the caller handle it, but raise to stop iteration
             raise RuntimeError(f"Cross-fading failed: {e}")


    def _synthesize_opus_webm_smoothed(self, pcm_np_producer, request_id, context):
        """Handles encoding Smoothed PCM (from crossfader) to Opus/WebM."""
        ffmpeg_process = None
        reader_thread = None
        output_queue = queue.Queue()
        total_encoded_chunks_yielded = 0

        # Create the cross-fader generator which consumes the numpy producer
        smoothed_pcm_bytes_producer = self._crossfade_pcm_chunks(
            pcm_np_producer,
            overlap_duration_sec=self.crossfade_duration_sec, # Use configured value
            sample_rate=SAMPLE_RATE,
            request_id=request_id,
            context=context # Pass context for potential aborts within crossfader (though unlikely)
        )

        try:
            ffmpeg_process = start_ffmpeg_opus_encoder() # Assumes SAMPLE_RATE f32le input
            reader_thread = threading.Thread(target=read_ffmpeg_output, args=(ffmpeg_process, output_queue, request_id))
            reader_thread.daemon = True # Allow main thread to exit even if this hangs (though cleanup tries to join)
            reader_thread.start()

            pcm_chunks_fed = 0
            # Consume SMOOTHED PCM byte chunks from the cross-fader and feed to FFmpeg
            for smoothed_pcm_chunk_bytes in smoothed_pcm_bytes_producer:
                if smoothed_pcm_chunk_bytes: # Should already be bytes
                    try:
                        # logging.debug(f"[ReqID: {request_id}] Feeding {len(smoothed_pcm_chunk_bytes)} smoothed PCM bytes to FFmpeg.")
                        ffmpeg_process.stdin.write(smoothed_pcm_chunk_bytes)
                        pcm_chunks_fed += 1
                    except (OSError, BrokenPipeError) as e:
                        logging.error(f"[ReqID: {request_id}] Error writing smoothed PCM to FFmpeg stdin: {e}. FFmpeg might have exited.")
                        # Attempt to read stderr in finally block
                        break # Stop feeding input

                # Check the output queue periodically while feeding input
                try:
                    while not output_queue.empty():
                        encoded_chunk = output_queue.get_nowait()
                        if isinstance(encoded_chunk, Exception):
                            logging.error(f"[ReqID: {request_id}] Error received from FFmpeg reader thread.")
                            raise encoded_chunk # Propagate reader error
                        if encoded_chunk is None: # Reader finished (might be premature)
                            logging.warning(f"[ReqID: {request_id}] Reader thread signaled completion early.")
                            output_queue.put(None) # Put it back for the final loop
                            break # Check input again or finish
                        # logging.debug(f"[ReqID: {request_id}] Yielding {len(encoded_chunk)} Opus/WebM bytes.")
                        yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)
                        total_encoded_chunks_yielded += 1
                except queue.Empty:
                    pass # No output ready yet, continue feeding input or finish loop

            logging.info(f"[ReqID: {request_id}] Finished feeding {pcm_chunks_fed} smoothed PCM chunks to FFmpeg.")

            # Signal FFmpeg that input is finished by closing stdin
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                logging.debug(f"[ReqID: {request_id}] Closing FFmpeg stdin.")
                try:
                    ffmpeg_process.stdin.close()
                except OSError as e:
                    logging.warning(f"[ReqID: {request_id}] Non-critical error closing FFmpeg stdin: {e}")


            # Read any remaining encoded output from the queue
            logging.debug(f"[ReqID: {request_id}] Reading remaining encoded output from FFmpeg...")
            while True:
                encoded_chunk = output_queue.get() # Blocking wait now
                if isinstance(encoded_chunk, Exception):
                     logging.error(f"[ReqID: {request_id}] Error received from FFmpeg reader thread during final read.")
                     raise encoded_chunk # Propagate reader error
                if encoded_chunk is None: # Sentinel indicating reader thread finished
                    logging.debug(f"[ReqID: {request_id}] Reached end of encoded stream from queue.")
                    break
                # logging.debug(f"[ReqID: {request_id}] Yielding final {len(encoded_chunk)} Opus/WebM bytes.")
                yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)
                total_encoded_chunks_yielded += 1

            logging.info(f"[ReqID: {request_id}] Smoothed Opus/WebM stream finished. Yielded {total_encoded_chunks_yielded} encoded chunks.")

        except Exception as e:
             # Includes errors raised from crossfader or reader thread
             logging.error(f"[ReqID: {request_id}] Error during Smoothed Opus/WebM encoding stream: {e}", exc_info=True)
             if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Opus/WebM encoding failed after smoothing: {e}")
             # Fall through to finally block for cleanup

        finally:
            # --- Cleanup FFmpeg process and reader thread ---
            if reader_thread and reader_thread.is_alive():
                logging.debug(f"[ReqID: {request_id}] Waiting for FFmpeg reader thread to join...")
                reader_thread.join(timeout=2.0) # Wait briefly
                if reader_thread.is_alive():
                    logging.warning(f"[ReqID: {request_id}] FFmpeg reader thread did not join cleanly.")

            if ffmpeg_process:
                logging.debug(f"[ReqID: {request_id}] Cleaning up FFmpeg process...")
                return_code = None
                stderr_output = ""
                try:
                    # Ensure pipes are closed before waiting (stdin might be closed already)
                    if ffmpeg_process.stdout and not ffmpeg_process.stdout.closed: ffmpeg_process.stdout.close()
                    if ffmpeg_process.stderr and not ffmpeg_process.stderr.closed:
                       try:
                            # Read stderr *after* process potentially finished or pipes closed
                            stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='replace')
                       except Exception as stderr_e:
                            logging.warning(f"[ReqID: {request_id}] Error reading FFmpeg stderr: {stderr_e}")
                       finally:
                            ffmpeg_process.stderr.close() # Ensure closed

                    # Wait for process completion with timeout
                    return_code = ffmpeg_process.wait(timeout=5.0)
                    logging.info(f"[ReqID: {request_id}] FFmpeg process exited with code {return_code}.")
                    if return_code != 0:
                        logging.error(f"[ReqID: {request_id}] FFmpeg process failed (code {return_code}). Stderr:\n{stderr_output if stderr_output else 'N/A'}")
                        # Don't abort here if stream might have finished ok, but log loudly
                    elif stderr_output:
                        # Log stderr even on success if it contains warnings etc.
                        logging.warning(f"[ReqID: {request_id}] FFmpeg stderr output (exit code 0):\n{stderr_output}")

                except subprocess.TimeoutExpired:
                    logging.error(f"[ReqID: {request_id}] FFmpeg process timed out on wait. Terminating.")
                    ffmpeg_process.terminate()
                    time.sleep(0.5) # Give terminate a chance
                    if ffmpeg_process.poll() is None: # Still running?
                        logging.warning(f"[ReqID: {request_id}] FFmpeg process did not terminate gracefully. Killing.")
                        ffmpeg_process.kill()
                    # Abort context if FFmpeg hung and we're still active
                    if context.is_active():
                        context.abort(grpc.StatusCode.INTERNAL, "FFmpeg process timed out during cleanup.")
                except Exception as e:
                    logging.error(f"[ReqID: {request_id}] Error during FFmpeg cleanup: {e}", exc_info=True)
                finally:
                    # Ensure poll is called to cleanup zombie process potential
                    if ffmpeg_process.poll() is None:
                        logging.warning(f"[ReqID: {request_id}] FFmpeg process still running after cleanup attempt, killing.")
                        ffmpeg_process.kill()


    def _synthesize_pcm_float32(self, pcm_bytes_producer, request_id, context):
        """Handles yielding raw PCM Float32 byte chunks."""
        pcm_chunks_yielded = 0
        try:
            for pcm_chunk_bytes in pcm_bytes_producer:
                if pcm_chunk_bytes: # Ensure not empty bytes
                    # logging.debug(f"[ReqID: {request_id}] Yielding {len(pcm_chunk_bytes)} raw PCM bytes.")
                    yield gateway_pb2.AudioChunk(audio_data=pcm_chunk_bytes)
                    pcm_chunks_yielded += 1
            logging.info(f"[ReqID: {request_id}] Raw PCM stream finished. Yielded {pcm_chunks_yielded} chunks.")
        except Exception as e:
             logging.error(f"[ReqID: {request_id}] Error during PCM streaming: {e}", exc_info=True)
             # Don't abort context if producer already did, but raise to signal SynthesizeSpeech
             raise RuntimeError(f"PCM streaming failed: {e}")

    # --- Main RPC Implementation ---
    def SynthesizeSpeech(self, request, context):
        """
        Handles request: splits text, calls Triton sequentially, gets PCM NumPy stream,
        optionally cross-fades and encodes to Opus, or yields raw PCM bytes, streams audio.
        """
        request_id = str(uuid.uuid4())
        output_format = request.output_format
        format_str = "UNKNOWN" # Default

        # Determine requested format and log
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
            format_str = "WAV_PCM_FLOAT32 (Forced Default)"

        logging.info(f"[ReqID: {request_id}] Received SynthesizeSpeech request. Format: {format_str}. Text len: {len(request.target_text)}")

        # --- 1. Select Template & Load Reference Audio ---
        try:
            selected_template_id = random.choice(self.template_ids)
            template_info = self.templates[selected_template_id]
            logging.info(f"[ReqID: {request_id}] Selected template ID: '{selected_template_id}'")
            reference_text = template_info.get("reference_text")
            reference_audio_path = template_info.get("reference_audio")
            if not reference_text or not reference_audio_path: raise ValueError("Template missing ref text/audio path")

            # Resolve and check reference audio path
            if not os.path.isabs(reference_audio_path):
                base_dir = os.path.dirname(self.templates_path)
                reference_audio_path = os.path.normpath(os.path.join(base_dir, reference_audio_path))
            if not os.path.exists(reference_audio_path): raise FileNotFoundError(f"Ref audio not found: {reference_audio_path}")

            # Load reference audio, ensuring correct format
            waveform, sr = sf.read(reference_audio_path, dtype='float32', always_2d=False)
            if sr != SAMPLE_RATE: raise ValueError(f"Reference audio must be {SAMPLE_RATE}kHz, found {sr}kHz.")
            # waveform is already float32 and 1D due to sf.read parameters
            logging.info(f"[ReqID: {request_id}] Loaded reference audio: {reference_audio_path}")

        except Exception as e:
            logging.error(f"[ReqID: {request_id}] Error preparing reference data: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Failed to prepare reference data: {e}")
            return # Exit RPC

        # --- 2. Split Target Text ---
        target_text = request.target_text
        text_chunks = []
        try:
            # Configure splitter (adjust chunk_size/overlap as needed)
            # SentenceSplitter is generally better than naive splitting
            splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0) # Chars, not words
            document = Document(text=target_text)
            nodes = splitter.get_nodes_from_documents([document])
            raw_chunks = [f"{node.get_content().strip()} " for node in nodes] # Strip whitespace
            # Logging text split results
            logging.info(f"\nText chunks after split:")
            for i in raw_chunks:
                logging.info(f"<<{i}>>")

            # Filter out empty chunks and potentially check length/word count
            final_chunks = []
            for chunk in raw_chunks:
                 if not chunk: continue
                 word_count = len(chunk.split())
                 # Add check for max words if SparkTTS has limitations
                 # if word_count > MAX_WORDS_PER_CHUNK: ... log warning/split further ...
                 if word_count > 75: # Example warning threshold
                      logging.warning(f"[ReqID: {request_id}] Splitter produced a potentially long chunk: {word_count} words.")
                 final_chunks.append(chunk)
            text_chunks = final_chunks

            if not text_chunks:
                 if target_text.strip(): # Original text was not empty
                      logging.warning(f"[ReqID: {request_id}] Text splitting yielded zero non-empty chunks, using original text as one chunk.")
                      text_chunks = [target_text.strip()]
                 else:
                      logging.info(f"[ReqID: {request_id}] Input text was empty or resulted in empty chunks. No audio.")
                      return # Yield nothing, client receives empty stream

            logging.info(f"[ReqID: {request_id}] Split text into {len(text_chunks)} chunks for processing.")

        except Exception as e:
            logging.error(f"[ReqID: {request_id}] Error during text splitting: {e}", exc_info=True)
            # Fallback: use the original text if splitting fails and text is not empty
            if target_text.strip():
                text_chunks = [target_text.strip()]
                logging.warning(f"[ReqID: {request_id}] Using original text due to splitting error.")
            else:
                logging.info(f"[ReqID: {request_id}] Empty text and splitting error. No audio.")
                return # Yield nothing

        # --- 3. Define PCM NumPy Producer ---
        # This generator handles interaction with Triton to get NumPy arrays
        pcm_np_gen = self.pcm_numpy_producer(
            text_chunks, waveform, reference_text, request_id, context
        )

        # --- 4. Process PCM Stream based on requested format ---
        try:
            if output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_OPUS_WEBM:
                logging.info(f"[ReqID: {request_id}] Starting Opus/WebM encoding process with pre-smoothing.")
                # This consumes the numpy generator, cross-fades, encodes, and yields Opus chunks
                yield from self._synthesize_opus_webm_smoothed(pcm_np_gen, request_id, context)

            else: # Default or explicit WAV_PCM_FLOAT32
                logging.info(f"[ReqID: {request_id}] Starting raw PCM Float32 byte streaming.")
                # Define simple inline generator to convert NumPy back to bytes for the PCM path
                def _np_to_bytes_gen(np_gen):
                    try:
                        for arr in np_gen:
                            if arr is not None and arr.size > 0:
                                yield arr.tobytes()
                    except Exception as e:
                        logging.error(f"[ReqID: {request_id}] Error in np_to_bytes generator: {e}", exc_info=True)
                        raise # Propagate error to _synthesize_pcm_float32

                # This consumes the numpy generator (via _np_to_bytes_gen) and yields raw PCM bytes
                yield from self._synthesize_pcm_float32(_np_to_bytes_gen(pcm_np_gen), request_id, context)

            logging.info(f"[ReqID: {request_id}] SynthesizeSpeech request processing completed successfully.")

        except Exception as e:
             # Catch errors raised from the producer chain or the format-specific handlers
             # Context should ideally be aborted by the function that failed. Log here just in case.
             if context.is_active():
                logging.error(f"[ReqID: {request_id}] Unhandled exception at SynthesizeSpeech top level: {e}", exc_info=True)
                try:
                    # Set generic internal error if not already set
                    context.abort(grpc.StatusCode.INTERNAL, f"Request processing failed unexpectedly: {e}")
                except Exception as abort_e:
                    # This might happen if context becomes inactive between check and abort
                    logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")
             else:
                 # Log that processing terminated due to an earlier handled error
                 logging.info(f"[ReqID: {request_id}] Request processing terminated due to an earlier reported error.")


# --- Server main execution ---
def serve(port, triton_url, model_name, templates_file, verbose, gateway_crossfade):
    # Optional: Check for FFmpeg before starting
    try:
        ffmpeg_version_proc = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True, timeout=5)
        logging.info(f"FFmpeg found: {ffmpeg_version_proc.stdout.splitlines()[0]}")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.error(f"FFmpeg check failed: {e}. Opus/WebM encoding will likely fail.")
        # Consider exiting if Opus is critical: sys.exit("FFmpeg not found or failed check.")

    # Configure server (consider adjusting max_workers based on load/cores)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Create servicer instance
    servicer = AudioGatewayServicer(
        triton_url,
        model_name,
        templates_file,
        verbose,
        crossfade_duration=gateway_crossfade # Pass crossfade duration
    )

    # Add servicer if templates loaded correctly
    if servicer.templates:
        gateway_pb2_grpc.add_AudioGatewayServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        logging.info(f"Starting gRPC Gateway Server on port {port}...")
        logging.info(f"Connecting to Triton at: {triton_url}")
        logging.info(f"Using Triton model: {model_name}")
        logging.info(f"Using templates from: {templates_file}")
        server.start()
        logging.info("Server started. Waiting for requests...")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logging.info("Server stopping...")
            server.stop(0)
            logging.info("Server stopped.")
    else:
        logging.error("Server startup failed due to template loading issues.")
        sys.exit(1)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server with Text Splitting, Format Selection, and Opus Smoothing")
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    parser.add_argument("--triton-url", type=str, default="inference:8001", help="URL of the Triton Inference Server (gRPC)")
    parser.add_argument("--model-name", type=str, default="spark_tts_decoupled", help="Name of the TTS model on Triton")
    parser.add_argument("--templates-file", type=str, default="templates/templates.json", help="Path to voice templates JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for Triton client")
    parser.add_argument("--gateway-crossfade", type=float, default=0.1, help="Overlap duration in seconds for gateway-side cross-fading before Opus encoding (e.g., 0.1 for 100ms)")


    args = parser.parse_args()

    # Start the server
    serve(
        args.port,
        args.triton_url,
        args.model_name,
        args.templates_file,
        args.verbose,
        args.gateway_crossfade
    )