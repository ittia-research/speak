# server_edge_tts.py
import argparse
import logging
import os
import queue
import uuid
import sys
import random
from concurrent import futures
import time
import subprocess
import threading
import asyncio

import grpc
import ffmpeg

try:
    import edge_tts
    from edge_tts import VoicesManager
except ImportError:
    logging.error("edge-tts is not installed. Please run: pip install edge-tts")
    sys.exit(1)

try:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document
except ImportError:
    logging.warning("llama-index-core not found. Text splitting will not occur.")
    SentenceSplitter = None
    Document = None

import gateway_pb2
import gateway_pb2_grpc

# --- Constants ---
OPUS_BITRATE = '64k'
# *** Define the output sample rate and channels we will force for Opus ***
OUTPUT_OPUS_SAMPLE_RATE = 16000
OUTPUT_OPUS_CHANNELS = 1
# *** Special value for default voice argument to trigger random ENGLISH selection PER REQUEST ***
RANDOM_ENGLISH_VOICE = "random-english"
# Fallback voice if random selection fails or no English voices exist
FALLBACK_ENGLISH_VOICE = "en-US-AriaNeural"


# --- FFmpeg Encoding/Decoding Helpers ---

def start_ffmpeg_encoder(input_format, output_format_options, request_id):
    """Starts FFmpeg process for converting audio streams."""
    input_args = {'format': input_format}
    logging.debug(f"[ReqID: {request_id}] Starting FFmpeg: Input={input_format}, Output Opts={output_format_options}")
    try:
        process = (
            ffmpeg
            .input('pipe:', **input_args)
            .output('pipe:', **output_format_options)
            .global_args('-hide_banner', '-loglevel', 'warning')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        logging.debug(f"[ReqID: {request_id}] FFmpeg process started (PID: {process.pid}) for {input_format} -> {output_format_options.get('format', 'unknown')}")
        return process
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please ensure FFmpeg is installed and in PATH.")
        raise
    except Exception as e:
        logging.error(f"[ReqID: {request_id}] Failed to start FFmpeg process: {e}", exc_info=True)
        raise

def read_ffmpeg_output(ffmpeg_process, output_queue, request_id):
    """Reads stdout from FFmpeg process and puts chunks into a queue."""
    # (No changes needed here)
    logging.debug(f"[ReqID: {request_id}] FFmpeg output reader thread started.")
    try:
        while True:
            chunk = ffmpeg_process.stdout.read(4096)
            if not chunk:
                logging.debug(f"[ReqID: {request_id}] FFmpeg stdout EOF reached.")
                break
            output_queue.put(chunk)
    except Exception as e:
        logging.error(f"[ReqID: {request_id}] Error reading FFmpeg stdout: {e}", exc_info=True)
        output_queue.put(e)
    finally:
        output_queue.put(None)
        logging.debug(f"[ReqID: {request_id}] FFmpeg output reader thread finished.")

# --- Edge TTS Helper ---

async def _edge_tts_streamer(text: str, voice: str, rate: str, volume: str, pitch: str, output_queue: queue.Queue, request_id: str):
    """Async function to stream TTS audio chunks and put them in a queue."""
    # (No changes needed here)
    logging.debug(f"[ReqID: {request_id}] Starting edge-tts stream generation for voice: {voice}")
    communicate = None
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_queue.put(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass
        logging.debug(f"[ReqID: {request_id}] Finished edge-tts stream generation.")
    except Exception as e:
        logging.error(f"[ReqID: {request_id}] Error during edge-tts streaming: {e}", exc_info=True)
        output_queue.put(e)
    finally:
        output_queue.put(None)

def run_edge_tts_in_thread(text: str, voice: str, rate_str: str, volume_str: str, pitch_str: str, output_queue: queue.Queue, request_id: str):
    """Runs the async streamer in a separate thread."""
    # (No changes needed here)
    logging.debug(f"[ReqID: {request_id}] Starting thread for edge-tts async task.")
    try:
        asyncio.run(_edge_tts_streamer(text, voice, rate_str, volume_str, pitch_str, output_queue, request_id))
        logging.debug(f"[ReqID: {request_id}] Edge-tts async task thread finished.")
    except Exception as e:
         logging.error(f"[ReqID: {request_id}] Error running edge-tts async task in thread: {e}", exc_info=True)
         output_queue.put(e)
         output_queue.put(None)


# --- Gateway Servicer Implementation ---
class AudioGatewayServicer(gateway_pb2_grpc.AudioGatewayServicer):
    def __init__(self, default_voice_setting, verbose=False):
        self.verbose = verbose
        self.available_voices = []
        self.default_voice_setting = default_voice_setting # Store the command line setting
        self.english_voices_list = [] # Pre-filtered list for random selection

        try:
            logging.info("Fetching available edge-tts voices during init...")
            voices_manager = asyncio.run(VoicesManager.create())
            self.available_voices = voices_manager.voices
            logging.info(f"Found {len(self.available_voices)} total voices.")

            # Pre-filter English voices for random selection mode
            self.english_voices_list = [
                v for v in self.available_voices if v['ShortName'].startswith('en-')
            ]
            logging.info(f"Found {len(self.english_voices_list)} English voices.")
            if not self.english_voices_list:
                 logging.warning("No English voices found in edge-tts list!")

            # Validate the default voice setting ONLY if it's NOT random mode
            if self.default_voice_setting != RANDOM_ENGLISH_VOICE:
                found = any(v['ShortName'] == self.default_voice_setting for v in self.available_voices)
                if not found:
                    logging.error(f"Specified default voice '{self.default_voice_setting}' not found in available edge-tts voices (checked ShortName).")
                    if self.available_voices:
                        available_short_names = [v['ShortName'] for v in self.available_voices[:10]]
                        logging.info(f"Some available ShortNames: {', '.join(available_short_names)}...")
                    sys.exit(1)
                else:
                    logging.info(f"Server configured to use FIXED voice: '{self.default_voice_setting}' for all requests.")
            else:
                 # Random mode: Check if we have English voices to choose from
                 if not self.english_voices_list:
                      logging.error(f"Random English voice mode selected, but no English voices found. Server cannot fulfill requests.")
                      # Decide whether to exit or maybe try the global fallback later per-request
                      # Exiting is safer to indicate a configuration problem.
                      sys.exit(1)
                 logging.info(f"Server configured for RANDOM English voice selection per request.")


        except Exception as e:
            logging.error(f"Failed during voice initialization: {e}", exc_info=True)
            sys.exit(1)

    def edge_tts_producer(self, text_chunks, voice, request_id, context):
        """
        Generator yields MP3 chunks from edge-tts for all text chunks using the specified voice.
        """
        # (No changes needed here - accepts the voice to use)
        logging.info(f"[ReqID: {request_id}] Starting Edge-TTS MP3 production for {len(text_chunks)} chunks using voice '{voice}'.")
        rate = "+0%"
        volume = "+0%"
        pitch = "+0Hz"

        for i, text_chunk in enumerate(text_chunks):
            chunk_req_id = f"{request_id}-C{i+1}"
            logging.info(f"[ReqID: {request_id}] Requesting MP3 for text chunk {i+1}/{len(text_chunks)} (EdgeTTSReqID: {chunk_req_id})")
            if not text_chunk or text_chunk.isspace():
                logging.warning(f"[ReqID: {request_id}] Skipping empty text chunk {i+1}.")
                continue

            mp3_chunk_queue = queue.Queue()
            tts_thread = threading.Thread(
                target=run_edge_tts_in_thread,
                args=(text_chunk, voice, rate, volume, pitch, mp3_chunk_queue, chunk_req_id),
                daemon=True)
            tts_thread.start()

            chunk_count = 0
            while True:
                if not context.is_active():
                    logging.warning(f"[ReqID: {request_id}] gRPC context cancelled during MP3 production for chunk {i+1}. Stopping.")
                    # Signal TTS thread to stop? Difficult with edge-tts's current structure. Rely on context check below.
                    break
                try:
                    mp3_chunk = mp3_chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    if not tts_thread.is_alive():
                        logging.debug(f"[ReqID: {request_id}] TTS thread finished for chunk {i+1} (queue empty).")
                        break
                    continue

                if isinstance(mp3_chunk, Exception):
                    logging.error(f"[ReqID: {request_id}] Error received from edge-tts thread for chunk {i+1}: {mp3_chunk}")
                    if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Edge-TTS synthesis failed for chunk {i+1}: {mp3_chunk}")
                    raise RuntimeError(f"Edge-TTS synthesis failed for chunk {i+1}") from mp3_chunk
                elif mp3_chunk is None:
                    logging.debug(f"[ReqID: {request_id}] Received None sentinel for chunk {i+1}, stream finished.")
                    break
                else:
                    yield mp3_chunk
                    chunk_count += 1

            # Wait for thread join, check context again
            if context.is_active():
                logging.debug(f"[ReqID: {request_id}] Waiting for TTS thread to join for chunk {i+1}...")
                tts_thread.join(timeout=5.0)
                if tts_thread.is_alive():
                    logging.warning(f"[ReqID: {request_id}] TTS thread for chunk {i+1} did not join cleanly.")
                else:
                    logging.debug(f"[ReqID: {request_id}] TTS thread joined successfully for chunk {i+1}.")
            else:
                 logging.warning(f"[ReqID: {request_id}] Context cancelled after processing chunk {i+1}. Stopping.")
                 # If thread is still running, it's daemon so shouldn't block shutdown, but it's not ideal.
                 break

        logging.info(f"[ReqID: {request_id}] Finished producing MP3 data for all processed text chunks.")


    def _synthesize_via_ffmpeg(self, input_producer, input_format, output_format_options, request_id, context):
        """
        Generic helper to pipe input producer data through FFmpeg and yield output chunks.
        """
        # (No changes needed in the mechanics here, only in the options passed to it)
        ffmpeg_process = None
        reader_thread = None
        output_queue = queue.Queue()
        total_encoded_chunks_yielded = 0
        input_fed_successfully = True

        try:
            ffmpeg_process = start_ffmpeg_encoder(input_format, output_format_options, request_id)
            reader_thread = threading.Thread(target=read_ffmpeg_output, args=(ffmpeg_process, output_queue, request_id))
            reader_thread.daemon = True
            reader_thread.start()

            input_chunks_fed = 0
            for input_chunk_bytes in input_producer:
                if not context.is_active():
                    logging.warning(f"[ReqID: {request_id}] Context cancelled while feeding FFmpeg. Stopping input.")
                    input_fed_successfully = False
                    break
                if input_chunk_bytes:
                    try:
                        ffmpeg_process.stdin.write(input_chunk_bytes)
                        input_chunks_fed += 1
                    except (OSError, BrokenPipeError) as e:
                        logging.error(f"[ReqID: {request_id}] Error writing {input_format} to FFmpeg stdin: {e}. FFmpeg might have exited.")
                        input_fed_successfully = False
                        break

                try:
                    while not output_queue.empty():
                        encoded_chunk = output_queue.get_nowait()
                        if isinstance(encoded_chunk, Exception):
                            logging.error(f"[ReqID: {request_id}] Error received from FFmpeg reader thread.")
                            raise encoded_chunk
                        if encoded_chunk is None:
                            logging.warning(f"[ReqID: {request_id}] Reader thread signaled completion early.")
                            output_queue.put(None)
                            break
                        yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)
                        total_encoded_chunks_yielded += 1
                except queue.Empty:
                    pass

            if input_fed_successfully:
                logging.info(f"[ReqID: {request_id}] Finished feeding {input_chunks_fed} {input_format} chunks to FFmpeg.")
            else:
                 logging.warning(f"[ReqID: {request_id}] Stopped feeding FFmpeg after {input_chunks_fed} chunks due to error or cancellation.")

            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed and input_fed_successfully:
                logging.debug(f"[ReqID: {request_id}] Closing FFmpeg stdin.")
                try:
                    ffmpeg_process.stdin.close()
                except OSError as e:
                    logging.warning(f"[ReqID: {request_id}] Non-critical error closing FFmpeg stdin: {e}")

            logging.debug(f"[ReqID: {request_id}] Reading remaining encoded output from FFmpeg...")
            while True:
                try:
                    encoded_chunk = output_queue.get(timeout=5.0)
                except queue.Empty:
                    logging.warning(f"[ReqID: {request_id}] Timeout waiting for final FFmpeg output.")
                    if reader_thread and reader_thread.is_alive():
                        logging.error(f"[ReqID: {request_id}] FFmpeg reader thread is still alive but queue is empty. Aborting stream.")
                        raise TimeoutError("FFmpeg reader thread stalled.")
                    else:
                         logging.warning(f"[ReqID: {request_id}] FFmpeg reader thread finished, but queue empty. Likely normal end.")
                    break

                if isinstance(encoded_chunk, Exception):
                     logging.error(f"[ReqID: {request_id}] Error received from FFmpeg reader thread during final read.")
                     raise encoded_chunk
                if encoded_chunk is None:
                    logging.debug(f"[ReqID: {request_id}] Reached end of encoded stream from queue.")
                    break
                yield gateway_pb2.AudioChunk(audio_data=encoded_chunk)
                total_encoded_chunks_yielded += 1

            logging.info(f"[ReqID: {request_id}] FFmpeg processing finished. Yielded {total_encoded_chunks_yielded} output chunks.")

        except Exception as e:
             logging.error(f"[ReqID: {request_id}] Error during FFmpeg processing stream: {e}", exc_info=True)
             if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, f"Audio processing/conversion failed: {e}")
             # Fall through to finally block for cleanup

        finally:
             # --- Cleanup FFmpeg process and reader thread ---
             # (Cleanup logic remains the same)
             if reader_thread and reader_thread.is_alive():
                 logging.debug(f"[ReqID: {request_id}] Waiting for FFmpeg reader thread to join...")
                 reader_thread.join(timeout=2.0)
                 if reader_thread.is_alive():
                     logging.warning(f"[ReqID: {request_id}] FFmpeg reader thread did not join cleanly.")

             if ffmpeg_process:
                 logging.debug(f"[ReqID: {request_id}] Cleaning up FFmpeg process...")
                 return_code = None
                 stderr_output = ""
                 try:
                    # ... (rest of cleanup logic is identical to previous versions) ...
                    if ffmpeg_process.stdout and not ffmpeg_process.stdout.closed: ffmpeg_process.stdout.close()
                    if ffmpeg_process.stderr and not ffmpeg_process.stderr.closed:
                       try:
                           stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='replace')
                       except Exception as stderr_e:
                           logging.warning(f"[ReqID: {request_id}] Error reading FFmpeg stderr: {stderr_e}")
                       finally:
                           if not ffmpeg_process.stderr.closed: ffmpeg_process.stderr.close()
                    return_code = ffmpeg_process.wait(timeout=5.0)
                    logging.info(f"[ReqID: {request_id}] FFmpeg process exited with code {return_code}.")
                    if return_code != 0: logging.error(f"[ReqID: {request_id}] FFmpeg process failed (code {return_code}). Stderr:\n{stderr_output if stderr_output else 'N/A'}")
                    elif stderr_output: logging.warning(f"[ReqID: {request_id}] FFmpeg stderr output (exit code 0):\n{stderr_output}")
                 except subprocess.TimeoutExpired:
                    logging.error(f"[ReqID: {request_id}] FFmpeg process timed out on wait. Terminating.")
                    ffmpeg_process.terminate(); time.sleep(0.5)
                    if ffmpeg_process.poll() is None: logging.warning(f"[ReqID: {request_id}] FFmpeg kill."); ffmpeg_process.kill()
                    if context.is_active(): context.abort(grpc.StatusCode.INTERNAL, "FFmpeg process timed out during cleanup.")
                 except Exception as e: logging.error(f"[ReqID: {request_id}] Error during FFmpeg cleanup: {e}", exc_info=True)
                 finally:
                    if ffmpeg_process.poll() is None:
                        try: logging.warning(f"[ReqID: {request_id}] FFmpeg killing leftover process."); ffmpeg_process.kill()
                        except Exception as kill_e: logging.error(f"[ReqID: {request_id}] Error killing FFmpeg process: {kill_e}")


    # --- Main RPC Implementation ---
    def SynthesizeSpeech(self, request, context):
        """
        Handles request: selects voice (fixed or random English), splits text,
        calls edge-tts for MP3, converts to Opus/WebM (16kHz mono) via FFmpeg, streams audio.
        """
        request_id = str(uuid.uuid4())
        output_format = request.output_format

        # --- Determine requested format and log (Still forces Opus) ---
        final_output_format = gateway_pb2.OutputFormat.OUTPUT_FORMAT_OPUS_WEBM
        format_str = "OPUS_WEBM"
        if output_format != gateway_pb2.OutputFormat.OUTPUT_FORMAT_OPUS_WEBM and \
           output_format != gateway_pb2.OutputFormat.OUTPUT_FORMAT_UNSPECIFIED:
            logging.warning(f"[ReqID: {request_id}] Received unsupported output format request ({output_format}). Forcing OPUS_WEBM.")
            format_str = "OPUS_WEBM (Forced)"
        elif output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_UNSPECIFIED:
             format_str = "OPUS_WEBM (Default)"

        logging.info(f"[ReqID: {request_id}] Received SynthesizeSpeech request. Outputting: {format_str}. Text len: {len(request.target_text)}")

        # --- 1. Select Voice FOR THIS REQUEST ---
        voice_to_use = None
        if self.default_voice_setting == RANDOM_ENGLISH_VOICE:
            if self.english_voices_list:
                chosen_voice_info = random.choice(self.english_voices_list)
                voice_to_use = chosen_voice_info['ShortName']
                logging.info(f"[ReqID: {request_id}] Randomly selected English voice: '{voice_to_use}'")
            else:
                # This case should ideally be caught at startup, but handle defensively
                logging.error(f"[ReqID: {request_id}] No English voices available for random selection!")
                # Try using the hardcoded fallback if it exists
                if any(v['ShortName'] == FALLBACK_ENGLISH_VOICE for v in self.available_voices):
                     voice_to_use = FALLBACK_ENGLISH_VOICE
                     logging.warning(f"[ReqID: {request_id}] Using fallback voice '{voice_to_use}'")
                else:
                     logging.error(f"[ReqID: {request_id}] Fallback voice '{FALLBACK_ENGLISH_VOICE}' not found either. Aborting request.")
                     context.abort(grpc.StatusCode.FAILED_PRECONDITION, "No suitable English voices available.")
                     return
        else:
            # Use the fixed voice specified via command line
            voice_to_use = self.default_voice_setting
            # Log only if verbose or maybe less frequently, as it's fixed
            logging.debug(f"[ReqID: {request_id}] Using fixed voice: '{voice_to_use}'")

        if not voice_to_use: # Should not happen if logic above is correct, but check
             logging.error(f"[ReqID: {request_id}] Failed to determine a voice to use. Aborting.")
             context.abort(grpc.StatusCode.INTERNAL, "Voice selection failed.")
             return

        # --- 2. Split Target Text (Optional but kept) ---
        # (Text splitting logic remains the same)
        target_text = request.target_text
        text_chunks = []
        if SentenceSplitter and Document:
            try:
                # Adjust chunk size if needed for different voices/languages potentially
                splitter = SentenceSplitter(chunk_size=250, chunk_overlap=0)
                document = Document(text=target_text)
                nodes = splitter.get_nodes_from_documents([document])
                raw_chunks = [node.get_content().strip() for node in nodes]
                text_chunks = [chunk for chunk in raw_chunks if chunk]
                if not text_chunks and target_text.strip():
                    text_chunks = [target_text.strip()]
                logging.info(f"[ReqID: {request_id}] Split text into {len(text_chunks)} chunks.")
            except Exception as e:
                logging.error(f"[ReqID: {request_id}] Error during text splitting: {e}", exc_info=True)
                if target_text.strip(): text_chunks = [target_text.strip()]
                else: logging.info(f"[ReqID: {request_id}] Empty text/splitting error."); return
        elif target_text.strip():
             logging.info(f"[ReqID: {request_id}] llama_index not found/failed. Using full text chunk.")
             text_chunks = [target_text.strip()]
        else:
             logging.info(f"[ReqID: {request_id}] Input text empty."); return
        if not text_chunks:
             logging.info(f"[ReqID: {request_id}] No valid text chunks."); return

        # --- 3. Define Edge-TTS MP3 Producer ---
        mp3_producer = self.edge_tts_producer(
            text_chunks, voice_to_use, request_id, context
        )

        # --- 4. Process MP3 Stream to Opus/WebM via FFmpeg ---
        try:
            logging.info(f"[ReqID: {request_id}] Starting Opus/WebM encoding from MP3 stream (forcing {OUTPUT_OPUS_SAMPLE_RATE}Hz / {OUTPUT_OPUS_CHANNELS}ch).")
            # *** Noise Fix: Re-add explicit 'ac' and 'ar' to force output format ***
            output_opts = {
                'format': 'webm',
                'acodec': 'libopus',
                'ac': OUTPUT_OPUS_CHANNELS,         # Force channels (e.g., 1 for mono)
                'ar': str(OUTPUT_OPUS_SAMPLE_RATE), # Force sample rate (e.g., 16000)
                'vbr': 'on',
                'b:a': OPUS_BITRATE
            }
            yield from self._synthesize_via_ffmpeg(mp3_producer, 'mp3', output_opts, request_id, context)

            logging.info(f"[ReqID: {request_id}] SynthesizeSpeech request processing completed successfully.")

        except Exception as e:
             # (Error handling remains the same)
             if context.is_active():
                logging.error(f"[ReqID: {request_id}] Unhandled exception at SynthesizeSpeech top level: {e}", exc_info=True)
                try: context.abort(grpc.StatusCode.INTERNAL, f"Request processing failed unexpectedly: {e}")
                except Exception as abort_e: logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")
             else:
                 logging.info(f"[ReqID: {request_id}] Request processing terminated due to an earlier reported error.")


# --- Server main execution ---
def serve(port, default_voice_arg, verbose):
    # (FFmpeg check remains the same)
    try:
        ffmpeg_version_proc = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True, timeout=5)
        logging.info(f"FFmpeg found: {ffmpeg_version_proc.stdout.splitlines()[0]}")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.warning(f"FFmpeg check failed: {e}. Audio conversion might fail if FFmpeg is not runnable.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    try:
        # Pass the command-line arg (which might be RANDOM_ENGLISH_VOICE)
        servicer = AudioGatewayServicer(
            default_voice_setting=default_voice_arg, # Pass the raw setting
            verbose=verbose,
        )
    except SystemExit:
         logging.error("Server startup failed during servicer initialization (e.g., voice validation).")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during servicer initialization: {e}", exc_info=True)
        sys.exit(1)

    gateway_pb2_grpc.add_AudioGatewayServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    logging.info(f"Starting gRPC Gateway Server on port {port}...")
    # Log the configured *mode* (fixed or random)
    if servicer.default_voice_setting == RANDOM_ENGLISH_VOICE:
         logging.info(f"Server configured for RANDOM English voice selection per request.")
    else:
         logging.info(f"Server configured to use FIXED voice: '{servicer.default_voice_setting}' for all requests.")
    logging.info(f"Output format: OPUS_WEBM @ {OUTPUT_OPUS_SAMPLE_RATE}Hz, {OUTPUT_OPUS_CHANNELS}ch (forced)")
    if verbose:
         logging.info("Verbose logging enabled.")

    server.start()
    logging.info("Server started. Waiting for requests...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopping due to keyboard interrupt...")
        server.stop(0)
        logging.info("Server stopped.")
    except Exception as e:
         logging.error(f"Server terminated unexpectedly: {e}", exc_info=True)
         server.stop(1)
         sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server using Edge-TTS (Opus/WebM output only)")
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    # *** Updated default voice argument help text ***
    parser.add_argument("--default-voice", type=str, default=RANDOM_ENGLISH_VOICE,
                        help=f"Default Edge-TTS voice setting. Use a specific ShortName (e.g., en-GB-SoniaNeural) "
                             f"for fixed voice, or '{RANDOM_ENGLISH_VOICE}' (default) to pick a random English voice per request.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    serve(
        args.port,
        args.default_voice, # Pass the argument value directly
        args.verbose
    )