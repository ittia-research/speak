# server_edge_tts.py
import argparse
import logging
import os
import queue
import uuid
import sys
import random
from concurrent import futures
import threading
import asyncio
import grpc

import edge_tts
from edge_tts import VoicesManager

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

import gateway_pb2
import gateway_pb2_grpc


# Foe the default voice argument
RANDOM_ENGLISH_VOICE = "random-english"
# Fallback voice if random selection fails or no English voices exist
FALLBACK_ENGLISH_VOICE = "en-US-AriaNeural"


# --- Edge TTS Helper ---

async def _edge_tts_streamer(text: str, voice: str, rate: str, volume: str, pitch: str, output_queue: queue.Queue, request_id: str):
    """Async function to stream TTS audio chunks and put them in a queue."""
    logging.debug(f"[ReqID: {request_id}] Starting edge-tts stream generation for voice: {voice}")
    communicate = None
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_queue.put(chunk["data"]) # MP3 data
            elif chunk["type"] == "WordBoundary":
                pass # Ignore word boundaries for now
        logging.debug(f"[ReqID: {request_id}] Finished edge-tts stream generation.")
    except Exception as e:
        logging.error(f"[ReqID: {request_id}] Error during edge-tts streaming: {e}", exc_info=True)
        output_queue.put(e) # Signal error
    finally:
        output_queue.put(None) # Signal completion

def run_edge_tts_in_thread(text: str, voice: str, rate_str: str, volume_str: str, pitch_str: str, output_queue: queue.Queue, request_id: str):
    """Runs the async streamer in a separate thread."""
    logging.debug(f"[ReqID: {request_id}] Starting thread for edge-tts async task.")
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_edge_tts_streamer(text, voice, rate_str, volume_str, pitch_str, output_queue, request_id))
        loop.close()
        logging.debug(f"[ReqID: {request_id}] Edge-tts async task thread finished.")
    except Exception as e:
         logging.error(f"[ReqID: {request_id}] Error running edge-tts async task in thread: {e}", exc_info=True)
         # Ensure queue gets signaled even on error
         output_queue.put(e)
         output_queue.put(None)


# --- Gateway Servicer ---
class AudioGatewayServicer(gateway_pb2_grpc.AudioGatewayServicer):
    def __init__(self, default_voice_setting, verbose=False):
        self.verbose = verbose
        self.available_voices = []
        self.default_voice_setting = default_voice_setting
        self.english_voices_list = []

        # --- Voice loading ---
        try:
            logging.info("Fetching available edge-tts voices during init...")
            voices_manager = asyncio.run(VoicesManager.create())
            self.available_voices = voices_manager.voices
            logging.info(f"Found {len(self.available_voices)} total voices.")

            self.english_voices_list = [
                v for v in self.available_voices if v['ShortName'].startswith('en-')
            ]
            logging.info(f"Found {len(self.english_voices_list)} English voices.")
            if not self.english_voices_list:
                 logging.warning("No English voices found in edge-tts list!")

            if self.default_voice_setting != RANDOM_ENGLISH_VOICE:
                found = any(v['ShortName'] == self.default_voice_setting for v in self.available_voices)
                if not found:
                    logging.error(f"Specified default voice '{self.default_voice_setting}' not found in available edge-tts voices (checked ShortName).")
                    sys.exit(1)
                else:
                    logging.info(f"Server configured to use FIXED voice: '{self.default_voice_setting}' for all requests.")
            else:
                 if not self.english_voices_list:
                      logging.error(f"Random English voice mode selected, but no English voices found. Server cannot fulfill requests.")
                      sys.exit(1)
                 logging.info(f"Server configured for RANDOM English voice selection per request.")

        except Exception as e:
            logging.error(f"Failed during voice initialization: {e}", exc_info=True)
            sys.exit(1)

    def edge_tts_producer(self, text_chunks, voice, request_id, context):
        """
        Generator yields MP3 chunks directly from edge-tts for all text chunks
        using the specified voice.
        """
        logging.info(f"[ReqID: {request_id}] Starting Edge-TTS MP3 production for {len(text_chunks)} chunks using voice '{voice}'.")
        rate = "+0%"   # Default rate
        volume = "+0%" # Default volume
        pitch = "+0Hz" # Default pitch

        total_yielded_chunks = 0
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

            chunk_count_for_this_text = 0
            while True:
                if not context.is_active():
                    logging.warning(f"[ReqID: {request_id}] gRPC context cancelled during MP3 production for chunk {i+1}. Stopping.")
                    # Best effort: rely on thread being daemon and checks below
                    break
                try:
                    # Use a slightly longer timeout to avoid busy-waiting, but short enough for responsiveness
                    mp3_chunk = mp3_chunk_queue.get(timeout=0.2)
                except queue.Empty:
                    # Check if thread finished while we were waiting
                    if not tts_thread.is_alive():
                        logging.debug(f"[ReqID: {request_id}] TTS thread finished for chunk {i+1} (queue empty check).")
                        break
                    # Thread still running, queue just empty for now, continue waiting
                    continue

                if isinstance(mp3_chunk, Exception):
                    logging.error(f"[ReqID: {request_id}] Error received from edge-tts thread for chunk {i+1}: {mp3_chunk}")
                    if context.is_active():
                        # Abort the gRPC stream if an error occurs during generation
                        context.abort(grpc.StatusCode.INTERNAL, f"Edge-TTS synthesis failed for chunk {i+1}: {mp3_chunk}")
                    # Reraise to stop the producer
                    raise RuntimeError(f"Edge-TTS synthesis failed for chunk {i+1}") from mp3_chunk
                elif mp3_chunk is None:
                    logging.debug(f"[ReqID: {request_id}] Received None sentinel for chunk {i+1}, stream for this text chunk finished.")
                    break # Move to the next text chunk
                else:
                    # Yield the raw MP3 bytes directly
                    yield mp3_chunk
                    chunk_count_for_this_text += 1
                    total_yielded_chunks += 1

            # After processing a text chunk's stream, wait for the thread to join cleanly
            if context.is_active():
                logging.debug(f"[ReqID: {request_id}] Waiting for TTS thread to join for chunk {i+1}...")
                tts_thread.join(timeout=5.0) # Give it some time to finish cleanup
                if tts_thread.is_alive():
                    logging.warning(f"[ReqID: {request_id}] TTS thread for chunk {i+1} did not join cleanly.")
                else:
                    logging.debug(f"[ReqID: {request_id}] TTS thread joined successfully for chunk {i+1}.")
            else:
                 logging.warning(f"[ReqID: {request_id}] Context cancelled after processing chunk {i+1}. Stopping outer loop.")
                 break # Exit the loop over text_chunks

        logging.info(f"[ReqID: {request_id}] Finished producing MP3 data. Total yielded chunks: {total_yielded_chunks}.")


    # --- Main RPC Implementation ---
    def SynthesizeSpeech(self, request, context):
        """
        Handles request: selects voice (fixed or random English), splits text,
        calls edge-tts for MP3, and streams MP3 audio chunks directly.
        """
        request_id = str(uuid.uuid4())
        output_format = request.output_format

        # --- Determine requested format and log (Only MP3 is supported) ---
        format_str = "MP3"
        if output_format != gateway_pb2.OutputFormat.OUTPUT_FORMAT_MP3 and \
           output_format != gateway_pb2.OutputFormat.OUTPUT_FORMAT_UNSPECIFIED:
            logging.warning(f"[ReqID: {request_id}] Received unsupported output format request ({output_format}). Forcing MP3.")
            format_str = "MP3 (Forced)"
        elif output_format == gateway_pb2.OutputFormat.OUTPUT_FORMAT_UNSPECIFIED:
             format_str = "MP3 (Default)"

        logging.info(f"[ReqID: {request_id}] Received SynthesizeSpeech request. Outputting: {format_str}. Text len: {len(request.target_text)}")

        # --- Select Voice FOR THIS REQUEST ---
        voice_to_use = None
        if self.default_voice_setting == RANDOM_ENGLISH_VOICE:
            if self.english_voices_list:
                chosen_voice_info = random.choice(self.english_voices_list)
                voice_to_use = chosen_voice_info['ShortName']
                logging.info(f"[ReqID: {request_id}] Randomly selected English voice: '{voice_to_use}'")
            else:
                logging.error(f"[ReqID: {request_id}] No English voices available for random selection!")
                if any(v['ShortName'] == FALLBACK_ENGLISH_VOICE for v in self.available_voices):
                     voice_to_use = FALLBACK_ENGLISH_VOICE
                     logging.warning(f"[ReqID: {request_id}] Using fallback voice '{voice_to_use}'")
                else:
                     logging.error(f"[ReqID: {request_id}] Fallback voice '{FALLBACK_ENGLISH_VOICE}' not found. Aborting.")
                     context.abort(grpc.StatusCode.FAILED_PRECONDITION, "No suitable English voices available.")
                     return
        else:
            voice_to_use = self.default_voice_setting
            logging.debug(f"[ReqID: {request_id}] Using fixed voice: '{voice_to_use}'")

        if not voice_to_use:
             logging.error(f"[ReqID: {request_id}] Failed to determine a voice to use. Aborting.")
             context.abort(grpc.StatusCode.INTERNAL, "Voice selection failed.")
             return

        # --- Split Target Text ---
        target_text = request.target_text
        text_chunks = []
        chunk_size = 250
        if SentenceSplitter and Document:
            try:
                splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
                document = Document(text=target_text)
                nodes = splitter.get_nodes_from_documents([document])
                raw_chunks = [f"{node.get_content().strip()} " for node in nodes] # Add one space at the end
                text_chunks = [chunk for chunk in raw_chunks if chunk] # Filter empty strings
                if not text_chunks and target_text.strip(): # Handle case where splitter returns nothing but input wasn't empty
                    text_chunks = [target_text.strip()]
                logging.info(f"[ReqID: {request_id}] Split text into {len(text_chunks)} chunks (target size ~{chunk_size}).")
            except Exception as e:
                logging.error(f"[ReqID: {request_id}] Error during text splitting: {e}", exc_info=True)
                # Fallback to using the whole text if splitting fails
                if target_text.strip(): text_chunks = [target_text.strip()]
                else: logging.info(f"[ReqID: {request_id}] Empty text provided."); return
        elif target_text.strip():
             logging.info(f"[ReqID: {request_id}] llama_index not found or failed. Using full text as one chunk.")
             text_chunks = [target_text.strip()]
        else:
             logging.info(f"[ReqID: {request_id}] Input text is empty."); return
        if not text_chunks:
             logging.warning(f"[ReqID: {request_id}] No valid text chunks after splitting/filtering."); return


        # --- Stream MP3 directly from Edge-TTS Producer ---
        try:
            logging.info(f"[ReqID: {request_id}] Starting direct MP3 streaming from edge-tts.")

            mp3_chunk_producer = self.edge_tts_producer(
                text_chunks, voice_to_use, request_id, context
            )

            total_yielded_chunks = 0
            for mp3_chunk_bytes in mp3_chunk_producer:
                 # Check context *before* yielding to potentially stop sooner
                 if not context.is_active():
                      logging.warning(f"[ReqID: {request_id}] Context became inactive while iterating producer. Stopping yield.")
                      break

                 # Wrap the raw MP3 bytes in the protobuf message
                 yield gateway_pb2.AudioChunk(audio_data=mp3_chunk_bytes)
                 total_yielded_chunks += 1
                 if self.verbose: # Log chunk yielding only if verbose
                     logging.debug(f"[ReqID: {request_id}] Yielded MP3 chunk {total_yielded_chunks}, size: {len(mp3_chunk_bytes)}")

            # Check context one last time after the loop finishes
            if context.is_active():
                 logging.info(f"[ReqID: {request_id}] SynthesizeSpeech request processing completed successfully. Total chunks yielded: {total_yielded_chunks}.")
            else:
                 logging.info(f"[ReqID: {request_id}] SynthesizeSpeech request finished due to context cancellation. Total chunks yielded: {total_yielded_chunks}.")


        except RuntimeError as e:
             # Catch the specific error raised by edge_tts_producer on TTS failure
             logging.error(f"[ReqID: {request_id}] Caught runtime error from TTS producer: {e}")
             # The context should have already been aborted by the producer in this case.
             # No need to abort again here, just log and exit the method.
             if context.is_active():
                 # This shouldn't happen if the producer aborted correctly, but log if it does
                 logging.warning(f"[ReqID: {request_id}] Context still active after producer raised error, aborting.")
                 try: context.abort(grpc.StatusCode.INTERNAL, f"Upstream TTS synthesis failed: {e}")
                 except Exception as abort_e: logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")

        except Exception as e:
             # Handle unexpected errors during the streaming process
             if context.is_active():
                logging.error(f"[ReqID: {request_id}] Unhandled exception during MP3 streaming: {e}", exc_info=True)
                try: context.abort(grpc.StatusCode.INTERNAL, f"Request processing failed unexpectedly: {e}")
                except Exception as abort_e: logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")
             else:
                 # Context was already inactive, likely due to client cancellation or previous error
                 logging.info(f"[ReqID: {request_id}] Request processing terminated due to an earlier error or cancellation.")


# --- Server main execution ---
def serve(port, default_voice_arg, verbose):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # Keep thread pool

    try:
        servicer = AudioGatewayServicer(
            default_voice_setting=default_voice_arg,
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
    # Log voice mode
    if servicer.default_voice_setting == RANDOM_ENGLISH_VOICE:
         logging.info(f"Server configured for RANDOM English voice selection per request.")
    else:
         logging.info(f"Server configured to use FIXED voice: '{servicer.default_voice_setting}' for all requests.")
    # Log output format
    logging.info(f"Output format: MP3 (Direct from edge-tts)")
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
         # Attempt graceful stop before exiting
         server.stop(1)
         sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server using Edge-TTS (MP3 output only)")
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    parser.add_argument("--default-voice", type=str, default=RANDOM_ENGLISH_VOICE,
                        help=f"Default Edge-TTS voice setting. Use a specific ShortName (e.g., en-GB-SoniaNeural) "
                             f"for fixed voice, or '{RANDOM_ENGLISH_VOICE}' (default) to pick a random English voice per request.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set higher level logging if not verbose
    if not args.verbose:
        logging.getLogger().setLevel(logging.INFO) # Default
    else:
        logging.getLogger().setLevel(logging.DEBUG) # Verbose enables DEBUG

    serve(
        args.port,
        args.default_voice,
        args.verbose
    )