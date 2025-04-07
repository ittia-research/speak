# gateway_server.py
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

import grpc
import numpy as np
import soundfile as sf

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
# UserData, callback, prepare_grpc_sdk_request remain the same as the previous version
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
        # You could initialize the splitter here if its config is static
        # self.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)


    def SynthesizeSpeech(self, request, context):
        """
        Handles request: splits text, calls Triton sequentially for chunks, streams audio.
        """
        request_id = str(uuid.uuid4()) # ID for the overall client request
        logging.info(f"[ReqID: {request_id}] Received SynthesizeSpeech request. Target text length: {len(request.target_text)}")

        # --- 1. Select Template & Load Reference Audio ---
        # (Same logic as before: random choice, load wav, checks)
        if not self.template_ids:
             logging.error(f"[ReqID: {request_id}] No template IDs available.")
             context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Server has no configured voice templates.")
             return
        try:
            selected_template_id = random.choice(self.template_ids)
            template_info = self.templates[selected_template_id]
            logging.info(f"[ReqID: {request_id}] Selected template ID: '{selected_template_id}'")
            reference_text = template_info.get("reference_text")
            reference_audio_path = template_info.get("reference_audio")
            # (Path validation and loading logic as before...)
            if not reference_text or not reference_audio_path: # Simplified check
                 raise ValueError("Template missing reference text or audio path")
            if not os.path.isabs(reference_audio_path):
                base_dir = os.path.dirname(self.templates_path)
                reference_audio_path = os.path.normpath(os.path.join(base_dir, reference_audio_path))
            if not os.path.exists(reference_audio_path):
                 raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

            # waveform, sr = sf.read(reference_audio_path, dtype='float32', always_2d=False)
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
        target_text = request.target_text
        text_chunks = []
        try:
            # Note: SentenceSplitter uses character count for chunk_size.
            # Estimate ~6 chars/word + space -> 64 words * 7 chars/word ~= 450 chars. Adjust as needed.
            # Overlap helps maintain context between chunks.
            # Using a callback to estimate word count if needed, or keep it simple.
            # TO-DO: better solution in edge cases
            splitter = SentenceSplitter(
                chunk_size=100,  # Target characters per chunk (tune this)
                chunk_overlap=0, # Character overlap (tune this)
                # separator=" ",    # Split by space primarily
                # paragraph_separator="\n\n", # Respect paragraphs
                # secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?", # Helps split on punctuation
            )
            # Wrap text in Document for splitter
            document = Document(text=target_text)
            # Split into text nodes, then extract text
            nodes = splitter.get_nodes_from_documents([document])
            text_chunks = [node.get_content() for node in nodes]
            logging.info(f"\nTexts after split:")
            for i in text_chunks:
                logging.info(f"<<{i}>>")

            # Simple Word Count Check (Optional but recommended for the 64-word rule)
            final_chunks = []
            for chunk in text_chunks:
                word_count = len(chunk.split())
                if word_count > 75: # Allow some buffer over 64
                    logging.warning(f"[ReqID: {request_id}] SentenceSplitter produced a chunk with {word_count} words (>{64}). Consider adjusting chunk_size/overlap or adding more robust splitting logic if this is frequent.")
                if chunk.strip(): # Avoid empty chunks
                    final_chunks.append(chunk.strip())
            text_chunks = final_chunks


            if not text_chunks:
                 # Handle case where input text is empty or splitting results in nothing
                 if target_text.strip(): # Input was not empty, but splitting failed
                      logging.warning(f"[ReqID: {request_id}] Text splitting resulted in zero chunks, using original text.")
                      text_chunks = [target_text] # Fallback to original text if splitting fails
                 else:
                      logging.info(f"[ReqID: {request_id}] Input text was empty. No audio to generate.")
                      # Return immediately, client will receive an empty stream
                      return

            logging.info(f"[ReqID: {request_id}] Split text into {len(text_chunks)} chunks.")

        except Exception as e:
            logging.error(f"[ReqID: {request_id}] Error during text splitting: {e}", exc_info=True)
            # Fallback: use the original text if splitting fails
            text_chunks = [target_text]
            logging.warning(f"[ReqID: {request_id}] Using original text due to splitting error.")

        # --- 3. Process Chunks Sequentially via Triton ---
        triton_client = None
        total_audio_chunks_sent = 0
        try:
            # Create Triton client once for the entire request
            triton_client = grpcclient.InferenceServerClient(url=self.triton_url, verbose=self.verbose)
            logging.info(f"[ReqID: {request_id}] Triton client created for {self.triton_url}")

            # Loop through each text chunk
            for i, text_chunk in enumerate(text_chunks):
                chunk_req_id = f"{request_id}-C{i+1}" # Unique ID for this chunk's Triton request
                logging.info(f"[ReqID: {request_id}] Processing text chunk {i+1}/{len(text_chunks)} (TritonReqID: {chunk_req_id})")
                # logging.debug(f"[ReqID: {request_id}] Chunk Text: '{text_chunk}'") # Uncomment for debugging chunk content

                # 3a. Prepare Inputs for this specific chunk
                try:
                    inputs = prepare_grpc_sdk_request(waveform, reference_text, text_chunk)
                except Exception as e:
                    logging.error(f"[ReqID: {request_id}] Error preparing inputs for chunk {i+1}: {e}", exc_info=True)
                    # Abort the whole request if input prep fails for a chunk
                    context.abort(grpc.StatusCode.INTERNAL, f"Failed to prepare inputs for chunk {i+1}: {e}")
                    # Ensure client is closed in finally block
                    return # Exit the SynthesizeSpeech method

                # 3b. Setup Triton Stream for this chunk
                user_data = UserData() # Fresh UserData for each chunk's stream
                stream_has_error = False
                try:
                    triton_client.start_stream(callback=partial(callback, user_data))

                    # 3c. Send Inference Request for this chunk
                    triton_client.async_stream_infer(
                        model_name=self.model_name,
                        inputs=inputs,
                        request_id=chunk_req_id, # Use the chunk-specific ID
                        outputs=[grpcclient.InferRequestedOutput("waveform")],
                        enable_empty_final_response=True,
                    )
                    logging.debug(f"[ReqID: {request_id}] Async infer request sent to Triton for chunk {i+1}.")

                    # 3d. Process Audio Stream for this chunk (Inner Loop)
                    audio_chunks_in_stream = 0
                    while True:
                        # Wait for a result/error from the callback for *this* chunk's stream
                        result_or_error = user_data._completed_requests.get() # Blocking wait

                        if isinstance(result_or_error, InferenceServerException):
                            logging.error(f"[ReqID: {request_id}] Triton error during chunk {i+1} stream (TritonReqID: {chunk_req_id}): {result_or_error}")
                            context.set_details(f"Triton inference failed on chunk {i+1}: {result_or_error}")
                            context.set_code(grpc.StatusCode.INTERNAL)
                            stream_has_error = True
                            break # Break inner loop for this chunk

                        try:
                            response = result_or_error.get_response()
                            # Optional: Check response.id == chunk_req_id for sanity
                        except Exception as e:
                             logging.error(f"[ReqID: {request_id}] Error getting response object for chunk {i+1}: {e}", exc_info=True)
                             context.set_details("Error processing Triton callback response.")
                             context.set_code(grpc.StatusCode.INTERNAL)
                             stream_has_error = True
                             break # Break inner loop

                        # Check for final marker for *this* chunk's stream
                        if response.parameters.get("triton_final_response", None) and \
                           response.parameters["triton_final_response"].bool_param:
                            logging.debug(f"[ReqID: {request_id}] Received final marker for chunk {i+1} stream.")
                            break # Successful end of this chunk's stream

                        # Process audio chunk
                        try:
                            audio_chunk_np = result_or_error.as_numpy("waveform")
                            if audio_chunk_np is None: continue # Skip null data

                            if audio_chunk_np.dtype != np.float32:
                                audio_chunk_np = audio_chunk_np.astype(np.float32)
                            audio_chunk_np = audio_chunk_np.reshape(-1) # Ensure 1D
                            if audio_chunk_np.size == 0: continue # Skip empty data

                            audio_bytes = audio_chunk_np.tobytes()

                            # >>> YIELD audio chunk bytes to the connected client <<<
                            yield gateway_pb2.AudioChunk(audio_data=audio_bytes)
                            audio_chunks_in_stream += 1
                            total_audio_chunks_sent += 1
                            # logging.debug(f"[ReqID: {request_id}] Sent audio chunk {total_audio_chunks_sent} (from text chunk {i+1}) to client.")

                        except InferenceServerException as e: # Error during as_numpy()
                             logging.error(f"[ReqID: {request_id}] Numpy conversion error for chunk {i+1}: {e}")
                             context.set_details(f"Numpy conversion error processing chunk {i+1}: {e}")
                             context.set_code(grpc.StatusCode.INTERNAL)
                             stream_has_error = True
                             break # Break inner loop
                        except Exception as e: # Other processing errors
                             logging.error(f"[ReqID: {request_id}] Unexpected error processing audio chunk {i+1}: {e}", exc_info=True)
                             context.set_details("Unexpected error processing audio chunk.")
                             context.set_code(grpc.StatusCode.INTERNAL)
                             stream_has_error = True
                             break # Break inner loop

                    # 3e. Clean up stream for this specific chunk (always attempt)
                    logging.debug(f"[ReqID: {request_id}] Stopping Triton stream for chunk {i+1}...")
                    triton_client.stop_stream()
                    logging.info(f"[ReqID: {request_id}] Finished processing stream for chunk {i+1}. Sent {audio_chunks_in_stream} audio chunks.")

                    # If an error occurred in this chunk's stream, stop processing subsequent chunks
                    if stream_has_error:
                        logging.error(f"[ReqID: {request_id}] Aborting processing due to error in chunk {i+1}.")
                        # The gRPC context details/code should already be set
                        return # Exit the SynthesizeSpeech method

                except InferenceServerException as e: # Error starting stream or async_infer
                    logging.error(f"[ReqID: {request_id}] Triton communication error for chunk {i+1}: {e}", exc_info=True)
                    context.abort(grpc.StatusCode.UNAVAILABLE, f"Triton communication failed for chunk {i+1}: {e}")
                    return
                except grpc.RpcError as e: # Underlying gRPC issues
                     logging.error(f"[ReqID: {request_id}] gRPC error during Triton call for chunk {i+1}: {e.code()} - {e.details()}")
                     context.abort(e.code(), f"gRPC error calling Triton for chunk {i+1}: {e.details()}")
                     return
                except Exception as e: # Other unexpected errors during chunk processing setup
                     logging.error(f"[ReqID: {request_id}] Unexpected error setting up chunk {i+1}: {e}", exc_info=True)
                     context.abort(grpc.StatusCode.INTERNAL, f"Unexpected error processing chunk {i+1}: {e}")
                     return

            # End of loop through text chunks
            logging.info(f"[ReqID: {request_id}] Successfully processed all {len(text_chunks)} text chunks. Total audio chunks sent: {total_audio_chunks_sent}.")

        except Exception as e:
            # Catch errors during outer loop setup or client creation
            logging.error(f"[ReqID: {request_id}] Unexpected error during main processing: {e}", exc_info=True)
            if not context.is_active(): # Check if context already aborted
                # Try to abort if not already done
                try: context.abort(grpc.StatusCode.INTERNAL, f"An unexpected gateway error occurred: {e}")
                except Exception as abort_e: logging.error(f"[ReqID: {request_id}] Error trying to abort context: {abort_e}")
            # Fall through to finally block for cleanup
        finally:
            # Ensure Triton client is closed if it was created
            if triton_client:
                try:
                    logging.debug(f"[ReqID: {request_id}] Closing Triton client connection.")
                    triton_client.close()
                    logging.info(f"[ReqID: {request_id}] Triton client connection closed.")
                except Exception as e:
                    logging.warning(f"[ReqID: {request_id}] Error closing Triton client: {e}")


# --- Server main execution ---
# (serve function and if __name__ == '__main__' block remain the same)
def serve(port, triton_url, model_name, templates_file, verbose):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = AudioGatewayServicer(triton_url, model_name, templates_file, verbose)
    if servicer.templates:
        gateway_pb2_grpc.add_AudioGatewayServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        logging.info(f"Starting gRPC Gateway Server on port {port}...")
        # (rest of logging and server.start/wait)
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
    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server with Text Splitting")
    # (Arguments remain the same)
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    parser.add_argument("--triton-url", type=str, default="inference:8001", help="URL of the Triton Inference Server (gRPC)")
    parser.add_argument("--model-name", type=str, default="spark_tts_decoupled", help="Name of the TTS model on Triton")
    parser.add_argument("--templates-file", type=str, default="templates/templates.json", help="Path to voice templates JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for Triton client")
    args = parser.parse_args()
    serve(args.port, args.triton_url, args.model_name, args.templates_file, args.verbose)