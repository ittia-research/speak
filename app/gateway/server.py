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

import grpc
import numpy as np
import soundfile as sf # Keep for checking reference audio

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException

# Import updated generated gRPC files (assuming gateway.proto is unchanged from last step)
import gateway_pb2
import gateway_pb2_grpc

# --- Triton client helper classes/functions ---
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
    target_text,
    sample_rate=16000,
):
    if waveform.dtype != np.float32:
        logging.warning(f"Input waveform dtype is {waveform.dtype}, converting to float32.")
        waveform = waveform.astype(np.float32)
    assert len(waveform.shape) == 1, "waveform should be 1D"
    assert sample_rate == 16000, "sample rate must be 16000"

    samples = waveform.reshape(1, -1) # Shape (1, N)
    lengths = np.array([[len(waveform)]], dtype=np.int32) # Shape (1, 1)

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

    def SynthesizeSpeech(self, request, context):
        request_id = str(uuid.uuid4()) # Unique ID for this specific gateway request handling
        logging.info(f"[Req ID: {request_id}] Received SynthesizeSpeech request for text: '{request.target_text[:50]}...'")

        # 1. Randomly select template
        if not self.template_ids:
             logging.error(f"[Req ID: {request_id}] No template IDs available.")
             context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Server has no configured voice templates.")
             return
        try:
            selected_template_id = random.choice(self.template_ids)
            logging.info(f"[Req ID: {request_id}] Randomly selected template ID: '{selected_template_id}'")
            template_info = self.templates[selected_template_id]
        except Exception as e:
             logging.error(f"[Req ID: {request_id}] Error selecting template: {e}", exc_info=True)
             context.abort(grpc.StatusCode.INTERNAL, "Failed to select a voice template.")
             return

        # 2. Get reference audio/text
        reference_text = template_info.get("reference_text")
        reference_audio_path = template_info.get("reference_audio")
        if not reference_text or not reference_audio_path:
            logging.error(f"[Req ID: {request_id}] Template '{selected_template_id}' is missing required info.")
            context.abort(grpc.StatusCode.INTERNAL, "Invalid template configuration for selected voice.")
            return
        if not os.path.isabs(reference_audio_path):
            base_dir = os.path.dirname(self.templates_path)
            reference_audio_path = os.path.normpath(os.path.join(base_dir, reference_audio_path))
        if not os.path.exists(reference_audio_path):
            logging.error(f"[Req ID: {request_id}] Reference audio file not found: {reference_audio_path}")
            context.abort(grpc.StatusCode.INTERNAL, "Reference audio not found for selected voice.")
            return

        # 3. Load reference audio (ensure float32)
        try:
            # Use always_2d=False to ensure 1D array if mono
            waveform, sr = sf.read(reference_audio_path, dtype='float32', always_2d=False)
            if sr != 16000:
                logging.error(f"[Req ID: {request_id}] Ref audio {reference_audio_path} has sr={sr}. Requires 16kHz.")
                context.abort(grpc.StatusCode.INTERNAL, "Reference audio must be 16kHz.")
                return
            if waveform.ndim > 1: # Handle potential stereo files - take first channel
                logging.warning(f"[Req ID: {request_id}] Reference audio seems to be stereo. Using only the first channel.")
                waveform = waveform[:, 0]
            if waveform.dtype != np.float32:
                 waveform = waveform.astype(np.float32) # Ensure float32
            logging.info(f"[Req ID: {request_id}] Loaded ref audio: {reference_audio_path} ({len(waveform)} samples, dtype: {waveform.dtype})")
        except Exception as e:
            logging.error(f"[Req ID: {request_id}] Error loading ref audio {reference_audio_path}: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to load reference audio.")
            return

        # 4. Prepare Triton request
        target_text = request.target_text
        try:
            inputs = prepare_grpc_sdk_request(waveform, reference_text, target_text)
        except Exception as e:
            logging.error(f"[Req ID: {request_id}] Error preparing Triton request: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to prepare inference request.")
            return

        # 5. Call Triton Inference Server (Streaming)
        user_data = UserData()
        triton_client = None
        triton_request_id = f"triton-{request_id}" # Link gateway req ID to Triton req ID

        try:
            triton_client = grpcclient.InferenceServerClient(url=self.triton_url, verbose=self.verbose)
            triton_client.start_stream(callback=partial(callback, user_data))
            outputs = [grpcclient.InferRequestedOutput("waveform")]

            triton_client.async_stream_infer(
                model_name=self.model_name,
                inputs=inputs,
                request_id=triton_request_id, # Use linked ID
                outputs=outputs,
                enable_empty_final_response=True,
            )
            logging.info(f"[Req ID: {request_id}] Sent request to Triton (TritonReqID: {triton_request_id}) for model '{self.model_name}'")

            # 6. Stream results back to the end user immediately
            processed_chunk_count = 0
            while True:
                result_or_error = user_data._completed_requests.get()

                if isinstance(result_or_error, InferenceServerException):
                    logging.error(f"[Req ID: {request_id}] Triton inference exception (TritonReqID: {triton_request_id}): {result_or_error}")
                    context.set_details(f"Triton inference failed: {result_or_error}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    break # Exit loop

                try:
                    response = result_or_error.get_response()
                    # Verify Triton request ID matches if needed (debugging)
                    # received_triton_id = response.id
                    # if received_triton_id != triton_request_id:
                    #    logging.warning(f"[Req ID: {request_id}] Mismatched Triton request ID! Expected {triton_request_id}, got {received_triton_id}")

                except Exception as e:
                     logging.error(f"[Req ID: {request_id}] Error getting response data from callback object: {e}", exc_info=True)
                     context.set_details("Error processing Triton callback response.")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     break

                # Check for final response marker
                if response.parameters.get("triton_final_response", None) and \
                   response.parameters["triton_final_response"].bool_param:
                    logging.info(f"[Req ID: {request_id}] Received final marker from Triton (TritonReqID: {triton_request_id}).")
                    break

                # Process audio chunk
                try:
                    # Get numpy array for 'waveform' output
                    audio_chunk_np = result_or_error.as_numpy("waveform")

                    if audio_chunk_np is None:
                        logging.debug(f"[Req ID: {request_id}] Received null 'waveform' numpy array. Skipping.")
                        continue

                    # *** Ensure float32 and reshape to 1D ***
                    if audio_chunk_np.dtype != np.float32:
                        audio_chunk_np = audio_chunk_np.astype(np.float32)
                    # Use reshape(-1) as in the example tts function
                    audio_chunk_np = audio_chunk_np.reshape(-1)

                    if audio_chunk_np.size == 0:
                         logging.debug(f"[Req ID: {request_id}] Received empty chunk after reshape. Skipping.")
                         continue

                    # *** Convert float32 numpy chunk to bytes ***
                    audio_bytes = audio_chunk_np.tobytes()

                    # Yield the raw bytes back to the connected client
                    yield gateway_pb2.AudioChunk(audio_data=audio_bytes)
                    processed_chunk_count += 1
                    logging.debug(f"[Req ID: {request_id}] Sent chunk {processed_chunk_count} ({len(audio_bytes)} bytes, {len(audio_chunk_np)} samples) as float32 bytes to client.")

                except InferenceServerException as e:
                     logging.error(f"[Req ID: {request_id}] Error extracting numpy 'waveform': {e}")
                     context.set_details(f"Error processing Triton response chunk: {e}")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     break
                except Exception as e:
                     logging.error(f"[Req ID: {request_id}] Unexpected error processing chunk: {e}", exc_info=True)
                     context.set_details("Unexpected error processing audio chunk.")
                     context.set_code(grpc.StatusCode.INTERNAL)
                     break

            logging.info(f"[Req ID: {request_id}] Finished processing Triton stream. Sent {processed_chunk_count} chunks to client.")

        except InferenceServerException as e:
            logging.error(f"[Req ID: {request_id}] Triton communication error: {e}", exc_info=True)
            context.abort(grpc.StatusCode.UNAVAILABLE, f"Could not communicate with Triton: {e}")
        except grpc.RpcError as e:
             logging.error(f"[Req ID: {request_id}] gRPC error during Triton communication: {e.code()} - {e.details()}")
             context.abort(e.code(), f"gRPC error calling Triton: {e.details()}")
        except Exception as e:
            logging.error(f"[Req ID: {request_id}] Unexpected error during gateway processing: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"An unexpected server error occurred: {e}")
        finally:
            if triton_client:
                try:
                    # Ensure stream cleanup
                    triton_client.stop_stream(wait_until_complete=False, gracefully=True)
                    triton_client.close()
                    logging.info(f"[Req ID: {request_id}] Closed Triton client connection.")
                except Exception as e:
                    logging.warning(f"[Req ID: {request_id}] Error closing Triton stream/client: {e}")

# --- Server main execution --- (serve function and if __name__ == '__main__' block remain the same as previous version)
def serve(port, triton_url, model_name, templates_file, verbose):
    """Starts the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Server")
    parser.add_argument("--port", type=int, default=30051, help="Port for the gateway server")
    parser.add_argument("--triton-url", type=str, default="inference:8001", help="URL of the Triton Inference Server (gRPC)")
    parser.add_argument("--model-name", type=str, default="spark_tts_decoupled", help="Name of the TTS model on Triton")
    parser.add_argument("--templates-file", type=str, default="templates/templates.json", help="Path to voice templates JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for Triton client")
    args = parser.parse_args()
    serve(args.port, args.triton_url, args.model_name, args.templates_file, args.verbose)