# gateway_client.py
import argparse
import logging
import grpc
import numpy as np
import soundfile as sf
import time
import sys

# Import updated generated gRPC files
import gateway_pb2
import gateway_pb2_grpc

# Define constants
SAMPLE_RATE = 16000
AUDIO_DTYPE = np.float32 # Data type received from gateway (as bytes)
OUTPUT_DTYPE = np.int16 # Final data type for saving WAV file
OUTPUT_SUBTYPE = 'PCM_16' # soundfile subtype for int16

def run_synthesis(gateway_address, text, output_file, chunk_overlap_duration):
    """Connects, sends request, receives stream, performs cross-fade, saves audio."""
    logging.info(f"Connecting to gateway at {gateway_address}...")
    # Note: Consider increasing max message size if chunks can be very large
    # options = [('grpc.max_receive_message_length', MAX_BYTES)]
    with grpc.insecure_channel(gateway_address) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=15)
            logging.info("Channel ready.")
        except grpc.FutureTimeoutError:
            logging.error(f"Timeout connecting to gateway server at {gateway_address}")
            return False

        stub = gateway_pb2_grpc.AudioGatewayStub(channel)
        request = gateway_pb2.SynthesizeRequest(target_text=text)
        logging.info(f"Sending request: text='{text[:50]}...'")

        # List to store received numpy chunks (float32)
        received_raw_chunks = []
        start_time = time.time()
        first_chunk_time = None
        received_bytes = 0

        try:
            response_stream = stub.SynthesizeSpeech(request)
            logging.info("Waiting for audio stream...")

            # Iterate through the streamed audio chunks
            for i, chunk in enumerate(response_stream):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfc = first_chunk_time - start_time
                    logging.info(f"Time to first chunk (TTFC): {ttfc:.3f} seconds")

                chunk_bytes = chunk.audio_data
                if not chunk_bytes:
                    logging.debug(f"Received empty audio chunk {i}. Skipping.")
                    continue

                # Convert received bytes back to float32 numpy array
                audio_segment = np.frombuffer(chunk_bytes, dtype=AUDIO_DTYPE)
                received_raw_chunks.append(audio_segment) # Store the raw chunk
                received_bytes += len(chunk_bytes)
                logging.debug(f"Received chunk {i+1}, samples: {len(audio_segment)}, bytes: {len(chunk_bytes)}")

            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"Stream finished. Received {len(received_raw_chunks)} chunks. Total time: {total_time:.3f} seconds")

        except grpc.RpcError as e:
            logging.error(f"gRPC Error during synthesis: Code={e.code()} Details='{e.details()}'")
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                 logging.error("Server unavailable. Is the gateway running?")
            elif e.code() == grpc.StatusCode.INTERNAL:
                 logging.error("Internal server error reported by gateway. Check gateway logs.")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during streaming: {e}", exc_info=True)
            return False

        # --- Post-processing: Cross-fading ---
        if not received_raw_chunks:
            logging.warning("No audio data received from the server.")
            return True # Technically succeeded, just no audio data

        logging.info(f"Starting cross-fading process with overlap: {chunk_overlap_duration * 1000:.1f} ms")
        try:
            # Calculate cross-fade parameters
            cross_fade_samples = int(chunk_overlap_duration * SAMPLE_RATE)
            if cross_fade_samples <= 0:
                 logging.warning(f"Cross-fade duration ({chunk_overlap_duration}s) too short, resulting in 0 overlap samples. Concatenating directly.")
                 final_audio_float32 = np.concatenate(received_raw_chunks)
            elif len(received_raw_chunks) == 1:
                 logging.info("Only one chunk received, no cross-fading needed.")
                 final_audio_float32 = received_raw_chunks[0]
            else:
                fade_out = np.linspace(1, 0, cross_fade_samples, dtype=AUDIO_DTYPE)
                fade_in = np.linspace(0, 1, cross_fade_samples, dtype=AUDIO_DTYPE)

                # Perform the cross-fading loop (identical to tts example)
                stitched_audio = None
                for i, audio_chunk in enumerate(received_raw_chunks):
                    # Ensure chunk has enough samples for overlap if it's not the last one
                    if i > 0 and len(received_raw_chunks[i-1]) < cross_fade_samples:
                        logging.warning(f"Previous chunk {i-1} is shorter than overlap ({len(received_raw_chunks[i-1])} < {cross_fade_samples}). Reducing overlap for this transition.")
                        effective_overlap = min(len(received_raw_chunks[i-1]), len(audio_chunk), cross_fade_samples)
                        if effective_overlap <=0: # Cannot overlap
                             segment_to_append = audio_chunk
                        else:
                             short_fade_out = np.linspace(1, 0, effective_overlap, dtype=AUDIO_DTYPE)
                             short_fade_in = np.linspace(0, 1, effective_overlap, dtype=AUDIO_DTYPE)
                             cross_faded_overlap = audio_chunk[:effective_overlap] * short_fade_in + \
                                                   stitched_audio[-effective_overlap:] * short_fade_out
                             segment_to_append = np.concatenate([cross_faded_overlap, audio_chunk[effective_overlap:]])
                             stitched_audio = stitched_audio[:-effective_overlap] # Remove the part that was faded out

                    elif i == 0: # First chunk
                        stitched_audio = audio_chunk
                        continue # Processed in the next iteration's overlap calculation

                    else: # Normal overlap case (i > 0)
                         # Check if current chunk is long enough
                        if len(audio_chunk) < cross_fade_samples:
                            logging.warning(f"Current chunk {i} is shorter than overlap ({len(audio_chunk)} < {cross_fade_samples}). Reducing overlap.")
                            effective_overlap = min(len(stitched_audio), len(audio_chunk), cross_fade_samples)
                            if effective_overlap <= 0:
                                segment_to_append = audio_chunk
                            else:
                                short_fade_out = np.linspace(1, 0, effective_overlap, dtype=AUDIO_DTYPE)
                                short_fade_in = np.linspace(0, 1, effective_overlap, dtype=AUDIO_DTYPE)
                                cross_faded_overlap = audio_chunk[:effective_overlap] * short_fade_in + \
                                                    stitched_audio[-effective_overlap:] * short_fade_out
                                segment_to_append = np.concatenate([cross_faded_overlap, audio_chunk[effective_overlap:]])
                                stitched_audio = stitched_audio[:-effective_overlap]
                        else:
                            # Standard overlap calculation from tts example
                            cross_faded_overlap = audio_chunk[:cross_fade_samples] * fade_in + \
                                                  stitched_audio[-cross_fade_samples:] * fade_out
                            # Append the faded overlap and the rest of the current chunk
                            segment_to_append = np.concatenate([cross_faded_overlap, audio_chunk[cross_fade_samples:]])
                            # Remove the last part of the previous audio that was faded out
                            stitched_audio = stitched_audio[:-cross_fade_samples]


                    # Append the processed segment
                    stitched_audio = np.concatenate([stitched_audio, segment_to_append])

                final_audio_float32 = stitched_audio # Rename for clarity

            # --- Conversion and Saving ---
            logging.info("Converting final audio to 16-bit integer.")
            # Clamp float32 audio to [-1.0, 1.0] to prevent wrap-around during int16 conversion
            final_audio_float32 = np.clip(final_audio_float32, -1.0, 1.0)
            # Convert to int16
            final_audio_int16 = (final_audio_float32 * 32767).astype(OUTPUT_DTYPE)

            duration = len(final_audio_int16) / SAMPLE_RATE
            logging.info(f"Final stitched audio duration: {duration:.3f} seconds")

            # Save as PCM_16 WAV file
            sf.write(output_file, final_audio_int16, SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
            logging.info(f"Audio saved successfully to {output_file}")
            return True # Indicate success

        except Exception as e:
            logging.error(f"Error during cross-fading or saving audio file: {e}", exc_info=True)
            return False # Indicate failure


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="gRPC Audio Gateway Client with Cross-fading")
    parser.add_argument("--gateway-address", type=str, default="localhost:50051", help="Address (host:port) of the gRPC gateway server")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output-file", type=str, default="gateway_output_faded.wav", help="Path to save the final WAV file")
    parser.add_argument("--chunk-overlap-duration", type=float, default=0.1, help="Overlap duration for cross-fading chunks in seconds (e.g., 0.1 for 100ms)")

    args = parser.parse_args()

    success = run_synthesis(args.gateway_address, args.text, args.output_file, args.chunk_overlap_duration)

    if success:
        logging.info("Client finished successfully.")
        sys.exit(0)
    else:
        logging.error("Client finished with errors.")
        sys.exit(1)