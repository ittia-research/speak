## Tech Stack
- Gateway: gRPC API server with Python
  - benifits:
    - Might need different resources allcation than resource heavy ML inference
    - Obstruct business layer on top of ML inference layer
- Inference engine: Triton + TensorRT-LLM

## Audio
### processing
- Add one space at end of every reference text, and end of every text chunks.
- Gateway-side cross-fading before opus encoding.
    - [ ] Shall we implement this for .wav output also?

## Workflow
### Opus
1. Text Splitting: The gateway splits the input text into smaller chunks (e.g., sentences).
2. Independent PCM Generation: The gateway sends each text chunk sequentially to the Triton server (running Spark-TTS). Triton generates the raw audio (PCM Float32) for each chunk independently.
3. PCM Collection (as NumPy): The gateway receives these raw PCM audio chunks back from Triton and converts them into NumPy arrays (pcm_numpy_producer).
4. Gateway Cross-fading (_crossfade_pcm_chunks): takes the NumPy array for the previous chunk and the NumPy array for the current chunk.
    * It identifies a small overlapping region (defined by --gateway-crossfade, e.g., 100ms).
    * It applies a fade-out to the end of the previous chunk's overlap region (volume goes from 100% to 0%).
    * It applies a fade-in to the beginning of the current chunk's overlap region (volume goes from 0% to 100%).
    * It adds these faded segments together, creating a smooth transition.
    * It yields the non-overlapping part of the previous chunk, then prepares the blended segment + rest of the current chunk for the next overlap.
5. Feed Smoothed PCM to FFmpeg: The gateway converts the smoothed audio segments back into raw bytes. These bytes, representing a much more continuous audio stream, are piped into the ffmpeg process.
6. Opus Encoding: ffmpeg encodes the smooth PCM stream into Opus audio within a WebM container.
7. Stream to Client: The gateway streams the resulting Opus/WebM chunks to the client.
