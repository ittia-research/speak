Change logs of this project.

2025.04.13
Changed TTS engine from Spark-TTS to EDGE TTS.
Although Spark-TTS can clone English (and Chinese) voices with good quality, EDGE TTS has below advantages for this project:
    - EDGE TTS has a collection of voices with almost perfectly correct pronunciation, whereas it's difficult to maintain voice with Spark-TTS voice clone, there will be far more EDGE cases.
    - Better audio quality, 24K sample rate instead of 16K.
    - Supports global languages.
    - No need for inference server. Spark-TTS needs GPU server with 8G VRAM.
    - Production ready, whereas Spark-TTS is experimental only without enough testing.
    - This project needs stable, high accurate engines for better service quality and easy maintainance.
