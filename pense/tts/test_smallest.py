import time
from smallestai.waves import TTSConfig, WavesStreamingTTS
from dotenv import load_dotenv
import os

load_dotenv(override=True)

config = TTSConfig(
    voice_id="aditi",
    api_key=os.getenv("SMALLEST_API_KEY"),
    sample_rate=24000,
    speed=1.0,
    max_buffer_flush_ms=50,
)

streaming_tts = WavesStreamingTTS(config)


def text_stream():
    text = "Streaming synthesis with chunked text input for Smallest SDK."
    for word in text.split():
        yield word + " "


audio_chunks = []
start_time = time.time()

for index, chunk in enumerate(streaming_tts.synthesize_streaming(text_stream())):
    print(f"Index: {index}")
    print(f"TTFT: {time.time() - start_time}")
    audio_chunks.append(chunk)
