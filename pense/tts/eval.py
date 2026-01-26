import asyncio
import argparse
import sys
import os
import json
import time
from os.path import join, exists
from pathlib import Path
from typing import Dict, List
import base64
import wave

from openai import AsyncOpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from groq import AsyncGroq
from cartesia import AsyncCartesia
from sarvamai import AsyncSarvamAI, AudioOutput
from google.cloud import texttospeech
from smallestai.waves import TTSConfig, WavesStreamingTTS

import numpy as np
from loguru import logger
import pandas as pd

from pense.utils import (
    configure_print_logger,
    log_and_print,
    get_tts_language_code,
    validate_tts_language,
)
from pense.tts.metrics import get_tts_llm_judge_score


# =============================================================================
# TTS Provider API Methods
# =============================================================================


def save_audio(audio_bytes: bytes, output_path: str, sample_rate: int = 24000):
    """Save audio bytes to a WAV file.

    Args:
        audio_bytes: Raw audio bytes (PCM or WAV format)
        output_path: Path to save the WAV file
        sample_rate: Audio sample rate (default: 24000)
    """
    import wave

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check if audio_bytes is already a WAV file
    if audio_bytes[:4] == b"RIFF":
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
    else:
        # Raw PCM data - wrap in WAV
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)


def convert_mp3_to_wav(mp3_path: str, wav_path: str, cleanup: bool = True):
    """Convert MP3 file to WAV format.

    Args:
        mp3_path: Path to the input MP3 file
        wav_path: Path to save the output WAV file
        cleanup: If True, delete the MP3 file after conversion (default: True)
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    if cleanup:
        os.remove(mp3_path)


async def synthesize_openai(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using OpenAI's TTS API and stream directly to file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI()

    start_time = time.time()
    ttfb = None

    # Stream directly to file
    with open(audio_path, "wb") as f:
        async with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            response_format="wav",
        ) as response:
            async for chunk in response.iter_bytes():
                if ttfb is None:
                    ttfb = time.time() - start_time
                f.write(chunk)

    return {"ttfb": ttfb}


async def synthesize_google(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Google Cloud Text-to-Speech API and save to file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    lang_code = get_tts_language_code(language, "google")

    client = texttospeech.TextToSpeechClient()

    # Sindhi requires synchronous API with Gemini-TTS model (streaming API doesn't support Sindhi)
    # See: https://cloud.google.com/text-to-speech/docs/gemini-tts
    if language.lower() == "sindhi":
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name="Charon",
            model_name="gemini-2.5-flash-lite-preview-tts",
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        start_time = time.time()
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        ttfb = time.time() - start_time

        # Save the audio content
        save_audio(response.audio_content, audio_path, sample_rate=24000)

        return {}

    # For other languages, use streaming API with Chirp3-HD voices
    streaming_audio_config = texttospeech.StreamingAudioConfig(
        audio_encoding=texttospeech.AudioEncoding.PCM,
        sample_rate_hertz=24000,
    )

    voice_params = texttospeech.VoiceSelectionParams(
        name=f"{lang_code}-Chirp3-HD-Charon",
        language_code=lang_code,
    )

    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=voice_params,
        streaming_audio_config=streaming_audio_config,
    )

    # Set the config for your stream. The first request must contain your config, and then each subsequent request must contain text.
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=streaming_config
    )

    start_time = time.time()
    ttfb = None

    # Request generator. Consider using Gemini or another LLM with output streaming as a generator.
    def request_generator():
        yield config_request
        # for text in text_iterator:
        yield texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(text=text)
        )

    streaming_responses = client.streaming_synthesize(request_generator())

    # Collect audio chunks and save to file
    audio_chunks = []
    for response in streaming_responses:
        if ttfb is None:
            ttfb = time.time() - start_time

        audio_chunks.append(response.audio_content)

    # Save combined PCM audio as WAV
    audio_bytes = b"".join(audio_chunks)
    save_audio(audio_bytes, audio_path, sample_rate=24000)

    return {"ttfb": ttfb}


async def synthesize_elevenlabs(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using ElevenLabs' TTS API and stream directly to file."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    start_time = time.time()
    ttfb = None

    elevenlabs = AsyncElevenLabs(api_key=api_key)

    voice_id = "m5qndnI7u4OAdXhH0Mr5"
    output_format = "mp3_24000_48"

    if language.lower() == "sindhi":
        model_id = "eleven_v3"

        response = elevenlabs.text_to_dialogue.stream(
            output_format=output_format,
            inputs=[
                {"text": text, "voice_id": voice_id},
            ],
            language_code="sd",
            model_id="eleven_v3",
        )

    else:
        model_id = "eleven_multilingual_v2"

        response = elevenlabs.text_to_speech.stream(
            voice_id=voice_id,  # Krishna pre-made voice
            output_format=output_format,
            text=text,
            model_id=model_id,
            # Optional voice settings that allow you to customize the output
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )

    mp3_path = audio_path.replace(".wav", ".mp3")
    with open(mp3_path, "wb") as f:
        async for chunk in response:
            if ttfb is None:
                ttfb = time.time() - start_time

            if chunk:
                f.write(chunk)

    convert_mp3_to_wav(mp3_path, audio_path)

    return {"ttfb": ttfb}


async def synthesize_cartesia(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Cartesia's TTS API and stream directly to file."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "cartesia")

    client = AsyncCartesia(api_key=api_key)

    # Default voice ID
    with open(audio_path, "wb") as f:
        start_time = time.time()
        ttfb = None

        bytes_iter = client.tts.bytes(
            model_id="sonic-3",
            transcript=text,
            voice={
                "mode": "id",
                "id": "faf0731e-dfb9-4cfc-8119-259a79b27e12",  # riya
            },
            language=lang_code,
            output_format={
                "container": "wav",
                "sample_rate": 24000,
                "encoding": "pcm_f32le",
            },
        )

        async for chunk in bytes_iter:
            if ttfb is None:
                ttfb = time.time() - start_time

            f.write(chunk)

    return {"ttfb": ttfb}


async def synthesize_groq(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Groq's TTS API and save to file."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = AsyncGroq(api_key=api_key)

    model = "canopylabs/orpheus-v1-english"
    voice = "troy"
    response_format = "wav"

    response = await client.audio.speech.create(
        model=model, voice=voice, input=text, response_format=response_format
    )

    log_and_print(f"\033[93mStoring generated audio to {audio_path}\033[0m")
    await response.write_to_file(audio_path)

    return {}


async def synthesize_sarvam(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Sarvam's TTS API and save to file."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "sarvam")

    client = AsyncSarvamAI(api_subscription_key=api_key)

    start_time = time.time()
    ttfb = None

    async with client.text_to_speech_streaming.connect(model="bulbul:v3-beta") as ws:
        await ws.configure(
            target_language_code=lang_code, speaker="aditya", output_audio_codec="wav"
        )

        await ws.convert(text)
        # print("Sent text message")

        await ws.flush()
        # print("Flushed buffer")

        chunk_count = 0
        with open(audio_path, "wb") as f:
            async for message in ws:
                if isinstance(message, AudioOutput):
                    if ttfb is None:
                        ttfb = time.time() - start_time
                        # Print "Started audio generation" in yellow using ANSI escape code for yellow
                        log_and_print(
                            f"\033[93mStoring generated audio to {audio_path}\033[0m",
                        )

                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)
                    f.write(audio_chunk)
                    f.flush()

        # print(f"All {chunk_count} chunks saved to output.mp3")
        log_and_print("\033[93mAudio generation complete\033[0m")
        if hasattr(ws, "_websocket") and not ws._websocket.closed:
            await ws._websocket.close()
            print("WebSocket connection closed.")

    return {
        "ttfb": ttfb,
    }


async def synthesize_smallest(text: str, language: str, audio_path: str) -> Dict:
    """Synthesize speech using Smallest AI's TTS API and save to file."""
    api_key = os.getenv("SMALLEST_API_KEY")
    if not api_key:
        raise ValueError("SMALLEST_API_KEY environment variable not set")

    lang_code = get_tts_language_code(language, "smallest")

    config = TTSConfig(
        voice_id="aditi",
        language=lang_code,
        api_key=api_key,
        sample_rate=24000,
        speed=1.0,
        max_buffer_flush_ms=100,
    )

    streaming_tts = WavesStreamingTTS(config)

    start_time = time.time()
    ttfb = None

    for chunk in streaming_tts.synthesize(text):
        if ttfb is None:
            ttfb = time.time() - start_time

        save_audio(chunk, audio_path, 24000)

    return {"ttfb": ttfb}


# =============================================================================
# Main Synthesis Router
# =============================================================================


async def synthesize_speech(
    text: str,
    provider: str,
    language: str,
    audio_path: str,
) -> Dict:
    """Route speech synthesis to the appropriate provider and save to audio_path."""
    provider_methods = {
        "openai": synthesize_openai,
        "google": synthesize_google,
        "elevenlabs": synthesize_elevenlabs,
        "cartesia": synthesize_cartesia,
        "groq": synthesize_groq,
        "sarvam": synthesize_sarvam,
        "smallest": synthesize_smallest,
    }

    if provider not in provider_methods:
        raise ValueError(f"Unsupported TTS provider: {provider}")

    method = provider_methods[provider]
    return await method(text, language, audio_path)


# =============================================================================
# TTS Evaluation Main
# =============================================================================


async def run_tts_eval(
    gt_data: List[Dict],
    provider: str,
    language: str,
    output_dir: str,
    results_csv_path: Path,
    overwrite: bool = False,
) -> int:
    """Process texts and synthesize speech, saving results immediately to CSV.

    Args:
        gt_data: List of {"id": ..., "text": ...} for each text to process
        provider: TTS provider name
        language: Language code
        output_dir: Directory to save audio files
        results_csv_path: Path to save results CSV
        overwrite: If True, overwrite existing results instead of resuming

    Returns:
        Number of texts successfully synthesized in this run.
    """
    # Load existing results to skip already processed texts (unless overwrite is True)
    if overwrite:
        processed_ids = set()
        # Remove existing results file if overwriting
        if exists(results_csv_path):
            os.remove(results_csv_path)
    elif exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        processed_ids = set(existing_df["id"].tolist())
    else:
        processed_ids = set()

    audio_output_dir = join(output_dir, "audios")
    os.makedirs(audio_output_dir, exist_ok=True)

    success_count = 0
    ttfb_values = []

    for i, item in enumerate(gt_data):
        _id = item["id"]
        text = item["text"]

        # Skip if already processed
        if _id in processed_ids:
            log_and_print(f"Skipping already processed: {_id}")
            continue

        log_and_print(f"Processing [{i+1}/{len(gt_data)}]: {_id}")

        try:
            audio_path = join(audio_output_dir, f"{_id}.wav")
            result = await synthesize_speech(text, provider, language, audio_path)

            # Handle optional ttfb (some providers may not return it)
            ttfb = result.get("ttfb")
            if ttfb is not None:
                ttfb_values.append(ttfb)

            # Prepare row data
            row_data = {
                "id": _id,
                "text": text,
                "audio_path": audio_path,
                "ttfb": ttfb,
            }

            # Append to CSV immediately for crash recovery
            row_df = pd.DataFrame([row_data])
            if exists(results_csv_path):
                row_df.to_csv(results_csv_path, mode="a", header=False, index=False)
            else:
                row_df.to_csv(results_csv_path, index=False)

            success_count += 1
            if ttfb is not None:
                log_and_print(f"\n\033[93m  TTFB: {ttfb:.3f}s\033[0m")

        except Exception as e:
            log_and_print(f"  Error synthesizing {_id}: {e}")
            logger.exception(f"Error synthesizing {_id}")
            continue

    return {
        "success_count": success_count,
        "ttfb_values": ttfb_values,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=True,
        choices=[
            "cartesia",
            "openai",
            "groq",
            "google",
            "elevenlabs",
            "sarvam",
            "smallest",
        ],
        help="TTS provider to use for evaluation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=[
            "english",
            "hindi",
            "kannada",
            "bengali",
            "malayalam",
            "marathi",
            "odia",
            "punjabi",
            "tamil",
            "telugu",
            "gujarati",
            "sindhi",
        ],
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the texts to synthesize",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first N texts only",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        help="Number of texts to run the evaluation on",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.provider)
    os.makedirs(output_dir, exist_ok=True)

    log_save_path = join(output_dir, "logs")
    if exists(log_save_path):
        os.remove(log_save_path)

    logger.remove()
    logger.add(log_save_path)

    print_log_save_path = join(output_dir, "results.log")
    if exists(print_log_save_path):
        os.remove(print_log_save_path)

    configure_print_logger(print_log_save_path)

    log_and_print("--------------------------------")
    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    # Validate language is supported by the provider
    try:
        validate_tts_language(args.language, args.provider)
    except ValueError as e:
        log_and_print(f"\033[31mError: {e}\033[0m")
        sys.exit(1)

    if not args.input.lower().endswith(".csv"):
        log_and_print(
            "\033[31mInput must be a CSV file path. Got: {}\033[0m".format(args.input)
        )
        sys.exit(1)

    df = pd.read_csv(args.input)

    ids = df["id"].tolist()
    texts = df["text"].tolist()

    if args.debug:
        ids = ids[: args.debug_count]
        texts = texts[: args.debug_count]

    gt_data = [{"id": _id, "text": text} for _id, text in zip(ids, texts)]

    results_csv_path = join(output_dir, "results.csv")

    log_and_print(f"Processing {len(gt_data)} texts with provider: {args.provider}")
    log_and_print("--------------------------------")

    # Run TTS evaluation
    eval_results = await run_tts_eval(
        gt_data=gt_data,
        provider=args.provider,
        language=args.language,
        output_dir=output_dir,
        results_csv_path=results_csv_path,
        overwrite=args.overwrite,
    )

    log_and_print("--------------------------------")
    log_and_print(f"Successfully synthesized: {eval_results['success_count']} texts")

    # Reload the final results from CSV
    if exists(results_csv_path):
        final_df = pd.read_csv(results_csv_path)
        all_ids = final_df["id"].tolist()
        all_texts = final_df["text"].tolist()
        all_audio_paths = final_df["audio_path"].tolist()
        all_ttfb = final_df["ttfb"].tolist()
    else:
        log_and_print("No results found")
        return

    # Run LLM judge evaluation
    log_and_print("Running LLM judge evaluation...")
    llm_judge_results = await get_tts_llm_judge_score(all_audio_paths, all_texts)
    log_and_print(f"LLM Judge Score: {llm_judge_results['score']}")

    # Build metrics data
    metrics_data = {
        "llm_judge_score": llm_judge_results["score"],
    }

    # Add ttfb metrics with mean, std, and values (filter out None/NaN values)
    valid_ttfb = [
        t
        for t in all_ttfb
        if t is not None and not (isinstance(t, float) and np.isnan(t))
    ]
    if valid_ttfb:
        metrics_data["ttfb"] = {
            "mean": float(np.mean(valid_ttfb)),
            "std": float(np.std(valid_ttfb)),
            "values": valid_ttfb,
        }

    # Save metrics
    metrics_save_path = join(output_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    log_and_print(f"Metrics saved to: {metrics_save_path}")

    # Update results CSV with LLM judge scores
    data = []
    for _id, text, audio_path, ttfb, llm_judge_score in zip(
        all_ids,
        all_texts,
        all_audio_paths,
        all_ttfb,
        llm_judge_results["per_row"],
    ):
        data.append(
            {
                "id": _id,
                "text": text,
                "audio_path": audio_path,
                "ttfb": ttfb,
                "llm_judge_score": llm_judge_score["match"],
                "llm_judge_reasoning": llm_judge_score["reasoning"],
            }
        )

    pd.DataFrame(data).to_csv(results_csv_path, index=False)
    log_and_print(f"Results saved to: {results_csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
