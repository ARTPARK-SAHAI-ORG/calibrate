import asyncio
import argparse
import sys
import os
import json
import base64
import httpx
from os.path import join, exists
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sarvamai import AsyncSarvamAI
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from groq import AsyncGroq
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from cartesia import AsyncCartesia

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
from google.api_core.client_options import ClientOptions

import time
import requests
from loguru import logger
import pandas as pd
import numpy as np
from natsort import natsorted

from calibrate.utils import (
    log_and_print,
    get_stt_language_code,
    validate_stt_language,
)
from calibrate.stt.metrics import (
    get_wer_score,
    get_llm_judge_score,
    get_string_similarity,
)


# =============================================================================
# STT Provider API Methods
# =============================================================================


def load_audio(audio_path: Path, as_file: bool = False):
    """
    Load audio file and convert to mono 16 kHz WAV format.

    Args:
        audio_path: Path to audio file.
        as_file: If True, return a file-like BytesIO object. If False, return bytes.

    Returns:
        Bytes or BytesIO of audio in mono, 16 kHz, 16-bit PCM WAV format.
    """
    import io

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio conversion. Install with 'pip install pydub'."
        )

    # Load audio using pydub (auto-detects format)
    audio = AudioSegment.from_file(audio_path)
    # Convert to mono, 16 kHz, 16-bit PCM
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio = audio.normalize()
    audio = audio.strip_silence(silence_len=100, silence_thresh=-40)

    # Export to WAV bytes
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")

    if as_file:
        out_io.seek(0)  # Reset position to start for reading
        out_io.name = "audio.wav"  # Set filename for APIs that need it
        return out_io

    return out_io.getvalue()


async def transcribe_deepgram(audio_path: Path, language: str) -> str:
    """Transcribe audio using Deepgram's REST API."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "deepgram")

    client = DeepgramClient(api_key=api_key)

    audio_file = load_audio(audio_path)

    options = PrerecordedOptions(model="nova-3", language=lang_code)

    payload: FileSource = {
        "buffer": audio_file,
    }

    response = await client.listen.asyncrest.v("1").transcribe_file(
        source=payload, options=options
    )
    transcript = response.results.channels[0].alternatives[0].transcript.strip()

    return {
        "transcript": transcript,
    }


async def transcribe_openai(audio_path: Path, language: str) -> str:
    """Transcribe audio using OpenAI's Whisper API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI()

    audio_file = load_audio(audio_path, as_file=True)

    response = await client.audio.transcriptions.create(
        model="gpt-4o-transcribe", file=audio_file
    )
    transcript = response.text

    return {
        "transcript": transcript,
    }


async def transcribe_groq(audio_path: Path, language: str) -> str:
    """Transcribe audio using Groq's Whisper API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "groq")

    client = AsyncGroq(api_key=api_key)

    audio_file = load_audio(audio_path, as_file=True)

    transcription = await client.audio.transcriptions.create(
        file=audio_file,  # Required audio file
        model="whisper-large-v3-turbo",  # Required model to use for transcription
        response_format="text",  # Optional
        language=lang_code,  # Optional
        temperature=0.0,  # Optional
    )

    return {
        "transcript": transcription.strip(),
    }


def _transcribe_google_streaming(
    audio_path: Path,
    lang_code: str,
    model: str = "chirp_3",
    region: str = "us",
) -> cloud_speech_types.StreamingRecognizeResponse:
    """Transcribes audio from an audio file stream using Google Cloud Speech-to-Text API.
    Args:
        stream_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        model (str): The model to use for transcription (default: chirp_3)
        region (str): The region for the API endpoint (default: us)
    Returns:
        list[cloud_speech_types.StreamingRecognizeResponse]: A list of objects.
            Each response includes the transcription results for the corresponding audio segment.
    """
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{region}-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    audio_content = load_audio(audio_path)

    # In practice, stream should be a generator yielding chunks of audio data
    # Chunk size must be < 25KB per Google STT API limitations
    # Use 24KB for a safe margin
    max_chunk_size = 24 * 1024  # 24KB = 24 * 1024 bytes
    stream = [
        audio_content[start : start + max_chunk_size]
        for start in range(0, len(audio_content), max_chunk_size)
    ]
    audio_requests = (
        cloud_speech_types.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech_types.RecognitionConfig(
        auto_decoding_config=cloud_speech_types.AutoDetectDecodingConfig(),
        language_codes=[lang_code],
        model=model,
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech_types.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{region}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech_types.RecognitionConfig, audio: list) -> list:
        yield config
        yield from audio

    # Transcribes the audio into text
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    all_interim_transcripts = []

    for response in responses_iterator:
        for result in response.results:
            interim_transcript = result.alternatives[0].transcript.strip()
            if not interim_transcript:
                continue

            all_interim_transcripts.append(interim_transcript)

    return " ".join(all_interim_transcripts)


async def transcribe_google(audio_path: Path, language: str) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text API."""
    from google.cloud import speech_v1 as speech

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    lang_code = get_stt_language_code(language, "google")

    # Use chirp_2 model and asia-southeast1 region for Sindhi
    if language.lower() == "sindhi":
        model = "chirp_2"
        region = "asia-southeast1"
    else:
        model = "chirp_3"
        region = "us"

    transcript = _transcribe_google_streaming(audio_path, lang_code, model, region)

    return {
        "transcript": transcript.strip(),
    }


async def transcribe_sarvam(audio_path: Path, language: str) -> str:
    """Transcribe audio using Sarvam's STT API."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "sarvam")

    audio_data = base64.b64encode(load_audio(audio_path)).decode("utf-8")

    client = AsyncSarvamAI(api_subscription_key=api_key)

    transcript = ""

    async with client.speech_to_text_streaming.connect(
        language_code=lang_code,
        model="saarika:v2.5",
        flush_signal=True,  # Enable manual control
    ) as ws:
        # Send audio
        await ws.transcribe(audio=audio_data, encoding="audio/wav", sample_rate=16000)

        # Force immediate processing
        await ws.flush()
        print("⚡ Processing forced - getting immediate results")
        # Get results
        async for message in ws:
            transcript = message.data.transcript
            processing_time = message.data.metrics.processing_latency
            break

    return {
        "transcript": transcript,
        "processing_time": processing_time,
    }


async def transcribe_elevenlabs(audio_path: Path, language: str) -> str:
    """Transcribe audio using ElevenLabs' STT API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "elevenlabs")

    elevenlabs = AsyncElevenLabs(api_key=api_key)

    audio_data = load_audio(audio_path)

    response = await elevenlabs.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v2",
        language_code=lang_code,
    )

    transcript = response.text

    return {
        "transcript": transcript,
    }


async def transcribe_cartesia(audio_path: Path, language: str) -> str:
    """Transcribe audio using Cartesia's STT API."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "cartesia")

    client = AsyncCartesia(api_key=api_key)

    try:
        # Create websocket connection with voice activity detection
        ws = await client.stt.websocket(
            model="ink-whisper",  # Model (required)
            language=lang_code,  # Language of your audio (required)
            encoding="pcm_s16le",  # Audio encoding format (required)
            sample_rate=16000,  # Audio sample rate (required)
            min_volume=0.15,  # Volume threshold for voice activity detection
            max_silence_duration_secs=0.3,  # Maximum silence duration before endpointing
        )

        # Simulate streaming audio data (replace with your audio source)
        async def audio_stream():
            """Simulate real-time audio streaming - replace with actual audio capture"""
            # Load audio file for simulation
            audio_data = load_audio(audio_path)

            # Stream in 100ms chunks (realistic for real-time processing)
            chunk_size = int(16000 * 0.1 * 2)  # 100ms at 16kHz, 16-bit

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if chunk:
                    yield chunk
                    # Simulate real-time streaming delay
                    await asyncio.sleep(0.1)

        # Send audio and receive results concurrently
        async def send_audio():
            """Send audio chunks to the STT websocket"""
            try:
                async for chunk in audio_stream():
                    await ws.send(chunk)
                    # print(f"Sent audio chunk of {len(chunk)} bytes")
                    # Small delay to simulate realtime applications
                    await asyncio.sleep(0.02)

                # Signal end of audio stream
                await ws.send("finalize")
                await ws.send("done")
                # print("Audio streaming completed")

            except Exception as e:
                print(f"Error sending audio: {e}")

        async def receive_transcripts():
            """Receive and process transcription results with word timestamps"""
            full_transcript = ""

            try:
                async for result in ws.receive():
                    if result["type"] == "transcript":
                        text = result["text"]
                        is_final = result["is_final"]

                        if is_final:
                            # Final result - this text won't change
                            full_transcript += text + " "
                            # print(f"FINAL: {text}")
                        # else:
                        # Partial result - may change as more audio is processed
                        # print(f"PARTIAL: {text}")

                    elif result["type"] == "done":
                        # print("Transcription completed")
                        break

            except Exception as e:
                print(f"Error receiving transcripts: {e}")

            return full_transcript.strip()

        # print("Starting streaming STT...")

        # Use asyncio.gather to run audio sending and transcript receiving concurrently
        _, (final_transcript) = await asyncio.gather(
            send_audio(), receive_transcripts()
        )

        # print(f"\nComplete transcript: {final_transcript}")
        # print(f"Total words with timestamps: {len(word_timestamps)}")

        # Clean up
        await ws.close()

        return {"transcript": final_transcript}

    except Exception as e:
        print(f"STT streaming error: {e}")
    finally:
        await client.close()


async def transcribe_smallest(audio_path: Path, language: str) -> str:
    """Transcribe audio using Smallest's STT API."""
    api_key = os.getenv("SMALLEST_API_KEY")
    if not api_key:
        raise ValueError("SMALLEST_API_KEY environment variable not set")

    lang_code = get_stt_language_code(language, "smallest")

    endpoint = "https://waves-api.smallest.ai/api/v1/pulse/get_text"
    params = {
        "model": "pulse",
        "language": lang_code,
        "word_timestamps": "false",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav",
    }

    audio = load_audio(audio_path)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            endpoint, params=params, headers=headers, content=audio
        )

    output = response.json()
    transcript = output.get("transcription", "")

    return {
        "transcript": transcript,
    }


# =============================================================================
# Main Transcription Router
# =============================================================================


async def transcribe_audio(
    audio_path: Path,
    provider: str,
    language: str,
) -> str:
    """Route audio transcription to the appropriate provider."""
    provider_methods = {
        "deepgram": transcribe_deepgram,
        "openai": transcribe_openai,
        "groq": transcribe_groq,
        "google": transcribe_google,
        "sarvam": transcribe_sarvam,
        "elevenlabs": transcribe_elevenlabs,
        "cartesia": transcribe_cartesia,
        "smallest": transcribe_smallest,
    }

    if provider not in provider_methods:
        raise ValueError(f"Unsupported STT provider: {provider}")

    method = provider_methods[provider]
    output = await method(audio_path, language)
    return output["transcript"].strip()


# =============================================================================
# STT Evaluation Main
# =============================================================================


async def run_stt_eval(
    gt_data: List[Dict],
    audio_dir: Path,
    provider: str,
    language: str,
    results_csv_path: Path,
) -> int:
    """Process audio files and save results immediately to CSV.

    Args:
        gt_data: List of {"id": ..., "gt": ...} for each file to process
        audio_dir: Directory containing audio files
        provider: STT provider name
        language: Language code
        results_csv_path: Path to save results CSV

    Returns:
        Number of files successfully transcribed (non-empty) in this run.
    """
    # Load existing results to skip already processed files
    if exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        results = existing_df.to_dict("records")
        processed_ids = set(existing_df["id"].tolist())
    else:
        results = []
        processed_ids = set()

    success_count = 0

    for i, gt_info in enumerate(gt_data):
        # Skip if already processed
        if gt_info["id"] in processed_ids:
            continue

        audio_path = audio_dir / f"{gt_info['id']}.wav"

        log_and_print(f"--------------------------------")
        log_and_print(f"Processing audio [{i + 1}/{len(gt_data)}]: {audio_path.name}")

        try:
            transcript = await transcribe_audio(audio_path, provider, language)
            log_and_print(f"\033[33mTranscript: {transcript}\033[0m")
            if transcript:
                success_count += 1
        except Exception as e:
            log_and_print(f"\033[91mFailed to transcribe {audio_path}: {e}\033[0m")
            raise e

        # Save immediately after each file
        results.append(
            {
                "id": gt_info["id"],
                "gt": gt_info["gt"],
                "pred": transcript,
            }
        )
        pd.DataFrame(results).to_csv(results_csv_path, index=False)

    return success_count


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=True,
        choices=[
            "deepgram",
            "openai",
            "cartesia",
            "smallest",
            "groq",
            "google",
            "sarvam",
            "elevenlabs",
        ],
        help="STT provider to use for evaluation",
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
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the audio files and stt.csv",
    )
    parser.add_argument(
        "-f",
        "--input-file-name",
        type=str,
        default="stt.csv",
        help="name of the input file containing the dataset to evaluate",
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
        help="Run the evaluation on the first 5 audio files",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of audio files to run the evaluation on",
    )
    parser.add_argument(
        "--ignore_retry",
        action="store_true",
        help="Ignore retrying if all the audios are not processed and move on to LLM judge",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )

    args = parser.parse_args()

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_dir = join(args.output_dir, args.provider)

    if not exists(output_dir):
        os.makedirs(output_dir)

    log_save_path = join(output_dir, "results.log")

    logger.remove()
    logger.add(
        log_save_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        colorize=False,
    )

    log_and_print("--------------------------------")

    command = " ".join(sys.argv)
    log_and_print(f"\033[33mRunning command\033[0m: {command}")

    # Validate language is supported by the provider
    try:
        validate_stt_language(args.language, args.provider)
    except ValueError as e:
        log_and_print(f"\033[31mError: {e}\033[0m")
        sys.exit(1)

    # Audio files are expected in audios/*.wav
    audio_dir = Path(args.input_dir) / "audios"
    gt_file = join(args.input_dir, args.input_file_name)
    results_csv_path = Path(output_dir) / "results.csv"

    # Delete existing results if overwrite is set
    if args.overwrite and exists(results_csv_path):
        os.remove(results_csv_path)
        log_and_print("Overwrite enabled - deleted existing results.csv")

    gt = pd.read_csv(gt_file)

    if args.debug:
        logger.debug(
            f"running in debug mode: using first {args.debug_count} audio files for evaluation"
        )
        gt = gt.head(args.debug_count)

    total_expected = len(gt)
    gt_data = [{"id": row["id"], "gt": row["text"]} for _, row in gt.iterrows()]

    # Process with retry loop
    previous_processed_count = -1

    while True:
        # Check current progress
        if exists(results_csv_path):
            current_df = pd.read_csv(results_csv_path)
            current_processed = len(current_df)

            if current_processed >= total_expected:
                log_and_print(f"All {total_expected} audio files processed")
                break

            log_and_print(f"Progress: {current_processed}/{total_expected} processed")
        else:
            current_processed = 0

        # Check if no progress was made
        if current_processed == previous_processed_count:
            log_and_print(
                f"No progress made - {total_expected - current_processed} files failed. "
                f"Saving empty transcripts and exiting."
            )
            # Add empty transcripts for unprocessed files
            if exists(results_csv_path):
                results = pd.read_csv(results_csv_path).to_dict("records")
                processed_ids = {r["id"] for r in results}
            else:
                results = []
                processed_ids = set()

            for gt_info in gt_data:
                if gt_info["id"] not in processed_ids:
                    results.append(
                        {"id": gt_info["id"], "gt": gt_info["gt"], "pred": ""}
                    )

            pd.DataFrame(results).to_csv(results_csv_path, index=False)
            break

        previous_processed_count = current_processed

        # Run transcription
        success_count = await run_stt_eval(
            gt_data=gt_data,
            audio_dir=audio_dir,
            provider=args.provider,
            language=args.language,
            results_csv_path=results_csv_path,
        )

        if args.ignore_retry:
            break

    # Load final results for metrics
    results_df = pd.read_csv(results_csv_path)
    all_ids = results_df["id"].tolist()
    all_gt_transcripts = results_df["gt"].tolist()
    all_pred_transcripts = results_df["pred"].fillna("").tolist()

    logger.info(f"gt_transcripts: {all_gt_transcripts}")
    logger.info(f"pred_transcripts: {all_pred_transcripts}")

    wer_results = get_wer_score(all_gt_transcripts, all_pred_transcripts)
    logger.info(f"WER: {wer_results['score']}")

    similarity_results = get_string_similarity(all_gt_transcripts, all_pred_transcripts)
    logger.info(f"String Similarity: {similarity_results['score']}")

    llm_results = await get_llm_judge_score(all_gt_transcripts, all_pred_transcripts)
    logger.info(f"LLM Judge Score: {llm_results['score']}")

    metrics_data = {
        "wer": wer_results["score"],
        "string_similarity": similarity_results["score"],
        "llm_judge_score": llm_results["score"],
    }

    data = []
    for _id, gt_text, pred_text, wer, sim, llm in zip(
        all_ids,
        all_gt_transcripts,
        all_pred_transcripts,
        wer_results["per_row"],
        similarity_results["per_row"],
        llm_results["per_row"],
    ):
        data.append(
            {
                "id": _id,
                "gt": gt_text,
                "pred": pred_text,
                "wer": wer,
                "string_similarity": sim,
                "llm_judge_score": llm["match"],
                "llm_judge_reasoning": llm["reasoning"],
            }
        )

    metrics_save_path = join(output_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    pd.DataFrame(data).to_csv(join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
