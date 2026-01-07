import asyncio
import websockets
import json
import csv
import itertools
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from jiwer import wer
import os
from os.path import join, exists
from loguru import logger

english_normalizer = EnglishTextNormalizer()
other_language_normalizer = BasicTextNormalizer()


async def transcribe_audio(audio_file, language):
    with open(audio_file, "rb") as f:
        audio_data = f.read()

    params = {
        "audioLanguage": language,  # Change to your language
        "audioEncoding": "linear16",  # 16-bit PCM
        "audioSampleRate": "16000",  # sample rate of the audio file
        "audioChannels": "1",
        "addPunctuation": "true",
        "api_key": os.getenv("SMALLEST_API_KEY"),
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"wss://waves-api.smallest.ai/api/v1/asr?{query_string}"

    transcription = []

    async with websockets.connect(url) as ws:

        async def listen():
            async for message in ws:
                response = json.loads(message)
                logger.debug(f"Response: {response}")
                if "text" in response and response["isFinal"]:
                    transcription.append(response["text"])

        listen_task = asyncio.create_task(listen())

        chunk_size = int(16000 * 2 * 0.3)  # 16kHz × 2 bytes × 0.3s
        while audio_data:
            chunk, audio_data = audio_data[:chunk_size], audio_data[chunk_size:]
            await ws.send(chunk)
            await asyncio.sleep(0.3)

        await ws.send(b"")  # End of stream
        await asyncio.sleep(2)
        listen_task.cancel()

    return " ".join(transcription)


def calculate_wer(reference, hypothesis, language="en"):
    if language == "en":
        ref_normalized = english_normalizer(reference)
        hyp_normalized = english_normalizer(hypothesis)
    else:
        ref_normalized = other_language_normalizer(reference)
        hyp_normalized = other_language_normalizer(hypothesis)
    return wer(ref_normalized, hyp_normalized)


async def main(input_csv, output_dir, language, debug):
    output_dir = join(output_dir, f"smallest_{language}")

    if not exists(output_dir):
        os.makedirs(output_dir)

    log_save_path = join(output_dir, "logs")

    if exists(log_save_path):
        os.remove(log_save_path)

    logger.add(log_save_path)

    output_csv = join(output_dir, "smallest_results.csv")

    results = []

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)

        if debug:
            reader = list(itertools.islice(reader, 5))

        for index, row in enumerate(reader):
            logger.debug(f"Processing row #{index + 1}: {row['text']}")
            audio_file = row["audio_path"]
            reference_text = row.get("text", "")

            transcript = await transcribe_audio(audio_file, language)
            row["transcript"] = transcript

            logger.debug(f"Transcript: {transcript}")

            row["wer"] = calculate_wer(reference_text, transcript, language)
            results.append(row)

    if results:
        with open(output_csv, "w", newline="") as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    wer_scores = [row["wer"] for row in results]
    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        logger.debug(f"Average WER: {avg_wer:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", type=str, default="./samples/dataset.csv")
    parser.add_argument("-o", "--output-dir", type=str, default="./sample_output")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="en",
        choices=[
            "en",
            "hi",
        ],
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.input_csv, args.output_dir, args.language, args.debug))
