"""
Sarvam LLM judges for STT — runnable example.

Demonstrates the two opt-in, LLM-based STT judges (both reached via the
``--sarvam-judges`` CLI flag) directly from the library:

  * LLM-WER / LLM-CER  (``get_llm_wer_cer_score``)
        Word-aligns reference vs. prediction, asks an LLM whether each
        *differing* segment is semantically / phonetically equivalent, forgives
        the equivalent ones, and re-scores WER/CER. Legitimate variation (e.g. a
        word written in English vs. the native script) stops counting as an
        error; genuine errors survive.

  * Intent & entity preservation  (``get_intent_entity_score``)
        Scores whether the transcription preserved the speaker's intent (0/1)
        and the key entities (0–1).

Both follow Sarvam AI's methodology:
  https://www.sarvam.ai/blogs/evaluating-indian-language-asr

Requirements:
  * ``OPENROUTER_API_KEY`` in your environment or a local ``.env`` (the judges
    call ``google/gemini-2.5-flash`` through OpenRouter).
  * First run downloads the ``openai/whisper-small`` processor used by the
    text normalizer.

Run from the repo root:
    uv run python examples/stt/sarvam_llm_judges.py
"""

import asyncio

from dotenv import load_dotenv

from calibrate_agent.stt.metrics import (
    get_wer_score,
    get_cer_score,
    get_llm_wer_cer_score,
    get_intent_entity_score,
)

# Hindi pairs: rows 0–1 differ only by legitimate variation an LLM should
# forgive; row 2 is a genuine error (closed -> open) that must survive.
LANGUAGE = "hindi"
REFERENCES = [
    "मुझे डॉक्टर से मिलना है",   # "I need to meet the doctor"
    "मेरा नाम अमित है",          # "my name is Amit"
    "बैंक कल बंद रहेगा",         # "the bank will be closed tomorrow"
]
PREDICTIONS = [
    "मुझे doctor से मिलना है",   # डॉक्टर -> doctor (cross-script) — equivalent
    "मेरा नाम अमित है",          # exact match
    "बैंक कल खुला रहेगा",        # बंद -> खुला (closed -> open) — genuine error
]


async def main() -> None:
    load_dotenv()

    # Baseline WER/CER — every difference counts as an error.
    base_wer = get_wer_score(REFERENCES, PREDICTIONS, language=LANGUAGE)
    base_cer = get_cer_score(REFERENCES, PREDICTIONS, language=LANGUAGE)
    print("Baseline (standard WER/CER)")
    print(f"  WER={base_wer['score']:.4f}  CER={base_cer['score']:.4f}")

    # LLM-WER/CER — legitimate variation forgiven, genuine errors kept.
    llm = await get_llm_wer_cer_score(REFERENCES, PREDICTIONS, language=LANGUAGE)
    print("\nLLM-WER / LLM-CER (equivalence-forgiven)")
    print(f"  LLM_WER={llm['llm_wer']:.4f}  LLM_CER={llm['llm_cer']:.4f}")
    for i, row in enumerate(llm["per_row"]):
        print(f"  row {i}: llm_wer={row['llm_wer']:.4f}")
        for seg in row["segments"]:
            verdict = "forgiven" if seg["equivalent"] else "kept as error"
            print(
                f"    {seg['reference']!r} vs {seg['prediction']!r} -> {verdict}: "
                f"{seg['reasoning']}"
            )

    # Intent & entity preservation.
    ie = await get_intent_entity_score(REFERENCES, PREDICTIONS, language=LANGUAGE)
    print("\nIntent & entity preservation")
    print(f"  intent={ie['intent']:.4f}  entity={ie['entity']:.4f}")

    print(
        "\nEquivalent from the CLI over a dataset of {id, gt, pred} rows:\n"
        "  calibrate-agent stt --eval-only \\\n"
        "      --dataset examples/stt/sample_dataset.json \\\n"
        "      -o ./out --language hindi --sarvam-judges"
    )


if __name__ == "__main__":
    asyncio.run(main())
