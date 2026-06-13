"""
Text normalization + score aggregation helpers.

Vendored from Sarvam AI's ``llm_intent_entity`` (src/llm_intent_entity/utilities.py).
Only the pieces the calibrate intent/entity flow uses are kept: the
``IndicNormalizer`` (applied to reference/prediction before judging) and the
``calculate_intent_accuracy`` / ``calculate_entity_metrics`` aggregators. The
upstream Google Sheets export and ``IndicASRPostProcessor`` are omitted.
"""

import re
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import WhisperProcessor

lang_to_code = {
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "odia": "or",
    "oria": "or",
    "assamese": "or",
    "punjabi": "pa",
    "english": "en",
}

indic_langs = {"hi", "bn", "ta", "te", "gu", "kn", "ml", "mr", "or", "pa"}


class IndicNormalizer:
    def __init__(self):
        self.indic_factory = IndicNormalizerFactory()
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_tokenizer = self.whisper_processor.tokenizer  # type: ignore

    def normalize_text(self, text: str, lang_code: str) -> str:
        lang_code = lang_to_code.get(lang_code, lang_code)
        if pd.isna(text) or not isinstance(text, str):
            return text
        if not text:
            return text
        base_lang = lang_code.split("-")[0].lower()
        text = re.sub(r"([,\-\.\(\)\[\]\{\}/\\])\B", r" ", text)
        INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´ۭ۪\u200b\u200c\u200d\u200e\u200f"
        text = text.translate(
            str.maketrans("", "", string.punctuation + INDIC_PUNCTUATION)
        ).lower()

        if base_lang in indic_langs and base_lang != "ur":  # urdu has special handling
            normalizer = self.indic_factory.get_normalizer(base_lang)
            text = normalizer.normalize(text)
        else:
            text = self.whisper_tokenizer.normalize(text)
        text = re.sub(" +", " ", text).strip()
        return text

    def _normalize_batch(
        self, text_batch: List[str], lang_batch: List[str]
    ) -> List[str]:
        return [
            self.normalize_text(text, lang)
            for text, lang in zip(text_batch, lang_batch)
        ]

    def normalize_texts(
        self,
        text_list: List[str],
        lang_list: List[str],
        n_jobs: int = -1,
        batch_size: int = 500,
    ) -> List[str]:
        if len(text_list) != len(lang_list):
            raise ValueError("text_list and lang_list must have the same length")

        if not text_list:
            return []

        batches: List[Tuple[List[str], List[str]]] = []
        for i in range(0, len(text_list), batch_size):
            batches.append((text_list[i : i + batch_size], lang_list[i : i + batch_size]))

        processed_batches = Parallel(n_jobs=n_jobs)(
            delayed(self._normalize_batch)(text_batch, lang_batch)
            for text_batch, lang_batch in tqdm(
                batches, desc="Normalizing text batches"
            )
        )

        if processed_batches:
            return [item for sublist in processed_batches for item in sublist]  # type: ignore
        return []


# METRICS FOR INTENT AND ENTITY EVALUATION


def calculate_intent_accuracy(intent_scores: List[int]) -> float:
    """Calculate accuracy for intent scores (0 or 1)"""
    if not intent_scores:
        return 0.0
    return sum(intent_scores) / len(intent_scores)


def calculate_entity_metrics(entity_scores: List[float]) -> dict:
    """Calculate metrics for entity scores (0 to 1)"""
    if not entity_scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    entity_scores_array = np.array(entity_scores)
    return {
        "mean": float(np.mean(entity_scores_array)),
        "median": float(np.median(entity_scores_array)),
        "std": float(np.std(entity_scores_array)),
    }
