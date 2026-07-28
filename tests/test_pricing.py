import unittest
from unittest.mock import patch


class TestSTTPricingResolver(unittest.TestCase):
    def test_resolves_default_stt_provider_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "openai")

        self.assertEqual(pricing["billing_unit"], "minute")
        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["native_rate"], 0.006)

    def test_resolves_explicit_stt_model_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "google", model="chirp_2")

        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "chirp_2")
        self.assertEqual(pricing["native_rate"], 0.016)

    def test_resolves_supported_stt_provider_defaults_with_pricing(self):
        from calibrate_agent.pricing import resolve_pricing
        from calibrate_agent.utils import STT_PROVIDER_MODELS

        providers_without_per_minute_pricing = {"gemini"}
        for provider, model in STT_PROVIDER_MODELS.items():
            if provider in providers_without_per_minute_pricing:
                continue
            with self.subTest(provider=provider):
                pricing = resolve_pricing("stt", provider)
                self.assertEqual(pricing["model"], model)
                self.assertIsInstance(pricing["native_rate"], float)

    def test_provider_lookup_is_case_insensitive_for_canonical_model(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "OpenAI", model="gpt-4o-transcribe")

        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["native_rate"], 0.006)


class TestTTSPricingResolver(unittest.TestCase):
    def test_resolves_char_billed_tts_provider(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "groq")

        self.assertEqual(pricing["billing_unit"], "character")
        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "canopylabs/orpheus-v1-english")
        self.assertEqual(pricing["native_rate"], 22.0)

    def test_resolves_minute_billed_tts_provider(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "openai")

        self.assertEqual(pricing["billing_unit"], "minute")
        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "gpt-4o-mini-tts")
        self.assertEqual(pricing["native_rate"], 0.015)

    def test_resolves_explicit_tts_model_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "google", model="gemini-2.5-flash-tts")

        self.assertEqual(pricing["billing_unit"], "minute")
        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "gemini-2.5-flash-tts")
        self.assertEqual(pricing["native_rate"], 0.015)

    def test_resolves_all_supported_tts_provider_defaults(self):
        from calibrate_agent.pricing import TTS_DEFAULT_MODELS, resolve_pricing

        for provider, model in TTS_DEFAULT_MODELS.items():
            with self.subTest(provider=provider):
                pricing = resolve_pricing("tts", provider)
                self.assertEqual(pricing["model"], model)
                self.assertIsInstance(pricing["native_rate"], float)
                self.assertGreater(pricing["native_rate"], 0)

    def test_provider_lookup_is_case_insensitive(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "Groq", model="canopylabs/orpheus-v1-english")

        self.assertEqual(pricing["model"], "canopylabs/orpheus-v1-english")
        self.assertEqual(pricing["native_rate"], 22.0)


class TestSarvamINRPricing(unittest.TestCase):
    def test_stt_sarvam_resolves_in_inr(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "sarvam")

        self.assertEqual(pricing["currency"], "INR")
        self.assertEqual(pricing["model"], "saaras:v3")
        self.assertEqual(pricing["native_rate"], 0.5)

    def test_tts_sarvam_resolves_in_inr(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "sarvam")

        self.assertEqual(pricing["currency"], "INR")
        self.assertEqual(pricing["model"], "bulbul:v3")
        self.assertEqual(pricing["native_rate"], 3000.0)


class TestCostBreakdown(unittest.TestCase):
    def test_usd_breakdown_has_no_fx_fields(self):
        from calibrate_agent.pricing import cost_breakdown

        pricing = {"currency": "USD", "native_rate": 0.006}
        fields = cost_breakdown(pricing, 2.0, "cost_per_minute")

        self.assertEqual(fields["currency"], "USD")
        self.assertEqual(fields["cost_per_minute_currency"], 0.006)
        self.assertEqual(fields["cost_usd"], 0.012)
        self.assertNotIn("cost_in_currency", fields)
        self.assertNotIn("conversion_rate", fields)

    def test_inr_breakdown_includes_native_and_conversion(self):
        from calibrate_agent import pricing as P

        entry = {"currency": "INR", "native_rate": 3000.0}
        with patch.object(P, "get_usd_to_inr_rate", return_value=100.0):
            fields = P.cost_breakdown(entry, 2.0, "cost_per_million_chars")

        self.assertEqual(fields["currency"], "INR")
        self.assertEqual(fields["cost_per_million_chars_currency"], 3000.0)
        self.assertEqual(fields["cost_in_currency"], 6000.0)
        self.assertEqual(fields["conversion_rate"], 100.0)
        self.assertEqual(fields["cost_usd"], 60.0)

    def test_inr_breakdown_omits_usd_when_fx_unavailable(self):
        from calibrate_agent import pricing as P

        entry = {"currency": "INR", "native_rate": 3000.0, "provider": "sarvam"}
        with patch.object(P, "get_usd_to_inr_rate", side_effect=RuntimeError("no FX")):
            fields = P.cost_breakdown(entry, 2.0, "cost_per_million_chars")

        # Native-currency cost is still reported; USD is skipped, not raised.
        self.assertEqual(fields["currency"], "INR")
        self.assertEqual(fields["cost_per_million_chars_currency"], 3000.0)
        self.assertEqual(fields["cost_in_currency"], 6000.0)
        self.assertNotIn("cost_usd", fields)
        self.assertNotIn("conversion_rate", fields)


class TestLLMPricingResolver(unittest.TestCase):
    EXPECTED_RATES = {
        "openai/gpt-5.4-mini": (0.75, 4.5),
        "openai/gpt-audio": (2.5, 10.0),
        "google/gemini-2.5-flash": (0.3, 2.5),
        "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    }

    def test_resolves_text_rates_for_every_source(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        for model, (input_rate, output_rate) in self.EXPECTED_RATES.items():
            for source in ("openrouter", "direct"):
                with self.subTest(model=model, source=source):
                    pricing = resolve_llm_pricing(model, source=source)

                    self.assertEqual(pricing["model"], model)
                    self.assertEqual(pricing["source"], source)
                    self.assertEqual(
                        pricing["input_price_per_million_tokens_usd"], input_rate
                    )
                    self.assertEqual(
                        pricing["output_price_per_million_tokens_usd"], output_rate
                    )

    def test_openrouter_is_the_default_source(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        self.assertEqual(
            resolve_llm_pricing("openai/gpt-5.4-mini")["source"], "openrouter"
        )

    def test_audio_rates_present_only_for_audio_priced_models(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        audio_pricing = resolve_llm_pricing("openai/gpt-audio")
        self.assertEqual(
            audio_pricing["audio_input_price_per_million_tokens_usd"], 32.0
        )
        self.assertEqual(
            audio_pricing["audio_output_price_per_million_tokens_usd"], 64.0
        )

        gemini_pricing = resolve_llm_pricing("google/gemini-2.5-flash")
        self.assertEqual(
            gemini_pricing["audio_input_price_per_million_tokens_usd"], 1.0
        )

        for model in ("openai/gpt-5.4-mini", "anthropic/claude-sonnet-4.5"):
            with self.subTest(model=model):
                pricing = resolve_llm_pricing(model)
                self.assertNotIn("audio_input_price_per_million_tokens_usd", pricing)
                self.assertNotIn("audio_output_price_per_million_tokens_usd", pricing)

    def test_reasoning_billed_as_output_only_for_gemini(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        self.assertTrue(
            resolve_llm_pricing("google/gemini-2.5-flash")[
                "reasoning_billed_as_output"
            ]
        )
        for model in (
            "openai/gpt-5.4-mini",
            "openai/gpt-audio",
            "anthropic/claude-sonnet-4.5",
        ):
            with self.subTest(model=model):
                self.assertFalse(
                    resolve_llm_pricing(model)["reasoning_billed_as_output"]
                )

    def test_unknown_model_returns_none(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        self.assertIsNone(resolve_llm_pricing("acme/does-not-exist"))

    def test_unknown_source_raises(self):
        from calibrate_agent.pricing import resolve_llm_pricing

        with self.assertRaises(ValueError):
            resolve_llm_pricing("openai/gpt-5.4-mini", source="bedrock")

    def test_every_judge_default_model_is_priced(self):
        from calibrate_agent.judges import (
            DEFAULT_AUDIO_JUDGE_MODEL,
            DEFAULT_TEXT_JUDGE_MODEL,
        )
        from calibrate_agent.pricing import resolve_llm_pricing
        from calibrate_agent.stt.sarvam_intent_entity import DEFAULT_INTENT_ENTITY_MODEL
        from calibrate_agent.stt.sarvam_llm_wer import DEFAULT_LLM_WER_MODEL
        from calibrate_agent.stt.semantic_wer import DEFAULT_SEMANTIC_WER_MODEL

        for model in (
            DEFAULT_TEXT_JUDGE_MODEL,
            DEFAULT_AUDIO_JUDGE_MODEL,
            DEFAULT_INTENT_ENTITY_MODEL,
            DEFAULT_LLM_WER_MODEL,
            DEFAULT_SEMANTIC_WER_MODEL,
        ):
            with self.subTest(model=model):
                self.assertIsNotNone(resolve_llm_pricing(model))


class TestPricingResolverGuards(unittest.TestCase):
    def test_unknown_provider_returns_none(self):
        from calibrate_agent.pricing import resolve_pricing

        self.assertIsNone(resolve_pricing("stt", "unknown"))
        self.assertIsNone(resolve_pricing("tts", "unknown"))

    def test_unsupported_component_raises(self):
        from calibrate_agent.pricing import resolve_pricing

        with self.assertRaises(ValueError):
            resolve_pricing("llm", "openai")


if __name__ == "__main__":
    unittest.main()
