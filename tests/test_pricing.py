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
    def test_resolves_default_tts_provider_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "openai")

        self.assertEqual(pricing["billing_unit"], "character")
        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "gpt-4o-mini-tts")
        self.assertEqual(pricing["native_rate"], 15.0)

    def test_resolves_explicit_tts_model_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "google", model="gemini-2.5-flash-tts")

        self.assertEqual(pricing["currency"], "USD")
        self.assertEqual(pricing["model"], "gemini-2.5-flash-tts")
        self.assertEqual(pricing["native_rate"], 30.0)

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

        pricing = resolve_pricing("tts", "OpenAI", model="gpt-4o-mini-tts")

        self.assertEqual(pricing["model"], "gpt-4o-mini-tts")
        self.assertEqual(pricing["native_rate"], 15.0)


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
        self.assertEqual(fields["cost_per_minute_usd"], 0.006)
        self.assertEqual(fields["cost_usd"], 0.012)
        self.assertNotIn("cost_in_currency", fields)
        self.assertNotIn("cost_per_usd", fields)

    def test_inr_breakdown_includes_native_and_fx(self):
        from calibrate_agent import pricing as P

        entry = {"currency": "INR", "native_rate": 3000.0}
        with patch.object(P, "get_usd_to_inr_rate", return_value=100.0):
            fields = P.cost_breakdown(entry, 2.0, "cost_per_million_chars")

        self.assertEqual(fields["currency"], "INR")
        self.assertEqual(fields["cost_per_million_chars_inr"], 3000.0)
        self.assertEqual(fields["cost_in_currency"], 6000.0)
        self.assertEqual(fields["cost_per_usd"], 100.0)
        self.assertEqual(fields["cost_usd"], 60.0)


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
