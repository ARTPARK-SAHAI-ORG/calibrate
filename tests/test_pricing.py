import unittest


class TestSTTPricingResolver(unittest.TestCase):
    def test_resolves_default_stt_provider_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "openai")

        self.assertEqual(pricing["billing_unit"], "minute")
        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["price_per_minute_usd"], 0.006)

    def test_resolves_explicit_stt_model_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "google", model="chirp_2")

        self.assertEqual(pricing["billing_unit"], "minute")
        self.assertEqual(pricing["model"], "chirp_2")
        self.assertEqual(pricing["price_per_minute_usd"], 0.016)

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
                self.assertIsInstance(pricing["price_per_minute_usd"], float)

    def test_provider_lookup_is_case_insensitive_for_canonical_model(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "OpenAI", model="gpt-4o-transcribe")

        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["price_per_minute_usd"], 0.006)


class TestTTSPricingResolver(unittest.TestCase):
    def test_resolves_default_tts_provider_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "openai")

        self.assertEqual(pricing["billing_unit"], "character")
        self.assertEqual(pricing["model"], "gpt-4o-mini-tts")
        self.assertEqual(pricing["price_per_million_chars_usd"], 15.0)

    def test_resolves_explicit_tts_model_pricing(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "google", model="gemini-2.5-flash-tts")

        self.assertEqual(pricing["billing_unit"], "character")
        self.assertEqual(pricing["model"], "gemini-2.5-flash-tts")
        self.assertEqual(pricing["price_per_million_chars_usd"], 30.0)

    def test_resolves_all_supported_tts_provider_defaults(self):
        from calibrate_agent.pricing import TTS_DEFAULT_MODELS, resolve_pricing

        for provider, model in TTS_DEFAULT_MODELS.items():
            with self.subTest(provider=provider):
                pricing = resolve_pricing("tts", provider)
                self.assertEqual(pricing["model"], model)
                self.assertIsInstance(pricing["price_per_million_chars_usd"], float)
                self.assertGreater(pricing["price_per_million_chars_usd"], 0)

    def test_provider_lookup_is_case_insensitive(self):
        from calibrate_agent.pricing import resolve_pricing

        pricing = resolve_pricing("tts", "OpenAI", model="gpt-4o-mini-tts")

        self.assertEqual(pricing["model"], "gpt-4o-mini-tts")
        self.assertEqual(pricing["price_per_million_chars_usd"], 15.0)


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
