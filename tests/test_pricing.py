import unittest


class TestPricingResolver(unittest.TestCase):
    def test_resolves_default_stt_provider_pricing(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "openai")

        self.assertEqual(pricing["billing_unit"], "audio_minute")
        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["price_per_unit_usd"], 0.006)
        self.assertEqual(pricing["pricing_source"], "calibrate_default")

    def test_resolves_default_stt_model_pricing(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing("stt", "openai", model="gpt-4o-transcribe")

        self.assertEqual(pricing["billing_unit"], "audio_minute")
        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["price_per_unit_usd"], 0.006)

    def test_nested_override_wins(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing(
            "stt",
            "openai",
            overrides={"stt": {"openai": {"price_per_minute_usd": 0.01}}},
        )

        self.assertEqual(pricing["price_per_unit_usd"], 0.01)
        self.assertEqual(pricing["pricing_source"], "config_override")

    def test_model_override_wins(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing(
            "stt",
            "openai",
            model="gpt-4o-transcribe",
            overrides={
                "stt": {
                    "openai": {
                        "models": {
                            "gpt-4o-transcribe": {
                                "billing_unit": "audio_minute",
                                "price_per_unit_usd": 0.03,
                            }
                        }
                    }
                }
            },
        )

        self.assertEqual(pricing["model"], "gpt-4o-transcribe")
        self.assertEqual(pricing["price_per_unit_usd"], 0.03)
        self.assertEqual(pricing["pricing_source"], "config_override")

    def test_bare_override_shape_is_supported(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing(
            "stt",
            "openai",
            overrides={"price_per_minute_usd": 0.02},
        )

        self.assertEqual(pricing["price_per_unit_usd"], 0.02)
        self.assertEqual(pricing["pricing_source"], "config_override")

    def test_invalid_override_falls_back_to_default(self):
        from calibrate.pricing import resolve_pricing

        pricing = resolve_pricing(
            "stt",
            "openai",
            overrides={"openai": {"price_per_minute_usd": "bad"}},
        )

        self.assertEqual(pricing["price_per_unit_usd"], 0.006)
        self.assertEqual(pricing["pricing_source"], "calibrate_default")

    def test_unknown_provider_without_override_returns_none(self):
        from calibrate.pricing import resolve_pricing

        self.assertIsNone(resolve_pricing("stt", "unknown"))
