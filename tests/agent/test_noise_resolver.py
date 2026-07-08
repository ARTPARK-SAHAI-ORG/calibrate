from calibrate_agent.agent.noise.schema import LOUDNESS_VOLUME, NoiseAtom
from calibrate_agent.agent.noise.resolver import ResolvedNoise, resolve_for_run


def _resolve(cfg, run_index=0, base_seed=42, language="english"):
    return resolve_for_run(
        cfg, language=language, run_index=run_index, base_seed=base_seed
    )


def test_off_and_none_return_none():
    assert _resolve(None) is None
    assert _resolve("off") is None
    assert _resolve({"mode": "off"}) is None


def test_determinism_same_inputs():
    cfg = {"mode": "random", "seed": 1}
    a = _resolve(cfg, run_index=3, base_seed=99)
    b = _resolve(cfg, run_index=3, base_seed=99)
    assert a == b
    assert isinstance(a, ResolvedNoise)


def test_varies_across_run_index():
    cfg = {"mode": "random"}
    results = {
        (
            _resolve(cfg, run_index=i, base_seed=5).atom.environment,
            _resolve(cfg, run_index=i, base_seed=5).atom.people,
            _resolve(cfg, run_index=i, base_seed=5).atom.loudness,
        )
        for i in range(20)
    }
    assert len(results) > 1


def test_fixed_returns_exact_atom():
    cfg = {
        "mode": "fixed",
        "environment": "office",
        "people": "light",
        "loudness": "loud",
    }
    res = _resolve(cfg)
    assert res.atom == NoiseAtom(
        environment="office", people="light", loudness="loud"
    )
    assert res.volume == LOUDNESS_VOLUME["loud"]


def test_fixed_loudness_list_sampled():
    cfg = {"mode": "fixed", "environment": "office", "loudness": ["faint", "harsh"]}
    res = _resolve(cfg)
    assert res.atom.loudness in {"faint", "harsh"}
    assert res.volume == LOUDNESS_VOLUME[res.atom.loudness]


def test_clean_fraction_half_seeded_count():
    cfg = {"mode": "random", "clean_fraction": 0.5}
    n = 200
    clean = sum(
        1
        for i in range(n)
        if _resolve(cfg, run_index=i, base_seed=123) is None
    )
    # Seeded -> deterministic; assert the exact count and that it's roughly half.
    assert 80 < clean < 120
    clean2 = sum(
        1
        for i in range(n)
        if _resolve(cfg, run_index=i, base_seed=123) is None
    )
    assert clean == clean2


def test_mixture_weight_one_always_chosen():
    cfg = {
        "mode": "mixture",
        "conditions": [
            {"weight": 1, "spec": {"environment": "rain", "loudness": "faint"}},
        ],
    }
    for i in range(30):
        res = _resolve(cfg, run_index=i)
        assert res is not None
        assert res.atom.environment == "rain"
        assert res.atom.loudness == "faint"


def test_mixture_off_condition_returns_none():
    cfg = {
        "mode": "mixture",
        "conditions": [{"weight": 1, "spec": "off"}],
    }
    for i in range(10):
        assert _resolve(cfg, run_index=i) is None


def test_random_env_none_maps_to_none():
    cfg = {"mode": "random", "environments": ["none"], "people": ["none"]}
    res = _resolve(cfg)
    assert res is not None
    assert res.atom.environment is None


def test_volume_matches_table():
    for level in ("faint", "moderate", "loud", "harsh"):
        cfg = {"mode": "fixed", "environment": "rain", "loudness": level}
        res = _resolve(cfg)
        assert res.volume == LOUDNESS_VOLUME[level]


def test_base_seed_none_ok():
    res = resolve_for_run(
        {"mode": "fixed", "environment": "rain"},
        language="hindi",
        run_index=0,
        base_seed=None,
    )
    assert res is not None
    assert res.seed == 0
