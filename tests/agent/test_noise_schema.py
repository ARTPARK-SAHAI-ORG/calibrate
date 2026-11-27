import pytest

from calibrate_agent.agent.noise.schema import (
    DENSITIES,
    DENSITY_VOICES,
    EVENT_SOUNDS,
    LANGUAGES,
    LOUDNESS_LEVELS,
    LOUDNESS_VOLUME,
    SAMPLE_RATE,
    SCENES,
    SINGLE_SOUNDS,
    NoiseAtom,
    normalize_noise_config,
)


def test_constants_sane():
    assert SAMPLE_RATE == 16000
    assert LANGUAGES == ["english", "hindi", "kannada"]
    assert EVENT_SOUNDS <= set(SINGLE_SOUNDS)
    assert set(DENSITY_VOICES) == set(DENSITIES)
    assert set(LOUDNESS_VOLUME) == set(LOUDNESS_LEVELS)
    for scene, sounds in SCENES.items():
        for s in sounds:
            assert s in SINGLE_SOUNDS, (scene, s)


def test_none_off_return_none():
    assert normalize_noise_config(None) is None
    assert normalize_noise_config("off") is None
    assert normalize_noise_config({"mode": "off"}) is None


def test_random_string_shorthand():
    out = normalize_noise_config("random")
    assert out["mode"] == "random"


def test_plain_atom_is_fixed():
    out = normalize_noise_config(
        {"environment": "rain", "people": "light", "loudness": "loud"}
    )
    assert out == {
        "mode": "fixed",
        "environment": "rain",
        "people": "light",
        "loudness": "loud",
    }


def test_fixed_defaults():
    out = normalize_noise_config({"mode": "fixed", "environment": "office"})
    assert out["mode"] == "fixed"
    assert out["people"] == "none"
    assert out["loudness"] == "moderate"


def test_fixed_scene_and_list_env():
    assert normalize_noise_config({"environment": "busy_street"})["environment"] == (
        "busy_street"
    )
    out = normalize_noise_config({"environment": ["rain", "wind"]})
    assert out["environment"] == ["rain", "wind"]


def test_fixed_env_none():
    out = normalize_noise_config({"environment": "none", "people": "heavy"})
    assert out["environment"] == "none"


def test_random_full():
    out = normalize_noise_config(
        {
            "mode": "random",
            "clean_fraction": 0.5,
            "environments": ["rain", "office"],
            "people": ["none", "heavy"],
            "loudness": ["faint", "loud"],
            "seed": 7,
        }
    )
    assert out["mode"] == "random"
    assert out["clean_fraction"] == 0.5
    assert out["environments"] == ["rain", "office"]
    assert out["people"] == ["none", "heavy"]
    assert out["loudness"] == ["faint", "loud"]
    assert out["seed"] == 7


def test_random_defaults_clean_fraction():
    out = normalize_noise_config({"mode": "random"})
    assert out["clean_fraction"] == 0.0


def test_random_loudness_any():
    out = normalize_noise_config({"mode": "random", "loudness": "any"})
    assert out["loudness"] == "any"


def test_mixture():
    out = normalize_noise_config(
        {
            "mode": "mixture",
            "conditions": [
                {"weight": 3, "spec": "off"},
                {"weight": 1, "spec": {"environment": "rain", "loudness": "loud"}},
            ],
        }
    )
    assert out["mode"] == "mixture"
    assert out["conditions"][0]["spec"] == "off"
    assert out["conditions"][0]["weight"] == 3.0
    assert out["conditions"][1]["spec"]["environment"] == "rain"


def test_sweep_rejected():
    with pytest.raises(ValueError, match="sweep"):
        normalize_noise_config({"mode": "sweep"})


def test_bad_env_name():
    with pytest.raises(ValueError, match="environment"):
        normalize_noise_config({"environment": "spaceship"})


def test_bad_env_list_entry():
    with pytest.raises(ValueError):
        normalize_noise_config({"environment": ["rain", "busy_street"]})


def test_bad_density():
    with pytest.raises(ValueError, match="density"):
        normalize_noise_config({"environment": "rain", "people": "tons"})


def test_bad_loudness():
    with pytest.raises(ValueError, match="loudness|Loudness"):
        normalize_noise_config({"environment": "rain", "loudness": "deafening"})


def test_bad_clean_fraction():
    with pytest.raises(ValueError):
        normalize_noise_config({"mode": "random", "clean_fraction": 1.5})


def test_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        normalize_noise_config({"mode": "wat"})


def test_mixture_needs_conditions():
    with pytest.raises(ValueError):
        normalize_noise_config({"mode": "mixture", "conditions": []})


def test_mixture_zero_total_weight():
    with pytest.raises(ValueError):
        normalize_noise_config(
            {"mode": "mixture", "conditions": [{"weight": 0, "spec": "off"}]}
        )


def test_atom_dataclass_frozen():
    atom = NoiseAtom(environment="rain")
    assert atom.people == "none"
    assert atom.loudness == "moderate"
    with pytest.raises(Exception):
        atom.people = "heavy"  # type: ignore[misc]
