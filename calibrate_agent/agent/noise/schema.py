"""Config schema, constants, and normalization for voice-sim background noise."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 16000
LANGUAGES = ["english", "hindi", "kannada"]
SINGLE_SOUNDS = [
    "rain",
    "wind",
    "engine",
    "vacuum",
    "train",
    "siren",
    "crying_baby",
    "footsteps",
    "keyboard_typing",
    "laughing",
    "dog",
    "car_horn",
]
EVENT_SOUNDS = {"dog", "car_horn"}
SCENES = {
    "busy_street": ["car_horn", "engine", "siren"],
    "vehicle": ["engine", "wind", "car_horn"],
    "railway_station": ["train", "footsteps"],
    "office": ["keyboard_typing"],
    "home_with_baby": ["crying_baby"],
    "housework": ["vacuum"],
    "rainy_street": ["rain", "wind", "car_horn"],
    "quiet_home": ["dog"],
}
DENSITY_VOICES = {"none": 0, "single": 1, "light": 3, "medium": 5, "heavy": 10}
LOUDNESS_VOLUME = {"faint": 0.15, "moderate": 0.30, "loud": 0.55, "harsh": 0.85}
DENSITIES = ["none", "single", "light", "medium", "heavy"]
LOUDNESS_LEVELS = ["faint", "moderate", "loud", "harsh"]


@dataclass(frozen=True)
class NoiseAtom:
    environment: str | list[str] | None
    people: str = "none"
    loudness: str = "moderate"


def _validate_environment(env):
    """Validate an environment value; return it unchanged if valid."""
    if env is None or env == "none":
        return env
    if isinstance(env, str):
        if env in SINGLE_SOUNDS or env in SCENES:
            return env
        raise ValueError(
            f"Unknown environment {env!r}. Must be a single sound "
            f"({', '.join(SINGLE_SOUNDS)}), a scene ({', '.join(SCENES)}), "
            f"'none', or a list of single sounds."
        )
    if isinstance(env, list):
        for item in env:
            if item not in SINGLE_SOUNDS:
                raise ValueError(
                    f"Environment list entry {item!r} is not a known single "
                    f"sound ({', '.join(SINGLE_SOUNDS)})."
                )
        return list(env)
    raise ValueError(
        f"Environment must be a string, list of sound names, or None; got "
        f"{type(env).__name__}."
    )


def _validate_people(people):
    if people not in DENSITIES:
        raise ValueError(
            f"Unknown people density {people!r}. Must be one of "
            f"{', '.join(DENSITIES)}."
        )
    return people


def _validate_loudness(loudness):
    """Validate loudness: a single level, a list of levels, or 'any'."""
    if loudness == "any":
        return loudness
    if isinstance(loudness, str):
        if loudness in LOUDNESS_LEVELS:
            return loudness
        raise ValueError(
            f"Unknown loudness {loudness!r}. Must be one of "
            f"{', '.join(LOUDNESS_LEVELS)}, a list of them, or 'any'."
        )
    if isinstance(loudness, list):
        for item in loudness:
            if item not in LOUDNESS_LEVELS:
                raise ValueError(
                    f"Loudness list entry {item!r} is not a known level "
                    f"({', '.join(LOUDNESS_LEVELS)})."
                )
        return list(loudness)
    raise ValueError(
        f"Loudness must be a string, list of levels, or 'any'; got "
        f"{type(loudness).__name__}."
    )


def _normalize_atom(raw):
    """Normalize a plain atom dict (environment/people/loudness)."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an atom dict, got {type(raw).__name__}.")
    environment = _validate_environment(raw.get("environment"))
    people = _validate_people(raw.get("people", "none"))
    loudness = _validate_loudness(raw.get("loudness", "moderate"))
    return {
        "environment": environment,
        "people": people,
        "loudness": loudness,
    }


def normalize_noise_config(raw) -> dict | None:
    """Normalize ``config.get("noise")`` into a canonical dict, or None.

    Returns None for a disabled config (None / "off" / {"mode": "off"}).
    Otherwise returns a dict with a ``"mode"`` key in
    {"fixed", "random", "mixture"}.
    """
    if raw is None:
        return None
    if raw == "off":
        return None
    if raw == "random":
        return {"mode": "random", "clean_fraction": 0.0}

    if not isinstance(raw, dict):
        raise ValueError(
            f"noise config must be None, 'off', 'random', or a dict; got "
            f"{type(raw).__name__}."
        )

    mode = raw.get("mode")
    if mode == "off":
        return None

    if mode is None:
        atom = _normalize_atom(raw)
        return {"mode": "fixed", **atom}

    if mode == "sweep":
        raise ValueError(
            "noise mode 'sweep' is not supported. Use 'fixed', 'random', or "
            "'mixture'."
        )

    if mode == "fixed":
        atom = _normalize_atom(raw)
        return {"mode": "fixed", **atom}

    if mode == "random":
        clean_fraction = raw.get("clean_fraction", 0.0)
        if not isinstance(clean_fraction, (int, float)) or isinstance(
            clean_fraction, bool
        ):
            raise ValueError("random clean_fraction must be a number in [0, 1].")
        if not 0.0 <= clean_fraction <= 1.0:
            raise ValueError(
                f"random clean_fraction must be in [0, 1]; got {clean_fraction}."
            )
        out: dict = {"mode": "random", "clean_fraction": float(clean_fraction)}
        if "environments" in raw:
            envs = raw["environments"]
            if not isinstance(envs, list):
                raise ValueError("random 'environments' pool must be a list.")
            out["environments"] = [_validate_environment(e) for e in envs]
        if "people" in raw:
            people = raw["people"]
            if not isinstance(people, list):
                raise ValueError("random 'people' pool must be a list.")
            out["people"] = [_validate_people(p) for p in people]
        if "loudness" in raw:
            loud = raw["loudness"]
            if loud == "any":
                out["loudness"] = "any"
            elif isinstance(loud, list):
                for item in loud:
                    if item not in LOUDNESS_LEVELS:
                        raise ValueError(
                            f"random loudness entry {item!r} is not a known "
                            f"level ({', '.join(LOUDNESS_LEVELS)})."
                        )
                out["loudness"] = list(loud)
            else:
                raise ValueError(
                    "random 'loudness' pool must be a list of levels or 'any'."
                )
        if "seed" in raw:
            seed = raw["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("random 'seed' must be an int.")
            out["seed"] = seed
        return out

    if mode == "mixture":
        conditions = raw.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(
                "mixture 'conditions' must be a non-empty list of "
                "{weight, spec} entries."
            )
        norm_conditions = []
        for cond in conditions:
            if not isinstance(cond, dict):
                raise ValueError("Each mixture condition must be a dict.")
            weight = cond.get("weight")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValueError("Each mixture condition needs a numeric 'weight'.")
            if weight < 0:
                raise ValueError("mixture condition 'weight' must be >= 0.")
            spec = cond.get("spec")
            if spec == "off" or spec is None:
                norm_spec: str | dict = "off"
            else:
                norm_spec = _normalize_atom(spec)
            norm_conditions.append({"weight": float(weight), "spec": norm_spec})
        total = sum(c["weight"] for c in norm_conditions)
        if total <= 0:
            raise ValueError("mixture condition weights must sum to > 0.")
        return {"mode": "mixture", "conditions": norm_conditions}

    raise ValueError(
        f"Unknown noise mode {mode!r}. Must be 'fixed', 'random', 'mixture', "
        f"or 'off'."
    )
