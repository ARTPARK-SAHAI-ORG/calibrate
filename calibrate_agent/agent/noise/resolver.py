"""Per-run resolution of a normalized noise config into a concrete draw."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrate_agent.agent.noise.schema import (
    DENSITIES,
    LOUDNESS_LEVELS,
    LOUDNESS_VOLUME,
    SCENES,
    SINGLE_SOUNDS,
    NoiseAtom,
    normalize_noise_config,
)


@dataclass(frozen=True)
class ResolvedNoise:
    atom: NoiseAtom
    volume: float
    seed: int


def _resolve_loudness(loudness, rng) -> str:
    """Collapse a loudness spec (level / list / 'any') to a single level."""
    if loudness == "any":
        return str(rng.choice(LOUDNESS_LEVELS))
    if isinstance(loudness, list):
        if not loudness:
            raise ValueError("Loudness pool is empty.")
        return str(rng.choice(loudness))
    return loudness


def resolve_for_run(
    noise_cfg,
    *,
    language: str,
    run_index: int,
    base_seed: int | None,
) -> ResolvedNoise | None:
    """Deterministically resolve the noise config for a single run.

    ``noise_cfg`` is the raw ``config["noise"]``. Returns None when this run
    should be clean. Deterministic: the same (run_index, base_seed) always
    yields the same result.
    """
    normalized = normalize_noise_config(noise_cfg)
    if normalized is None:
        return None

    seed = (base_seed or 0) * 1_000_003 + run_index
    rng = np.random.default_rng(seed)

    mode = normalized["mode"]

    if mode == "fixed":
        loudness = _resolve_loudness(normalized.get("loudness", "moderate"), rng)
        atom = NoiseAtom(
            environment=normalized.get("environment"),
            people=normalized.get("people", "none"),
            loudness=loudness,
        )
        return ResolvedNoise(atom=atom, volume=LOUDNESS_VOLUME[loudness], seed=seed)

    if mode == "random":
        clean_fraction = normalized.get("clean_fraction", 0.0)
        if clean_fraction > 0.0 and float(rng.random()) < clean_fraction:
            return None
        env_pool = normalized.get(
            "environments", SINGLE_SOUNDS + list(SCENES) + ["none"]
        )
        people_pool = normalized.get("people", DENSITIES)
        loud_pool = normalized.get("loudness", LOUDNESS_LEVELS)

        environment = str(rng.choice(env_pool))
        if environment == "none":
            environment = None  # type: ignore[assignment]
        people = str(rng.choice(people_pool))
        loudness = _resolve_loudness(loud_pool, rng)
        atom = NoiseAtom(environment=environment, people=people, loudness=loudness)
        return ResolvedNoise(atom=atom, volume=LOUDNESS_VOLUME[loudness], seed=seed)

    if mode == "mixture":
        conditions = normalized["conditions"]
        weights = np.array([c["weight"] for c in conditions], dtype=float)
        weights = weights / weights.sum()
        idx = int(rng.choice(len(conditions), p=weights))
        spec = conditions[idx]["spec"]
        if spec == "off":
            return None
        loudness = _resolve_loudness(spec.get("loudness", "moderate"), rng)
        atom = NoiseAtom(
            environment=spec.get("environment"),
            people=spec.get("people", "none"),
            loudness=loudness,
        )
        return ResolvedNoise(atom=atom, volume=LOUDNESS_VOLUME[loudness], seed=seed)

    raise ValueError(f"Unhandled noise mode {mode!r}.")
