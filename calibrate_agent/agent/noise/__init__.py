"""Background-noise injection for voice simulations.

Generates a per-simulation background noise track (environmental sounds +
distant "backgroundified" speaker chatter) that is mixed under the simulated
caller's audio, so the tested agent's STT hears a realistic noisy caller.

See design/noise_injection_spec.md for the full design.
"""
