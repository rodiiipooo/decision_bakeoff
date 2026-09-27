"""Decider protocol and architecture registry."""

from __future__ import annotations

from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable

from decision_bakeoff.schema import Candidate, Example


@runtime_checkable
class Decider(Protocol):
    """score(state, candidates) -> {id: prob}."""

    name: str

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ...


def get_architecture(arch_id: str, **kwargs):
    """Factory for A0–A5 architectures."""
    from decision_bakeoff.deciders import arch as _arch

    factories = {
        "A0": _arch.RouterBaselineArch,
        "A1": _arch.FlatMean,
        "A2": _arch.ZScoreMean,
        "A3": _arch.StackedGate,
        "A4": _arch.MixtureOfDeciders,
        "A5": _arch.AdapterBank,
    }
    if arch_id not in factories:
        raise KeyError(f"unknown architecture: {arch_id}")
    return factories[arch_id](**kwargs)


__all__ = ["Decider", "get_architecture"]
