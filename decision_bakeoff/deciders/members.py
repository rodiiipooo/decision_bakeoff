"""Member deciders used inside ensemble architectures."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from decision_bakeoff.deciders.base import (
    cosine,
    hash_embed,
    softmax_from_scores,
    state_text,
    tokenize,
    uniform,
)
from decision_bakeoff.schema import Candidate, Example


class RouterBaseline:
    """A0 / member: prefer example baseline_choice, else first candidate."""

    name = "RouterBaseline"

    def __init__(self) -> None:
        self._baseline: Optional[str] = None

    def bind_example(self, example: Example) -> None:
        self._baseline = example.baseline_choice

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        if not candidates:
            return {}
        choice = self._baseline
        ids = [c.id for c in candidates]
        if choice is None or choice not in ids:
            choice = ids[0]
        out = {cid: 0.0 for cid in ids}
        out[choice] = 1.0
        return out


class CosineDecider:
    """B2: BoW / hash-embedding cosine similarity between state and candidate text."""

    name = "CosineDecider"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        if not candidates:
            return {}
        q = hash_embed(state_text(state), self.dim)
        raw = {c.id: cosine(q, hash_embed(c.text, self.dim)) for c in candidates}
        # Shift into positive range then softmax
        shifted = {k: v + 1.0 for k, v in raw.items()}
        return softmax_from_scores(shifted)


class MockCLMDecider:
    """C0: Mock contrastive-LM — hash/BoW embeddings + cosine.

    Plug-in point for a real contrastive LM: replace `embed()` with a model
    forward that returns fixed-dim vectors; keep `score()` unchanged.
    """

    name = "MockCLMDecider"

    def __init__(self, dim: int = 96) -> None:
        self.dim = dim

    def embed(self, text: str) -> list:
        """Mock embedding. Swap this method for a real CLM encoder."""
        # Combine hash embed with light length / token-count features folded in
        base = hash_embed(text, self.dim)
        n_tok = float(len(tokenize(text)))
        # Mild length bias in last dim
        if self.dim:
            base = list(base)
            base[-1] = base[-1] + 0.01 * min(n_tok, 50.0)
            # re-normalize lightly
            from decision_bakeoff.deciders.base import cosine as _  # noqa: F401
            import math

            norm = math.sqrt(sum(v * v for v in base)) or 1.0
            base = [v / norm for v in base]
        return base

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        if not candidates:
            return {}
        q = self.embed(state_text(state))
        raw = {c.id: cosine(q, self.embed(c.text)) for c in candidates}
        shifted = {k: (v + 1.0) * 2.0 for k, v in raw.items()}
        return softmax_from_scores(shifted)


def default_members() -> list:
    return [RouterBaseline(), CosineDecider(), MockCLMDecider()]
