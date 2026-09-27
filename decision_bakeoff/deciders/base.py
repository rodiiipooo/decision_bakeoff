"""Shared helpers for deciders."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from decision_bakeoff.schema import Candidate


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bow_vector(text: str, vocab: Mapping[str, int] | None = None) -> Dict[str, float]:
    toks = tokenize(text)
    counts = Counter(toks)
    if not counts:
        return {}
    total = float(sum(counts.values()))
    vec = {t: c / total for t, c in counts.items()}
    if vocab is not None:
        return {t: vec.get(t, 0.0) for t in vocab}
    return vec


def hash_embed(text: str, dim: int = 64) -> List[float]:
    """Simple deterministic hash embedding (no external deps)."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hash(tok)  # stable within process; fine for offline smoke
        # Use a more portable murmur-like mix with built-in hash of salted key
        h = 0
        for ch in tok:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def softmax_from_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    ids = list(scores.keys())
    vals = [float(scores[i]) for i in ids]
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps) or 1.0
    return {i: e / s for i, e in zip(ids, exps)}


def mean_probs(
    member_probs: Iterable[Mapping[str, float]], candidate_ids: Sequence[str]
) -> Dict[str, float]:
    ids = list(candidate_ids)
    if not ids:
        return {}
    acc = {cid: 0.0 for cid in ids}
    n = 0
    for probs in member_probs:
        n += 1
        for cid in ids:
            acc[cid] += float(probs.get(cid, 0.0))
    if n == 0:
        u = 1.0 / len(ids)
        return {cid: u for cid in ids}
    return {cid: acc[cid] / n for cid in ids}


def state_text(state: Mapping) -> str:
    """Flatten state dict into a query-like string."""
    parts: List[str] = []
    for k in sorted(state.keys()):
        v = state[k]
        if isinstance(v, (str, int, float)):
            parts.append(f"{k} {v}")
        elif isinstance(v, list):
            parts.append(f"{k} " + " ".join(str(x) for x in v))
        elif isinstance(v, dict):
            parts.append(f"{k} " + " ".join(f"{a}:{b}" for a, b in sorted(v.items())))
    return " ".join(parts)


def uniform(candidates: Sequence[Candidate]) -> Dict[str, float]:
    if not candidates:
        return {}
    u = 1.0 / len(candidates)
    return {c.id: u for c in candidates}
