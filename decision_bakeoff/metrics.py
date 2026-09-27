"""Evaluation metrics for decision bakeoff (stdlib + optional math)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_probs(scores: Mapping[str, float], candidate_ids: Sequence[str]) -> Dict[str, float]:
    """Normalize scores to a probability distribution over candidate_ids."""
    vals = [float(scores.get(cid, 0.0)) for cid in candidate_ids]
    # Softmax for stability if values look like logits / unnormalized
    if not vals:
        return {}
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    if s <= 0:
        n = len(candidate_ids)
        return {cid: 1.0 / n for cid in candidate_ids}
    return {cid: e / s for cid, e in zip(candidate_ids, exps)}


def top1_accuracy(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
) -> float:
    if not predictions:
        return 0.0
    correct = 0
    for probs, gold, cids in zip(predictions, golds, candidate_id_lists):
        if not cids:
            continue
        # Prefer explicit probs; if missing, treat as 0
        ranked = sorted(cids, key=lambda c: float(probs.get(c, 0.0)), reverse=True)
        if ranked and ranked[0] == gold:
            correct += 1
    return correct / len(predictions)


def pick_gold_rate(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
) -> float:
    """Fraction of examples where gold is among the scored candidates with mass > 0."""
    if not predictions:
        return 0.0
    hits = 0
    for probs, gold, cids in zip(predictions, golds, candidate_id_lists):
        if gold in cids and float(probs.get(gold, 0.0)) > 0:
            hits += 1
    return hits / len(predictions)


def ndcg_at_k(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
    k: int = 3,
) -> float:
    if not predictions:
        return 0.0
    scores: List[float] = []
    for probs, gold, cids in zip(predictions, golds, candidate_id_lists):
        ranked = sorted(cids, key=lambda c: float(probs.get(c, 0.0)), reverse=True)[:k]
        dcg = 0.0
        for i, cid in enumerate(ranked):
            rel = 1.0 if cid == gold else 0.0
            dcg += rel / math.log2(i + 2)
        # Ideal DCG for single relevant item is 1/log2(2) = 1
        idcg = 1.0
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores)


def ece_stub(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error stub using top-1 confidence bins."""
    if not predictions:
        return 0.0
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for probs, gold, cids in zip(predictions, golds, candidate_id_lists):
        if not cids:
            continue
        ranked = sorted(cids, key=lambda c: float(probs.get(c, 0.0)), reverse=True)
        conf = float(probs.get(ranked[0], 0.0))
        correct = 1 if ranked[0] == gold else 0
        idx = min(n_bins - 1, max(0, int(conf * n_bins)))
        bins[idx].append((conf, correct))
    ece = 0.0
    n = len(predictions)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(a for _, a in b) / len(b)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)
    return ece


def epc_score(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
) -> float:
    """Expected Probability of Correct — mean assigned probability on gold."""
    if not predictions:
        return 0.0
    total = 0.0
    for probs, gold, cids in zip(predictions, golds, candidate_id_lists):
        if gold in cids:
            total += float(probs.get(gold, 0.0))
    return total / len(predictions)


@dataclass
class LatencyTracker:
    times_ms: List[float] = field(default_factory=list)

    def record(self, start: float, end: Optional[float] = None) -> float:
        end = end if end is not None else time.perf_counter()
        ms = (end - start) * 1000.0
        self.times_ms.append(ms)
        return ms

    def summary(self) -> Dict[str, float]:
        if not self.times_ms:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        xs = sorted(self.times_ms)
        n = len(xs)

        def pct(p: float) -> float:
            i = min(n - 1, max(0, int(math.ceil(p * n) - 1)))
            return xs[i]

        return {
            "count": float(n),
            "mean_ms": sum(xs) / n,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
        }


def compute_all(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    candidate_id_lists: Sequence[Sequence[str]],
    latency: Optional[LatencyTracker] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "top1": top1_accuracy(predictions, golds, candidate_id_lists),
        "ndcg@3": ndcg_at_k(predictions, golds, candidate_id_lists, k=3),
        "pick_gold": pick_gold_rate(predictions, golds, candidate_id_lists),
        "ece_stub": ece_stub(predictions, golds, candidate_id_lists),
        "epc": epc_score(predictions, golds, candidate_id_lists),
        "n": len(predictions),
    }
    if latency is not None:
        out["latency"] = latency.summary()
    return out
