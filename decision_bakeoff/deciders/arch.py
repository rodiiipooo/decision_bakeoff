"""Architectures A0–A5 implementing the Decider protocol."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from decision_bakeoff.deciders.base import (
    mean_probs,
    softmax_from_scores,
    state_text,
    uniform,
)
from decision_bakeoff.deciders.members import (
    CosineDecider,
    MockCLMDecider,
    RouterBaseline,
    default_members,
)
from decision_bakeoff.schema import Candidate, Example


def _member_scores(
    members: Sequence[Any],
    state: Mapping,
    candidates: Sequence[Candidate],
    example: Optional[Example] = None,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for m in members:
        if isinstance(m, RouterBaseline) and example is not None:
            m.bind_example(example)
        out.append(m.score(state, candidates))
    return out


class RouterBaselineArch:
    """A0: use example baseline_choice or first candidate."""

    name = "A0_RouterBaseline"

    def __init__(self, **kwargs: Any) -> None:
        self._inner = RouterBaseline()
        self._example: Optional[Example] = None

    def fit(self, calib: Sequence[Example]) -> "RouterBaselineArch":
        return self

    def bind_example(self, example: Example) -> None:
        self._example = example
        self._inner.bind_example(example)

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        if self._example is not None:
            self._inner.bind_example(self._example)
        return self._inner.score(state, candidates)


class FlatMean:
    """A1: mean of member probabilities over shared candidate ids."""

    name = "A1_FlatMean"

    def __init__(self, members: Optional[list] = None, **kwargs: Any) -> None:
        self.members = members or default_members()
        self._example: Optional[Example] = None

    def fit(self, calib: Sequence[Example]) -> "FlatMean":
        return self

    def bind_example(self, example: Example) -> None:
        self._example = example

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ids = [c.id for c in candidates]
        probs = _member_scores(self.members, state, candidates, self._example)
        return mean_probs(probs, ids)


class ZScoreMean:
    """A2: per-model z-score on calib split, then mean of z-scored probs."""

    name = "A2_ZScoreMean"

    def __init__(self, members: Optional[list] = None, **kwargs: Any) -> None:
        self.members = members or default_members()
        self._mu: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}
        self._example: Optional[Example] = None
        self._fitted = False

    def fit(self, calib: Sequence[Example]) -> "ZScoreMean":
        # Collect top-1 prob per member on calib for z-score stats
        vals: Dict[str, List[float]] = defaultdict(list)
        for ex in calib:
            for m in self.members:
                if isinstance(m, RouterBaseline):
                    m.bind_example(ex)
                probs = m.score(ex.state, ex.candidates)
                if not probs:
                    continue
                top = max(probs.values())
                vals[m.name].append(float(top))
        for name, xs in vals.items():
            if not xs:
                self._mu[name] = 0.0
                self._sigma[name] = 1.0
                continue
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
            self._mu[name] = mu
            self._sigma[name] = math.sqrt(var) if var > 1e-12 else 1.0
        self._fitted = True
        return self

    def bind_example(self, example: Example) -> None:
        self._example = example

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ids = [c.id for c in candidates]
        if not ids:
            return {}
        z_maps: List[Dict[str, float]] = []
        for m in self.members:
            if isinstance(m, RouterBaseline) and self._example is not None:
                m.bind_example(self._example)
            probs = m.score(state, candidates)
            mu = self._mu.get(m.name, 0.0)
            sigma = self._sigma.get(m.name, 1.0) or 1.0
            # z-score each candidate's prob, then softmax to re-normalize
            z = {cid: (float(probs.get(cid, 0.0)) - mu) / sigma for cid in ids}
            z_maps.append(softmax_from_scores(z))
        return mean_probs(z_maps, ids)


class StackedGate:
    """A3: logistic (pure-numpy fallback) on member features.

    Features per example: for each member — top1 score, margin, entropy —
    plus a router-match flag. Optional sklearn if present; else hand-rolled
    logistic regression with gradient descent.
    """

    name = "A3_StackedGate"

    def __init__(self, members: Optional[list] = None, **kwargs: Any) -> None:
        self.members = members or default_members()
        self.weights: Optional[List[float]] = None
        self.bias: float = 0.0
        self._example: Optional[Example] = None
        self._use_sklearn = False
        self._sk_model = None

    @staticmethod
    def _entropy(probs: Mapping[str, float]) -> float:
        h = 0.0
        for p in probs.values():
            if p > 1e-12:
                h -= p * math.log(p + 1e-12)
        return h

    def _features(
        self,
        member_probs: Sequence[Mapping[str, float]],
        candidates: Sequence[Candidate],
        example: Optional[Example],
    ) -> List[float]:
        feats: List[float] = []
        ids = [c.id for c in candidates]
        for probs in member_probs:
            vals = sorted((float(probs.get(cid, 0.0)) for cid in ids), reverse=True)
            top1 = vals[0] if vals else 0.0
            second = vals[1] if len(vals) > 1 else 0.0
            margin = top1 - second
            ents = self._entropy({cid: float(probs.get(cid, 0.0)) for cid in ids})
            feats.extend([top1, margin, ents])
        # router match flag
        router_match = 0.0
        if example is not None and example.baseline_choice:
            if member_probs:
                router_probs = member_probs[0]
                ranked = sorted(ids, key=lambda c: float(router_probs.get(c, 0.0)), reverse=True)
                if ranked and ranked[0] == example.baseline_choice:
                    router_match = 1.0
        feats.append(router_match)
        return feats

    def fit(self, calib: Sequence[Example]) -> "StackedGate":
        X: List[List[float]] = []
        y: List[int] = []
        for ex in calib:
            mprobs = _member_scores(self.members, ex.state, ex.candidates, ex)
            feats = self._features(mprobs, ex.candidates, ex)
            # Label: whether flat-mean top1 matches gold (gate learns to trust ensemble)
            mean_p = mean_probs(mprobs, [c.id for c in ex.candidates])
            ranked = sorted(mean_p, key=mean_p.get, reverse=True)  # type: ignore[arg-type]
            label = 1 if ranked and ranked[0] == ex.gold else 0
            # Also train per-candidate ranking signal: use gold vs not via repeated rows
            X.append(feats)
            y.append(1 if ex.gold in {c.id for c in ex.candidates} and label == 1 else label)
            # Enrich: second view — gold pick indicator from cosine member
            X.append(feats)
            y.append(1 if ranked and ranked[0] == ex.gold else 0)

        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore

            clf = LogisticRegression(max_iter=200, solver="lbfgs")
            clf.fit(X, y)
            self._sk_model = clf
            self._use_sklearn = True
            self.weights = list(map(float, clf.coef_[0]))
            self.bias = float(clf.intercept_[0])
        except Exception:
            self._use_sklearn = False
            self.weights, self.bias = self._fit_logistic(X, y)
        return self

    @staticmethod
    def _fit_logistic(
        X: List[List[float]], y: List[int], lr: float = 0.1, epochs: int = 200
    ) -> Tuple[List[float], float]:
        if not X:
            return [], 0.0
        d = len(X[0])
        w = [0.0] * d
        b = 0.0
        n = float(len(X))
        for _ in range(epochs):
            gw = [0.0] * d
            gb = 0.0
            for x, yi in zip(X, y):
                z = b + sum(wj * xj for wj, xj in zip(w, x))
                # sigmoid
                if z >= 0:
                    p = 1.0 / (1.0 + math.exp(-z))
                else:
                    ez = math.exp(z)
                    p = ez / (1.0 + ez)
                err = p - yi
                for j in range(d):
                    gw[j] += err * x[j]
                gb += err
            for j in range(d):
                w[j] -= lr * (gw[j] / n)
            b -= lr * (gb / n)
        return w, b

    def _gate_prob(self, feats: List[float]) -> float:
        if self._use_sklearn and self._sk_model is not None:
            import numpy as np  # type: ignore

            return float(self._sk_model.predict_proba([feats])[0][1])
        if not self.weights:
            return 0.5
        z = self.bias + sum(wj * xj for wj, xj in zip(self.weights, feats))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def bind_example(self, example: Example) -> None:
        self._example = example

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ids = [c.id for c in candidates]
        if not ids:
            return {}
        mprobs = _member_scores(self.members, state, candidates, self._example)
        feats = self._features(mprobs, candidates, self._example)
        gate = self._gate_prob(feats)
        ens = mean_probs(mprobs, ids)
        # Blend ensemble with router (member 0) using gate
        router = mprobs[0] if mprobs else uniform(candidates)
        return {
            cid: gate * float(ens.get(cid, 0.0)) + (1.0 - gate) * float(router.get(cid, 0.0))
            for cid in ids
        }


class MixtureOfDeciders:
    """A4: mixture weights by problem_type (learned on calib or heuristic)."""

    name = "A4_MixtureOfDeciders"

    def __init__(self, members: Optional[list] = None, **kwargs: Any) -> None:
        self.members = members or default_members()
        # problem_type -> member_name -> weight
        self.table: Dict[str, Dict[str, float]] = {}
        self._example: Optional[Example] = None
        self._problem_type: Optional[str] = None

    def fit(self, calib: Sequence[Example]) -> "MixtureOfDeciders":
        # Count top-1 hits per (problem_type, member)
        hits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: Dict[str, int] = defaultdict(int)
        for ex in calib:
            totals[ex.problem_type] += 1
            for m in self.members:
                if isinstance(m, RouterBaseline):
                    m.bind_example(ex)
                probs = m.score(ex.state, ex.candidates)
                if not probs:
                    continue
                top = max(probs, key=probs.get)  # type: ignore[arg-type]
                if top == ex.gold:
                    hits[ex.problem_type][m.name] += 1
        names = [m.name for m in self.members]
        for pt, n in totals.items():
            raw = {name: hits[pt].get(name, 0) + 1.0 for name in names}  # Laplace
            s = sum(raw.values()) or 1.0
            self.table[pt] = {k: v / s for k, v in raw.items()}
        # Heuristic fallback for unseen types
        if not self.table:
            u = 1.0 / max(1, len(names))
            self.table["_default"] = {name: u for name in names}
        return self

    def bind_example(self, example: Example) -> None:
        self._example = example
        self._problem_type = example.problem_type

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ids = [c.id for c in candidates]
        if not ids:
            return {}
        mprobs = _member_scores(self.members, state, candidates, self._example)
        pt = self._problem_type or "_default"
        weights = self.table.get(pt) or self.table.get("_default")
        if not weights:
            return mean_probs(mprobs, ids)
        acc = {cid: 0.0 for cid in ids}
        for m, probs in zip(self.members, mprobs):
            w = float(weights.get(m.name, 0.0))
            for cid in ids:
                acc[cid] += w * float(probs.get(cid, 0.0))
        s = sum(acc.values()) or 1.0
        return {cid: acc[cid] / s for cid in ids}


class AdapterBank:
    """A5: MockCLM adapter bank — hash/BoW embeddings + cosine.

    Real contrastive-LM plug-in: replace MockCLMDecider.embed with your encoder;
    optionally keep a bank of adapters keyed by problem_type.
    """

    name = "A5_AdapterBank"

    def __init__(self, dim: int = 96, **kwargs: Any) -> None:
        self.dim = dim
        self.bank: Dict[str, MockCLMDecider] = {
            "default": MockCLMDecider(dim=dim),
        }
        self._cosine = CosineDecider(dim=64)
        self._example: Optional[Example] = None
        self._problem_type: Optional[str] = None
        # Per-type blend weight toward MockCLM vs Cosine (fit on calib)
        self.blend: Dict[str, float] = defaultdict(lambda: 0.7)

    def fit(self, calib: Sequence[Example]) -> "AdapterBank":
        # Instantiate a dedicated MockCLM adapter per problem_type seen
        for ex in calib:
            if ex.problem_type not in self.bank:
                self.bank[ex.problem_type] = MockCLMDecider(dim=self.dim)
        # Learn blend: compare MockCLM vs Cosine hit rates
        clm_hits: Dict[str, int] = defaultdict(int)
        cos_hits: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)
        for ex in calib:
            totals[ex.problem_type] += 1
            adapter = self.bank.get(ex.problem_type, self.bank["default"])
            p_clm = adapter.score(ex.state, ex.candidates)
            p_cos = self._cosine.score(ex.state, ex.candidates)
            if p_clm and max(p_clm, key=p_clm.get) == ex.gold:  # type: ignore[arg-type]
                clm_hits[ex.problem_type] += 1
            if p_cos and max(p_cos, key=p_cos.get) == ex.gold:  # type: ignore[arg-type]
                cos_hits[ex.problem_type] += 1
        for pt, n in totals.items():
            c = clm_hits[pt] + 1.0
            o = cos_hits[pt] + 1.0
            self.blend[pt] = c / (c + o)
        return self

    def bind_example(self, example: Example) -> None:
        self._example = example
        self._problem_type = example.problem_type

    def score(
        self, state: Mapping, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        ids = [c.id for c in candidates]
        if not ids:
            return {}
        pt = self._problem_type or "default"
        adapter = self.bank.get(pt, self.bank["default"])
        p_clm = adapter.score(state, candidates)
        p_cos = self._cosine.score(state, candidates)
        w = float(self.blend.get(pt, 0.7))
        blended = {
            cid: w * float(p_clm.get(cid, 0.0)) + (1.0 - w) * float(p_cos.get(cid, 0.0))
            for cid in ids
        }
        s = sum(blended.values()) or 1.0
        return {cid: blended[cid] / s for cid in ids}
