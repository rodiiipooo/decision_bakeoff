"""Eval harness: fit on calib, score shadow/holdout, write reports."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from decision_bakeoff.deciders import get_architecture
from decision_bakeoff.metrics import LatencyTracker, compute_all
from decision_bakeoff.schema import Example, filter_split, load_data_dir

CHI = ZoneInfo("America/Chicago")


def _now_stamp() -> str:
    return datetime.now(CHI).strftime("%Y%m%d_%H%M%S")


def _bind_and_score(arch: Any, ex: Example) -> Dict[str, float]:
    if hasattr(arch, "bind_example"):
        arch.bind_example(ex)
    return arch.score(ex.state, ex.candidates)


def evaluate_arch(
    arch_id: str,
    examples: Sequence[Example],
    eval_splits: Sequence[str] = ("shadow", "holdout"),
) -> Dict[str, Any]:
    calib = filter_split(examples, "calib")
    # Also allow train to supplement calib if calib empty
    if not calib:
        calib = filter_split(examples, "train")

    arch = get_architecture(arch_id)
    if hasattr(arch, "fit"):
        arch.fit(calib)

    per_split: Dict[str, Any] = {}
    all_preds: List[Dict[str, float]] = []
    all_golds: List[str] = []
    all_cids: List[List[str]] = []
    latency = LatencyTracker()
    by_type: Dict[str, Dict[str, List]] = {}

    for split in eval_splits:
        subset = filter_split(examples, split)
        if not subset:
            continue
        preds: List[Dict[str, float]] = []
        golds: List[str] = []
        cids: List[List[str]] = []
        split_lat = LatencyTracker()
        for ex in subset:
            t0 = time.perf_counter()
            probs = _bind_and_score(arch, ex)
            split_lat.record(t0)
            latency.record(t0)
            preds.append(probs)
            golds.append(ex.gold)
            cids.append(ex.candidate_ids())
            bt = by_type.setdefault(ex.problem_type, {"preds": [], "golds": [], "cids": []})
            bt["preds"].append(probs)
            bt["golds"].append(ex.gold)
            bt["cids"].append(ex.candidate_ids())
        metrics = compute_all(preds, golds, cids, latency=split_lat)
        per_split[split] = metrics
        all_preds.extend(preds)
        all_golds.extend(golds)
        all_cids.extend(cids)

    overall = compute_all(all_preds, all_golds, all_cids, latency=latency)
    type_metrics = {
        pt: compute_all(v["preds"], v["golds"], v["cids"]) for pt, v in by_type.items()
    }
    return {
        "arch": arch_id,
        "name": getattr(arch, "name", arch_id),
        "n_calib": len(calib),
        "overall": overall,
        "per_split": per_split,
        "per_problem_type": type_metrics,
    }


def run_bakeoff(
    data_dir: Path | str,
    arch_ids: Sequence[str],
    results_root: Path | str = "results",
    eval_splits: Sequence[str] = ("shadow", "holdout"),
) -> Path:
    examples = load_data_dir(data_dir)
    if not examples:
        raise FileNotFoundError(f"no examples loaded from {data_dir}")

    stamp = _now_stamp()
    out_dir = Path(results_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for arch_id in arch_ids:
        reports.append(evaluate_arch(arch_id, examples, eval_splits=eval_splits))

    metrics_path = out_dir / "metrics.json"
    payload = {
        "timestamp": datetime.now(CHI).isoformat(),
        "data_dir": str(data_dir),
        "n_examples": len(examples),
        "arch_ids": list(arch_ids),
        "eval_splits": list(eval_splits),
        "results": reports,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "report.md"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return out_dir


def _render_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Decision Bakeoff Report")
    lines.append("")
    lines.append(f"- **Timestamp:** {payload['timestamp']}")
    lines.append(f"- **Data:** `{payload['data_dir']}` ({payload['n_examples']} examples)")
    lines.append(f"- **Eval splits:** {', '.join(payload['eval_splits'])}")
    lines.append(f"- **Architectures:** {', '.join(payload['arch_ids'])}")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| Arch | Top-1 | nDCG@3 | Pick-gold | ECE stub | EPC | Mean latency (ms) | N |")
    lines.append("|------|------:|-------:|----------:|---------:|----:|------------------:|--:|")
    for r in payload["results"]:
        o = r["overall"]
        lat = o.get("latency") or {}
        lines.append(
            "| {arch} | {top1:.3f} | {ndcg:.3f} | {pg:.3f} | {ece:.3f} | {epc:.3f} | {lat:.2f} | {n} |".format(
                arch=r["arch"],
                top1=o["top1"],
                ndcg=o["ndcg@3"],
                pg=o["pick_gold"],
                ece=o["ece_stub"],
                epc=o["epc"],
                lat=float(lat.get("mean_ms", 0.0)),
                n=int(o["n"]),
            )
        )
    lines.append("")
    lines.append("## Per problem type (top-1)")
    lines.append("")
    # collect types
    types = sorted(
        {
            pt
            for r in payload["results"]
            for pt in r.get("per_problem_type", {})
        }
    )
    if types:
        header = "| Arch | " + " | ".join(types) + " |"
        sep = "|------|" + "|".join(["------:" for _ in types]) + "|"
        lines.append(header)
        lines.append(sep)
        for r in payload["results"]:
            cells = [r["arch"]]
            for pt in types:
                m = r.get("per_problem_type", {}).get(pt)
                cells.append(f"{m['top1']:.3f}" if m else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Offline/shadow eval only; do not tune on holdout or promote foreign "
        "tuned weights without a fresh local calib fit."
    )
    lines.append("- A3 uses pure-Python logistic fallback when sklearn is absent.")
    lines.append("- A5 MockCLM is a hash/BoW stand-in; swap `MockCLMDecider.embed` for a real CLM.")
    lines.append("")
    return "\n".join(lines)
