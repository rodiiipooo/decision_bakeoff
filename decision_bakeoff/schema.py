"""JSONL example schema and helpers for decision_bakeoff."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


VALID_SPLITS = frozenset({"train", "calib", "shadow", "holdout"})
VALID_PROBLEM_TYPES = frozenset(
    {
        "route",
        "retrieve_rerank",
        "bon_verify",
        "triage",
        "guided_questionnaire",
    }
)


@dataclass
class Candidate:
    id: str
    text: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        return cls(id=str(d["id"]), text=str(d.get("text", "")))


@dataclass
class Example:
    id: str
    problem_type: str
    state: Dict[str, Any]
    candidates: List[Candidate]
    gold: str
    baseline_choice: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    split: str = "shadow"
    questions: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Example":
        cands = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        pt = str(d["problem_type"])
        split = str(d.get("split", "shadow"))
        if pt not in VALID_PROBLEM_TYPES:
            raise ValueError(f"unknown problem_type: {pt}")
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {split}")
        return cls(
            id=str(d["id"]),
            problem_type=pt,
            state=dict(d.get("state") or {}),
            candidates=cands,
            gold=str(d["gold"]),
            baseline_choice=(
                str(d["baseline_choice"]) if d.get("baseline_choice") is not None else None
            ),
            meta=dict(d.get("meta") or {}),
            split=split,
            questions=d.get("questions"),
        )

    def candidate_ids(self) -> List[str]:
        return [c.id for c in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "problem_type": self.problem_type,
            "state": self.state,
            "candidates": [{"id": c.id, "text": c.text} for c in self.candidates],
            "gold": self.gold,
            "baseline_choice": self.baseline_choice,
            "meta": self.meta,
            "split": self.split,
        }
        if self.questions is not None:
            out["questions"] = self.questions
        return out


def load_jsonl(path: Path | str) -> List[Example]:
    path = Path(path)
    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(Example.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return examples


def load_data_dir(data_dir: Path | str) -> List[Example]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    examples: List[Example] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        examples.extend(load_jsonl(path))
    return examples


def filter_split(examples: Sequence[Example], split: str) -> List[Example]:
    return [e for e in examples if e.split == split]


def write_jsonl(path: Path | str, examples: Iterable[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
