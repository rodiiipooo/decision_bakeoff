# EDIT_LOG

## 2026-09-27 08:52:26 CDT
- **Files:** decision_bakeoff/__init__.py, decision_bakeoff/schema.py, decision_bakeoff/metrics.py, CHANGELOG.md, EDIT_LOG.md
- **What:** Added package init, JSONL Example/Candidate schema with loaders, and eval metrics (top-1, nDCG@K, pick-gold, ECE stub, EPC, latency tracker).
- **Why:** First logical chunk — core data model and metrics before deciders/harness.

## 2026-09-27 08:53:09 CDT
- **Files:** decision_bakeoff/deciders/__init__.py, base.py, members.py, arch.py; CHANGELOG.md
- **What:** Decider protocol; members RouterBaseline, CosineDecider (B2), MockCLMDecider (C0); architectures A0–A5 (RouterBaseline, FlatMean, ZScoreMean, StackedGate, MixtureOfDeciders, AdapterBank).
- **Why:** Second logical chunk — scoring backends before harness/data.
