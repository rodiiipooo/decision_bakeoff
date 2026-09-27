# EDIT_LOG

## 2026-09-27 08:52:26 CDT
- **Files:** decision_bakeoff/__init__.py, decision_bakeoff/schema.py, decision_bakeoff/metrics.py, CHANGELOG.md, EDIT_LOG.md
- **What:** Added package init, JSONL Example/Candidate schema with loaders, and eval metrics (top-1, nDCG@K, pick-gold, ECE stub, EPC, latency tracker).
- **Why:** First logical chunk — core data model and metrics before deciders/harness.

## 2026-09-27 08:53:09 CDT
- **Files:** decision_bakeoff/deciders/__init__.py, base.py, members.py, arch.py; CHANGELOG.md
- **What:** Decider protocol; members RouterBaseline, CosineDecider (B2), MockCLMDecider (C0); architectures A0–A5 (RouterBaseline, FlatMean, ZScoreMean, StackedGate, MixtureOfDeciders, AdapterBank).
- **Why:** Second logical chunk — scoring backends before harness/data.

## 2026-09-27 08:53:27 CDT
- **Files:** data/samples/*.jsonl (5 files); CHANGELOG.md
- **What:** Synthetic ≥20 examples per problem type (route, retrieve_rerank, bon_verify, triage, guided_questionnaire) with train/calib/shadow/holdout splits; guided_questionnaire includes questions[].
- **Why:** Third logical chunk — offline smoke data for harness.

## 2026-09-27 08:53:45 CDT
- **Files:** decision_bakeoff/harness.py, decision_bakeoff/__main__.py; CHANGELOG.md
- **What:** Eval harness (fit calib, score shadow/holdout, metrics.json + report.md) and CLI `python -m decision_bakeoff run --data ... --arch A0,A1,...`.
- **Why:** Fourth logical chunk — runnable bakeoff entrypoint.

## 2026-09-27 08:54:07 CDT
- **Files:** tests/test_smoke.py, docs/EVAL_PROTOCOL.md, README.md, .gitignore, results/.gitkeep; CHANGELOG.md
- **What:** Smoke tests, eval protocol docs, architecture-table README with foreign-tuning warning, gitignore for generated results.
- **Why:** Fifth logical chunk — docs/tests polish and success criteria coverage.
