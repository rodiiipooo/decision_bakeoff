# decision_bakeoff

Offline / shadow evaluation of decision architectures **A0â€“A5** across five
problem types. Stdlib-first; no GPU or network required for the smoke path.

## Architectures

| ID | Name | Members / method |
|----|------|------------------|
| **A0** | RouterBaseline | Use example `baseline_choice`, else first candidate |
| **A1** | FlatMean | Mean of member probabilities over shared ids |
| **A2** | ZScoreMean | Per-model z-score on `calib`, then mean |
| **A3** | StackedGate | Logistic on top1 / margin / entropy (+ router match); pure-Python fallback, optional sklearn |
| **A4** | MixtureOfDeciders | Mixture weights by `problem_type` (calib hits or heuristic) |
| **A5** | AdapterBank | MockCLM hash/BoW embeddings + cosine; real CLM plug-in at `MockCLMDecider.embed` |

**Members:** RouterBaseline, CosineDecider (B2), MockCLMDecider (C0).

## Problem types

`route` Â· `retrieve_rerank` Â· `bon_verify` Â· `triage` Â· `guided_questionnaire`

## Quick start

```bash
cd /workspace/decision_bakeoff
python -m decision_bakeoff run --data data/samples
# or select arches:
python -m decision_bakeoff run --data data/samples --arch A0,A1,A2,A3,A4,A5
```

Results: `results/<timestamp>/report.md` and `metrics.json`.

```bash
python -m unittest tests.test_smoke -v
```

## Layout

```text
decision_bakeoff/
  schema.py          # JSONL Example / Candidate
  metrics.py         # top-1, nDCG@K, pick-gold, ECE stub, latency, EPC
  harness.py         # fit + eval + report writer
  __main__.py        # CLI
  deciders/          # members + A0â€“A5
data/samples/*.jsonl
docs/EVAL_PROTOCOL.md
tests/test_smoke.py
```

## Foreign-tuning warning

**Do not** reuse thresholds or weights tuned on a foreign dataset, tenant, or
holdout fold without a fresh local `calib` fit. This repo is an offline shadow
bakeoff only â€” it is not a trading system and must not be wired to market
execution paths.

See [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) for splits, metrics, and
the MockCLM plug-in point.

## License

MIT — see [LICENSE](LICENSE).

