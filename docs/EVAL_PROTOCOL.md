# Evaluation Protocol

## Goal

Offline / shadow comparison of decision architectures **A0–A5** across five
problem types without requiring GPU or network access for the smoke path.

## Problem types

| Type | Decision |
|------|----------|
| `route` | Pick destination / department |
| `retrieve_rerank` | Rank retrieved documents |
| `bon_verify` | Accept / reject / abstain / revise a claim |
| `triage` | Assign severity / priority |
| `guided_questionnaire` | Next action given answers (+ optional `questions[]`) |

## Splits

| Split | Role |
|-------|------|
| `train` | Reserved for heavier learners (unused by default smoke) |
| `calib` | Fit z-scores, stacked gate, mixture tables, adapter blends |
| `shadow` | Primary offline eval |
| `holdout` | Secondary frozen check — **do not tune on this** |

## Decider protocol

```text
score(state, candidates) -> {candidate_id: probability}
```

Probabilities should be non-negative and ideally sum to ~1 over the candidate
set for that example.

## Architectures

| ID | Name | Summary |
|----|------|---------|
| A0 | RouterBaseline | `baseline_choice` or first candidate |
| A1 | FlatMean | Mean of member probs |
| A2 | ZScoreMean | Per-member z-score on calib, then mean |
| A3 | StackedGate | Logistic on top1/margin/entropy + router flag (sklearn optional) |
| A4 | MixtureOfDeciders | Weights by `problem_type` from calib hits |
| A5 | AdapterBank | MockCLM hash/BoW + cosine; plug-in via `MockCLMDecider.embed` |

### Members

- **RouterBaseline** — example baseline pointer
- **CosineDecider (B2)** — hash-embedding cosine
- **MockCLMDecider (C0)** — mock contrastive LM (same embed API as a real CLM)

## Metrics

- **Top-1** accuracy
- **nDCG@K** (default K=3, single relevant gold)
- **Pick-gold** — gold has positive mass
- **ECE stub** — binned top-1 confidence vs accuracy
- **EPC** — mean probability on gold
- **Latency** — mean / p50 / p95 ms per score call

## Foreign-tuning warning

Do **not** import weights or thresholds tuned on another tenant, market, or
holdout fold and treat them as production-ready. Always re-fit on local
`calib` (or a fresh shadow slice) before comparing architectures. This package
is for **shadow bakeoff**, not live trading or external market systems.

## Reproducing the smoke run

```bash
cd /workspace/decision_bakeoff
python -m decision_bakeoff run --data data/samples --arch A0,A1,A2,A3,A4,A5
```

Artifacts land in `results/<YYYYMMDD_HHMMSS>/report.md` and `metrics.json`.
