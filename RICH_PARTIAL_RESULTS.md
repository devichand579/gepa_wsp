# Partial rich-feedback results

Seed0 only; seeds1 and2 have not started. All values are selection accuracy. No final-test evaluation has occurred. The target remains30 mutation calls per seed.

| Task | One-shot | GEPA so far | Adam | Completed mutation responses |
|---|---:|---:|---:|---:|
| mnist | 88.1% | 89.0% | 93.8% | 4 |
| fashion | 24.5% | 69.1% | 82.7% | 6 |
| sst2 | 71.2% | 71.2% | 75.1% | 4 |

Fashion seed0 construction failed and retained random initialization, so its improvement includes recovery from failed construction. The selected MNIST program internally compares multiple closed-form fits using training-only cross-validation. This is a result of the whole LLM-written weight-update program; it does not isolate GEPA selection from statistical fitting or internal model selection. Program CPU time is recorded, and semantic review remains part of final interpretation.

Current protocol and exact data access: RICH_PROTOCOL.md. Raw program sources and numeric checkpoints are preserved under rich_runs/rich_main/.
