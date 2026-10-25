# Result provenance

The compact tables in this directory were copied from the saved Google Drive runs and cleaned only for naming and duplicate rows.

| Stage | Saved run | Status |
|---|---|---|
| Reward regression | [`reward_model_v1`](https://drive.google.com/drive/folders/1pHZLh4PKdwkR1Jo3vywG0SaezEOeO6ft) | Complete |
| Psychology-informed RL | [`psych_reward_rl_v1`](https://drive.google.com/drive/folders/1oHzsurY9B2-Eikyep2V-JuEgX6soeZNY) | One accepted checkpoint |
| Constrained reranker | [`psych_constrained_reranker_v1`](https://drive.google.com/drive/folders/1zox4AuflEiKLs2zDE1AjG0o2_zNgIUY4) | Complete; safe GPT-2 fallback selected |
| Supervised Pareto | [`psych_supervised_pareto_v2`](https://drive.google.com/drive/folders/1b3N7aC4G1_enFaqaoIOm3K5PtfX8FeFW) | Complete; no hard-feasible alternative |
| Exact minimal repair | [`psych_minimal_repair_final_v3`](https://drive.google.com/drive/folders/1hMuC-3Byb7n5ELhSGphQAaYLO8d2GR2f) | Incomplete after Stage 1 |

The V06 bootstrap deltas are all zero because the selected policy is the unchanged GPT-2 fallback, not because bootstrap evaluation failed.
