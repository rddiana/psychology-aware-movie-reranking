# Experiment history

## 1. Frozen GPT-2 baseline

The sequential model established ranking, coverage and exposure references. All later guardrails are defined relative to this frozen policy.

## 2. CatBoost reward regression

The model predicted normalized watched percentage. On validation, RMSE improved from `0.3770` for the user-item blend to `0.3670` for CatBoost.

## 3. Psychology-informed RL

RL increased the scalar reward and expected completion, but extra optimization degraded NDCG and coverage. Increasing epochs was therefore rejected.

## 4. Supervised constrained reranker

The supervised warm start improved Recall, NDCG, expected completion, novelty, repeat rate, OOD rate and diversity. However, HHI increased by about 9%, revealing a global exposure problem that user-local scoring could not solve.

Full constrained variants strongly increased expected completion (`0.609–0.621`) and diversity (`0.662–0.691`), but all failed the NDCG, coverage and exposure guardrails. The correct selected action was the unchanged GPT-2 fallback.

## 5. Pareto Top-10

The search evaluated 97 first-stage and 192 second-stage configurations. No candidate passed every hard constraint. A nearly feasible candidate improved watch and coverage but missed the NDCG floor by approximately `1.8e-5`; it was not promoted.

## 6. Exact minimal repair

The final idea preserves a supervised seed slate, locks Top-3 or Top-5, permits at most one or two replacements per user, and enforces per-item global exposure budgets with CP-SAT. The Drive run ended after Stage 1, so no final result is claimed yet.
