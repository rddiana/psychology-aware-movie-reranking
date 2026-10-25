# Methodology

## Pipeline

1. A frozen GPT-2 next-item model generates 64 candidates.
2. Choice and completion models estimate behavioral utility from historical signals available before the cutoff.
3. A supervised reranker produces a Top-10 seed slate.
4. Pareto search studies trade-offs among relevance, expected completion, novelty, diversity, popularity and OOD risk.
5. The proposed final stage applies exact, minimal-edit global exposure repair while locking the strongest prefix.

## Optimization target

The score explored during reranking is

\[
s = s_{supervised} + lpha\,watched + eta\,choice
    + \gamma\,diversity - \lambda\,popularity - \mu\,OOD.
\]

This scalar score proposes candidates; it does not replace guardrails. Constraint violations are normalized by their tolerance:

\[
v_{normalized} = rac{v}{tolerance}.
\]

The exact-repair formulation is lexicographic: first minimize discounted supervised relevance loss, then maximize behavioral utility among solutions satisfying global exposure limits.

## Leakage control

- Temporal splits are fixed in advance.
- Historical features use events strictly before the evaluation cutoff.
- V06 and V07 never ingest test interactions into the feature store.
- The final recommender test is kept closed while validation configurations are compared.
