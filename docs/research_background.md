# Research background

## Non-compensatory movie choice

[*Non-compensatory decision making for movie choice*](https://cau.scholarworks.kr/item/54c55e7e-0a00-474d-b777-3ee64717b837) motivates treating some user requirements as thresholds rather than assuming that every weak attribute can be compensated by a strong one. This informed the move from a single reward to explicit guardrails.

## Ratings and social information

[*The Influence of Online Ratings on Film Choice*](https://doaj.org/article/a8adc400f6524cb99f5c45380962f84d) supports including popularity and social evidence in the analysis, while also treating them carefully because they can dominate individual preference.

## Mood management

[*Computationally modeling mood management theory*](https://doi.org/10.1093/joc/jqad020) motivated context-dependent behavioral proxies. KION does not contain mood labels, so the project uses only observable proxies: recent completion, gaps between views, genre transitions, novelty preference and repeat behavior. Time of day and true session duration were excluded after the audit showed that the timestamps contain dates only.

## Popularity bias

[*A survey on popularity bias in recommender systems*](https://link.springer.com/article/10.1007/s11257-024-09406-0) motivates measuring more than catalogue coverage. The project tracks exposure Gini and HHI and proposes per-item global exposure budgets to prevent the feedback loop “shown more → watched more → shown even more”.

## Design consequence

Together these works led to a two-level design: supervised behavioral utility proposes useful slates, while hard constraints protect relevance, coverage, exposure, repetition and OOD safety.
