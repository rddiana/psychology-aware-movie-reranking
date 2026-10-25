# Model artifacts

Large weights are intentionally excluded from Git. The notebooks expect:

```text
weights/best_model/
├── config.json
└── model.safetensors

psych_constrained_reranker_v1/
├── choice_model.cbm
├── choice_metrics.json
├── completion_model.cbm
├── completion_metrics.json
├── ood_support.json
├── reranker_normalization.json
└── reranker_supervised.pt
```

Optional candidate caches (`*.npz`) accelerate reruns but are not required for the repository history.
