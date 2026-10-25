# Data

The KION dataset is not redistributed in this repository. Place prepared files in a local directory and set its path in `configs/paths.yaml` or directly in a notebook.

Expected files:

- `train_interactions.csv`
- `validation_interactions.csv`
- `validation_examples.npz`
- `test_examples.npz`
- `item_mapping.json`
- `resolved_config.json`
- `split_boundaries.json`
- `input_sha256.json`

The final recommender experiments use validation for model selection. Test remains closed until one configuration is frozen.
