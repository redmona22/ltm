# LTM: Open-Source Large Tabular Model

A pretrained transformer for tabular data, released under Apache-2.0.

**Goal:** competitive with gradient-boosted trees on small tables, with no tuning, through a scikit-learn-style `fit` / `predict` API.

## Status
Early development. Nothing is released yet.

## Plan
1. Synthetic table prior (random structural causal models)
2. Model v0 (10-50M params): column and row attention, categorical embeddings
3. In-context prediction head
4. Evaluation harness (TabArena, OpenML-CC18) against LightGBM, CatBoost, TabPFN
5. Public weights and a demo Space

## Demo
https://huggingface.co/spaces/ciphemic/ltm-demo

## License
Code: Apache-2.0. Pretraining data: synthetic tables plus permissively licensed datasets only (see `DATA_LICENSES.md`).

## Contributing
Issues and pull requests are welcome.
