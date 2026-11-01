# LTM: Open-Source Large Tabular Model

A small pretrained transformer for tabular data with a scikit-learn style `fit` / `predict` API. It learns in context: `fit` stores your training rows and `predict` runs the model, with no gradient training and no tuning. Apache-2.0.

**Status: early research prototype (v2, 2.4M parameters).** It was pretrained only on synthetic tables, with no real data.

## Results

Compared with scikit-learn's default `HistGradientBoosting` (untuned) on public datasets. Each dataset was subsampled to at most 1500 rows, split 70/30 at random, and averaged over 3 splits. LTM used 400 context rows. See `benchmark.py`.

| Dataset | Task | LTM v2 | HistGB |
|---|---|---|---|
| iris | accuracy | 0.985 | 0.970 |
| wine | accuracy | 0.994 | 1.000 |
| breast_cancer | accuracy | 0.953 | 0.953 |
| digits | accuracy | 0.692 | 0.965 |
| credit-g | accuracy | 0.741 | 0.776 |
| diabetes (Pima) | accuracy | 0.756 | 0.756 |
| blood-transfusion | accuracy | 0.800 | 0.751 |
| vehicle | accuracy | 0.730 | 0.781 |
| ilpd | accuracy | 0.716 | 0.703 |
| banknote-authentication | accuracy | 0.995 | 0.990 |
| phoneme | accuracy | 0.773 | 0.861 |
| segment | accuracy | 0.835 | 0.975 |
| diabetes (sklearn) | R² | 0.369 | 0.310 |
| california | R² | 0.644 | 0.736 |
| **Mean, 12 classification datasets** | accuracy | **0.831** | **0.873** |
| **Mean, 2 regression datasets** | R² | **0.506** | **0.523** |

In short: **competitive with default gradient boosting on small tables, still behind on larger tables, many-feature tables and multi-class problems** (digits, segment, phoneme, california). Using 700 context rows instead of 400 did not change this (mean accuracy 0.832, R² 0.509).

## Limits

- At most 10 classes and 128 features. Trained on tables of up to 1000 rows and 64 features.
- Baseline is default HistGradientBoosting only. It has not been compared with tuned LightGBM or CatBoost, or with TabPFN.
- Small benchmark (14 datasets, 3 splits each). Differences of a point or two are within noise.
- Categorical columns must be integer-coded (0 to 11); you pass their indices in `cat_features`.

## Usage

```python
from ltm import LTMClassifier, LTMRegressor

clf = LTMClassifier(checkpoint="ltm_v2.pt").fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
```

The weights (`ltm_v2.pt`) are not stored in this repository. Download them from the Hugging Face model page and place the file next to your script.

## Training

```
python -m ltm.train --steps 20000 --batch 4 --workers 2 --out ltm_v1.pt
python -m ltm.train --steps 15000 --batch 4 --workers 2 --max_rows 1000 --max_feats 64 --ckpt --lr 2e-4 --init ltm_v1.pt --out ltm_v2.pt
```

Both runs used a single free Colab T4 GPU (about 1 and 1.7 hours). Training tables come from random structural causal models in `ltm/prior.py`. See `DATA_LICENSES.md`.

## Demo

https://huggingface.co/spaces/ciphemic/ltm-demo

## License

Apache-2.0.
