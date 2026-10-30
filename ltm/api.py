"""scikit-learn style API: fit stores the training rows as context, predict runs the model."""
import numpy as np
import torch

from .model import LTM

MAX_CTX = 400    # context rows kept at fit time (v1 was trained on tables up to 600 rows)
CHUNK = 200      # query rows per forward pass (exact: query rows never see each other)


def _load(path):
    ck = torch.load(path, map_location="cpu")
    model = LTM(d=ck["d"], layers=ck["layers"])
    model.load_state_dict(ck["model"])
    return model.eval()


class _Base:
    task = None

    def __init__(self, checkpoint="ltm_v1.pt", cat_features=None, max_ctx=MAX_CTX, random_state=0):
        self.checkpoint = checkpoint
        self.cat_features = cat_features
        self.max_ctx = max_ctx
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if len(X) > self.max_ctx:
            idx = np.random.default_rng(self.random_state).choice(len(X), self.max_ctx, replace=False)
            X, y = X[idx], y[idx]
        cat = np.zeros(X.shape[1], dtype=bool)
        if self.cat_features is not None:
            cat[list(self.cat_features)] = True
        self.n_features_in_ = X.shape[1]
        self.X_ = torch.tensor(X)
        self.cat_mask_ = torch.tensor(cat)
        self._fit_target(y)
        self.model_ = _load(self.checkpoint)
        return self

    def _run(self, X):
        Xq = torch.tensor(np.asarray(X, dtype=np.float32))
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xq), CHUNK):
                full = torch.cat([self.X_, Xq[i:i + CHUNK]], 0)
                outs.append(self.model_(full, self.cat_mask_, self.y_ctx_, self.task, self.n_classes_))
        return torch.cat(outs)


class LTMClassifier(_Base):
    task = "classification"

    def _fit_target(self, y):
        self.classes_, enc = np.unique(y, return_inverse=True)
        if len(self.classes_) > 10:
            raise ValueError("LTM v0 supports at most 10 classes")
        self.n_classes_ = len(self.classes_)
        self.y_ctx_ = torch.tensor(enc, dtype=torch.long)

    def predict_proba(self, X):
        return torch.softmax(self._run(X), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class LTMRegressor(_Base):
    task = "regression"

    def _fit_target(self, y):
        y = y.astype(np.float32)
        self.ym_, self.ys_ = float(y.mean()), max(float(y.std()), 1e-6)
        self.n_classes_ = 0
        self.y_ctx_ = torch.tensor(y)

    def predict(self, X):
        return self._run(X).numpy() * self.ys_ + self.ym_
