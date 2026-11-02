"""Synthetic table prior: random structural causal models (SCMs).

Each call samples a random DAG, pushes noise through random nonlinear node
functions, then picks feature nodes and one target node. Some features are
binned into categoricals, and values are dropped at random to mimic missing data.
"""
from dataclasses import dataclass

import numpy as np

_ACTS = (lambda x: x, np.tanh, lambda x: np.maximum(x, 0.0), np.sin, np.abs)


@dataclass
class Table:
    X: np.ndarray         # (n_rows, n_features) float; categoricals are integer codes; NaN = missing
    y: np.ndarray         # (n_rows,) int labels (classification) or float (regression)
    cat_mask: np.ndarray  # (n_features,) bool, True = categorical column
    task: str             # "classification" or "regression"
    n_classes: int        # 0 for regression


def _std(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-8 else 1.0)


def _root(rng, n):
    kind = rng.integers(3)
    if kind == 0:
        v = rng.normal(size=n)
    elif kind == 1:
        v = rng.uniform(-1, 1, size=n)
    else:
        v = rng.lognormal(0.0, 0.7, size=n)
    return _std(v)


_TRANSFORMS = (
    lambda z: z,
    lambda z: np.exp(0.7 * np.clip(z, -6, 6)),
    lambda z: np.sign(z) * np.abs(z) ** 1.5,
    lambda z: z ** 3,
)


def _tree(rng, P, n):
    """Random decision tree on the parent columns: piecewise-constant output."""
    depth = int(rng.integers(2, 4))
    code = np.zeros(n, dtype=np.int64)
    for _ in range(depth):
        new = code * 2
        for c in np.unique(code):
            idx = np.flatnonzero(code == c)
            if len(idx) < 4:
                continue
            j = int(rng.integers(P.shape[1]))
            thr = np.quantile(P[idx, j], rng.uniform(0.25, 0.75))
            new[idx] += P[idx, j] > thr
        code = new
    return rng.normal(size=int(code.max()) + 1)[code]


def _node(rng, parents, n, p_tree):
    P = np.stack(parents, axis=1)
    if rng.random() < p_tree:
        z = sum(_tree(rng, P, n) for _ in range(int(rng.integers(1, 3))))
    else:
        w = rng.normal(size=P.shape[1])
        act = _ACTS[rng.integers(len(_ACTS))]
        z = act(P @ w + rng.normal())
        if P.shape[1] >= 2 and rng.random() < 0.3:
            i, j = rng.choice(P.shape[1], 2, replace=False)
            z = z + rng.normal() * P[:, i] * P[:, j]
    z = _std(z) + rng.uniform(0.05, 0.6) * rng.normal(size=n)
    return _std(z)


def _bin(rng, v, k):
    props = rng.dirichlet(np.ones(k) * rng.choice([0.5, 1.0, 2.0, 5.0]))
    edges = np.quantile(v, np.cumsum(props)[:-1])
    codes = np.searchsorted(edges, v)
    return rng.permutation(k)[codes]


def generate_table(rng, n_rows, n_features, task="classification"):
    n_hidden = int(rng.integers(0, n_features + 1))
    n_nodes = n_features + 1 + n_hidden
    edge_p = rng.uniform(0.1, 0.5)
    p_tree = rng.uniform(0.0, 0.7)
    adj = np.triu(rng.random((n_nodes, n_nodes)) < edge_p, k=1)  # adj[j, i]: j -> i

    vals, has_pa = [], []
    for i in range(n_nodes):
        pa = np.flatnonzero(adj[:, i])
        if len(pa) == 0 and i > 0 and rng.random() < 0.7:
            pa = np.array([rng.integers(i)])
        has_pa.append(len(pa) > 0)
        vals.append(_node(rng, [vals[j] for j in pa], n_rows, p_tree) if len(pa) else _root(rng, n_rows))

    cand = ([i for i in range(n_nodes // 2, n_nodes) if has_pa[i]]
            or [i for i in range(n_nodes) if has_pa[i]]
            or [n_nodes - 1])
    t = int(rng.choice(cand))
    others = [i for i in range(n_nodes) if i != t]
    feat = rng.choice(others, n_features, replace=False)

    X = np.stack([vals[i] for i in feat], axis=1)

    # correlated features: some columns become noisy mixes of two others
    q = rng.uniform(0.0, 0.5)
    if n_features >= 3:
        for c in range(n_features):
            if rng.random() < q:
                a, b = rng.choice([i for i in range(n_features) if i != c], 2, replace=False)
                mix = rng.normal() * X[:, a] + rng.normal() * X[:, b]
                X[:, c] = _std(mix) + rng.uniform(0.0, 0.4) * rng.normal(size=n_rows)

    # skewed and heavy-tailed columns
    p_tf = rng.uniform(0.0, 0.5)
    for c in range(n_features):
        if rng.random() < p_tf:
            X[:, c] = _TRANSFORMS[rng.integers(len(_TRANSFORMS))](X[:, c])
    cat_mask = rng.random(n_features) < rng.uniform(0, 0.5)
    for c in np.flatnonzero(cat_mask):
        X[:, c] = _bin(rng, X[:, c], int(rng.integers(2, 10))).astype(float)

    if task == "classification":
        k = 2 if rng.random() < 0.4 else int(rng.integers(3, 11))
        y = _bin(rng, vals[t], k)
        _, y = np.unique(y, return_inverse=True)
        n_classes = int(y.max()) + 1
    else:
        y, n_classes = vals[t], 0

    for c in range(n_features):
        rate = rng.choice([0.0, 0.0, 0.05, 0.2])
        X[rng.random(n_rows) < rate, c] = np.nan

    return Table(X, y, cat_mask, task, n_classes)


def sample_table(rng):
    """Sample one random table with random shape and task."""
    n_rows = int(np.exp(rng.uniform(np.log(50), np.log(2000))))
    n_features = int(np.exp(rng.uniform(np.log(2), np.log(100))))
    task = "classification" if rng.random() < 0.6 else "regression"
    return generate_table(rng, n_rows, n_features, task)
