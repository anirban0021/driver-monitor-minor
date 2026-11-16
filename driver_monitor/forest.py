"""A Random Forest classifier in plain NumPy (no scikit-learn).

Same algorithm as scikit-learn's ``RandomForestClassifier``:
  * every tree is grown on a bootstrap sample of the training windows (bagging);
  * at every split only a random subset of sqrt(n_features) features is considered;
  * the split is the cut that lowers the Gini impurity most;
  * a window's class probabilities are the average of the trees' leaf distributions.

It exists because compiled packages such as scikit-learn are blocked on some Windows setups
(Smart App Control / Application Control), while NumPy is not. Training data here is a few
thousand windows of ~20 features, so plain NumPy is fast enough.

Models are saved with ``pickle``: only load model files you trained yourself.
"""

from __future__ import annotations

import math

import numpy as np


def _grow(X: np.ndarray, yi: np.ndarray, w: np.ndarray, k: int, max_features: int, min_leaf: int, max_depth: int,
          rng: np.random.Generator):
    """Grow one tree. Returns (feature, threshold, left, right, value, importance) arrays; feature == -1 marks a leaf."""
    n, d = X.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), yi] = w                      # each sample carries its class weight
    feat, thr, left, right, value = [], [], [], [], []
    importance = np.zeros(d)

    def new_node() -> int:
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(None)
        return len(feat) - 1

    stack = [(new_node(), np.arange(n), 0)]
    while stack:
        node, idx, depth = stack.pop()
        dist = onehot[idx].sum(axis=0)
        total = float(dist.sum())
        value[node] = dist / total
        if depth >= max_depth or len(idx) < 2 * min_leaf or np.count_nonzero(dist) < 2:
            continue                                   # leaf
        parent_cost = total - float((dist ** 2).sum()) / total      # total * Gini(parent)
        best = None
        m = len(idx)
        lo, hi = min_leaf - 1, m - min_leaf            # cut after sorted position i, for i in [lo, hi)
        if hi <= lo:
            continue
        for f in rng.permutation(d)[:max_features]:
            xs_all = X[idx, f]
            order = np.argsort(xs_all, kind="stable")
            xs = xs_all[order]
            valid = xs[lo:hi] < xs[lo + 1:hi + 1]      # cannot cut between equal values
            if not valid.any():
                continue
            cum = np.cumsum(onehot[idx][order], axis=0)
            left_c = cum[lo:hi]
            right_c = dist - left_c
            nl, nr = left_c.sum(axis=1), right_c.sum(axis=1)
            cost = (nl - (left_c ** 2).sum(axis=1) / nl) + (nr - (right_c ** 2).sum(axis=1) / nr)
            cost = np.where(valid, cost, np.inf)
            j = int(np.argmin(cost))
            if best is None or cost[j] < best[0]:
                best = (float(cost[j]), int(f), float((xs[lo + j] + xs[lo + j + 1]) / 2))
        if best is None or best[0] >= parent_cost - 1e-12:
            continue
        cost, f, t = best
        go_left = X[idx, f] <= t
        li, ri = idx[go_left], idx[~go_left]
        if len(li) == 0 or len(ri) == 0:
            continue
        importance[f] += parent_cost - cost
        feat[node], thr[node] = f, t
        l, r = new_node(), new_node()
        left[node], right[node] = l, r
        stack.append((l, li, depth + 1))
        stack.append((r, ri, depth + 1))

    return (np.array(feat, dtype=np.int32), np.array(thr), np.array(left, dtype=np.int32),
            np.array(right, dtype=np.int32), np.vstack(value), importance)


def _tree_proba(tree, X: np.ndarray) -> np.ndarray:
    feat, thr, left, right, value, _ = tree
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        f = feat[node]
        active = np.flatnonzero(f >= 0)
        if active.size == 0:
            return value[node]
        n = node[active]
        go_left = X[active, f[active]] <= thr[n]
        node[active] = np.where(go_left, left[n], right[n])


class RandomForest:
    def __init__(self, n_trees: int = 300, min_leaf: int = 3, max_depth: int = 14, max_features: int | None = None,
                 class_weight: str | None = "balanced", seed: int = 0) -> None:
        self.n_trees, self.min_leaf, self.max_depth = n_trees, min_leaf, max_depth
        self.max_features, self.class_weight, self.seed = max_features, class_weight, seed
        self.classes_: np.ndarray = np.array([])
        self.feature_importances_: np.ndarray = np.array([])
        self._trees: list = []

    def fit(self, X, y) -> RandomForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        n, d = X.shape
        k = len(self.classes_)
        counts = np.bincount(yi, minlength=k)
        class_w = n / (k * counts) if self.class_weight == "balanced" else np.ones(k)
        w = class_w[yi]
        max_features = self.max_features or max(1, int(math.sqrt(d)))
        rng = np.random.default_rng(self.seed)
        self._trees = []
        importance = np.zeros(d)
        for _ in range(self.n_trees):
            boot = rng.integers(0, n, n)
            tree = _grow(X[boot], yi[boot], w[boot], k, max_features, self.min_leaf, self.max_depth, rng)
            self._trees.append(tree)
            if tree[5].sum() > 0:
                importance += tree[5] / tree[5].sum()
        self.feature_importances_ = importance / max(importance.sum(), 1e-12)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return sum(_tree_proba(t, X) for t in self._trees) / len(self._trees)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
