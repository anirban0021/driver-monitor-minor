"""Train the Random Forest from the CSVs written by ``extract.py``.

    python -m driver_monitor.train --frames data/frames --out models/drowsiness_rf.pkl

What it does:
  * cuts every video into overlapping windows and computes the features of ``features.py``;
  * measures each person's open-eye baseline from their own alert videos;
  * cross-validates with *different people in training and test* (GroupKFold by subject), which is the
    only honest way to score it: if frames of one person sit on both sides the score is inflated;
  * scores a simple rule-based baseline on the same windows, so the report can compare the two;
  * fits the final model on all windows and saves it (a pickle: only load model files you trained yourself).
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import MODELS_DIR
from .features import CLASSES, COLUMNS, FEATURES, VERSION, make_windows, upper_half_median
from .forest import RandomForest
from .state import Config

DEFAULT_OUT = MODELS_DIR / "drowsiness_rf.pkl"


@dataclass
class Video:
    name: str
    subject: str
    label: str
    samples: np.ndarray


def load_videos(frames_dir: Path) -> list[Video]:
    videos = []
    for path in sorted(frames_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            rows = list(reader)
        if not header or not rows or tuple(header[3:]) != COLUMNS:
            print(f"Skipping {path.name}: empty or not a feature CSV", file=sys.stderr)
            continue
        label = rows[0][2]
        if label not in CLASSES:
            continue                                  # "skip" and anything unknown
        samples = np.array([[float(x) for x in r[3:]] for r in rows], dtype=float)
        videos.append(Video(path.stem, rows[0][1], label, samples))
    return videos


def subject_baselines(videos: list[Video]) -> dict[str, float]:
    """Each person's open-eye EAR, from their alert videos (all their videos if they have none)."""
    pools: dict[str, dict[str, list]] = defaultdict(lambda: {"alert": [], "all": []})
    for v in videos:
        face = v.samples[:, 1] > 0.5
        ear = v.samples[face, 2]
        pools[v.subject]["all"].append(ear)
        if v.label == "alert":
            pools[v.subject]["alert"].append(ear)
    out = {}
    for subject, p in pools.items():
        chosen = p["alert"] or p["all"]
        b = upper_half_median(np.concatenate(chosen)) if chosen else None
        if b:
            out[subject] = b
    return out


def build_windows(videos: list[Video], window_s: float, stride_s: float):
    base = subject_baselines(videos)
    X, y, groups = [], [], []
    for v in videos:
        if v.subject not in base:
            continue
        for _, vec in make_windows(v.samples, base[v.subject], window_s, stride_s):
            X.append(vec)
            y.append(v.label)
            groups.append(v.subject)
    return np.array(X), np.array(y), np.array(groups)


def rule_predict(X: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """What the app's own rules would say about each window (the baseline the model has to beat).

    The two rules that fit inside one window: a mouth held open for yawn_min_s is a yawn, eyes shut for
    microsleep_s is drowsiness. The app's PERCLOS and yawn-count rules need minutes of history, so they
    cannot be judged on an 8-second window and are left out.
    """
    cfg = cfg or Config()
    col = {name: i for i, name in enumerate(FEATURES)}
    out = []
    for row in X:
        if row[col["mouth_open_longest_s"]] >= cfg.yawn_min_s:
            out.append("yawning")
        elif row[col["closed_longest_s"]] >= cfg.microsleep_s:
            out.append("drowsy")
        else:
            out.append("alert")
    return np.array(out)


def group_folds(groups: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (train, test) index splits where every person is entirely in the training side or entirely in the test side."""
    names, counts = np.unique(groups, return_counts=True)
    load = np.zeros(k)
    fold_of = {}
    for i in np.argsort(-counts, kind="stable"):           # biggest people first, each into the lightest fold
        j = int(np.argmin(load))
        fold_of[names[i]] = j
        load[j] += counts[i]
    fold = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(fold != j), np.flatnonzero(fold == j)) for j in range(k)]


def _report(title: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    """Accuracy, per-class scores and a confusion matrix over *every* window.

    A prediction outside the true classes (e.g. the rules saying "drowsy" on a dataset that has no drowsy
    windows) gets its own column and counts as an error, instead of silently disappearing.
    """
    extra = sorted(set(y_pred.tolist()) - set(labels))
    cols = list(labels) + extra
    cm = np.array([[int(((y_true == a) & (y_pred == b)).sum()) for b in cols] for a in labels])
    tp = np.array([cm[i, i] for i in range(len(labels))])
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = tp / np.maximum(predicted[:len(labels)], 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    print(f"\n=== {title} ===")
    print(f"accuracy {tp.sum() / max(cm.sum(), 1):.3f}   macro-F1 {f1.mean():.3f}   ({int(cm.sum())} windows)")
    print(f"{'':>12}{'precision':>10}{'recall':>10}{'f1':>10}{'windows':>10}")
    for i, lab in enumerate(labels):
        print(f"{lab:>12}{precision[i]:>10.2f}{recall[i]:>10.2f}{f1[i]:>10.2f}{support[i]:>10}")
    print("confusion matrix (rows = true, columns = predicted)")
    print("            " + "".join(f"{lab:>10}" for lab in cols))
    for lab, row in zip(labels, cm):
        print(f"{lab:>12}" + "".join(f"{n:>10}" for n in row))
    if extra:
        print(f"(columns {extra} are predictions that are not a class in this data: always wrong)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.train", description="Train the drowsiness / yawning Random Forest.")
    p.add_argument("--frames", type=Path, default=Path("data/frames"), help="folder of CSVs from `extract frames`")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window-s", type=float, default=8.0)
    p.add_argument("--stride-s", type=float, default=4.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    videos = load_videos(args.frames)
    if not videos:
        raise SystemExit(f"No usable feature CSVs in {args.frames}. Run `python -m driver_monitor.extract frames` first.")
    X, y, groups = build_windows(videos, args.window_s, args.stride_s)
    if len(X) == 0:
        raise SystemExit("No windows could be built (videos too short, or the face was rarely found).")

    labels = [c for c in CLASSES if c in set(y)]
    n_people = len(set(groups))
    print(f"{len(videos)} videos, {n_people} people, {len(X)} windows of {args.window_s:g} s: "
          + ", ".join(f"{c} {int((y == c).sum())}" for c in labels))
    if len(labels) < 2 or n_people < 2:
        raise SystemExit("Need at least two classes and two different people to train and test honestly.")

    def make_model() -> RandomForest:
        return RandomForest(n_trees=args.trees, min_leaf=3, class_weight="balanced", seed=args.seed)

    oof = np.empty(len(y), dtype=object)
    for train, test in group_folds(groups, max(2, min(args.folds, n_people))):
        oof[test] = make_model().fit(X[train], y[train]).predict(X[test])
    _report("Random Forest (people never seen in training)", y, oof.astype(str), labels)
    _report("Rule-based baseline (the app's own rules that fit in a window, same windows)", y, rule_predict(X), labels)

    model = make_model().fit(X, y)
    print("\n=== What the model relies on ===")
    for i in np.argsort(model.feature_importances_)[::-1][:8]:
        print(f"{FEATURES[i]:<22}{model.feature_importances_[i]:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump({"model": model, "features": FEATURES, "classes": [str(c) for c in model.classes_],
                     "window_s": args.window_s, "stride_s": args.stride_s, "version": VERSION}, fh)
    print(f"\nSaved {args.out}. Use it with:  python -m driver_monitor --rf {args.out}")


if __name__ == "__main__":
    main()
