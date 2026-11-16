"""Random Forest pipeline: features, label guessing, video extraction, training and live use.

The training data here is synthetic and trivially separable: these tests prove the plumbing
(files, windows, subject-wise splitting, saving, loading, alert wiring), not real-world accuracy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from driver_monitor import extract, features, forest, train
from driver_monitor.detector import FaceReading
from driver_monitor.features import COLUMNS, FEATURES, make_windows, upper_half_median, window_features
from driver_monitor.state import Config, Frame, Level, Status

FPS = 15.0
BASE = 0.30


def samples(kind: str, seconds: float = 12.0, seed: int = 0, ear: float = BASE) -> np.ndarray:
    """Synthetic per-frame signals: 'alert' (with brief talking), 'yawning' (jaw held open) or 'drowsy' (long closures)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    s = np.full((n, len(COLUMNS)), np.nan)
    s[:, 0], s[:, 1] = t, 1.0
    s[:, 2] = ear + rng.normal(0, 0.005, n)
    s[:, 3] = 0.12 + rng.normal(0, 0.01, n)
    s[:, 4] = 0.03 + abs(rng.normal(0, 0.01, n))
    s[:, 5] = 0.05 + abs(rng.normal(0, 0.01, n))
    s[:, 6:9] = rng.normal(0, 1.0, (n, 3))
    if kind == "alert":                                   # talking: short mouth openings
        for start in np.arange(1.0, seconds - 1, 1.5):
            m = (t >= start) & (t < start + 0.3)
            s[m, 4], s[m, 3] = 0.6, 0.5
    elif kind == "yawning":
        m = (t >= seconds / 3) & (t < seconds / 3 + 3.0)
        s[m, 4], s[m, 3] = 0.85, 0.7
    elif kind == "drowsy":
        for start in (2.0, 6.5):
            m = (t >= start) & (t < start + 2.0)
            s[m, 2], s[m, 5] = 0.10, 0.9
    return s


# ---- features ----------------------------------------------------------------------------------

def test_window_features_separate_the_three_behaviours():
    idx = {name: i for i, name in enumerate(FEATURES)}
    alert, yawn, drowsy = (window_features(samples(k), BASE) for k in ("alert", "yawning", "drowsy"))
    assert alert[idx["perclos"]] == 0 and alert[idx["closed_longest_s"]] == 0
    assert alert[idx["mouth_open_longest_s"]] < 0.5 and alert[idx["mouth_open_episodes"]] >= 3     # talking = many short openings
    assert 2.8 <= yawn[idx["mouth_open_longest_s"]] <= 3.3 and yawn[idx["mouth_open_episodes"]] == 1
    assert 1.8 <= drowsy[idx["closed_longest_s"]] <= 2.3 and drowsy[idx["perclos"]] > 0.25


def test_eye_openness_is_relative_to_the_drivers_own_baseline():
    s = samples("plain", ear=0.20)                            # a narrow-eyed driver whose normal EAR is 0.20, no talking
    s[:, 5] = np.nan                                          # no blendshapes: the rule is plain EAR < baseline * ratio
    idx = {name: i for i, name in enumerate(FEATURES)}
    narrow = window_features(s, 0.20)
    assert narrow[idx["perclos"]] == 0 and narrow[idx["ear_rel_mean"]] == pytest.approx(1.0, abs=0.05)
    assert window_features(s, 0.30)[idx["perclos"]] > 0.9     # judged against a wide-eyed baseline they would look shut


def test_lowered_lids_are_not_shut_when_the_blink_score_says_open():
    """With blendshapes present, EAR alone must fall far below the threshold, exactly as DriverState decides it."""
    idx = {name: i for i, name in enumerate(FEATURES)}
    s = samples("plain", ear=0.20)                            # EAR 0.20 against a 0.30 baseline, blink score ~0.05
    assert window_features(s, 0.30)[idx["perclos"]] == 0


def test_squeezed_eyes_at_the_start_of_a_yawn_are_not_a_microsleep():
    idx = {name: i for i, name in enumerate(FEATURES)}
    yawn = samples("yawning")                                 # jaw open from 4 s to 7 s
    t = yawn[:, 0]
    squeeze = (t >= 4.0) & (t < 5.0)
    yawn[squeeze, 2], yawn[squeeze, 5] = 0.10, 0.9
    assert window_features(yawn, BASE)[idx["closed_longest_s"]] < 0.2
    plain = samples("yawning")
    shut = (t >= 1.0) & (t < 2.0)                             # the same closure with the mouth closed is a real closure
    plain[shut, 2], plain[shut, 5] = 0.10, 0.9
    assert 0.8 <= window_features(plain, BASE)[idx["closed_longest_s"]] <= 1.2


def test_a_turned_head_makes_ear_unreliable_so_only_the_blink_score_counts():
    idx = {name: i for i, name in enumerate(FEATURES)}
    t = samples("plain")[:, 0]
    low_ear = (t >= 2.0) & (t < 4.0)
    turned = samples("plain")
    turned[low_ear, 2], turned[low_ear, 7] = 0.05, 45.0       # EAR collapses, but the head is turned 45 degrees
    assert window_features(turned, BASE)[idx["closed_longest_s"]] == 0
    facing = samples("plain")
    facing[low_ear, 2] = 0.05                                 # same EAR facing the camera: eyes really are shut
    assert 1.8 <= window_features(facing, BASE)[idx["closed_longest_s"]] <= 2.3


def test_window_features_refuse_unjudgeable_windows():
    assert window_features(samples("alert"), None) is None
    assert window_features(samples("alert")[:5], BASE) is None
    gone = samples("alert")
    gone[: int(len(gone) * 0.6) + 5, 1] = 0.0                 # face missing for most of the window
    assert window_features(gone, BASE) is None


def test_make_windows_slides_over_a_video():
    wins = list(make_windows(samples("alert", seconds=20), BASE, window_s=8.0, stride_s=4.0))
    assert [round(t) for t, _ in wins] == [0, 4, 8, 12] and all(v.shape == (len(FEATURES),) for _, v in wins)
    assert list(make_windows(np.empty((0, len(COLUMNS))), BASE, 8.0, 4.0)) == []


def test_upper_half_median_matches_calibration():
    assert upper_half_median([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.35)
    assert upper_half_median([np.nan, 0.3]) == pytest.approx(0.3) and upper_half_median([]) is None


# ---- labels and extraction ---------------------------------------------------------------------

@pytest.mark.parametrize("path, expected", [
    ("Mirror/Female_mirror/1-FemaleNoGlasses-Yawning.avi", ("female1", "yawning")),
    ("Mirror/Male_mirror/12-MaleGlasses-Talking.avi", ("male12", "alert")),
    ("Mirror/Male_mirror/12-MaleGlasses-Normal.avi", ("male12", "alert")),
    ("Mirror/Male_mirror/3-MaleNoGlasses-Talking&Yawning.avi", ("male3", "skip")),
    ("Dash/Female/4-FemaleNoGlasses.avi", ("female4", "skip")),
    ("P07/0.mp4", ("P07", "alert")), ("P07/5.mp4", ("P07", "skip")), ("P07/10.mp4", ("P07", "drowsy")),
])
def test_guess_label(path, expected):
    assert extract.guess_label(Path(path)) == expected


def test_labels_command_lists_every_video(tmp_path, capsys):
    root = tmp_path / "ds" / "Mirror" / "Female_mirror"
    root.mkdir(parents=True)
    for name in ("1-FemaleNoGlasses-Normal.avi", "1-FemaleNoGlasses-Yawning.avi"):
        (root / name).write_bytes(b"")
    extract.main(["labels", "--root", str(tmp_path / "ds"), "--out", str(tmp_path / "labels.csv")])
    rows = list(csv.DictReader((tmp_path / "labels.csv").open()))
    assert {r["label"] for r in rows} == {"alert", "yawning"} and {r["subject"] for r in rows} == {"female1"}
    assert "2 videos" in capsys.readouterr().out


class FakeTracker:
    """Stands in for FaceTracker (the model file can't be assumed): yawning face on every frame but the 3rd."""

    def __init__(self, *_):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def process(self, frame, t_ms):
        self.n += 1
        if self.n % 3 == 0:
            return None
        return FaceReading(points=np.zeros((478, 3)), ear=0.3, mar=0.7, pitch=1.0, yaw=2.0, roll=0.0, gaze=0.1,
                           blendshapes={"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1, "jawOpen": 0.9})


def test_extract_video_writes_one_row_per_sampled_frame(tmp_path):
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(60):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    out = tmp_path / "frames" / "v.csv"
    rows, face_frac = extract.extract_video(video, out, "s1", "yawning", stride=2, tracker_factory=FakeTracker)
    assert rows == 30 and face_frac == pytest.approx(2 / 3, abs=0.05) and not out.with_suffix(".part").exists()
    data = list(csv.reader(out.open()))
    assert tuple(data[0]) == extract.FRAME_HEADER and data[1][:3] == ["v.mp4", "s1", "yawning"]
    assert float(data[2][3]) == pytest.approx(2 / 30, abs=1e-3)


# ---- training and live use ---------------------------------------------------------------------

def write_dataset(folder: Path, people: int = 6) -> None:
    folder.mkdir(parents=True)
    for p in range(people):
        for kind in ("alert", "yawning", "drowsy"):
            for rep in range(2):
                s = samples(kind, seconds=17, seed=p * 10 + rep, ear=0.22 + 0.03 * p)
                with (folder / f"p{p}_{kind}_{rep}.csv").open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(extract.FRAME_HEADER)
                    for row in s:
                        w.writerow((f"p{p}_{kind}_{rep}.avi", f"p{p}", kind, *row))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("rf")
    write_dataset(root / "frames")
    out = root / "rf.pkl"
    train.main(["--frames", str(root / "frames"), "--out", str(out), "--trees", "40", "--folds", "3"])
    return out


def test_training_reports_people_not_seen_in_training_and_saves_a_model(trained, capsys):
    assert trained.exists()
    # (the summary is printed by the fixture run; run again quietly on the same data to read it)
    train.main(["--frames", str(trained.parent / "frames"), "--out", str(trained.parent / "again.pkl"), "--trees", "20", "--folds", "3"])
    text = capsys.readouterr().out
    assert "6 people" in text and "people never seen in training" in text and "Rule-based baseline" in text
    assert "mouth_open_longest_s" in text or "closed_longest_s" in text        # feature importance table


def test_training_needs_two_people(tmp_path):
    write_dataset(tmp_path / "frames", people=1)
    with pytest.raises(SystemExit, match="two different people"):
        train.main(["--frames", str(tmp_path / "frames"), "--out", str(tmp_path / "m.pkl")])


def run_stream(model, kind: str, seconds: float = 14.0, status_level: Level = Level.OK):
    from driver_monitor.classifier import ModelAlert

    cfg = Config()
    alert = ModelAlert(model, cfg)
    s = samples(kind, seconds=seconds, seed=99, ear=BASE)
    last = None
    for row in s:
        f = Frame(t=row[0], face=True, ear=row[2], mar=row[3], jaw_open=row[4], blink=row[5], pitch=row[6], yaw=row[7], gaze=row[8])
        status = Status(status_level, [], False, False, 0.2, 0.0, 0, 0.0, 0.0)
        alert.apply(status, f.t, f, BASE)
        last = status
    return last


def test_live_model_adds_a_warn_for_yawning_and_fatigue_but_not_for_alert(trained):
    y = run_stream(trained, "yawning")
    assert y.level == Level.WARN and y.reasons and y.reasons[0].startswith("Yawn: model")
    d = run_stream(trained, "drowsy")
    assert d.level == Level.WARN and d.reasons[0].startswith("Fatigue: model")
    a = run_stream(trained, "alert")
    assert a.level == Level.OK and a.reasons == []


def test_live_model_never_lowers_an_alert(trained):
    assert run_stream(trained, "alert", status_level=Level.ALERT).level == Level.ALERT


def test_live_model_waits_for_a_baseline_and_a_full_window(trained):
    from driver_monitor.classifier import ModelAlert

    m = ModelAlert(trained, Config())
    f = Frame(t=0.0, face=True, ear=0.3, mar=0.1, jaw_open=0.0, blink=0.0)
    assert m.update(0.0, f, None) is None                       # not calibrated yet
    assert m.update(1.0, Frame(t=1.0, face=True, ear=0.3), BASE) is None    # window not full


def test_a_model_trained_with_other_features_is_refused(tmp_path):
    import pickle

    from driver_monitor.classifier import ModelAlert

    with (tmp_path / "old.pkl").open("wb") as fh:
        pickle.dump({"model": None, "features": ("x",), "classes": [], "window_s": 8.0}, fh)
    with pytest.raises(ValueError, match="different features"):
        ModelAlert(tmp_path / "old.pkl", Config())


def test_rule_baseline_uses_the_apps_own_thresholds():
    idx = {n: i for i, n in enumerate(features.FEATURES)}
    X = np.zeros((3, len(FEATURES)))
    X[0, idx["mouth_open_longest_s"]] = 2.0
    X[1, idx["closed_longest_s"]] = 1.5
    assert list(train.rule_predict(X)) == ["yawning", "drowsy", "alert"]


# ---- the forest itself --------------------------------------------------------------------------

def blobs(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    X = rng.normal(0, 1, (n, 8))
    X[:, 0] += (y == "b") * 6.0                      # feature 0 separates b, feature 3 separates c
    X[:, 3] -= (y == "c") * 6.0
    return X, y


def test_forest_learns_and_finds_the_informative_features():
    X, y = blobs()
    f = forest.RandomForest(n_trees=60, seed=1).fit(X[:200], y[:200])
    assert (f.predict(X[200:]) == y[200:]).mean() > 0.9
    assert list(f.classes_) == ["a", "b", "c"] and f.feature_importances_.sum() == pytest.approx(1.0)
    assert set(np.argsort(f.feature_importances_)[::-1][:2]) == {0, 3}
    p = f.predict_proba(X[:10])
    assert p.shape == (10, 3) and np.allclose(p.sum(axis=1), 1.0)


def test_forest_is_repeatable_and_survives_pickling():
    import pickle

    X, y = blobs()
    a = forest.RandomForest(n_trees=20, seed=5).fit(X, y)
    b = forest.RandomForest(n_trees=20, seed=5).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    c = pickle.loads(pickle.dumps(a))
    assert np.array_equal(a.predict_proba(X), c.predict_proba(X))


def test_forest_handles_a_single_pure_class_and_constant_features():
    f = forest.RandomForest(n_trees=5).fit(np.ones((20, 4)), np.array(["x"] * 20))
    assert list(f.predict(np.ones((3, 4)))) == ["x"] * 3


def test_group_folds_never_split_a_person():
    groups = np.array(["p1"] * 10 + ["p2"] * 8 + ["p3"] * 6 + ["p4"] * 4 + ["p5"] * 2)
    folds = train.group_folds(groups, 3)
    assert len(folds) == 3
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]) and len(tr) + len(te) == len(groups)
    assert sorted(i for _, te in folds for i in te) == list(range(len(groups)))     # every window is tested once
