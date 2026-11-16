"""Embedding-based driver recognition: alignment points, cosine matching, mixed databases, the check tool."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from driver_monitor import idcheck
from driver_monitor.embedder import EMBED_DIM, FaceEmbedder, alignment_row, cosine_distance
from driver_monitor.profiles import (
    Identifier,
    Phase,
    ProfileStore,
    best_match,
    face_signature,
    signature_distance,
    thresholds_for,
)
from driver_monitor.state import Config

from .test_profiles import BASELINE, make_face


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(0, 1, EMBED_DIM)
    return v / np.linalg.norm(v)


def feed(ident: Identifier, sig: np.ndarray, seconds: float, t0: float = 0.0, baseline=BASELINE):
    status = None
    for i in range(int(seconds * 30)):
        status = ident.update(t0 + i / 30, sig, None, baseline)
        if status.event:
            return status
    return status


# ---- alignment row ------------------------------------------------------------------------------

def test_alignment_row_has_the_layout_opencv_expects():
    row = alignment_row(make_face(1))
    assert row.shape == (1, 15) and row.dtype == np.float32
    r = row[0]
    assert r[4] < r[6]                       # first eye is the one on the image's left
    assert r[10] < r[12]                     # first mouth corner is on the image's left
    assert r[2] > 0 and r[3] > 0 and r[14] == 1.0


def test_alignment_row_is_the_same_for_a_mirrored_frame():
    pts = make_face(1)
    mirrored = pts.copy()
    mirrored[:, 0] = 640 - mirrored[:, 0]
    a, b = alignment_row(pts)[0], alignment_row(mirrored)[0]
    assert a[4] < a[6] and b[4] < b[6] and b[10] < b[12]


def test_alignment_row_rejects_unusable_points():
    assert alignment_row(np.zeros((10, 3))) is None
    flat = make_face(1)
    flat[:, 0] = 100.0                       # both eyes at the same x
    assert alignment_row(flat) is None


def test_embedder_without_its_model_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model"):
        FaceEmbedder(tmp_path / "missing.onnx")


# ---- distances and database ---------------------------------------------------------------------

def test_cosine_distance_basics():
    a = unit(1)
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert cosine_distance(a, -a) == pytest.approx(2.0)
    assert cosine_distance(a, 5 * a) == pytest.approx(0.0, abs=1e-9)    # the length does not matter
    assert cosine_distance(a, np.zeros(EMBED_DIM)) == 1.0
    assert signature_distance(a, unit(2)) == pytest.approx(cosine_distance(a, unit(2)))


def test_thresholds_follow_the_kind_of_signature():
    cfg = Config()
    assert thresholds_for(unit(1), cfg) == (cfg.embed_threshold, cfg.embed_margin)
    assert thresholds_for(face_signature(make_face(1)), cfg) == (cfg.match_threshold, cfg.match_margin)


def test_store_keeps_embeddings_and_old_signatures_side_by_side():
    with ProfileStore(":memory:") as store:
        store.add("old", face_signature(make_face(1)), BASELINE)
        store.add("new", unit(1), BASELINE)
        sizes = sorted(len(d.signature) for d in store.drivers())
        assert sizes == [EMBED_DIM, len(face_signature(make_face(1)))]


def test_matching_only_compares_like_with_like():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        store.add("old", face_signature(make_face(1)), BASELINE)
        e = store.add("new", unit(1), BASELINE)
        m = best_match(unit(1), store.drivers(), cfg.embed_threshold, cfg.embed_margin)
        assert m.driver.id == e.id
        assert best_match(unit(99), [store.drivers()[0]], cfg.embed_threshold, cfg.embed_margin).driver is None


# ---- recognition flow with embeddings -----------------------------------------------------------

def test_embedding_driver_is_enrolled_then_recognised_and_a_stranger_is_not():
    cfg = Config()
    me, stranger = unit(1), unit(2)
    with ProfileStore(":memory:") as store:
        s = feed(Identifier(store, cfg, name="Anirban"), me, 11.0)
        assert s.is_new and s.event == "enrolled"
        again = feed(Identifier(store, cfg), me + 0.05 * unit(7), 3.0, baseline=None)
        assert again.event == "recognized" and again.driver.name == "Anirban" and again.distance < cfg.embed_threshold
        other = Identifier(store, cfg)
        s = feed(other, stranger, 3.0, baseline=None)
        assert other.phase is Phase.ENROLLING and s.event is None      # not "Anirban"
        s = feed(other, stranger, 11.0, t0=3.0)
        assert s.event == "enrolled" and len(store.drivers()) == 2


def test_an_old_landmark_profile_does_not_block_or_capture_an_embedding_driver():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        store.add("Anirban (old method)", face_signature(make_face(1)), BASELINE)
        ident = Identifier(store, cfg, name="Anirban")
        feed(ident, unit(5), 3.0, baseline=None)
        assert ident.phase is Phase.ENROLLING                          # the old row was ignored, a new one is made


# ---- the check tool -----------------------------------------------------------------------------

def test_person_is_the_leading_number_of_the_file_name(tmp_path):
    assert idcheck.person_of(tmp_path / "12-Drowsy.mp4") == "12"
    assert idcheck.person_of(tmp_path / "Kaif_clip.mp4") == "Kaif_clip"


def test_verdict_says_when_people_can_and_cannot_be_separated():
    ok = idcheck.verdict([0.10, 0.20], [0.60, 0.80])
    assert "SEPARABLE" in ok and "NOT" not in ok and "0.400" in ok
    bad = idcheck.verdict([0.10, 0.70], [0.60, 0.80])
    assert "NOT separable" in bad
    assert "need at least" in idcheck.verdict([], [0.5])


def test_pair_distances_split_same_and_different_people():
    t = {
        "1-a.mp4": {"person": "1", "v": np.array([1.0, 0.0])},
        "1-b.mp4": {"person": "1", "v": np.array([0.9, 0.1])},
        "2-a.mp4": {"person": "2", "v": np.array([0.0, 1.0])},
        "3-a.mp4": {"person": "3", "v": None},                          # no clear frames: left out
    }
    same, diff = idcheck.pair_distances(t, "v", cosine_distance)
    assert len(same) == 1 and len(diff) == 2 and max(same) < min(diff)


class _Tracker:
    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def process(self, frame, t_ms):
        from driver_monitor.detector import FaceReading  # noqa: F401  (documents what a real reading is)
        return type("F", (), {"points": make_face(1), "yaw": 0.0, "pitch": 0.0, "blink": 0.0, "jaw_open": 0.0})()


class _Embedder:
    def embed(self, frame, points):
        return unit(3)


def test_video_templates_keep_only_clear_frames(tmp_path):
    video = tmp_path / "1-Normal.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(60):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    t = idcheck.video_templates(video, Config(), tracker_factory=_Tracker, embedder=_Embedder(), stride=2)
    assert t["frames"] == 30 and t["embedding"] is not None and t["landmarks"] is not None
    assert cosine_distance(t["embedding"], unit(3)) == pytest.approx(0.0, abs=1e-9)
    none = idcheck.video_templates(video, Config(), tracker_factory=_Tracker, embedder=None, stride=2)
    assert none["embedding"] is None
