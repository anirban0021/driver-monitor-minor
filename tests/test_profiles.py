"""Driver recognition: face signature, database, matching and the 10 s enrolment flow (no camera needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driver_monitor import hud
from driver_monitor.profiles import (
    SIGNATURE_DIM,
    Identifier,
    Phase,
    ProfileStore,
    best_match,
    face_signature,
    signature_distance,
)
from driver_monitor.state import Config, DriverState, Frame

BASELINE = {"ear": 0.30, "pitch": -2.0, "yaw": 1.0, "gaze": 0.05}


def make_face(seed: int) -> np.ndarray:
    """A synthetic 478-point 'face' in pixel units with the landmarks the signature relies on in sensible places."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 40, (478, 3))
    pts[:, 2] *= 0.5
    pts[33], pts[133] = (-45, 0, 0), (-15, 0, 0)       # right eye corners (outer, inner)
    pts[362], pts[263] = (15, 0, 0), (45, 0, 0)        # left eye corners (inner, outer)
    pts[10], pts[152] = (0, -70, 0), (0, 85, 0)        # forehead, chin
    return pts + (320, 240, 0)


def rotate(pts: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    a, b, c = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return (pts - pts.mean(0)) @ (rz @ rx @ ry).T + pts.mean(0)


def noisy(sig: np.ndarray, sd: float, seed: int) -> np.ndarray:
    return sig + np.random.default_rng(seed).normal(0, sd, sig.shape)


# ---- signature ----------------------------------------------------------------------------------

def test_signature_ignores_position_scale_and_head_rotation():
    face = make_face(1)
    ref = face_signature(face)
    assert ref is not None and ref.shape == (SIGNATURE_DIM,)
    moved = rotate(face, 25, -15, 10) * 1.7 + (100, -50, 30)
    assert signature_distance(ref, face_signature(moved)) < 1e-6


def test_signature_rejects_unusable_meshes():
    assert face_signature(np.zeros((100, 3))) is None
    assert face_signature(np.zeros((478, 3))) is None


def test_different_faces_are_further_apart_than_the_same_face():
    a, b = face_signature(make_face(1)), face_signature(make_face(2))
    same = signature_distance(a, noisy(a, 0.01, 7))
    assert signature_distance(a, b) > 5 * same


# ---- database -----------------------------------------------------------------------------------

def test_store_roundtrip_and_management(tmp_path):
    sig = face_signature(make_face(1))
    with ProfileStore(tmp_path / "sub" / "d.db") as store:
        d = store.add(None, sig, BASELINE)
        assert d.name == f"Driver {d.id}"
        named = store.add("Anirban", face_signature(make_face(2)), None)
        assert named.name == "Anirban"
    with ProfileStore(tmp_path / "sub" / "d.db") as store:       # survives a restart
        a, b = store.drivers()
        assert signature_distance(a.signature, sig) < 1e-5       # stored as float32
        assert a.baseline == pytest.approx(BASELINE) and b.baseline is None
        store.seen(a.id)
        store.set_baseline(a.id, {**BASELINE, "ear": 0.28})
        assert store.rename(b.id, "Agniva")
        assert store.delete(a.id) and not store.delete(a.id)
        assert [d.name for d in store.drivers()] == ["Agniva"]


def test_store_skips_signatures_with_another_layout(tmp_path):
    with ProfileStore(tmp_path / "d.db") as store:
        store.add("old", np.zeros(12), None)
        assert store.drivers() == []


# ---- matching -----------------------------------------------------------------------------------

def test_best_match_threshold_and_margin():
    with ProfileStore(":memory:") as store:
        a = store.add("A", face_signature(make_face(1)), BASELINE)
        b = store.add("B", face_signature(make_face(2)), BASELINE)
        drivers = store.drivers()
        probe = noisy(a.signature, 0.01, 3)
        m = best_match(probe, drivers, threshold=0.05, margin=0.015)
        assert m.driver is not None and m.driver.id == a.id and m.distance < 0.05
        assert best_match(face_signature(make_face(3)), drivers, 0.05, 0.015).driver is None   # a stranger
        assert best_match(probe, drivers, threshold=1e-6, margin=0.0).driver is None           # too strict
        assert best_match(probe, [a, b], threshold=10.0, margin=1e9).driver is None            # ambiguous
        assert best_match(probe, [], 0.05, 0.015).driver is None                               # empty database


# ---- the start-up flow --------------------------------------------------------------------------

def run(ident: Identifier, sig, seconds: float, t0: float = 0.0, baseline=BASELINE, fps: float = 30.0):
    """Feed `seconds` of frames; returns the last status and the statuses that carried an event."""
    status, events = None, []
    for i in range(int(seconds * fps)):
        status = ident.update(t0 + i / fps, noisy(sig, 0.005, i) if sig is not None else None, None, baseline)
        if status.event:
            events.append(status)
    return status, events


def test_new_driver_is_scanned_for_ten_seconds_then_saved():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        ident = Identifier(store, cfg, name="Kaif")
        sig = face_signature(make_face(1))
        s, events = run(ident, sig, 6.0)
        assert s.phase is Phase.ENROLLING and 0.5 < s.progress < 0.7 and not events and store.drivers() == []
        s, events = run(ident, sig, 5.0, t0=6.0)
        assert s.phase is Phase.READY and s.is_new and [e.event for e in events] == ["enrolled"]
        saved = store.drivers()
        assert len(saved) == 1 and saved[0].name == "Kaif" and saved[0].baseline == pytest.approx(BASELINE)
        assert events[0].driver.id == saved[0].id
        # enrolment finished only after enroll_s seconds of clear frames
        assert events[0].progress == 1.0


def test_enrolment_waits_for_a_baseline():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        ident = Identifier(store, cfg)
        s, _ = run(ident, face_signature(make_face(1)), 12.0, baseline=None)
        assert s.phase is Phase.ENROLLING and s.progress == 1.0 and store.drivers() == []
        s, events = run(ident, face_signature(make_face(1)), 0.2, t0=12.0)
        assert s.phase is Phase.READY and events


def test_unclear_frames_do_not_count_towards_enrolment():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        ident = Identifier(store, cfg)
        s, _ = run(ident, None, 20.0)               # nobody in front of the camera
        assert s.phase is Phase.SCANNING and s.progress == 0.0 and store.drivers() == []


def test_returning_driver_is_recognised_without_a_new_scan():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        sig = face_signature(make_face(1))
        run(Identifier(store, cfg), sig, 11.0)                      # first visit: enrolled
        second = Identifier(store, cfg)                              # next day: new session
        s, events = run(second, sig, 3.0, baseline=None)             # ... before any calibration has finished
        assert [e.event for e in events] == ["recognized"]
        assert s.phase is Phase.READY and not s.is_new and s.distance < cfg.match_threshold
        assert len(store.drivers()) == 1 and store.drivers()[0].visits == 2


def test_a_second_person_gets_their_own_profile_and_the_first_is_still_recognised():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        a, b = face_signature(make_face(1)), face_signature(make_face(2))
        run(Identifier(store, cfg), a, 11.0)
        _, events = run(Identifier(store, cfg), b, 11.0)
        assert [e.event for e in events] == ["enrolled"] and len(store.drivers()) == 2
        _, events = run(Identifier(store, cfg), a, 3.0)
        assert events[0].event == "recognized" and events[0].driver.id == store.drivers()[0].id


def test_manual_recalibration_updates_the_saved_baseline():
    cfg = Config()
    with ProfileStore(":memory:") as store:
        sig = face_signature(make_face(1))
        ident = Identifier(store, cfg)
        run(ident, sig, 11.0)
        ident.refresh_baseline()
        run(ident, sig, 0.1, t0=20.0, baseline={**BASELINE, "ear": 0.26})
        assert store.drivers()[0].baseline["ear"] == pytest.approx(0.26)


# ---- state and HUD hooks ------------------------------------------------------------------------

def test_driver_state_can_adopt_a_saved_baseline():
    st = DriverState(Config())
    assert st.baseline() is None and not st.calibrated
    st.load_baseline(**BASELINE)
    assert st.calibrated and st.baseline() == BASELINE
    assert st.threshold() == pytest.approx(0.30 * Config().ear_ratio)
    s = st.update(Frame(t=0.0, face=True, ear=0.3))
    assert not s.calibrating


def test_alerts_use_the_loaded_drivers_own_eye_threshold():
    """Same EAR reading: 'closed' for a wide-eyed driver, 'open' for someone whose eyes are naturally narrow."""
    wide, narrow = DriverState(Config()), DriverState(Config())
    wide.load_baseline(ear=0.40, pitch=0.0, yaw=0.0, gaze=0.0)
    narrow.load_baseline(ear=0.22, pitch=0.0, yaw=0.0, gaze=0.0)
    f = Frame(t=0.0, face=True, ear=0.26)
    assert wide.update(f).eyes_closed and not narrow.update(f).eyes_closed


def test_hud_draws_every_identity_phase():
    from driver_monitor.profiles import Driver, IdentityStatus

    d = Driver(1, "Driver 1", np.zeros(SIGNATURE_DIM), BASELINE)
    state = DriverState(Config())
    status = state.update(Frame(t=0.0, face=False))
    for who in (IdentityStatus(), IdentityStatus(Phase.ENROLLING, progress=0.4, hint="Close your mouth"),
                IdentityStatus(Phase.ENROLLING, progress=0.4), IdentityStatus(Phase.READY, d, True),
                IdentityStatus(Phase.READY, d, False, distance=0.031)):
        img = np.zeros((720, 1280, 3), np.uint8)
        hud.draw(img, None, status, 30.0, cfg=Config(), who=who)
        assert img.any()
