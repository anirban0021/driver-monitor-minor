"""Per-driver recognition: who is behind the wheel, and what does their normal face look like?

Three parts, all independent of the camera so they can be unit tested. A driver is described either by an
SFace embedding (``embedder.py``, preferred: it is trained to tell people apart) or, when that model
is not downloaded, by the landmark-shape ``face_signature`` below, which separates people far less reliably:

* ``face_signature``  turns the 478 FaceLandmarker points into a pose- and scale-free vector
  that describes the *shape* of the face (eyes, brows, nose, mouth, jaw line).
* ``ProfileStore``    keeps one row per driver in SQLite: the signature plus the personal
  baseline (open-eye EAR, neutral head pose, gaze) that ``DriverState`` learns during calibration.
* ``Identifier``      the start-up state machine. It looks at the first seconds of clear frames:
  a known face is recognised straight away and its stored baseline is reused, an unknown face is
  scanned for ``enroll_s`` seconds (10 by default) and saved as a new driver.

Only landmark numbers are stored, never photos. Run ``python -m driver_monitor.profiles list``
to see the saved drivers and ``... delete ID`` to remove one.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import metrics
from .embedder import EMBED_DIM, cosine_distance

if TYPE_CHECKING:
    from .detector import FaceReading
    from .state import Config


def _default_db() -> Path:
    """data/drivers.db in a source checkout, else ~/.driver-monitor/drivers.db (pip install)."""
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "drivers.db" if (root / "pyproject.toml").exists() else Path.home() / ".driver-monitor" / "drivers.db"


DEFAULT_DB = _default_db()

# --------------------------------------------------------------------------- face signature

_FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
              176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
_BROWS = (70, 63, 105, 66, 107, 300, 293, 334, 296, 336)
_NOSE = (1, 2, 4, 5, 6, 168, 197, 195, 98, 327, 129, 358, 49, 279, 64, 294, 48, 278)
_CHEEKS = (205, 425, 50, 280, 116, 345)
_LIPS = (61, 291, 0, 17, 37, 267, 84, 314)

# The landmarks that carry identity: jaw line, brows, eye shape, nose and mouth structure.
SIGNATURE_POINTS: tuple[int, ...] = tuple(sorted(set(
    _FACE_OVAL + _BROWS + _NOSE + _CHEEKS + _LIPS
    + metrics.RIGHT_EYE_CONTOUR + metrics.LEFT_EYE_CONTOUR + metrics.MOUTH_CONTOUR
)))
SIGNATURE_DIM = len(SIGNATURE_POINTS) * 3
Z_WEIGHT = 0.5  # MediaPipe's depth is the noisiest axis, so it counts for less


def face_signature(points: np.ndarray) -> np.ndarray | None:
    """Pose- and scale-free description of the face shape, or None if the mesh is unusable.

    The landmarks are moved into the face's own frame (origin between the eyes, x across the
    eyes, y towards the chin, z out of the face) and divided by the outer eye-corner distance.
    Turning the head, leaning towards the camera or moving the seat therefore changes nothing;
    what is left is the geometry that differs between people.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] < 468 or p.shape[1] < 3:
        return None
    r_out, r_in = p[metrics.RIGHT_EYE_CORNERS[0]], p[metrics.RIGHT_EYE_CORNERS[1]]
    l_in, l_out = p[metrics.LEFT_EYE_CORNERS[0]], p[metrics.LEFT_EYE_CORNERS[1]]
    across = l_out - r_out
    scale = float(np.linalg.norm(across))
    if scale < 1e-6:
        return None
    ex = across / scale
    down = p[metrics.CHIN] - p[metrics.FOREHEAD]
    down = down - float(down @ ex) * ex            # make it perpendicular to ex
    n = float(np.linalg.norm(down))
    if n < 1e-6:
        return None
    ey = down / n
    ez = np.cross(ex, ey)
    origin = (r_out + r_in + l_in + l_out) / 4.0
    local = (p[list(SIGNATURE_POINTS), :3] - origin) @ np.stack([ex, ey, ez]).T / scale
    local[:, 2] *= Z_WEIGHT
    return local.ravel()


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two stored faces of the same kind.

    Landmark signatures: RMS gap per landmark, in units of the distance between the eyes' outer corners.
    SFace embeddings (128 numbers): cosine distance, 1 minus the similarity.
    """
    if len(a) == EMBED_DIM:
        return cosine_distance(a, b)
    return float(np.linalg.norm(a - b) / math.sqrt(len(a) / 3))


def thresholds_for(sig: np.ndarray, cfg: Config) -> tuple[float, float]:
    """(match_threshold, match_margin) that fit this kind of signature."""
    return (cfg.embed_threshold, cfg.embed_margin) if len(sig) == EMBED_DIM else (cfg.match_threshold, cfg.match_margin)


def identity_signature(face: FaceReading | None, cfg: Config) -> tuple[np.ndarray | None, str | None]:
    """Signature of this frame, or (None, why) if the frame is not clear enough to identify anyone.

    Only a frontal face with open eyes and a closed mouth counts: a yawn or a sideways glance
    changes the geometry we are trying to measure.
    """
    if face is None:
        return None, "Face the camera"
    if abs(face.yaw) > cfg.id_pose_deg or abs(face.pitch) > cfg.id_pose_deg:
        return None, "Look at the camera"
    if face.blink > cfg.id_blink_max:
        return None, "Open your eyes"
    if face.jaw_open > cfg.id_jaw_max:
        return None, "Close your mouth"
    sig = face_signature(face.points)
    return (sig, None) if sig is not None else (None, "Face the camera")


# --------------------------------------------------------------------------- database

@dataclass
class Driver:
    id: int
    name: str
    signature: np.ndarray
    baseline: dict[str, float] | None   # keys: ear, pitch, yaw, gaze
    visits: int = 1
    created_at: str = ""
    last_seen: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    signature     BLOB    NOT NULL,
    baseline_ear   REAL,
    baseline_pitch REAL,
    baseline_yaw   REAL,
    baseline_gaze  REAL,
    visits        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL,
    last_seen     TEXT    NOT NULL
)
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProfileStore:
    """SQLite-backed list of drivers. One connection, used from the main thread only."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(_SCHEMA)
        self._db.commit()

    def drivers(self) -> list[Driver]:
        out = []
        rows = self._db.execute(
            "SELECT id, name, signature, baseline_ear, baseline_pitch, baseline_yaw, baseline_gaze, visits, created_at, last_seen "
            "FROM drivers ORDER BY id"
        ).fetchall()
        for id_, name, blob, ear, pitch, yaw, gaze, visits, created, seen in rows:
            sig = np.frombuffer(blob, dtype=np.float32).astype(float)
            if sig.size not in (SIGNATURE_DIM, EMBED_DIM):
                print(f"Skipping driver {id_} ({name}): saved with a different signature layout", file=sys.stderr)
                continue
            base = None if ear is None else {"ear": ear, "pitch": pitch, "yaw": yaw, "gaze": gaze}
            out.append(Driver(id_, name, sig, base, visits, created, seen))
        return out

    def add(self, name: str | None, signature: np.ndarray, baseline: dict[str, float] | None) -> Driver:
        now = _now()
        b = baseline or {}
        cur = self._db.execute(
            "INSERT INTO drivers (name, signature, baseline_ear, baseline_pitch, baseline_yaw, baseline_gaze, created_at, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name or "", np.asarray(signature, dtype=np.float32).tobytes(),
             b.get("ear"), b.get("pitch"), b.get("yaw"), b.get("gaze"), now, now),
        )
        driver_id = int(cur.lastrowid)
        if not name:
            name = f"Driver {driver_id}"
            self._db.execute("UPDATE drivers SET name = ? WHERE id = ?", (name, driver_id))
        self._db.commit()
        return Driver(driver_id, name, np.asarray(signature, dtype=np.float32).astype(float), baseline, 1, now, now)

    def seen(self, driver_id: int) -> None:
        self._db.execute("UPDATE drivers SET visits = visits + 1, last_seen = ? WHERE id = ?", (_now(), driver_id))
        self._db.commit()

    def set_baseline(self, driver_id: int, baseline: dict[str, float]) -> None:
        self._db.execute(
            "UPDATE drivers SET baseline_ear = ?, baseline_pitch = ?, baseline_yaw = ?, baseline_gaze = ? WHERE id = ?",
            (baseline["ear"], baseline["pitch"], baseline["yaw"], baseline["gaze"], driver_id),
        )
        self._db.commit()

    def rename(self, driver_id: int, name: str) -> bool:
        cur = self._db.execute("UPDATE drivers SET name = ? WHERE id = ?", (name, driver_id))
        self._db.commit()
        return cur.rowcount > 0

    def delete(self, driver_id: int) -> bool:
        cur = self._db.execute("DELETE FROM drivers WHERE id = ?", (driver_id,))
        self._db.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ProfileStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- matching

@dataclass
class Match:
    driver: Driver | None      # the recognised driver, or None for "someone new"
    distance: float            # to the closest saved driver (inf when the database is empty)
    runner_up: float = math.inf


def best_match(probe: np.ndarray, drivers: list[Driver], threshold: float, margin: float) -> Match:
    """Closest saved driver, accepted only if it is close enough and clearly closer than the next one."""
    drivers = [d for d in drivers if len(d.signature) == len(probe)]     # only compare like with like
    if not drivers:
        return Match(None, math.inf)
    ranked = sorted(((signature_distance(probe, d.signature), d) for d in drivers), key=lambda x: x[0])
    dist, best = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else math.inf
    known = dist <= threshold and runner - dist >= margin
    return Match(best if known else None, dist, runner)


# --------------------------------------------------------------------------- start-up state machine

class Phase(Enum):
    SCANNING = "scanning"    # collecting clear frames to find out who this is
    ENROLLING = "enrolling"  # unknown face: scanning it for enroll_s seconds
    READY = "ready"          # driver known (recognised or just saved)


@dataclass
class IdentityStatus:
    phase: Phase = Phase.SCANNING
    driver: Driver | None = None
    is_new: bool = False
    progress: float = 0.0          # enrolment, 0..1
    distance: float | None = None  # to the closest saved driver at the moment of the decision
    hint: str | None = None        # why the last frame did not count
    event: str | None = None       # "recognized" or "enrolled", only on the frame it happens


@dataclass
class Identifier:
    """Decides who is driving. Time is passed in, so it is deterministic and testable.

    Feed it one ``update`` per frame with that frame's signature (or None) and the baseline that
    ``DriverState`` has learned so far (or None). The ``event`` on the returned status tells the
    caller when to load a stored baseline ("recognized") or that a new profile was saved ("enrolled").
    """

    store: ProfileStore
    cfg: Config
    name: str | None = None                 # name for a newly enrolled driver
    phase: Phase = Phase.SCANNING
    driver: Driver | None = None
    is_new: bool = False
    distance: float | None = None
    _sigs: deque = field(default_factory=lambda: deque(maxlen=900))
    _good: float = 0.0                      # seconds of clear frames collected
    _last_t: float | None = None
    _refresh: bool = False

    def refresh_baseline(self) -> None:
        """After a manual recalibration: save the new baseline to the current driver once it is learned."""
        self._refresh = True

    def update(self, t: float, sig: np.ndarray | None, hint: str | None, baseline: dict[str, float] | None) -> IdentityStatus:
        cfg = self.cfg
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.25)
        self._last_t = t
        event = None

        if self.phase is Phase.READY:
            if self._refresh and baseline is not None and self.driver is not None:
                self.store.set_baseline(self.driver.id, baseline)
                self.driver.baseline = baseline
                self._refresh = False
            return self._status(hint, event)

        if sig is not None:
            self._sigs.append(sig)
            self._good += dt

        if self.phase is Phase.SCANNING and self._good >= cfg.identify_s and len(self._sigs) >= 10:
            template = np.median(np.stack(self._sigs), axis=0)
            m = best_match(template, self.store.drivers(), *thresholds_for(template, cfg))
            self.distance = None if math.isinf(m.distance) else m.distance
            if m.driver is not None:
                self.phase, self.driver, self.is_new, event = Phase.READY, m.driver, False, "recognized"
                self.store.seen(m.driver.id)
                return self._status(hint, event)
            self.phase = Phase.ENROLLING

        if self.phase is Phase.ENROLLING and self._good >= cfg.enroll_s and baseline is not None:
            template = np.median(np.stack(self._sigs), axis=0)
            self.driver = self.store.add(self.name, template, baseline)
            self.phase, self.is_new, event = Phase.READY, True, "enrolled"

        return self._status(hint, event)

    def _status(self, hint: str | None, event: str | None) -> IdentityStatus:
        progress = 1.0 if self.phase is Phase.READY else min(1.0, self._good / self.cfg.enroll_s) if self.phase is Phase.ENROLLING else 0.0
        return IdentityStatus(self.phase, self.driver, self.is_new, progress, self.distance, hint, event)


# --------------------------------------------------------------------------- management CLI

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.profiles", description="Manage the saved drivers.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"database file (default: {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every saved driver")
    r = sub.add_parser("rename", help="rename a driver")
    r.add_argument("id", type=int)
    r.add_argument("name")
    d = sub.add_parser("delete", help="delete a driver and their face data")
    d.add_argument("id", type=int)
    sub.add_parser("compare", help="distance between every pair of saved drivers (for tuning the match thresholds)")
    args = p.parse_args(argv)

    with ProfileStore(args.db) as store:
        drivers = store.drivers()
        if args.cmd == "list":
            if not drivers:
                print("No drivers saved yet.")
            for dr in drivers:
                ear = f"EAR {dr.baseline['ear']:.3f}" if dr.baseline else "no baseline"
                print(f"{dr.id:>3}  {dr.name:<20} {ear:<12} visits {dr.visits:<4} last seen {dr.last_seen}")
        elif args.cmd == "rename":
            print("Renamed." if store.rename(args.id, args.name) else f"No driver with id {args.id}.")
        elif args.cmd == "delete":
            print("Deleted." if store.delete(args.id) else f"No driver with id {args.id}.")
        elif args.cmd == "compare":
            if len(drivers) < 2:
                print("Need at least two saved drivers.")
            for i, a in enumerate(drivers):
                for b in drivers[i + 1:]:
                    if len(a.signature) != len(b.signature):
                        print(f"{a.name} <-> {b.name}: saved with different recognition methods, cannot compare")
                        continue
                    print(f"{a.name} <-> {b.name}: {signature_distance(a.signature, b.signature):.4f}")


if __name__ == "__main__":
    main()
