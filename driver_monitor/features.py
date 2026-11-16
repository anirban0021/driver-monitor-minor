"""Window features for the learned drowsiness / yawning classifier.

The same code turns a short stretch of per-frame signals into one feature vector in training
(``train.py``) and in the live app (``classifier.py``), so the model never sees numbers that were
computed differently from the ones it learned on. Eye openness is divided by the driver's own
open-eye baseline, which is what per-driver calibration (or a recognised driver's saved profile)
provides, so a narrow-eyed and a wide-eyed driver look the same to the model.

Pure numpy: no camera and no compiled ML package needed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .state import Config

# One row per frame: time (s), face found (0/1), then the raw signals. Missing values are NaN.
COLUMNS = ("t", "face", "ear", "mar", "jaw", "blink", "pitch", "yaw", "gaze")
T, FACE, EAR, MAR, JAW, BLINK, PITCH, YAW, GAZE = range(len(COLUMNS))

CLASSES = ("alert", "yawning", "drowsy")  # "skip" is allowed in label files and is dropped
MIN_FACE_FRAC = 0.6                       # windows where the face is mostly missing are not judged
VERSION = 2                               # bump when a feature's meaning changes: older model files are then refused

FEATURES = (
    "ear_rel_mean", "ear_rel_min", "ear_rel_std",
    "perclos", "closed_longest_s", "closures_per_min", "closure_mean_s",
    "mar_mean", "mar_max", "jaw_mean", "jaw_max",
    "mouth_open_frac", "mouth_open_longest_s", "mouth_open_episodes",
    "pitch_std", "pitch_range", "yaw_std", "yaw_range", "gaze_std",
    "face_frac",
)

_CFG = Config()  # thresholds that define "closed" and "mouth open" come from the app's own defaults


def upper_half_median(values) -> float | None:
    """Open-eye baseline: median of the upper half of the samples, as DriverState calibration does it."""
    v = np.asarray(values, dtype=float)
    v = np.sort(v[~np.isnan(v)])
    return float(np.median(v[len(v) // 2:])) if len(v) else None


def _runs(mask: np.ndarray, dt: np.ndarray) -> list[float]:
    """Durations (s) of each stretch where mask stays True."""
    out, cur = [], 0.0
    for m, d in zip(mask, dt):
        if m:
            cur += d
        elif cur > 0:
            out.append(cur)
            cur = 0.0
    if cur > 0:
        out.append(cur)
    return out


def _run_so_far(mask: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """For every frame, how long the current stretch of True has lasted (0 where mask is False)."""
    out = np.zeros(len(mask))
    cur = 0.0
    for i, (m, d) in enumerate(zip(mask, dt)):
        cur = cur + d if m else 0.0
        out[i] = cur
    return out


def _std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) else 0.0


def _range(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x)) if len(x) else 0.0


def window_features(samples: np.ndarray, baseline_ear: float | None) -> np.ndarray | None:
    """Feature vector (len(FEATURES)) for one window of (N, len(COLUMNS)) samples, or None if it can't be judged."""
    s = np.asarray(samples, dtype=float)
    if baseline_ear is None or baseline_ear <= 0 or len(s) < 10:
        return None
    face = s[:, FACE] > 0.5
    face_frac = float(face.mean())
    if face_frac < MIN_FACE_FRAC:
        return None

    t = s[:, T]
    dt = np.diff(t, append=t[-1])
    dt[-1] = float(np.median(dt[:-1])) if len(dt) > 1 else 0.0
    dt = np.clip(dt, 0.0, 0.25)                 # a dropped stretch of frames must not count as a long closure
    span = max(float(dt.sum()), 1e-6)
    face_time = max(float(dt[face].sum()), 1e-6)

    ear_rel = s[face, EAR] / baseline_ear
    cfg = _CFG

    # Mouth open: the jawOpen blendshape, or MAR when the blendshapes are missing (exactly as DriverState does).
    jaw_raw = s[:, JAW]
    with np.errstate(invalid="ignore"):
        mouth = face & np.where(np.isnan(jaw_raw), s[:, MAR] > cfg.yawn_mar, jaw_raw > cfg.yawn_jaw)
    jaw = np.nan_to_num(jaw_raw)

    # Eyes shut: the same rule as DriverState, so the features mean what the app's alerts mean.
    #  - head turned or tipped far from the camera: EAR is unreliable, only a confident blink counts;
    #  - blendshapes present: a confident blink, or EAR far below the threshold (drooping lids != shut);
    #  - no blendshapes: EAR below the threshold.
    # Eyes squeezed shut during the first yawn_squint_s of a real yawn are not a microsleep.
    blink_raw = s[:, BLINK]
    has_blink = ~np.isnan(blink_raw)
    blink = np.nan_to_num(blink_raw)
    thr = max(cfg.ear_floor, baseline_ear * cfg.ear_ratio)
    with np.errstate(invalid="ignore"):
        blink_closed = has_blink & (blink > cfg.blink_score)
        off_axis = (np.abs(np.nan_to_num(s[:, YAW])) > cfg.eye_pose_yaw_deg) | (np.nan_to_num(s[:, PITCH]) < -cfg.eye_pose_pitch_deg)
        shut = np.where(off_axis, blink_closed,
                        np.where(has_blink, blink_closed | (s[:, EAR] < thr * cfg.ear_sure_ratio), s[:, EAR] < thr))
    squint = mouth & (_run_so_far(mouth, dt) < cfg.yawn_squint_s)
    closed = face & ~squint & shut
    closures, openings = _runs(closed, dt), _runs(mouth, dt)

    f = {
        "ear_rel_mean": float(np.nanmean(ear_rel)),
        "ear_rel_min": float(np.nanmin(ear_rel)),
        "ear_rel_std": _std(ear_rel[~np.isnan(ear_rel)]),
        "perclos": float(dt[closed].sum()) / face_time,
        "closed_longest_s": max(closures, default=0.0),
        "closures_per_min": len(closures) * 60.0 / span,
        "closure_mean_s": float(np.mean(closures)) if closures else 0.0,
        "mar_mean": float(np.nanmean(s[face, MAR])),
        "mar_max": float(np.nanmax(s[face, MAR])),
        "jaw_mean": float(np.mean(jaw[face])),
        "jaw_max": float(np.max(jaw[face])),
        "mouth_open_frac": float(dt[mouth].sum()) / face_time,
        "mouth_open_longest_s": max(openings, default=0.0),
        "mouth_open_episodes": float(len(openings)),
        "pitch_std": _std(s[face, PITCH]),
        "pitch_range": _range(s[face, PITCH]),
        "yaw_std": _std(s[face, YAW]),
        "yaw_range": _range(s[face, YAW]),
        "gaze_std": _std(s[face, GAZE]),
        "face_frac": face_frac,
    }
    return np.array([f[k] for k in FEATURES], dtype=float)


def make_windows(samples: np.ndarray, baseline_ear: float | None, window_s: float, stride_s: float
                 ) -> Iterator[tuple[float, np.ndarray]]:
    """Slide a window over one video: yields (start time, feature vector). The last window may be a bit short."""
    s = np.asarray(samples, dtype=float)
    if len(s) == 0:
        return
    start, end = float(s[0, T]), float(s[-1, T])
    while start + 0.75 * window_s <= end:
        m = (s[:, T] >= start) & (s[:, T] < start + window_s)
        v = window_features(s[m], baseline_ear)
        if v is not None:
            yield start, v
        start += stride_s
