"""Live use of the trained Random Forest (the forest itself is plain NumPy, see forest.py).

``ModelAlert`` keeps the last ``window_s`` seconds of signals, asks the model what they look like
once per ``rf_every_s`` seconds, and adds a WARN reason when it is confident the driver is yawning
or drowsy. It can only raise caution, never lower it: the rule-based ALERTs in ``state.py`` stay
the safety layer, so a wrong or missing prediction can never silence an alarm.
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURES, VERSION, window_features
from .state import Config, Frame, Level, Status


@dataclass
class Prediction:
    label: str
    prob: float


# model label -> (reason prefix the alert policy already knows, spoken/banner text comes from alerts.py)
_REASONS = {"yawning": "Yawn", "drowsy": "Fatigue"}


class ModelAlert:
    def __init__(self, path: Path | str, cfg: Config) -> None:
        with Path(path).open("rb") as fh:
            bundle = pickle.load(fh)   # a model file you trained yourself
        if tuple(bundle["features"]) != FEATURES or bundle.get("version") != VERSION:
            raise ValueError("This model was trained with different features: retrain it with `python -m driver_monitor.train`.")
        self.model = bundle["model"]
        self.classes: list[str] = list(bundle["classes"])
        self.window_s: float = float(bundle["window_s"])
        self.cfg = cfg
        self._rows: deque[tuple] = deque()
        self._last_eval = -math.inf
        self.current: Prediction | None = None

    def update(self, t: float, f: Frame, baseline_ear: float | None) -> Prediction | None:
        """Add this frame; every rf_every_s seconds, refresh the prediction. Returns the latest one (or None)."""
        nan = math.nan
        self._rows.append((t, 1.0 if f.face else 0.0, f.ear if f.face else nan, f.mar if f.face else nan,
                           nan if f.jaw_open is None else f.jaw_open, nan if f.blink is None else f.blink,
                           f.pitch, f.yaw, f.gaze))
        while self._rows and t - self._rows[0][0] > self.window_s:
            self._rows.popleft()
        if baseline_ear is None or t - self._last_eval < self.cfg.rf_every_s:
            return self.current
        if t - self._rows[0][0] < 0.8 * self.window_s:
            return self.current                      # not enough history yet
        self._last_eval = t
        arr = np.array(self._rows, dtype=float)
        vec = window_features(arr, baseline_ear)
        if vec is None:
            self.current = None                      # face mostly missing: the rules handle "driver not visible"
            return None
        proba = self.model.predict_proba(vec.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        self.current = Prediction(self.classes[i], float(proba[i]))
        return self.current

    def apply(self, status: Status, t: float, f: Frame, baseline_ear: float | None) -> Prediction | None:
        """Update, then add a WARN reason to the status when the model is confident and the rules haven't said it already."""
        pred = self.update(t, f, baseline_ear)
        if pred is None or pred.prob < self.cfg.rf_min_prob:
            return pred
        prefix = _REASONS.get(pred.label)
        if prefix is not None and not any(r.startswith(prefix) for r in status.reasons):
            status.reasons.append(f"{prefix}: model {round(pred.prob * 100)}%")
            status.level = max(status.level, Level.WARN)
        return pred
