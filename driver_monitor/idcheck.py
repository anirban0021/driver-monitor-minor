"""Check how well face recognition separates people, using your own clips.

Put videos of several people in one folder and name them ``<person>-<anything>.mp4``, for example
``1-Normal.mp4``, ``1-Yawning.mp4``, ``2-Normal.mp4``, ``3-Drowsy.mp4`` (the leading number is the person).

    python -m driver_monitor.idcheck --root C:\\Datasets\\Own

For every video it keeps the clear frontal frames, describes the face in two ways (landmark shape and
the SFace embedding, if its model is downloaded) and compares videos with each other. What matters:
distances between videos of the SAME person must all be smaller than distances between DIFFERENT people.
The report prints both ranges and, when they do not overlap, a threshold that separates them.
"""

from __future__ import annotations

import argparse
import itertools
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .detector import DEFAULT_MODEL, FaceTracker
from .embedder import FaceEmbedder, cosine_distance
from .extract import VIDEO_EXT
from .profiles import identity_signature, signature_distance
from .state import Config


def person_of(video: Path) -> str:
    m = re.match(r"(\d+)", video.stem)
    return m.group(1) if m else video.stem


def video_templates(video: Path, cfg: Config, tracker_factory: Callable = FaceTracker, model: Path = DEFAULT_MODEL,
                    embedder: FaceEmbedder | None = None, stride: int = 3, max_frames: int = 120) -> dict:
    """Median landmark signature and median embedding over the clear frames of one video."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarks: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    n = -1
    try:
        with tracker_factory(model) as tracker:
            while len(landmarks) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if n % stride:
                    continue
                face = tracker.process(frame, int(n / fps * 1000))
                sig, _ = identity_signature(face, cfg)
                if sig is None:
                    continue
                landmarks.append(sig)
                if embedder is not None:
                    emb = embedder.embed(frame, face.points)
                    if emb is not None:
                        embeddings.append(emb)
    finally:
        cap.release()
    return {
        "frames": len(landmarks),
        "landmarks": np.median(np.stack(landmarks), axis=0) if len(landmarks) >= 5 else None,
        "embedding": np.median(np.stack(embeddings), axis=0) if len(embeddings) >= 5 else None,
    }


def pair_distances(templates: dict[str, dict], key: str, dist: Callable) -> tuple[list[float], list[float]]:
    """(same-person distances, different-person distances) between every pair of videos."""
    same, diff = [], []
    for ta, tb in itertools.combinations(templates.values(), 2):
        if ta[key] is None or tb[key] is None:
            continue
        (same if ta["person"] == tb["person"] else diff).append(dist(ta[key], tb[key]))
    return same, diff


def verdict(same: list[float], diff: list[float]) -> str:
    def rng(v: list[float]) -> str:
        return f"min {min(v):.3f}  mean {statistics.fmean(v):.3f}  max {max(v):.3f}  ({len(v)} pairs)"

    if not same or not diff:
        return "  need at least two videos of one person and videos of at least two people"
    lines = [f"  same person:       {rng(same)}", f"  different people:  {rng(diff)}"]
    if max(same) < min(diff):
        lines.append(f"  -> SEPARABLE. A threshold between {max(same):.3f} and {min(diff):.3f} works; "
                     f"the middle is {(max(same) + min(diff)) / 2:.3f}.")
    else:
        lines.append(f"  -> NOT separable: the largest same-person distance ({max(same):.3f}) is not below the "
                     f"smallest different-person distance ({min(diff):.3f}), so no threshold avoids mistakes.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.idcheck", description="How well does face recognition separate people?")
    p.add_argument("--root", type=Path, required=True, help="folder of <person>-<name>.mp4 videos")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--stride", type=int, default=3, help="look at every Nth frame")
    args = p.parse_args(argv)

    videos = sorted(v for v in args.root.rglob("*") if v.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"No videos under {args.root}")
    cfg = Config.load(args.config)
    try:
        embedder = FaceEmbedder()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"SFace not available ({exc}); checking the landmark signature only.", file=sys.stderr)
        embedder = None

    templates: dict[str, dict] = {}
    for v in videos:
        t = video_templates(v, cfg, embedder=embedder, stride=args.stride)
        t["person"] = person_of(v)
        templates[v.name] = t
        print(f"{v.name:<28} person {t['person']:<4} clear frames {t['frames']}")

    print("\nLandmark shape (current fallback; threshold match_threshold in config.toml):")
    print(verdict(*pair_distances(templates, "landmarks", signature_distance)))
    if embedder is not None:
        print("\nSFace embedding (cosine distance; threshold embed_threshold in config.toml):")
        print(verdict(*pair_distances(templates, "embedding", cosine_distance)))


if __name__ == "__main__":
    main()
