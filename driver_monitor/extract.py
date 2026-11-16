"""Turn dataset videos into per-frame feature CSVs, the input of ``train.py``.

Two steps:

  1. python -m driver_monitor.extract labels --root D:\\YawDD --out data/labels.csv
       Walks the dataset folder and *guesses* subject and label from file and folder names.
       Open the CSV and correct it: this file decides what the model learns.
  2. python -m driver_monitor.extract frames --root D:\\YawDD --labels data/labels.csv --out data/frames
       Runs the same FaceLandmarker and metrics as the live app on every video and writes one CSV
       per video. Videos that already have a CSV are skipped, so you can stop and resume.

labels.csv columns: video (path relative to --root), subject (one id per person), label
(alert, yawning, drowsy, or skip to leave the video out).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

import cv2

from .detector import DEFAULT_MODEL, FaceTracker
from .features import COLUMNS

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
LABELS = {"alert", "yawning", "drowsy", "skip"}
FRAME_HEADER = ("video", "subject", "label", *COLUMNS)


def guess_label(rel: Path) -> tuple[str, str]:
    """Best-effort (subject, label) from a dataset-relative path. Always review the result.

    YawDD, camera-under-mirror set: '.../Mirror/Female_mirror/1-FemaleNoGlasses-Yawning.avi', one situation
    per video. The dashboard set mixes situations inside one video, so it is skipped.
    UTA-RLDD: '<participant>/0.mp4' alert, '5.mp4' low vigilance (skipped: self-reported and ambiguous),
    '10.mp4' drowsy.
    """
    parts = [p.lower() for p in rel.parts]
    stem = rel.stem.lower()
    if stem in {"0", "5", "10"}:                                   # UTA-RLDD
        subject = "/".join(rel.parts[:-1]) or "rldd"
        return subject, {"0": "alert", "5": "skip", "10": "drowsy"}[stem]

    gender = "female" if any("female" in p for p in parts) else "male" if any("male" in p for p in parts) else "x"
    m = re.match(r"(\d+)", rel.stem)
    subject = f"{gender}{m.group(1)}" if m else rel.stem
    if "dash" in parts:
        return subject, "skip"
    yawn, talk = "yawn" in stem, ("talk" in stem or "sing" in stem)
    if yawn and talk:
        return subject, "skip"                                     # mixed inside one video
    if yawn:
        return subject, "yawning"
    if talk or "normal" in stem:
        return subject, "alert"
    return subject, "skip"


def cmd_labels(root: Path, out: Path) -> None:
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"No videos found under {root}")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("video", "subject", "label"))
        for v in videos:
            rel = v.relative_to(root)
            subject, label = guess_label(rel)
            counts[label] = counts.get(label, 0) + 1
            w.writerow((rel.as_posix(), subject, label))
    print(f"Wrote {out}: {len(videos)} videos, {counts}. Open it and fix any wrong subject or label before step 2.")


def extract_video(video: Path, out_csv: Path, subject: str, label: str, stride: int = 2,
                  tracker_factory: Callable = FaceTracker, model: Path = DEFAULT_MODEL) -> tuple[int, float]:
    """Write one CSV of per-frame signals. Returns (frames written, fraction with a face)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = faces = 0
    tmp = out_csv.with_suffix(".part")
    try:
        with tracker_factory(model) as tracker, tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FRAME_HEADER)
            n = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if n % stride:
                    continue
                t = n / fps
                face = tracker.process(frame, int(t * 1000))
                if face is None:
                    w.writerow((video.name, subject, label, f"{t:.3f}", 0, *(["nan"] * 7)))
                else:
                    faces += 1
                    w.writerow((video.name, subject, label, f"{t:.3f}", 1, f"{face.ear:.4f}", f"{face.mar:.4f}",
                                f"{face.jaw_open:.4f}", f"{face.blink:.4f}", f"{face.pitch:.2f}", f"{face.yaw:.2f}",
                                f"{face.gaze:.4f}"))
                rows += 1
    finally:
        cap.release()
    tmp.replace(out_csv)                  # only a finished video leaves a .csv, so a stopped run resumes cleanly
    return rows, (faces / rows if rows else math.nan)


def cmd_frames(root: Path, labels: Path, out_dir: Path, stride: int, limit: int | None) -> None:
    with labels.open(newline="", encoding="utf-8") as fh:
        entries = [r for r in csv.DictReader(fh)]
    bad = [r for r in entries if r["label"] not in LABELS]
    if bad:
        raise SystemExit(f"Unknown label {bad[0]['label']!r} for {bad[0]['video']}. Use one of {sorted(LABELS)}.")
    todo = [r for r in entries if r["label"] != "skip"]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} videos to process ({len(entries) - len(todo)} skipped), every {stride}. frame")
    for i, r in enumerate(todo, 1):
        out_csv = out_dir / (Path(r["video"]).with_suffix("").as_posix().replace("/", "__") + ".csv")
        if out_csv.exists():
            print(f"[{i}/{len(todo)}] {r['video']}: already done")
            continue
        try:
            rows, face_frac = extract_video(root / r["video"], out_csv, r["subject"], r["label"], stride)
        except OSError as exc:
            print(f"[{i}/{len(todo)}] {r['video']}: {exc}", file=sys.stderr)
            continue
        print(f"[{i}/{len(todo)}] {r['video']}: {rows} frames, face in {face_frac:.0%}, label {r['label']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="driver_monitor.extract", description="Dataset videos -> per-frame feature CSVs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("labels", help="step 1: guess a labels.csv from file names")
    a.add_argument("--root", type=Path, required=True, help="dataset folder")
    a.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    b = sub.add_parser("frames", help="step 2: extract features from every labelled video")
    b.add_argument("--root", type=Path, required=True, help="dataset folder (same as step 1)")
    b.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    b.add_argument("--out", type=Path, default=Path("data/frames"))
    b.add_argument("--stride", type=int, default=2, help="analyse every Nth frame (2 = twice as fast, plenty for 30 fps video)")
    b.add_argument("--limit", type=int, help="only the first N videos, to try the pipeline quickly")
    args = p.parse_args(argv)
    if args.cmd == "labels":
        cmd_labels(args.root, args.out)
    else:
        cmd_frames(args.root, args.labels, args.out, args.stride, args.limit)


if __name__ == "__main__":
    main()
