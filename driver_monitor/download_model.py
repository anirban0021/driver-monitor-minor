"""Fetch the models into ./models: MediaPipe face, phone and hand models, and OpenCV's SFace for driver recognition."""

from __future__ import annotations

import urllib.request

from .detector import DEFAULT_MODEL, MODEL_URL
from .embedder import EMBED_MODEL, EMBED_URL
from .phone import HAND_MODEL, HAND_MODEL_URL, OBJECT_MODEL, OBJECT_MODEL_URL


def main() -> None:
    for path, url in ((DEFAULT_MODEL, MODEL_URL), (OBJECT_MODEL, OBJECT_MODEL_URL), (HAND_MODEL, HAND_MODEL_URL),
                      (EMBED_MODEL, EMBED_URL)):
        if path.exists():
            print(f"{path.name} already present")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
        if path.stat().st_size < 100_000:      # an error page or a Git LFS pointer instead of the model
            path.unlink()
            raise SystemExit(f"Download of {path.name} failed (got a tiny file). Check the URL or your connection.")
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
