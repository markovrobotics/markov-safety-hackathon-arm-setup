"""Latest frames from the ZED Box camera server (box/camera_server.py)."""

import io
import urllib.request

import numpy as np
from PIL import Image

DEPTH_NEAR, DEPTH_FAR = 0.1, 2.0  # metres: a depth image is white at DEPTH_NEAR, black at DEPTH_FAR (box/camera_server.py)


def frame(box_url: str, name: str) -> np.ndarray:
    """The latest frame of one eye (name: "wrist" is its left eye, "wrist_right" its right, "wrist_depth" the depth seen
    from its left eye), HxWx3 uint8 RGB."""
    with urllib.request.urlopen(f"{box_url}/{name}.jpg", timeout=2) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
