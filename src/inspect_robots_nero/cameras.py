"""Latest frames from the ZED Box camera server (box/camera_server.py)."""

import io
import urllib.request

import numpy as np
from PIL import Image


def frame(box_url: str, name: str) -> np.ndarray:
    """The latest frame of one eye (name: "wrist" is its left eye, "wrist_right" its right), HxWx3 uint8 RGB."""
    with urllib.request.urlopen(f"{box_url}/{name}.jpg", timeout=2) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
