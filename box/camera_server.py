"""ZED Box camera server: opens the ZED X Nanos and serves each one's latest frames, 640x360 JPEG:
  http://<box>:8080/<name>.jpg          left eye (what the embodiment reads)
  http://<box>:8080/<name>_right.jpg    right eye
  http://<box>:8080/<name>_depth.jpg    depth seen from the left eye, grey: white at DEPTH_NEAR, black at DEPTH_FAR and beyond
                                        (or where the SDK has no depth), linear between
  ...jpg?next                           the next new frame (waits up to 1 s for it)
  http://<box>:8080/                    every eye and depth of every camera live in a grid, a row per camera

    python3 camera_server.py --list                      # serial numbers found
    python3 camera_server.py wrist=<serial> tripod=<serial>
"""

import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyzed.sl as sl

latest: dict[str, tuple[float, bytes]] = {}  # eye -> (monotonic time grabbed, JPEG)
fresh = threading.Condition()
names: list[str] = []
# A frame is served only while it is current. On 11-11 at 12:10:55 both cameras' GMSL links dropped; grab() failed on every
# call from then on and the server kept answering with the 12:11 frames, so the grid showed them as live for 25 min.
MAX_AGE = 1.0
# The watchdog in __main__ ends the process when a camera has given no frame for LOST_S once open (or OPEN_S while it
# opens): Argus keeps a dropped camera's dead session ("Device 0 (of 1) is in use"), so no reopen inside this process brings
# it back. It exits 3, on which the unit restarts zed_x_daemon and nvargus-daemon before systemd starts it again.
LOST_S, OPEN_S = 3.0, 60.0
# The ZED X Nano has no depth sensor: the SDK computes depth from the two eyes on this box's GPU, where the full-size frames
# are, rather than on the Air from the 640x360 JPEGs it gets.
# SPEC-AMBIGUITY: (2026-11-11 15:50 Kaya asked for "it could be done if that's helpful, but it should be wherever it's faster")
# which depth model and how to show depth to an agent that reads images could mean
# 1. NEURAL_LIGHT (the SDK's fast learned model) as a linear grey image over a fixed range
# 2. NEURAL / NEURAL_PLUS (sharper, slower) or a colour map
# Choice defaulted to: 1 because it keeps both cameras near their frame rate, and a fixed linear grey reads as distance
DEPTH_MODE = sl.DEPTH_MODE.NEURAL_LIGHT
DEPTH_NEAR, DEPTH_FAR = 0.1, 2.0  # metres, as in src/inspect_robots_nero/cameras.py: the tripod sees the far table edge at about 1.8 m
last_ok: dict[str, float] = {}  # camera -> monotonic time of its last good grab (or of the start, until it opens)
is_open: dict[str, bool] = {}


def grab(name: str, serial: int) -> None:
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.AUTO, camera_fps=30, depth_mode=DEPTH_MODE,
                             coordinate_units=sl.UNIT.METER, depth_minimum_distance=DEPTH_NEAR)
    init.set_from_serial_number(serial)
    cam = sl.Camera()
    # an open can fail while nvargus-daemon is still settling (a service restart): keep trying, or that camera is gone until the next restart
    while (err := cam.open(init)) != sl.ERROR_CODE.SUCCESS:
        print(f"{name} ({serial}): {err}; retrying in 2 s", file=sys.stderr, flush=True)
        time.sleep(2)
    is_open[name] = True
    last_ok[name] = time.monotonic()
    mat, depth = sl.Mat(), sl.Mat()
    while True:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            now = time.monotonic()
            for view, key in ((sl.VIEW.LEFT, name), (sl.VIEW.RIGHT, f"{name}_right")):
                cam.retrieve_image(mat, view, sl.MEM.CPU, sl.Resolution(640, 360))
                latest[key] = (now, cv2.imencode(".jpg", mat.get_data()[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, sl.Resolution(640, 360))
            grey = np.nan_to_num((DEPTH_FAR - depth.get_data()) / (DEPTH_FAR - DEPTH_NEAR) * 255, nan=0, posinf=0, neginf=0)
            latest[f"{name}_depth"] = (now, cv2.imencode(".jpg", grey.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            last_ok[name] = now
            with fresh:
                fresh.notify_all()


# The grid fetches each eye's latest frame one request at a time, each given 2 s, rather than holding one MJPEG
# stream open per eye: a stream whose server went away without a reset (the box rebooted on 11-10) sends nothing more
# and raises no error, so its tile kept its last frame until the page was reloaded. A request that fails or times out
# is just retried, so every tile comes back by itself once the server answers again, and one with no new frame for
# 2 s is dimmed.
GRID = """<!doctype html><title>cameras</title><style>
html,body{margin:0;height:100%;background:#000}
body{display:grid;grid-template-columns:1fr 1fr 1fr;grid-auto-rows:1fr;gap:2px}
img{width:100%;height:100%;object-fit:contain;transition:opacity .3s}img.stale{opacity:.3}</style>
<script>
async function live(img) {
  let shown = performance.now();
  setInterval(() => img.classList.toggle("stale", performance.now() - shown > 2000), 500);
  for (;;) {
    try {
      const r = await fetch(img.dataset.src + "?next&t=" + Date.now(), {signal: AbortSignal.timeout(2000), cache: "no-store"});
      if (!r.ok) throw new Error(r.status);
      const url = URL.createObjectURL(await r.blob());
      await new Promise(done => { img.onload = img.onerror = done; img.src = url; });
      URL.revokeObjectURL(url);
      shown = performance.now();
    } catch (e) {
      await new Promise(go => setTimeout(go, 500));
    }
  }
}
addEventListener("DOMContentLoaded", () => document.querySelectorAll("img").forEach(live));
</script>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.strip("/")
        if path == "":
            page = GRID + "".join(f'<img data-src="/{n}{eye}.jpg">' for n in names for eye in ("", "_right", "_depth"))
            self._send("text/html", page.encode())
            return
        path, _, query = path.partition("?")
        if "next" in query.split("&"):  # the grid's: answer with the next new frame, so it fetches at the camera's rate
            with fresh:
                fresh.wait(1.0)
        frame = latest.get(path.removesuffix(".jpg"))
        if frame is None:
            self.send_error(404, f"cameras: {sorted(latest)}")
            return
        age = time.monotonic() - frame[0]
        if age > MAX_AGE:  # the grid dims a tile that gets no frame; every other reader gets the error, never an old frame
            self.send_error(503, f"no frame for {age:.0f} s")
            return
        self._send("image/jpeg", frame[1])

    def _send(self, kind: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if sys.argv[1:] == ["--list"]:
        for d in sl.Camera.get_device_list():
            print(d.serial_number, d.camera_model, d.camera_state)
        sys.exit()
    for arg in sys.argv[1:]:
        name, serial = arg.split("=")
        names.append(name)
        last_ok[name] = time.monotonic()
        threading.Thread(target=grab, args=(name, int(serial)), daemon=True).start()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever, daemon=True).start()
    while True:
        time.sleep(0.5)
        for name in names:
            quiet = time.monotonic() - last_ok[name]
            if quiet > (LOST_S if is_open.get(name) else OPEN_S):
                print(f"{name}: no frame for {quiet:.0f} s; exiting so systemd resets the camera daemons and starts again",
                      file=sys.stderr, flush=True)
                os._exit(3)
