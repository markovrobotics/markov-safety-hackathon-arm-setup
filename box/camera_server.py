"""ZED Box camera server: opens the ZED X Nanos and serves each one's latest frames, 640x360 JPEG:
  http://<box>:8080/<name>.jpg          left eye (what the embodiment reads)
  http://<box>:8080/<name>_right.jpg    right eye
  ...jpg?next                           the next new frame (waits up to 1 s for it)
  http://<box>:8080/                    every eye of every camera live in a grid, a row per camera

    python3 camera_server.py --list                      # serial numbers found
    python3 camera_server.py wrist=<serial> tripod=<serial>
"""

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import pyzed.sl as sl

latest: dict[str, bytes] = {}
fresh = threading.Condition()
names: list[str] = []


def grab(name: str, serial: int) -> None:
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.AUTO, camera_fps=30, depth_mode=sl.DEPTH_MODE.NONE)
    init.set_from_serial_number(serial)
    cam = sl.Camera()
    # an open can fail while nvargus-daemon is still settling (a service restart): keep trying, or that camera is gone until the next restart
    while (err := cam.open(init)) != sl.ERROR_CODE.SUCCESS:
        print(f"{name} ({serial}): {err}; retrying in 2 s", file=sys.stderr, flush=True)
        time.sleep(2)
    mat = sl.Mat()
    while True:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            for view, key in ((sl.VIEW.LEFT, name), (sl.VIEW.RIGHT, f"{name}_right")):
                cam.retrieve_image(mat, view, sl.MEM.CPU, sl.Resolution(640, 360))
                latest[key] = cv2.imencode(".jpg", mat.get_data()[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            with fresh:
                fresh.notify_all()


# The grid fetches each eye's latest frame one request at a time, each given 2 s, rather than holding one MJPEG
# stream open per eye: a stream whose server went away without a reset (the box rebooted on 11-10) sends nothing more
# and raises no error, so its tile kept its last frame until the page was reloaded. A request that fails or times out
# is just retried, so every tile comes back by itself once the server answers again, and one with no new frame for
# 2 s is dimmed.
GRID = """<!doctype html><title>cameras</title><style>
html,body{margin:0;height:100%;background:#000}
body{display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:1fr;gap:2px}
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
            page = GRID + "".join(f'<img data-src="/{n}{eye}.jpg">' for n in names for eye in ("", "_right"))
            self._send("text/html", page.encode())
            return
        path, _, query = path.partition("?")
        if "next" in query.split("&"):  # the grid's: answer with the next new frame, so it fetches at the camera's rate
            with fresh:
                fresh.wait(1.0)
        body = latest.get(path.removesuffix(".jpg"))
        if body is None:
            self.send_error(404, f"cameras: {sorted(latest)}")
            return
        self._send("image/jpeg", body)

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
        threading.Thread(target=grab, args=(name, int(serial)), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
