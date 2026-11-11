"""Controller-side reset for the NERO, run on the arm controller itself (Arm.recover sends it there over ssh as agx):
log in to the controller's web API, Ethernet control mode, clean its errors, enable all joints, wait for External CAN
Push, then hand control back to CAN. Prints one JSON line."""
import json, time, urllib.parse, urllib.request

BASE = "http://127.0.0.1"


def call(path, token=None, body=None, **q):
    url = BASE + path + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=3).read())


steps = []
try:
    token = call("/api/login", body={"username": "admin", "password": "123456"})["data"]["accessToken"]
    for path, q in (("/api/setting/setCtrlMode", {"ctrlMode": "3"}), ("/api/cleanError", {"action": "1"}),
                    ("/api/set/enable", {"jointNum": "8", "action": "1"})):
        r = call(path, token, **q)
        steps.append(f"{path.rsplit('/', 1)[1]} {'ok' if r.get('success') else r}")
    deadline = time.monotonic() + 5
    while True:
        d = call("/api/setting/getExternalPush", token).get("data")
        if d is True or d in (1, "1", "true") or (isinstance(d, dict) and any(d.get(k) in (True, 1) for k in ("enable", "enabled", "externalPush"))):
            steps.append("external push on"); break
        if time.monotonic() > deadline:
            steps.append(f"external push not confirmed: {d}"); break
        time.sleep(0.1)
    call("/api/setting/setCtrlMode", token, ctrlMode="1")
    deadline = time.monotonic() + 5
    while (mode := call("/api/setting/getMode", token).get("data", {}).get("ctrlMode")) != 1 and time.monotonic() < deadline:
        time.sleep(0.1)
    steps.append(f"ctrlMode {mode}")
    print(json.dumps({"ok": mode == 1, "steps": steps}))
except Exception as e:
    print(json.dumps({"ok": False, "steps": steps, "error": f"{type(e).__name__}: {e}"}))
