# Markov safety hackathon arm setup

An [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment for this rig, so any LLM or VLA can drive it. The rig has:
- an AgileX NERO 7-DOF arm,
- an Inspire RH56DFTP-2R right hand on its flange,
- two ZED X Nano stereo cameras, one on the wrist and one on a tripod,
- a ZED Box that serves the cameras and the arm's CAN bus over Ethernet.

The laptop (macOS or Linux) runs everything else. It needs no CAN adapter or driver.

## From a fresh laptop to a moving arm

You need [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`) and git. Do steps 1-3 in order.

1. **Plug in.** Power the arm, the hand and the ZED Box. The box, the hand and the arm controller each take one Ethernet cable to the laptop: a USB Ethernet adapter per cable, or one switch. Any cable can go in any port.
2. **Clone and install.**
   ```bash
   git clone https://github.com/markovrobotics/markov-safety-hackathon-arm-setup && cd markov-safety-hackathon-arm-setup
   uv sync
   ```
   This installs Python, this package, Inspect Robots and its agent policy into `.venv/`.
3. **Set up the laptop's network.**
   - Mac: `sudo laptop/net-macos.sh`. It bridges every USB Ethernet adapter into `bridge1`.
   - Linux: `sudo laptop/net-linux.sh <port> <port> [<port>]`, naming the rig ports (`ip -br link` lists them). It makes the bridge `br-rig`.

   Run it again after you plug in another adapter. Both scripts give the laptop the addresses below and NAT the box out to the internet for installs.

Then check the hardware, and run a task:

```bash
uv run bringup.py     # MOVES THE ARM: clear the space around it first
uv run inspect-robots run --instruction "pick up the cup" --embodiment nero --policy agent -P model=anthropic/claude-fable-5
```

`bringup.py` saves `wrist.jpg` and `tripod.jpg`, then closes and opens the hand. Then it jogs each arm joint 5° and back, away from the nearer limit, printing each joint as it goes. If it gets through all seven joints, the rig works end to end.

The agent policy reads `ANTHROPIC_API_KEY` from the environment or from a `.env` file in the directory you run from (`OPENROUTER_API_KEY` for other models). To run it on a logged-in Claude Code or Codex subscription instead of a key:
1. Start `uv run llm_bridge.py` in another terminal.
2. Pass `-P model=claude/opus -P base_url=http://127.0.0.1:8765/v1`, or `model=codex/` for the Codex login's default model.

Each turn is then one `claude -p` or `codex exec` call. In our runs a `codex` turn took about 11 s, and the first one over a minute. `claude` must be logged in in the session that runs the bridge: over ssh to a Mac it can't reach the login keychain and answers "Not logged in", while `codex` works there.

A run stops at whichever comes first: `--max-steps` (default 300 steps, which in our runs was about 18 model calls, since each call's move runs about 17 steps) or the agent's model-call budget, `-P max_llm_calls` (default 100). For a long run, raise both, e.g. `--max-steps 30000 -P max_llm_calls=800`.

To skip `--embodiment` and `--policy`, run `uv run inspect-robots setup` once and make them the defaults.

## Addresses

| Device | Address | Notes |
|---|---|---|
| Laptop | 192.168.123.10/24 and 192.168.11.10/24 | set by the `laptop/` scripts |
| ZED Box | 192.168.123.50 | cameras on port 8080, CAN (socketcand) on port 29536 |
| Hand | 192.168.11.210 | its factory address; Modbus TCP, port 6000 |
| Arm controller | IPv6 link-local only (`fe80::6ce8:14ff:fe96:c66`) | ssh as `agx`; used only by the arm reset, see below |

Pass other addresses to `inspect-robots` with `-E`, and to `bringup.py` as bare `key=value` (e.g. `uv run bringup.py hand_ip=10.0.0.5`):
- `-E box_url=http://<box>:8080`
- `-E hand_ip=<ip>`
- `-E can=<interface on the box>@<box>[:port]`
- `-E controller=agx@<address>`, or `-E controller=` to turn off the controller reset

Other settings: `-E speed=50` (percent of the arm's top speed) and `-E control_hz=20`.

## Embodiment

Two ways to command the arm, picked with `-E control=`:

- **`pose` (the default).** The action is 12 numbers: the palm's `x, y, z` in metres and `roll, pitch, yaw` in radians, then the 6 hand DOFs (pinky, ring, middle, index, thumb bend, thumb rotation; 0 closed, 1 open). The agent gets a `move_to` tool over these.
  - The palm point is the middle of the palm, on the hand's axis, 10.5 cm out from the arm's flange face (the printed mount plus the hand to mid-palm).
  - `x, y, z` are in the arm's base frame. The base stands on the table, and the table top is about z = 0.03.
  - `roll, pitch, yaw` turn the palm about the base frame's x, y and z axes, about the palm point. They count from the palm's orientation when the trial started, so `0, 0, 0` is how it started.
  - The palm stays inside a box: x and y within ±0.7 m, z from 0.09 to 0.8 m. Pitch stays within ±1.3 rad and roll and yaw within ±1.6 rad.
  - No pose may bring the hand within 1 cm of the table, counted with the fingers open, from the flange face to the fingertips. With the fingers pointing down, the palm stays about 17 cm up.
  - The arm's joints come from inverse kinematics on AgileX's URDF (`kinematics.py`). It matches the arm's own flange readings to 0.07 mm and 0.01°. If a motion reaches a pose the arm cannot reach (too far, past a joint limit, too close to the table, or needing the arm to flip to another configuration), the arm stops at the last reachable point. The model is told why and how close it got.
- **`joint`.** The action is 13 numbers: the 7 arm joint targets (rad, base to wrist), then the 6 hand DOFs. The agent gets a `move_joints` tool.

The observation is the same either way:
- `joint_pos`: the 7 arm joints and the 6 hand DOFs.
- `eef_pose`: the palm pose as above, then the 6 hand DOFs.
- `hand_force`: per-DOF grip force, in the hand's raw units.
- Both eyes and the depth of both cameras, 640x360: `wrist` and `tripod` (left eyes), `wrist_right` and `tripod_right`, and `wrist_depth` and `tripod_depth`: the depth the left eye sees, computed by the ZED SDK from the two eyes on the box's GPU, grey from white at 0.1 m to black at 2.0 m and beyond (black also where there is no depth). The agent sees all six on every call.

Steps run at 20 Hz. Joint targets go to the arm's firmware position mode (`move_j`), which smooths the motion and clamps each joint to its limits.

### Without the rig

`-E sim=true` runs the same embodiment on a MuJoCo stand-in (`sim.py`). It has the NERO's kinematics drawn as capsules, a box-and-finger hand, the table and a few objects, and both cameras are rendered. Install it with `uv sync --extra sim`:

```bash
uv run inspect-robots run --instruction "touch the orange bottle" --embodiment nero -E sim=true --policy agent -P model=anthropic/claude-fable-5
```

In the sim, joints move straight to their targets at 1 rad/s. Nothing collides, stalls or faults.

## When something goes wrong

**A joint is blocked.** A joint can stop more than 0.05 rad short of its target and not move for 0.5 s, because something is in its way or it is at a limit. The arm then:
1. backs that joint off 0.05 rad and holds there;
2. drops the rest of the current motion;
3. tells the model which joint stopped and where, and which way to move it (in `pose` control, to move the palm back the way it came).

The joint is not driven further in the blocked direction until the model commands it the other way. Without this, the motors push at their current cap until a joint driver latches a fault and the arm loses power.

**A joint driver reports a warning** (over-current, over-temperature, and the like). The arm holds where it is and the model is told what was reported.

**The arm dies.** This covers an emergency stop, a joint that loses power, a latched driver error or collision status, or no CAN feedback for 1 s. On the next step the embodiment resets the arm where it stands:
1. Over CAN: normal mode, then enable.
2. If that does not bring it back: the controller's own reset over ssh (clear errors, enable all joints, hand control back to CAN), then enable again.

When the reset works, the model is told the arm died, what the reset did, and which joints moved while it was unpowered (a joint can sag under the hand's weight). It then plans from where the arm is now. No stall is called for 3 s after a reset, since a re-enabled joint can take 2 s to start moving.

When neither reset brings it back, the trial ends with an `EmbodimentFault` naming what failed, and the eval stops. Power-cycle the arm and run again.

The controller reset needs the laptop's ssh key in `agx`'s `~/.ssh/authorized_keys` on the arm controller. The controller has no published password, so a new laptop's public key has to be added from a machine that can already log in. Without the key, only the CAN reset runs, and everything else works as above.

**A cable comes out.**
- **Arm CAN cable, the CAN stick, or the box's Ethernet:** the arm sends no feedback for 1 s and counts as dead (see above). The CAN connection is not reopened in the middle of a run, so the reset fails and the trial ends with an `EmbodimentFault`. Plug it back in and run the command again.
- **Hand cable:** the next hand read fails ("Inspire hand at ... did not answer a read") and the trial ends with an `EmbodimentFault`. Plug it back in and run again.
- **Box Ethernet, cameras:** the next frame fetch times out after 2 s and the trial ends the same way.
- **Arm controller's Ethernet:** nothing changes until the arm dies; then only the CAN reset is available.

If a replugged USB Ethernet adapter does not come back, rerun the `laptop/` script for your OS.

## Setting up the ZED Box (once)

The ZED Box in the kit is already set up. To set up another:

**Cameras.** The ZED X Nano needs ZED Box GMSL driver 1.4.2 or newer and ZED SDK 5.3 or newer. With older versions the box sees no camera (`dmesg`: "Error -1 setting gmsl link") or reports it as Unknown. Get both from stereolabs.com/developers; for the ZED Box Mini on JetPack 6.1 that is `stereolabs-zedbox-mini_1.4.4-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0_arm64.deb`, then reboot. Then copy `box/` over and run `./install.sh`:
- With no arguments it lists the camera serial numbers.
- `./install.sh wrist=<serial> tripod=<serial>` installs `camera_server.py` as a systemd service that starts on boot. It serves each camera's left image at `http://<box>:8080/<name>.jpg` the right at `<name>_right.jpg` and the depth at `<name>_depth.jpg` (the embodiment reads all three), and every eye live in a grid at `http://<box>:8080/`. A frame older than 1 s is never served: the request gets a 503. When a camera gives no frame for 3 s (a GMSL link drop), the server exits, and the service restarts the ZED camera daemons and then the server, which brings the camera back.

**CAN.** Plug the arm's CAN cable into the candleLight USB-CAN stick (USB id 1d50:606f), and the stick into the box. Then run `box/install-can.sh` on the box. JetPack's kernel has no driver for the stick, so the script first builds the kernel's own `gs_usb` driver from the matching stable kernel source (it needs the box online, which the `laptop/` scripts give it). It brings the stick up as `nerocan` at 1 Mbit whenever it is plugged in, and serves it on port 29536 with [socketcand](https://github.com/linux-can/socketcand). The laptop opens that with python-can's `socketcand` interface.

To plug the stick into the laptop instead:
- Linux: `sudo ip link set can0 up type can bitrate 1000000`, then `-E can_interface=socketcan -E can=can0`.
- Mac: `-E can_interface=gs_usb -E can=0`. macOS has no CAN driver, so python-can drives the stick over USB itself. Use a USB data port: a charge-only USB-C port does not enumerate it.

## Files

- `src/inspect_robots_nero/embodiment.py`: the `nero` embodiment. It also runs the stall stop, the fault hold and the arm reset, on a watcher thread that runs while the model thinks too.
- `src/inspect_robots_nero/arm.py`: the NERO through AgileX's [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm): joint targets, feedback, driver status and the reset.
- `src/inspect_robots_nero/controller_reset.py`: the controller-side reset, sent to the controller over ssh and run there.
- `src/inspect_robots_nero/hand.py`: the Inspire hand over Modbus TCP, through [`inspire-rh56dftp`](https://pypi.org/project/inspire-rh56dftp/).
- `src/inspect_robots_nero/cameras.py`: the latest frame from each camera on the box.
- `bringup.py`: the first-contact hardware check above.
- `jog.py`: jogs joints one at a time from stdin and prints the arm's state, limits and health as JSON lines. Its header has the commands.
- `llm_bridge.py`: an OpenAI-style endpoint on localhost that answers through a logged-in `claude` or `codex` CLI.
- `box/`: the camera server and the CAN setup for the ZED Box.
- `laptop/`: the network setup for macOS and Linux.
