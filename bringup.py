"""Bring-up on hardware: save a frame from each camera, close and open the hand,
jog each NERO joint 5 degrees (away from its nearer limit) and back. Prints what it reads as it goes. Takes
the embodiment's -E settings as k=v:

    uv run bringup.py [can=nerocan@192.168.123.50] [can_interface=socketcand] [hand_ip=192.168.11.210] [box_url=http://192.168.123.50:8080]
"""

import math
import sys
import time

from PIL import Image
from inspect_robots_nero import cameras
from inspect_robots_nero.arm import LIMITS, Arm
from inspect_robots_nero.embodiment import CAMERAS, NeroEmbodiment
from inspect_robots_nero.hand import Hand

rig = NeroEmbodiment(**dict(a.split("=", 1) for a in sys.argv[1:]))

for c in CAMERAS:
    Image.fromarray(cameras.frame(rig.box_url, c)).save(f"{c}.jpg")
    print(f"saved {c}.jpg")

hand = Hand(rig.hand_ip)
for pose in ([0.2] * 5 + [1.0], [1.0] * 5 + [0.0]):
    hand.move(pose)
    time.sleep(1.5)
    print("hand", hand.angles(), "force", hand.forces())

arm = Arm(rig.can, 20, rig.can_interface)
start = arm.joints()
print("arm deg", [round(math.degrees(q), 1) for q in start], "flange", arm.flange_pose())
for i in range(7):
    moved = []
    step = math.radians(5) if start[i] + math.radians(5) <= LIMITS[i][1] else -math.radians(5)  # away from its limit
    for d in (step, 0):
        arm.move([q + d * (j == i) for j, q in enumerate(start)])
        time.sleep(2)
        moved.append(round(math.degrees(arm.joints()[i]), 1))
    print(f"j{i + 1} deg: start {round(math.degrees(start[i]), 1)}, {math.degrees(step):+.0f} -> {moved[0]}, back -> {moved[1]}")
arm.close()
hand.close()
