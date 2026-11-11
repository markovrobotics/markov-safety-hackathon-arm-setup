"""The Inspire RH56DFTP hand over Modbus TCP, through inspire-rh56dftp.

Six DOFs in the driver's order: pinky, ring, middle, index, thumb_bend,
thumb_rotation. Values here are the hand's raw 0..1000 scaled to 0..1:
1 is a finger open (extended), 0 closed.
"""

import numpy as np
from inspire_rh56dftp import InspireHand, registers

DOFS = tuple(registers.DOF_ORDER)


class Hand:
    def __init__(self, ip: str = "192.168.11.210", speed: int = 1000):
        # No send throttling: every step's target goes out. speed is the hand's own motor speed, 0..1000.
        self.hand = InspireHand(ip=ip, default_speed=speed, min_send_interval_s=0, min_change=0)

    def angles(self) -> list[float]:
        return [v / 1000 for v in self._read(registers.ANGLE_ACT)]

    def forces(self) -> list[float]:
        """Per-DOF grip force, raw units (signed)."""
        return np.array(self._read(registers.FORCE_ACT), dtype=np.uint16).view(np.int16).astype(float).tolist()

    def move(self, angles: list[float]) -> None:
        self.hand.send_raw([round(a * 1000) for a in angles])

    def close(self) -> None:
        self.hand.close()

    def _read(self, address: int) -> list[int]:
        values = self.hand.read_holding(address, 6)
        if values is None:
            raise RuntimeError(f"Inspire hand at {self.hand.ip} did not answer a read")
        return values
