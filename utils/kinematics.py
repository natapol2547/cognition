from __future__ import annotations

import math

WHEEL_RADIUS = 0.0205  # e-puck wheel radius [m]
AXLE_LENGTH = 0.052    # distance between left and right wheels [m]

# If the map rotates the wrong way when you turn the robot, set this to -1.
ROTATION_SIGN = 0.75


def calculate_diff_drive_velocities(vx: float, w: float) -> list[float]:
    """Return linear wheel velocities [left, right] for desired body motion.

    Compatible with set_velocity which converts to angular by dividing by WHEEL_RADIUS.
    """
    half_axle = AXLE_LENGTH / 2.0
    return [
        vx - w * half_axle,
        vx + w * half_axle,
    ]


class DiffDriveOdometry:
    """Dead-reckoning pose estimator for a differential-drive robot."""

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._prev_left: float | None = None
        self._prev_right: float | None = None

    def update(self, left_pos: float, right_pos: float) -> None:
        """Update the pose estimate from current wheel encoder positions (radians)."""
        if self._prev_left is None or self._prev_right is None:
            self._prev_left = left_pos
            self._prev_right = right_pos
            return

        dl = (left_pos - self._prev_left) * WHEEL_RADIUS
        dr = (right_pos - self._prev_right) * WHEEL_RADIUS
        self._prev_left = left_pos
        self._prev_right = right_pos

        dc = (dl + dr) / 2.0
        dtheta = ROTATION_SIGN * (dr - dl) / AXLE_LENGTH

        # Use midpoint angle for position update to reduce error when rotating.
        theta_mid = self.theta + 0.5 * dtheta
        self.theta += dtheta
        self.x += dc * math.cos(theta_mid)
        self.y += dc * math.sin(theta_mid)

    def get_pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta
