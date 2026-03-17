from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from controller import Camera, Keyboard, Motor, Robot

from utils.kinematics import calculate_diff_drive_velocities
from utils.robot_func import (
    get_rgb_cam_frame,
    load_webots_robot_class,
    rotate,
    set_position,
    set_velocity,
    stop_robot,
)

LINEAR_SPEED = 0.1  # m/s
ANGULAR_SPEED = 2.0  # rad/s


def run_robot() -> None:
    RobotClass = load_webots_robot_class()
    robot: Robot = RobotClass()
    timestep = int(robot.getBasicTimeStep())

    camera = cast("Camera", robot.getDevice("camera"))
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()

    left_motor = cast("Motor", robot.getDevice("left wheel motor"))
    right_motor = cast("Motor", robot.getDevice("right wheel motor"))
    wheels: list[Motor] = [left_motor, right_motor]

    set_position(wheels, [float("inf")] * 2)
    stop_robot(wheels)

    keyboard = cast("Keyboard", robot.getKeyboard())
    keyboard.enable(timestep)

    print("Robot ready. Use WS to move, AD to rotate. Press 'X' in Webots to quit.")

    while robot.step(timestep) != -1:
        vx, w = 0.0, 0.0

        key = keyboard.getKey()
        while key != -1:
            char = chr(key) if 0 < key < 128 else ""
            if char in ("W", "w"):
                vx += LINEAR_SPEED
            elif char in ("S", "s"):
                vx -= LINEAR_SPEED
            elif char in ("A", "a"):
                w += ANGULAR_SPEED
            elif char in ("D", "d"):
                w -= ANGULAR_SPEED
            elif char in ("X", "x"):
                stop_robot(wheels)
                cv2.destroyAllWindows()
                return
            key = keyboard.getKey()

        velocities = calculate_diff_drive_velocities(vx, w)
        set_velocity(wheels, velocities)

        image = get_rgb_cam_frame(camera)
        if image is not None:
            cv2.imshow("Robot Vision", image)
            cv2.waitKey(1)

    stop_robot(wheels)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot()
