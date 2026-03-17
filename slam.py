from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2

if TYPE_CHECKING:
    from controller import Keyboard, Lidar, Motor, PositionSensor, Robot

from utils.kinematics import DiffDriveOdometry, calculate_diff_drive_velocities
from utils.map import OccupancyGrid
from utils.robot_func import (
    load_webots_robot_class,
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

    left_motor = cast("Motor", robot.getDevice("left wheel motor"))
    right_motor = cast("Motor", robot.getDevice("right wheel motor"))
    wheels: list[Motor] = [left_motor, right_motor]
    set_position(wheels, [float("inf")] * 2)
    stop_robot(wheels)

    left_encoder = cast("PositionSensor", robot.getDevice("left wheel sensor"))
    right_encoder = cast("PositionSensor", robot.getDevice("right wheel sensor"))
    left_encoder.enable(timestep)
    right_encoder.enable(timestep)

    lidar = cast("Lidar", robot.getDevice("lidar"))
    lidar.enable(timestep)
    lidar_fov = lidar.getFov()
    lidar_max_range = lidar.getMaxRange()

    keyboard = cast("Keyboard", robot.getKeyboard())
    keyboard.enable(timestep)

    odometry = DiffDriveOdometry()
    grid = OccupancyGrid(size=400, resolution=0.03)

    print("SLAM running. Use WS to move, AD to rotate. Press 'X' to quit.")

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

        odometry.update(left_encoder.getValue(), right_encoder.getValue())
        pose = odometry.get_pose()

        ranges = lidar.getRangeImage()
        if ranges:
            grid.update(pose, list(ranges), lidar_fov, lidar_max_range)

        map_img = grid.render()
        cv2.imshow("SLAM Map", map_img)
        cv2.waitKey(1)

    stop_robot(wheels)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot()
