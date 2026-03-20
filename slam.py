from __future__ import annotations

from typing import TYPE_CHECKING
import cv2

from mapping.kinematics import DiffDriveOdometry, calculate_diff_drive_velocities
from mapping.grid import OccupancyGrid

from devices.motor import MotorActuator
from devices.encoder import EncoderSensor
from devices.lidar import LidarSensor
from devices.compass import CompassSensor

from utils.keyboard import WebotsKeyboard
from utils.robot import get_webots_robot

LINEAR_SPEED = 0.1  # m/s
ANGULAR_SPEED = 2.0  # rad/s

def stop_robot(wheels: list[MotorActuator]) -> None:
    for wheel in wheels:
        wheel.stop()

def set_velocity(wheels: list[MotorActuator], velocity: list[float]) -> None:
    for wheel, vel in zip(wheels, velocity):
        wheel.setVelocity(vel)

def run_robot() -> None:
    robot = get_webots_robot()
    timestep = int(robot.getBasicTimeStep())

    # Motors and encoders
    left_motor = MotorActuator(robot, "left wheel motor")
    right_motor = MotorActuator(robot, "right wheel motor")
    wheels: list[MotorActuator] = [left_motor, right_motor]

    left_encoder = EncoderSensor(robot, "left wheel sensor")
    right_encoder = EncoderSensor(robot, "right wheel sensor")
    
    # Compass
    compass = CompassSensor(robot)
    
    # Lidar
    lidar = LidarSensor(robot)
    lidar_fov = lidar.getFov()
    lidar_max_range = lidar.getMaxRange()

    # Keyboard
    keyboard = WebotsKeyboard(robot)

    # Odometry and grid
    odometry = DiffDriveOdometry(left_encoder, right_encoder, compass)
    grid = OccupancyGrid(world_min=(-4.0, -4.0), world_max=(2.0, 2.0), resolution=0.01)

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

        odometry.update()
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
