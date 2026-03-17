from __future__ import annotations
from typing import TYPE_CHECKING

import os
import sys
import importlib
from dotenv import load_dotenv
import numpy as np

if TYPE_CHECKING:
    from controller import Motor, Robot, Camera, RangeFinder

from utils.kinematics import AXLE_LENGTH, WHEEL_RADIUS

MAX_SPEED = 6.28  # max motor angular velocity [rad/s]


def load_webots_robot_class() -> type[Robot]:
    load_dotenv()
    """Resolve the WEBOTS_HOME path, add it to sys.path, and return the Robot class."""
    webots_home = os.getenv("WEBOTS_HOME")
    if not webots_home:
        raise ValueError("WEBOTS_HOME not found in .env file!")

    controller_path = os.path.join(webots_home, "lib", "controller", "python")
    if controller_path not in sys.path:
        sys.path.append(controller_path)

    controller = importlib.import_module("controller")
    return controller.Robot

def fps_to_sampling_rate(fps: float) -> int:
    if fps <= 0:
        raise ValueError("FPS must be greater than 0")
    return int(1000 / fps)

def set_velocity(wheels: list[Motor], velocity: list[float]) -> None:
    if len(wheels) != len(velocity):
        raise ValueError("Length of wheels and velocity must be the same")
    
    angular_velocity = [v / WHEEL_RADIUS for v in velocity]
    for wheel, vel in zip(wheels, angular_velocity):
        wheel.setVelocity(vel)

def set_position(wheels: list[Motor], position: list[float]) -> None:
    if len(wheels) != len(position):
        raise ValueError("Length of wheels and position must be the same")

    for wheel, pos in zip(wheels, position):
        wheel.setPosition(pos)

def stop_robot(wheels: list[Motor]) -> None:
    set_velocity(wheels, [0.0] * len(wheels))

def moveL(wheels: list[Motor], velocity: float) -> None:
    set_velocity(wheels, [velocity] * len(wheels))

def rotate(wheels: list[Motor], rps: float) -> None:
    half_axle = AXLE_LENGTH / 2.0
    left_vel = -rps * half_axle / WHEEL_RADIUS
    right_vel = rps * half_axle / WHEEL_RADIUS
    left_vel = np.clip(left_vel, -MAX_SPEED, MAX_SPEED)
    right_vel = np.clip(right_vel, -MAX_SPEED, MAX_SPEED)
    wheels[0].setVelocity(float(left_vel))
    wheels[1].setVelocity(float(right_vel))


def get_rgb_cam_frame(camera: Camera) -> np.ndarray | None:
    """Retrieve an RGB frame from a Webots Camera device."""
    raw_image = camera.getImage()
    if raw_image:
        width = camera.getWidth()
        height = camera.getHeight()
        image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        # Drop the Alpha channel
        return image_array[:, :, :3]
    return None

def get_depth_cam_frame(camera: RangeFinder) -> np.ndarray | None:
    """Retrieve a depth frame from a Webots RangeFinder device."""
    raw_image = camera.getRangeImage(data_type="buffer")
    if raw_image:
        width = camera.getWidth()
        height = camera.getHeight()
        image_array = np.frombuffer(raw_image, np.float32).reshape((height, width))  # type: ignore
        return image_array
    return None