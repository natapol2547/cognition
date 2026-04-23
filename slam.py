from __future__ import annotations

from typing import TYPE_CHECKING
import cv2
import numpy as np

from mapping.kinematics import DiffDriveOdometry, calculate_diff_drive_velocities
from mapping.graph_slam import render_graph, render_poses, render_lidar_scan
from mapping.graph_slam_session import GraphSlamSession

from devices.motor import MotorActuator
from devices.encoder import EncoderSensor
from devices.lidar import LidarSensor
from devices.compass import CompassSensor

from utils.keyboard import WebotsKeyboard
from utils.robot import get_webots_robot

LINEAR_SPEED = 0.3  # m/s
ANGULAR_SPEED = 2.0  # rad/s

def stop_robot(wheels: list[MotorActuator]) -> None:
    for wheel in wheels:
        wheel.stop()

def set_velocity(wheels: list[MotorActuator], velocity: list[float]) -> None:
    for wheel, vel in zip(wheels, velocity):
        wheel.setVelocity(vel)


def _blank_after_panel(size: int) -> np.ndarray:
    """Placeholder for the 'after optimization' panel shown before loop closure."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.putText(
        img, "After optimization", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    msg = "(awaiting loop closure)"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(
        img, msg, ((size - tw) // 2, size // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA,
    )
    return img

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
    odometry = DiffDriveOdometry(left_encoder, right_encoder, compass) # , compass

    # Graph SLAM
    session = GraphSlamSession(loop_return_radius=0.15)

    print("SLAM running. Use WS to move, AD to rotate. Press 'X' to quit.")
    print("Drive a loop; graph is solved automatically when you return to the start.")

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

        pose_vec = np.array([pose[0], pose[1], pose[2]], dtype=float)
        scan = np.asarray(ranges, dtype=float) if ranges is not None else None
        solved = session.step(pose_vec, scan)
        if solved:
            res = session.optimize_result
            if res is not None and res.chi2_history:
                chi2_0 = res.chi2_history[0]
                print(
                    f"Loop closed: nodes={len(session.graph.nodes)} "
                    f"edges={len(session.graph.edges)} "
                    f"chi2: {chi2_0:.4f} -> {res.final_chi2:.4f} "
                    f"over {res.iterations} iters "
                    f"({res.accepted} acc / {res.rejected} rej, "
                    f"converged={res.converged})"
                )
            else:
                print(
                    f"Loop closed: nodes={len(session.graph.nodes)} "
                    f"edges={len(session.graph.edges)} chi2={session.last_chi2:.4f}"
                )
        if scan is not None and scan.size > 0:
            lidar_img = render_lidar_scan(scan, lidar_fov, max_range=lidar_max_range)
            cv2.imshow("Lidar", lidar_img)

        panel_size = 600
        node_scans = [n.scan for n in session.graph.nodes]

        if (
            session.closed
            and session.pre_optim_poses is not None
            and session.pre_optim_edges is not None
        ):
            post_poses = session.graph.poses()
            left_panel = render_poses(
                session.pre_optim_poses,
                session.pre_optim_edges,
                size=panel_size,
                title="Before optimization",
                loop_radius=session.loop_return_radius,
                extra_poses=post_poses,
                scans=node_scans,
                scan_fov=lidar_fov,
                scan_max_range=lidar_max_range,
            )
            right_panel = render_poses(
                post_poses,
                [(e.i, e.j, e.kind) for e in session.graph.edges],
                size=panel_size,
                title="After optimization",
                loop_radius=session.loop_return_radius,
                extra_poses=session.pre_optim_poses,
                scans=node_scans,
                scan_fov=lidar_fov,
                scan_max_range=lidar_max_range,
            )
        else:
            left_panel = render_graph(
                session.graph,
                size=panel_size,
                title="Live (before optimization)",
                loop_radius=session.loop_return_radius,
                scan_fov=lidar_fov,
                scan_max_range=lidar_max_range,
                draw_scans=True,
            )
            right_panel = _blank_after_panel(panel_size)

        sep = np.full((panel_size, 2, 3), 60, dtype=np.uint8)
        cv2.imshow("Graph SLAM", np.hstack([left_panel, sep, right_panel]))
        cv2.waitKey(1)

    stop_robot(wheels)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot()
