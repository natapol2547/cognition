from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from mapping.kinematics import DiffDriveOdometry, calculate_diff_drive_velocities
from mapping.graph_omg import GraphSession, GraphSLAM, scan_to_points

from devices.motor import MotorActuator
from devices.encoder import EncoderSensor
from devices.lidar import LidarSensor
from devices.compass import CompassSensor

from utils.keyboard import WebotsKeyboard
from utils.robot import get_supervisor, get_webots_robot

LINEAR_SPEED = 0.3  # m/s
ANGULAR_SPEED = 2.0  # rad/s

PANEL_SIZE = 600


def stop_robot(wheels: list[MotorActuator]) -> None:
    for wheel in wheels:
        wheel.stop()


def set_velocity(wheels: list[MotorActuator], velocity: list[float]) -> None:
    for wheel, vel in zip(wheels, velocity):
        wheel.setVelocity(vel)


def lerp(start, end, t):
    """Linear interpolation between two single values."""
    return start + (end - start) * t

def lerp_3d(start_pos, end_pos, t):
    """Applies lerp to a 3D coordinate list [x, y, z]."""
    return [lerp(start_pos[i], end_pos[i], t) for i in range(3)]


# ---------------------------------------------------------------------------
# Rendering helpers (live in slam.py; mapping.graph_omg owns the SLAM math).
# ---------------------------------------------------------------------------

def render_lidar_scan(
    ranges: np.ndarray,
    fov: float,
    max_range: float = 2.0,
    size: int = 360,
) -> np.ndarray:
    """Robot-centered top-down view of a single lidar scan."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    scale = (size / 2 - 10) / max(max_range, 1e-6)

    cv2.circle(img, (cx, cy), int(max_range * scale), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(0.5 * max_range * scale), (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (int(cx + 0.5 * max_range * scale), cy), (80, 80, 80), 1)

    pts = scan_to_points(ranges, fov, max_range=max_range)
    for p in pts:
        px = int(cx + p[0] * scale)
        py = int(cy - p[1] * scale)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (255, 255, 0)

    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
    cv2.line(img, (cx, cy), (cx + 12, cy), (0, 255, 0), 1)

    cv2.putText(
        img, f"lidar  fov={math.degrees(fov):.0f}deg  max={max_range:.1f}m",
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def _auto_scale(points: np.ndarray, size: int, margin: int, default_scale: float = 80.0) -> float:
    if points.size == 0:
        return default_scale
    max_abs = float(np.max(np.abs(points)))
    if max_abs <= 0:
        return default_scale
    return min(default_scale, (size / 2 - margin) / max_abs)


def _scan_world_points(
    graph: GraphSLAM,
    fov: float,
    max_range: Optional[float],
    poses_override: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """For each node, return its scan projected into world coords (or (0,2))."""
    ids = sorted(graph.nodes)
    out: list[np.ndarray] = []
    for k, nid in enumerate(ids):
        node = graph.nodes[nid]
        if node.scan is None:
            out.append(np.zeros((0, 2)))
            continue
        pose = poses_override[k] if poses_override is not None and k < poses_override.shape[0] else node.pose
        body = scan_to_points(node.scan, fov, max_range)
        if body.shape[0] == 0:
            out.append(body)
            continue
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        out.append(body @ R.T + pose[:2])
    return out


def _draw_pose_graph(
    img: np.ndarray,
    poses: np.ndarray,
    edges: list[tuple[int, int, str]],
    scale: float,
    cx: int,
    cy: int,
    node_color: tuple[int, int, int] = (0, 255, 0),
    start_color: tuple[int, int, int] = (255, 128, 0),
    odom_color: tuple[int, int, int] = (200, 200, 200),
    loop_color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 1.0,
) -> None:
    """Draw ``poses`` + ``edges`` onto ``img`` in-place, blended with ``alpha``."""
    if poses.shape[0] == 0:
        return

    overlay = img if alpha >= 1.0 else img.copy()

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return int(cx + p[0] * scale), int(cy - p[1] * scale)

    for i, j, kind in edges:
        if i >= poses.shape[0] or j >= poses.shape[0]:
            continue
        color = odom_color if kind == "odom" else loop_color
        thickness = 1 if kind == "odom" else 2
        cv2.line(overlay, to_px(poses[i]), to_px(poses[j]), color, thickness, cv2.LINE_AA)

    for idx in range(poses.shape[0]):
        p = poses[idx]
        color = start_color if idx == 0 else node_color
        radius = 5 if idx == 0 else 3
        px, py = to_px(p)
        cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)
        hx = int(px + 10 * math.cos(p[2]))
        hy = int(py - 10 * math.sin(p[2]))
        cv2.line(overlay, (px, py), (hx, hy), color, 1, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def render_graph(
    graph: GraphSLAM,
    size: int = PANEL_SIZE,
    title: Optional[str] = None,
    loop_radius: Optional[float] = None,
    pre_poses: Optional[np.ndarray] = None,
    margin: int = 40,
    scan_fov: Optional[float] = None,
    scan_max_range: Optional[float] = None,
    draw_scans: bool = True,
    draw_pre_scans: bool = False,
) -> np.ndarray:
    """Draw a pose graph with optional lidar scan aggregation.

    If ``pre_poses`` is given, overlay the previous (pre-optimization) graph in
    faded grey. If ``draw_scans`` and ``scan_fov`` are set, each node's stored
    scan is projected into world coordinates using that node's current pose.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    poses = graph.poses()

    # Pre-compute scan world points now so they contribute to the auto-fit bounds.
    scans_world: list[np.ndarray] = []
    pre_scans_world: list[np.ndarray] = []
    if draw_scans and scan_fov is not None:
        scans_world = _scan_world_points(graph, scan_fov, scan_max_range)
        if draw_pre_scans and pre_poses is not None and pre_poses.shape[0] > 0:
            pre_scans_world = _scan_world_points(graph, scan_fov, scan_max_range, pre_poses)

    # Gather every point we want to fit into the view.
    bounds: list[np.ndarray] = []
    if poses.shape[0] > 0:
        bounds.append(poses[:, :2])
    if pre_poses is not None and pre_poses.shape[0] > 0:
        bounds.append(pre_poses[:, :2])
    if loop_radius:
        bounds.append(np.array([[loop_radius, loop_radius], [-loop_radius, -loop_radius]]))
    for pts in scans_world + pre_scans_world:
        if pts.shape[0] > 0:
            bounds.append(pts)
    all_pts = np.concatenate(bounds, axis=0) if bounds else np.zeros((0, 2))
    scale = _auto_scale(all_pts, size, margin)
    cx = cy = size // 2

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return int(cx + p[0] * scale), int(cy - p[1] * scale)

    # Pre-optim scans (drawn first, darkest).
    for pts in pre_scans_world:
        if pts.shape[0] == 0:
            continue
        px = (cx + pts[:, 0] * scale).astype(np.int32)
        py = (cy - pts[:, 1] * scale).astype(np.int32)
        mask = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        img[py[mask], px[mask]] = (70, 70, 70)

    # Current scans in light grey so nodes/edges sit on top.
    for pts in scans_world:
        if pts.shape[0] == 0:
            continue
        px = (cx + pts[:, 0] * scale).astype(np.int32)
        py = (cy - pts[:, 1] * scale).astype(np.int32)
        mask = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        img[py[mask], px[mask]] = (180, 180, 180)

    if loop_radius and loop_radius > 0:
        r_px = max(1, int(loop_radius * scale))
        cv2.circle(img, (cx, cy), r_px, (0, 200, 200), 1, cv2.LINE_AA)

    if pre_poses is not None and pre_poses.shape[0] > 0:
        pre_edges = [(e.i, e.j, e.edge_type) for e in graph.edges]
        _draw_pose_graph(
            img, pre_poses, pre_edges, scale, cx, cy,
            node_color=(120, 120, 120),
            start_color=(140, 140, 140),
            odom_color=(90, 90, 90),
            loop_color=(90, 90, 140),
            alpha=0.6,
        )

    edges = [(e.i, e.j, e.edge_type) for e in graph.edges]
    _draw_pose_graph(img, poses, edges, scale, cx, cy)

    header = f"nodes: {poses.shape[0]}  edges: {len(edges)}"
    cv2.putText(
        img, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    if title:
        cv2.putText(
            img, title, (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def _blank_after_panel(size: int) -> np.ndarray:
    """Placeholder panel shown on the right before loop closure happens."""
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_robot() -> None:
    robot = get_webots_robot()
    supervisor = get_supervisor()
    timestep = int(robot.getBasicTimeStep())

    left_motor = MotorActuator(robot, "left wheel motor")
    right_motor = MotorActuator(robot, "right wheel motor")
    wheels: list[MotorActuator] = [left_motor, right_motor]

    left_encoder = EncoderSensor(robot, "left wheel sensor")
    right_encoder = EncoderSensor(robot, "right wheel sensor")

    compass = CompassSensor(robot)

    lidar = LidarSensor(robot)
    lidar_fov = lidar.getFov()
    lidar_max_range = lidar.getMaxRange()

    keyboard = WebotsKeyboard(robot)

    odometry = DiffDriveOdometry(left_encoder, right_encoder)

    session = GraphSession(
        loop_radius=0.15,
        lidar_fov=lidar_fov,
        lidar_max_range=lidar_max_range,
    )
    
    # Object
    target_node = supervisor.getFromDef('BALL')
    if target_node is None:
        print("Error: Node 'BALL' not found in the scene tree!")
        return
        
    translation_field = target_node.getField('translation')
    
    add_position = [0.5, 0.0, 0.0]
    start_pos = translation_field.getSFVec3f()
    end_pos = [val + add for val, add in zip(translation_field.getSFVec3f(), add_position)]
    
    # Total time for a full A -> B -> A cycle (in seconds)
    cycle_time = 4.0             
    half_cycle = cycle_time / 2.0

    print("SLAM running. Use WS to move, AD to rotate. Press 'X' to quit.")
    print("Drive a loop; the graph is optimized on each loop closure (start-return or ICP match).")

    while robot.step(timestep) != -1:
        # Ball
        # Get current elapsed simulation time
        current_time = supervisor.getTime()
        
        # Determine where we are in the current cycle (bounds: 0.0 to cycle_time)
        time_in_cycle = current_time % cycle_time
        
        # Calculate the interpolation factor 't' (0.0 to 1.0)
        if time_in_cycle < half_cycle:
            # First half: Moving from start_pos to end_pos
            t = time_in_cycle / half_cycle
        else:
            # Second half: Moving back from end_pos to start_pos
            # We invert the math so 't' goes from 1.0 back down to 0.0
            t = 1.0 - ((time_in_cycle - half_cycle) / half_cycle)
            
        # Calculate the new 3D position using our lerp function
        new_position = lerp_3d(start_pos, end_pos, t)
        
        # Apply the new coordinates to the object
        translation_field.setSFVec3f(new_position)
        
        # Robot
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
        pose_vec = np.array([pose[0], pose[1], pose[2]], dtype=float)

        ranges = lidar.getRangeImage()
        scan = np.asarray(ranges, dtype=float) if ranges is not None else None

        optimized = session.step(pose_vec, scan)
        if optimized:
            tensions = session.last_tensions
            initial = tensions[0] if tensions else float("nan")
            final = tensions[-1] if tensions else float("nan")
            loops = ", ".join(f"({i}->{j})" for i, j in session.last_loop_edges)
            print(
                f"Loop closed: nodes={len(session.graph.nodes)} "
                f"edges={len(session.graph.edges)} "
                f"tension {initial:.4f} -> {final:.4f} "
                f"({len(tensions)} iters) loops=[{loops}]"
            )

        # Lidar visualization window.
        if scan is not None and scan.size > 0:
            cv2.imshow("Lidar", render_lidar_scan(scan, lidar_fov, max_range=lidar_max_range))

        # Graph visualization: live on the left, after-optimization on the right.
        left_panel = render_graph(
            session.graph,
            size=PANEL_SIZE,
            title="Live (pre-optimization if loop not closed)",
            loop_radius=session.loop_radius,
            scan_fov=lidar_fov,
            scan_max_range=lidar_max_range,
            draw_scans=True,
        )
        if session.pre_optim_poses is not None:
            right_panel = render_graph(
                session.graph,
                size=PANEL_SIZE,
                title="After optimization (grey = before)",
                loop_radius=session.loop_radius,
                pre_poses=session.pre_optim_poses,
                scan_fov=lidar_fov,
                scan_max_range=lidar_max_range,
                draw_scans=True,
                draw_pre_scans=True,
            )
        else:
            right_panel = _blank_after_panel(PANEL_SIZE)

        sep = np.full((PANEL_SIZE, 2, 3), 60, dtype=np.uint8)
        cv2.imshow("Graph SLAM", np.hstack([left_panel, sep, right_panel]))
        cv2.waitKey(1)

    stop_robot(wheels)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot()
