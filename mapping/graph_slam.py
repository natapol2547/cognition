from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import cv2

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# SE(2) helpers

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def v2t(p: np.ndarray) -> np.ndarray:
    """Vector (x, y, theta) -> 3x3 homogeneous SE(2) matrix."""
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([
        [c, -s, p[0]],
        [s,  c, p[1]],
        [0,  0, 1.0],
    ])


def t2v(T: np.ndarray) -> np.ndarray:
    """3x3 SE(2) matrix -> (x, y, theta) vector."""
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])


def relative_pose(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Relative pose of pj expressed in the frame of pi."""
    return t2v(np.linalg.inv(v2t(pi)) @ v2t(pj))


def compose(pi: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compose pi with a relative delta expressed in pi's frame."""
    return t2v(v2t(pi) @ v2t(delta))


# Graph data structures

@dataclass
class OptimizeResult:
    """Summary of a single call to `PoseGraph.optimize`.

    `chi2_history` holds the chi2 of the initial estimate followed by the
    chi2 after every *accepted* LM step, so it is monotonically non-increasing.
    """
    chi2_history: List[float] = field(default_factory=list)
    iterations: int = 0
    accepted: int = 0
    rejected: int = 0
    converged: bool = False
    final_chi2: float = 0.0
    lambda_final: float = 0.0


@dataclass
class PoseNode:
    id: int
    pose: np.ndarray
    scan: Optional[np.ndarray] = None


@dataclass
class PoseEdge:
    i: int
    j: int
    z: np.ndarray
    information: np.ndarray
    kind: str = "odom"  # "odom" or "loop"


class PoseGraph:
    def __init__(self) -> None:
        self.nodes: List[PoseNode] = []
        self.edges: List[PoseEdge] = []
        self.last_result: Optional[OptimizeResult] = None

    def add_node(self, pose: np.ndarray, scan: Optional[np.ndarray] = None) -> int:
        node_id = len(self.nodes)
        self.nodes.append(PoseNode(id=node_id, pose=np.asarray(pose, dtype=float).copy(), scan=scan))
        return node_id

    def add_edge(
        self,
        i: int,
        j: int,
        z: np.ndarray,
        information: Optional[np.ndarray] = None,
        kind: str = "odom",
    ) -> None:
        if information is None:
            information = np.diag([50.0, 50.0, 100.0])
        self.edges.append(
            PoseEdge(
                i=i,
                j=j,
                z=np.asarray(z, dtype=float).copy(),
                information=np.asarray(information, dtype=float).copy(),
                kind=kind,
            )
        )

    def poses(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros((0, 3))
        return np.stack([n.pose for n in self.nodes], axis=0)

    # --- Global Error Minimization: Levenberg-Marquardt on SE(2) ---

    def _edge_error(self, edge: PoseEdge) -> np.ndarray:
        xi = self.nodes[edge.i].pose
        xj = self.nodes[edge.j].pose
        e_mat = np.linalg.inv(v2t(edge.z)) @ np.linalg.inv(v2t(xi)) @ v2t(xj)
        e = t2v(e_mat)
        e[2] = wrap_angle(e[2])
        return e

    def _chi2(
        self,
        robust_kernel: Literal["none", "huber"] = "none",
        huber_delta: float = 1.0,
    ) -> float:
        """Total weighted error across all edges (optionally robustified)."""
        total = 0.0
        for edge in self.edges:
            e = self._edge_error(edge)
            s = float(e.T @ edge.information @ e)
            if robust_kernel == "huber" and s > huber_delta ** 2:
                # rho(s) for the (squared) Huber kernel: quadratic below delta^2,
                # linear above it.
                total += 2.0 * huber_delta * math.sqrt(s) - huber_delta ** 2
            else:
                total += s
        return total

    def _linearize(
        self,
        robust_kernel: Literal["none", "huber"] = "none",
        huber_delta: float = 1.0,
    ) -> tuple[lil_matrix, np.ndarray, float]:
        """Build (H, b, chi2) at the current pose estimate."""
        n = len(self.nodes)
        H = lil_matrix((3 * n, 3 * n))
        b = np.zeros(3 * n)
        chi2 = 0.0

        for edge in self.edges:
            xi = self.nodes[edge.i].pose
            xj = self.nodes[edge.j].pose
            z = edge.z
            Omega = edge.information

            e = self._edge_error(edge)
            s = float(e.T @ Omega @ e)

            # Robust kernel weight (Huber): scale both the information matrix
            # AND the implicit error so that H = sum w * A^T Omega A, b = sum w * A^T Omega e
            if robust_kernel == "huber" and s > huber_delta ** 2:
                sqrt_s = math.sqrt(s)
                # rho'(s) for squared Huber: huber_delta / sqrt(s)
                weight = huber_delta / sqrt_s
                chi2_edge = 2.0 * huber_delta * sqrt_s - huber_delta ** 2
            else:
                weight = 1.0
                chi2_edge = s

            Omega_eff = weight * Omega

            # Jacobians A = de/dxi, B = de/dxj (Grisetti SE(2) form)
            si, ci = math.sin(xi[2]), math.cos(xi[2])
            dx = xj[0] - xi[0]
            dy = xj[1] - xi[1]

            zc, zs = math.cos(z[2]), math.sin(z[2])
            Rz_T = np.array([[zc, zs], [-zs, zc]])
            Ri_T = np.array([[ci, si], [-si, ci]])

            A = np.zeros((3, 3))
            A[0:2, 0:2] = -Rz_T @ Ri_T
            dRiT_dtheta = np.array([[-si, ci], [-ci, -si]])
            A[0:2, 2] = Rz_T @ dRiT_dtheta @ np.array([dx, dy])
            A[2, 2] = -1.0

            B = np.zeros((3, 3))
            B[0:2, 0:2] = Rz_T @ Ri_T
            B[2, 2] = 1.0

            i_slice = slice(3 * edge.i, 3 * edge.i + 3)
            j_slice = slice(3 * edge.j, 3 * edge.j + 3)

            H_ii = A.T @ Omega_eff @ A
            H_ij = A.T @ Omega_eff @ B
            H_jj = B.T @ Omega_eff @ B
            b_i = A.T @ Omega_eff @ e
            b_j = B.T @ Omega_eff @ e

            H[i_slice, i_slice] = H[i_slice, i_slice] + H_ii
            H[i_slice, j_slice] = H[i_slice, j_slice] + H_ij
            H[j_slice, i_slice] = H[j_slice, i_slice] + H_ij.T
            H[j_slice, j_slice] = H[j_slice, j_slice] + H_jj
            b[i_slice] += b_i
            b[j_slice] += b_j

            chi2 += chi2_edge

        return H, b, chi2

    def _apply_delta(self, dx_all: np.ndarray) -> None:
        for k, node in enumerate(self.nodes):
            delta = dx_all[3 * k:3 * k + 3]
            node.pose[0] += delta[0]
            node.pose[1] += delta[1]
            node.pose[2] = wrap_angle(node.pose[2] + delta[2])

    def optimize(
        self,
        iterations: int = 50,
        tol: float = 1e-6,
        lm_lambda_init: float = 1e-3,
        lm_factor: float = 2.0,
        robust_kernel: Literal["none", "huber"] = "huber",
        huber_delta: float = 1.0,
        verbose: bool = False,
    ) -> float:
        """Levenberg-Marquardt global error minimization.

        Returns the final chi2 (also available on `self.last_result` with
        the full history, accept/reject counts, and convergence flag).
        """
        n = len(self.nodes)
        result = OptimizeResult()
        self.last_result = result

        if n == 0 or not self.edges:
            result.converged = True
            return 0.0

        lam = lm_lambda_init
        chi2_current = self._chi2(robust_kernel, huber_delta)
        result.chi2_history.append(chi2_current)

        if verbose:
            print(f"[optimize] iter=0 chi2={chi2_current:.6f} lambda={lam:.2e}")

        for it in range(1, iterations + 1):
            result.iterations = it

            H, b, _ = self._linearize(robust_kernel, huber_delta)

            # Gauge fix on node 0: strong prior prevents the whole graph
            # from floating freely.
            for k in range(3):
                H[k, k] = H[k, k] + 1e6

            # Marquardt's diagonal damping: H + lambda * diag(H)
            H_csr = csr_matrix(H)
            diag_H = H_csr.diagonal()
            damping = diag_H * lam
            H_damped = H_csr + csr_matrix(
                (damping, (np.arange(3 * n), np.arange(3 * n))),
                shape=(3 * n, 3 * n),
            )

            try:
                dx_all = np.asarray(spsolve(H_damped, -b)).ravel()
            except Exception:
                if verbose:
                    print(f"[optimize] iter={it} spsolve failed, stopping")
                break

            # Tentatively apply; snapshot first so we can roll back on reject.
            pose_snapshot = [node.pose.copy() for node in self.nodes]
            self._apply_delta(dx_all)
            chi2_new = self._chi2(robust_kernel, huber_delta)

            if chi2_new < chi2_current:
                # Accept: record new chi2, decrease damping.
                delta_chi2 = chi2_current - chi2_new
                chi2_current = chi2_new
                result.chi2_history.append(chi2_current)
                result.accepted += 1
                lam = max(lam / lm_factor, 1e-12)

                if verbose:
                    print(
                        f"[optimize] iter={it} ACCEPT chi2={chi2_current:.6f} "
                        f"d_chi2={delta_chi2:.2e} |dx|={np.linalg.norm(dx_all):.2e} "
                        f"lambda={lam:.2e}"
                    )

                if (
                    delta_chi2 < tol
                    or float(np.linalg.norm(dx_all)) < tol
                ):
                    result.converged = True
                    break
            else:
                # Reject: roll back poses, increase damping.
                for node, snap in zip(self.nodes, pose_snapshot):
                    node.pose[:] = snap
                result.rejected += 1
                lam = lam * lm_factor

                if verbose:
                    print(
                        f"[optimize] iter={it} REJECT chi2={chi2_new:.6f} "
                        f"(>{chi2_current:.6f}) lambda={lam:.2e}"
                    )

                if lam > 1e12:
                    break

        result.final_chi2 = chi2_current
        result.lambda_final = lam
        return chi2_current


def scan_to_body_points(
    ranges: np.ndarray,
    fov: float,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """Convert a 1-D lidar range array into (N, 2) points in the robot body frame.

    Assumes samples are evenly spaced across `fov`. Infinite and over-range
    returns are filtered out. Webots' Lidar returns angles going from
    +fov/2 down to -fov/2 (i.e. counter-clockwise looking down), so we match
    that convention here.
    """
    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0:
        return np.zeros((0, 2))
    n = ranges.size
    if n == 1:
        angles = np.array([0.0])
    else:
        angles = np.linspace(fov / 2.0, -fov / 2.0, n)
    valid = np.isfinite(ranges) & (ranges > 0)
    if max_range is not None:
        valid &= ranges < max_range
    r = ranges[valid]
    a = angles[valid]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def scan_to_world_points(
    pose: np.ndarray,
    ranges: np.ndarray,
    fov: float,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """Transform a lidar scan at `pose` into world-frame (N, 2) points."""
    body = scan_to_body_points(ranges, fov, max_range)
    if body.shape[0] == 0:
        return body
    c, s = math.cos(pose[2]), math.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return body @ R.T + pose[:2]


def render_lidar_scan(
    ranges: np.ndarray,
    fov: float,
    max_range: float = 2.0,
    size: int = 360,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Top-down, robot-centered view of a single lidar scan.

    Robot sits at the center looking +x (right). Scan points are cyan dots,
    the max-range ring and heading tick are drawn for reference.
    """
    img = np.full((size, size, 3), background, dtype=np.uint8)
    cx = size // 2
    cy = size // 2
    scale = (size / 2 - 10) / max(max_range, 1e-6)

    cv2.circle(img, (cx, cy), int(max_range * scale), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(0.5 * max_range * scale), (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (int(cx + 0.5 * max_range * scale), cy), (80, 80, 80), 1)

    pts = scan_to_body_points(ranges, fov, max_range=max_range)
    for p in pts:
        px = int(cx + p[0] * scale)
        py = int(cy - p[1] * scale)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (255, 255, 0)

    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
    hx = int(cx + 12)
    cv2.line(img, (cx, cy), (hx, cy), (0, 255, 0), 1)

    cv2.putText(
        img, f"lidar  fov={math.degrees(fov):.0f}deg  max={max_range:.1f}m",
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def _render_poses(
    poses: np.ndarray,
    edges: list[tuple[int, int, str]],
    size: int,
    scale: float,
    margin: int,
    title: Optional[str] = None,
    loop_radius: Optional[float] = None,
    extra_poses: Optional[np.ndarray] = None,
    scans: Optional[list[Optional[np.ndarray]]] = None,
    scan_fov: Optional[float] = None,
    scan_max_range: Optional[float] = None,
) -> np.ndarray:
    """Shared renderer given raw arrays of poses and (i, j, kind) edge tuples.

    `extra_poses`, if provided, is used only to expand the auto-fit bounds
    so two renders (e.g. before/after) share the same scale.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if poses.shape[0] == 0:
        return img

    cx = size // 2
    cy = size // 2

    # Pre-compute per-node world-frame scan points (needed for bounds + render)
    scan_world_points: list[np.ndarray] = []
    if scans is not None and scan_fov is not None:
        for idx in range(poses.shape[0]):
            s = scans[idx] if idx < len(scans) else None
            if s is None:
                scan_world_points.append(np.zeros((0, 2)))
            else:
                scan_world_points.append(
                    scan_to_world_points(poses[idx], s, scan_fov, scan_max_range)
                )

    # Auto-fit scale so all nodes (and optional extras / scans) fit with margin
    bounds_src = poses[:, :2]
    if extra_poses is not None and extra_poses.shape[0] > 0:
        bounds_src = np.concatenate([bounds_src, extra_poses[:, :2]], axis=0)
    for pts in scan_world_points:
        if pts.shape[0] > 0:
            bounds_src = np.concatenate([bounds_src, pts], axis=0)

    max_abs = float(np.max(np.abs(bounds_src))) if bounds_src.size else 0.0
    # Include loop circle in bounds too
    if loop_radius is not None:
        max_abs = max(max_abs, loop_radius)
    if max_abs > 0:
        fit_scale = (size / 2 - margin) / max_abs
        scale = min(scale, fit_scale)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        px = int(cx + p[0] * scale)
        py = int(cy - p[1] * scale)
        return px, py

    # Scan points (drawn first so nodes/edges sit on top)
    for pts in scan_world_points:
        if pts.shape[0] == 0:
            continue
        px = (cx + pts[:, 0] * scale).astype(np.int32)
        py = (cy - pts[:, 1] * scale).astype(np.int32)
        mask = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        img[py[mask], px[mask]] = (180, 180, 180)

    # Loop return radius circle at start (node 0 in world frame = (0,0))
    if loop_radius is not None and loop_radius > 0:
        r_px = max(1, int(loop_radius * scale))
        cv2.circle(img, (cx, cy), r_px, (0, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            f"r={loop_radius:.2f}m",
            (cx + r_px + 4, cy - r_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Edges
    for i, j, kind in edges:
        if i >= poses.shape[0] or j >= poses.shape[0]:
            continue
        color = (200, 200, 200) if kind == "odom" else (0, 0, 255)
        thickness = 1 if kind == "odom" else 2
        cv2.line(img, to_px(poses[i]), to_px(poses[j]), color, thickness)

    # Nodes
    for idx in range(poses.shape[0]):
        p = poses[idx]
        color = (255, 128, 0) if idx == 0 else (0, 255, 0)
        radius = 5 if idx == 0 else 3
        px, py = to_px(p)
        cv2.circle(img, (px, py), radius, color, -1)
        hx = int(px + 10 * math.cos(p[2]))
        hy = int(py - 10 * math.sin(p[2]))
        cv2.line(img, (px, py), (hx, hy), color, 1)

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


def render_graph(
    graph: PoseGraph,
    size: int = 600,
    scale: float = 80.0,
    margin: int = 40,
    title: Optional[str] = None,
    loop_radius: Optional[float] = None,
    extra_poses: Optional[np.ndarray] = None,
    scan_fov: Optional[float] = None,
    scan_max_range: Optional[float] = None,
    draw_scans: bool = False,
) -> np.ndarray:
    """Draw the pose graph: odom edges gray, loop edges red, nodes green, start blue.

    If `draw_scans` is True and `scan_fov` is provided, each node's stored
    lidar scan is projected into the world using that node's pose and drawn
    as light-gray points (an aggregated occupancy-style view).
    """
    if not graph.nodes:
        return np.zeros((size, size, 3), dtype=np.uint8)
    poses = graph.poses()
    edges = [(e.i, e.j, e.kind) for e in graph.edges]
    scans = [n.scan for n in graph.nodes] if draw_scans else None
    return _render_poses(
        poses, edges, size, scale, margin,
        title=title, loop_radius=loop_radius, extra_poses=extra_poses,
        scans=scans, scan_fov=scan_fov, scan_max_range=scan_max_range,
    )


def render_poses(
    poses: np.ndarray,
    edges: list[tuple[int, int, str]],
    size: int = 600,
    scale: float = 80.0,
    margin: int = 40,
    title: Optional[str] = None,
    loop_radius: Optional[float] = None,
    extra_poses: Optional[np.ndarray] = None,
    scans: Optional[list[Optional[np.ndarray]]] = None,
    scan_fov: Optional[float] = None,
    scan_max_range: Optional[float] = None,
) -> np.ndarray:
    """Public renderer for a raw poses array (used for the pre-optimization snapshot)."""
    return _render_poses(
        poses, edges, size, scale, margin,
        title=title, loop_radius=loop_radius, extra_poses=extra_poses,
        scans=scans, scan_fov=scan_fov, scan_max_range=scan_max_range,
    )
