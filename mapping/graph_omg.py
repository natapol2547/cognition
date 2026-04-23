"""Simple pose-graph SLAM (Gauss-Newton + online session driver).

LiDAR loop-closure matching is delegated to `mapping.icp.icp` (Lu & Milios style
2D ICP downloaded from https://github.com/richardos/icp).

Public surface:
    - Node, Edge: graph primitives.
    - GraphSLAM: Gauss-Newton optimiser (numerical Jacobian, scalar weights).
    - GraphSession: online driver that adds nodes on motion thresholds, builds
      odometry edges, detects loop closures (proximity to start + ICP on
      lidar scans), and triggers optimisation.
    - Helpers: wrap_angle, relative_pose, compose_se2, scan_to_points, icp_match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from mapping.icp import icp_match


# ---------------------------------------------------------------------------
# Tiny SE(2) helpers (pure trig; no matrix work needed)
# ---------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def relative_pose(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Return pose of ``pj`` expressed in ``pi``'s frame: [dx, dy, dtheta]."""
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    c, s = math.cos(pi[2]), math.sin(pi[2])
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        wrap_angle(pj[2] - pi[2]),
    ])


def compose_se2(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Return world pose: ``pose`` composed with body-frame ``delta`` [dx, dy, dtheta]."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    dx, dy, dth = float(delta[0]), float(delta[1]), float(delta[2])
    c, s = math.cos(th), math.sin(th)
    return np.array([
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        wrap_angle(th + dth),
    ])


# ---------------------------------------------------------------------------
# Graph data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A dot on the map: the robot's estimated pose at a moment in time."""
    id: int
    pose: np.ndarray  # [x, y, theta]
    scan: Optional[np.ndarray] = None  # raw lidar ranges (optional)


@dataclass
class Edge:
    """A rubber band between two nodes.

    ``measurement`` is [dx, dy, dtheta] — the relative pose of node ``j``
    in node ``i``'s frame. ``weight`` scales the information matrix used
    by the optimiser (Omega = weight * I).
    """
    i: int
    j: int
    measurement: np.ndarray
    weight: float = 1.0
    edge_type: Literal["odom", "loop"] = "odom"


# ---------------------------------------------------------------------------
# Gauss-Newton optimiser (directly follows the reference implementation)
# ---------------------------------------------------------------------------

class GraphSLAM:
    """Minimal pose-graph optimiser (Gauss-Newton, numerical Jacobians)."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def poses(self) -> np.ndarray:
        """Return (N, 3) array of poses, ordered by node id."""
        if not self.nodes:
            return np.zeros((0, 3))
        return np.stack([self.nodes[i].pose for i in sorted(self.nodes)], axis=0)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return wrap_angle(angle)

    def _calculate_error(
        self,
        pose_i: np.ndarray,
        pose_j: np.ndarray,
        measurement: np.ndarray,
    ) -> np.ndarray:
        """How stretched is this rubber band right now?"""
        xi, yi, theta_i = pose_i
        xj, yj, theta_j = pose_j
        meas_x, meas_y, meas_theta = measurement

        diff_x = xj - xi
        diff_y = yj - yi
        pred_x = diff_x * math.cos(theta_i) + diff_y * math.sin(theta_i)
        pred_y = -diff_x * math.sin(theta_i) + diff_y * math.cos(theta_i)
        pred_theta = wrap_angle(theta_j - theta_i)

        return np.array([
            pred_x - meas_x,
            pred_y - meas_y,
            wrap_angle(pred_theta - meas_theta),
        ])

    def _calculate_jacobian(
        self,
        pose_i: np.ndarray,
        pose_j: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nudge test: finite-difference Jacobian w.r.t. pose_i and pose_j."""
        epsilon = 1e-6
        base_error = self._calculate_error(pose_i, pose_j, measurement)

        J_i = np.zeros((3, 3))
        J_j = np.zeros((3, 3))

        for k in range(3):
            nudged = pose_i.copy()
            nudged[k] += epsilon
            J_i[:, k] = (
                self._calculate_error(nudged, pose_j, measurement) - base_error
            ) / epsilon

        for k in range(3):
            nudged = pose_j.copy()
            nudged[k] += epsilon
            J_j[:, k] = (
                self._calculate_error(pose_i, nudged, measurement) - base_error
            ) / epsilon

        return J_i, J_j, base_error

    def optimize(
        self,
        iterations: int = 10,
        anchor_weight: float = 1e6,
        verbose: bool = False,
    ) -> list[float]:
        """Gauss-Newton loop that relaxes every rubber band at once.

        Assumes node ids are contiguous from 0. Anchors node 0 so the graph
        cannot drift freely. Returns the per-iteration total tension.
        """
        if not self.nodes or not self.edges:
            return []

        num_nodes = len(self.nodes)
        dim = num_nodes * 3
        tensions: list[float] = []

        for iteration in range(iterations):
            # Sparse COO accumulators: one entry per 3×3 block contribution.
            # Each edge produces four 3×3 blocks (ii, ij, ji, jj) → 4×9 = 36
            # scalar entries.  Pre-allocate generously.
            max_entries = len(self.edges) * 36 + 9  # +9 for anchor
            rows = np.empty(max_entries, dtype=np.int32)
            cols = np.empty(max_entries, dtype=np.int32)
            vals = np.empty(max_entries, dtype=np.float64)
            ptr = 0

            b = np.zeros(dim)
            total_tension = 0.0

            for edge in self.edges:
                pose_i = self.nodes[edge.i].pose
                pose_j = self.nodes[edge.j].pose

                J_i, J_j, error = self._calculate_jacobian(
                    pose_i, pose_j, edge.measurement
                )
                total_tension += float(error @ error) * edge.weight

                w = edge.weight
                i3, j3 = edge.i * 3, edge.j * 3

                # Four 3×3 contributions to H (Omega = w * I so J.T @ Omega @ J = w * J.T @ J)
                blocks = [
                    (i3, i3, J_i.T @ J_i * w),
                    (i3, j3, J_i.T @ J_j * w),
                    (j3, i3, J_j.T @ J_i * w),
                    (j3, j3, J_j.T @ J_j * w),
                ]
                for r0, c0, blk in blocks:
                    for dr in range(3):
                        for dc in range(3):
                            rows[ptr] = r0 + dr
                            cols[ptr] = c0 + dc
                            vals[ptr] = blk[dr, dc]
                            ptr += 1

                b[i3:i3+3] += J_i.T @ (error * w)
                b[j3:j3+3] += J_j.T @ (error * w)

            # Pin node 0 (anchor).
            for k in range(3):
                rows[ptr] = k
                cols[ptr] = k
                vals[ptr] = anchor_weight
                ptr += 1

            H = scipy.sparse.csc_matrix(
                (vals[:ptr], (rows[:ptr], cols[:ptr])),
                shape=(dim, dim),
            )

            try:
                movement = scipy.sparse.linalg.spsolve(H, -b)
                if not np.isfinite(movement).all(): # type: ignore
                    break
            except Exception:
                break

            for i in range(num_nodes):
                self.nodes[i].pose[0] += movement[i*3 + 0]
                self.nodes[i].pose[1] += movement[i*3 + 1]
                self.nodes[i].pose[2] = wrap_angle(
                    self.nodes[i].pose[2] + movement[i*3 + 2]
                )

            tensions.append(total_tension)
            if verbose:
                print(f"[graph_omg] iter {iteration+1}/{iterations} tension={total_tension:.4f}")

        return tensions


# ---------------------------------------------------------------------------
# LiDAR helpers
# ---------------------------------------------------------------------------

def scan_to_points(
    ranges,
    fov: float,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """Convert a 1-D lidar range array into (M, 2) points in the robot body frame.

    Webots' Lidar sweeps from +fov/2 down to -fov/2, so we match that convention.
    Invalid and over-range returns are dropped.
    """
    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0:
        return np.zeros((0, 2))
    n = ranges.size
    angles = np.linspace(fov / 2.0, -fov / 2.0, n) if n > 1 else np.array([0.0])
    valid = np.isfinite(ranges) & (ranges > 0)
    if max_range is not None:
        valid &= ranges < max_range
    r = ranges[valid]
    a = angles[valid]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


# ---------------------------------------------------------------------------
# Online session driver
# ---------------------------------------------------------------------------

class GraphSession:
    """Online pose-graph builder.

    - Adds a node whenever the robot has moved / rotated past the thresholds.
    - Every new node is initialised in map frame by composing the *optimised* pose
      of the previous node with the raw incremental odometry since that node;
      each step also gets an odometry edge from the previous node.
    - Loop closures are proposed two ways:
        1. Proximity to node 0 (once past the warm-up).
        2. ICP of the current lidar scan against any older, nearby node.
    - If any loop edge is added, the whole graph is re-optimised.
    """

    def __init__(
        self,
        node_dist_thresh: float = 0.5,
        node_angle_thresh: float = math.radians(25),
        loop_radius: float = 0.15,
        loop_warmup_nodes: int = 8,
        icp_radius: float = 0.25,
        icp_min_gap: int = 8,
        icp_max_residual: float = 0.05,
        icp_search_max_distance: float = 0.5,
        lidar_fov: float = 2 * math.pi,
        lidar_max_range: Optional[float] = None,
        odom_weight: float = 1.0,
        loop_weight: float = 5.0,
        optimize_iterations: int = 10,
    ) -> None:
        self.node_dist_thresh = node_dist_thresh
        self.node_angle_thresh = node_angle_thresh
        self.loop_radius = loop_radius
        self.loop_warmup_nodes = loop_warmup_nodes
        self.icp_radius = icp_radius
        self.icp_min_gap = icp_min_gap
        self.icp_max_residual = icp_max_residual
        self.icp_search_max_distance = icp_search_max_distance
        self.lidar_fov = lidar_fov
        self.lidar_max_range = lidar_max_range
        self.odom_weight = odom_weight
        self.loop_weight = loop_weight
        self.optimize_iterations = optimize_iterations

        self.graph = GraphSLAM()
        self._next_id: int = 0
        self._last_odom_pose: Optional[np.ndarray] = None
        self._last_node_id: Optional[int] = None
        self._scan_points_cache: dict[int, np.ndarray] = {}

        self.closed_to_start: bool = False
        self.optimized_this_tick: bool = False
        self.last_tensions: list[float] = []
        self.last_loop_edges: list[tuple[int, int]] = []
        self.pre_optim_poses: Optional[np.ndarray] = None
        self.pre_optim_edges: Optional[list[tuple[int, int, str]]] = None

    # ---- internal helpers -------------------------------------------------

    def _should_add(self, odom: np.ndarray) -> bool:
        if self._last_odom_pose is None:
            return True
        dx = odom[0] - self._last_odom_pose[0]
        dy = odom[1] - self._last_odom_pose[1]
        dtheta = wrap_angle(odom[2] - self._last_odom_pose[2])
        return (
            math.hypot(dx, dy) >= self.node_dist_thresh
            or abs(dtheta) >= self.node_angle_thresh
        )

    def _cached_points(self, node_id: int) -> Optional[np.ndarray]:
        if node_id in self._scan_points_cache:
            return self._scan_points_cache[node_id]
        node = self.graph.nodes.get(node_id)
        if node is None or node.scan is None:
            return None
        pts = scan_to_points(node.scan, self.lidar_fov, self.lidar_max_range)
        self._scan_points_cache[node_id] = pts
        return pts

    def _add_node(self, pose: np.ndarray, scan: Optional[np.ndarray]) -> int:
        nid = self._next_id
        self._next_id += 1
        stored = None if scan is None else np.asarray(scan, dtype=float).copy()
        self.graph.add_node(Node(id=nid, pose=pose.copy(), scan=stored))
        return nid

    # ---- main entry point -------------------------------------------------

    def step(self, pose: np.ndarray, scan: Optional[np.ndarray] = None) -> bool:
        """Feed the latest odometry pose and lidar scan.

        The first node uses the raw odometry pose as the map frame origin; later
        node poses are initialised from the previous node's (possibly optimised)
        map pose and the odometry delta since that keyframe.

        Returns True on the tick where a loop closure was added and the graph
        re-optimised; False otherwise.
        """
        odom = np.asarray(pose, dtype=float)
        self.optimized_this_tick = False
        self.last_loop_edges = []

        if self._last_odom_pose is None:
            nid = self._add_node(odom, scan)
            self._last_odom_pose = odom.copy()
            self._last_node_id = nid
            return False

        if not self._should_add(odom):
            return False

        prev_id = self._last_node_id
        assert prev_id is not None and self._last_odom_pose is not None

        delta = relative_pose(self._last_odom_pose, odom)
        prev_map = self.graph.nodes[prev_id].pose
        new_map = compose_se2(prev_map, delta)
        new_id = self._add_node(new_map, scan)
        self.graph.add_edge(Edge(
            i=prev_id,
            j=new_id,
            measurement=delta,
            weight=self.odom_weight,
            edge_type="odom",
        ))
        self._last_odom_pose = odom.copy()
        self._last_node_id = new_id

        loop_added = self._maybe_close_to_start(new_id)
        if scan is not None:
            loop_added = self._maybe_close_via_icp(new_id) or loop_added

        if loop_added:
            self.pre_optim_poses = self.graph.poses().copy()
            self.pre_optim_edges = [
                (e.i, e.j, e.edge_type) for e in self.graph.edges
            ]
            self.last_tensions = self.graph.optimize(
                iterations=self.optimize_iterations,
            )
            self.optimized_this_tick = True
            return True

        return False

    # ---- loop-closure strategies -----------------------------------------

    def _maybe_close_to_start(self, new_id: int) -> bool:
        if self.closed_to_start:
            return False
        if new_id <= self.loop_warmup_nodes:
            return False
        start = self.graph.nodes.get(0)
        if start is None:
            return False
        new_pose = self.graph.nodes[new_id].pose
        dx = new_pose[0] - start.pose[0]
        dy = new_pose[1] - start.pose[1]
        if math.hypot(dx, dy) >= self.loop_radius:
            return False

        # Physically back at the start: the two nodes share the same world
        # pose, so the relative-pose measurement is identity.
        self.graph.add_edge(Edge(
            i=new_id,
            j=0,
            measurement=np.zeros(3),
            weight=self.loop_weight,
            edge_type="loop",
        ))
        self.last_loop_edges.append((new_id, 0))
        self.closed_to_start = True
        return True

    def _maybe_close_via_icp(self, new_id: int) -> bool:
        new_pts = self._cached_points(new_id)
        if new_pts is None or new_pts.shape[0] < 3:
            return False

        new_pose = self.graph.nodes[new_id].pose
        loop_added = False
        cutoff_id = new_id - self.icp_min_gap
        for cand_id in range(0, cutoff_id + 1):
            cand = self.graph.nodes.get(cand_id)
            if cand is None or cand.scan is None:
                continue

            dist = math.hypot(
                new_pose[0] - cand.pose[0], new_pose[1] - cand.pose[1]
            )
            if dist > self.icp_search_max_distance or dist > self.icp_radius:
                continue

            cand_pts = self._cached_points(cand_id)
            if cand_pts is None or cand_pts.shape[0] < 3:
                continue

            # ICP init: current map-frame guess of the new node in the candidate's
            # frame. ICP refines this; the result IS the measurement for edge
            # (cand -> new).
            init = relative_pose(cand.pose, new_pose)
            z_icp, residual = icp_match(
                ref_pts=cand_pts,
                pts=new_pts,
                init=(float(init[0]), float(init[1]), float(init[2])),
            )
            if residual >= self.icp_max_residual:
                continue

            self.graph.add_edge(Edge(
                i=cand_id,
                j=new_id,
                measurement=z_icp,
                weight=self.loop_weight,
                edge_type="loop",
            ))
            self.last_loop_edges.append((cand_id, new_id))
            loop_added = True

        return loop_added
