from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

from mapping.graph_slam import (
    OptimizeResult,
    PoseGraph,
    relative_pose,
    wrap_angle,
)


class GraphSlamSession:
    """Online driver that builds a pose graph and closes the loop on start return.

    Workflow:
    1. First `step` adds node 0 at the starting pose.
    2. Each subsequent `step` adds a new node + odometry edge when the robot
       has moved beyond the configured linear/angular thresholds since the
       last node.
    3. Once the robot has travelled far enough from the start (so we don't
       immediately "close" at spawn), returning within `loop_return_radius`
       adds a single loop-closure edge back to node 0 and optimizes the graph.
    """

    def __init__(
        self,
        node_linear_thresh: float = 0.15,
        node_angular_thresh: float = 0.3,
        loop_return_radius: float = 0.12,
        min_travel_for_loop: float = 0.5,
        odom_information: Optional[np.ndarray] = None,
        loop_information: Optional[np.ndarray] = None,
        optimize_iterations: int = 50,
        loop_constraint: Literal["identity", "odom_delta"] = "identity",
        loop_constrain_heading: bool = False,
        robust_kernel: Literal["none", "huber"] = "huber",
        huber_delta: float = 1.0,
        lm_lambda_init: float = 1e-3,
    ) -> None:
        self.node_linear_thresh = node_linear_thresh
        self.node_angular_thresh = node_angular_thresh
        self.loop_return_radius = loop_return_radius
        self.min_travel_for_loop = min_travel_for_loop
        self.optimize_iterations = optimize_iterations
        self.loop_constraint: Literal["identity", "odom_delta"] = loop_constraint
        self.loop_constrain_heading = loop_constrain_heading
        self.robust_kernel: Literal["none", "huber"] = robust_kernel
        self.huber_delta = huber_delta
        self.lm_lambda_init = lm_lambda_init

        self.odom_information = (
            odom_information if odom_information is not None
            else np.diag([50.0, 50.0, 100.0])
        )
        # Loop-closure information. By default we only constrain (x, y): the
        # robot may re-enter the start region from any direction, so we must
        # not force its heading to match node 0's. Heading is already pinned
        # reliably by the odometry chain (which reads from the compass).
        if loop_information is not None:
            self.loop_information = loop_information
        elif loop_constrain_heading:
            self.loop_information = np.diag([20.0, 20.0, 50.0])
        else:
            self.loop_information = np.diag([20.0, 20.0, 0.0])

        self.graph = PoseGraph()
        self.start_pose: Optional[np.ndarray] = None
        self.last_node_pose: Optional[np.ndarray] = None
        self.last_node_id: int = -1
        self.left_start: bool = False
        self.closed: bool = False
        self.last_chi2: float = 0.0
        self.optimize_result: Optional[OptimizeResult] = None

        # Snapshot of the graph state captured just before optimize() runs.
        # Stored as arrays so subsequent graph mutations cannot alter them.
        self.pre_optim_poses: Optional[np.ndarray] = None
        self.pre_optim_edges: Optional[list[tuple[int, int, str]]] = None

    def _should_add_node(self, pose: np.ndarray) -> bool:
        if self.last_node_pose is None:
            return True
        dx = pose[0] - self.last_node_pose[0]
        dy = pose[1] - self.last_node_pose[1]
        dt = wrap_angle(pose[2] - self.last_node_pose[2])
        dist = math.hypot(dx, dy)
        return dist >= self.node_linear_thresh or abs(dt) >= self.node_angular_thresh

    def _dist_from_start(self, pose: np.ndarray) -> float:
        if self.start_pose is None:
            return 0.0
        return math.hypot(pose[0] - self.start_pose[0], pose[1] - self.start_pose[1])

    def step(self, pose: np.ndarray, scan: Optional[np.ndarray] = None) -> bool:
        """Feed the current odometry pose and latest lidar scan.

        Returns True on the tick when the loop was closed and the graph
        was optimized, False otherwise.
        """
        pose = np.asarray(pose, dtype=float).copy()

        if self.start_pose is None or self.last_node_pose is None:
            node_id = self.graph.add_node(pose, scan=scan)
            self.start_pose = pose.copy()
            self.last_node_pose = pose.copy()
            self.last_node_id = node_id
            return False

        assert self.start_pose is not None and self.last_node_pose is not None

        if self._should_add_node(pose):
            new_id = self.graph.add_node(pose, scan=scan)
            z = relative_pose(self.last_node_pose, pose)
            self.graph.add_edge(
                self.last_node_id,
                new_id,
                z,
                information=self.odom_information,
                kind="odom",
            )
            self.last_node_id = new_id
            self.last_node_pose = pose.copy()

        dist_start = self._dist_from_start(pose)
        if not self.left_start and dist_start > self.min_travel_for_loop:
            self.left_start = True

        if (
            self.left_start
            and not self.closed
            and dist_start < self.loop_return_radius
        ):
            if self.loop_constraint == "identity":
                # Robot is physically back at the start, so the last node and
                # node 0 are co-located; the measurement is the SE(2) identity.
                z_loop = np.zeros(3)
            else:
                # Trust raw odometry: the measured relative pose reproduces
                # whatever odometry currently says (vacuous for optimization,
                # kept for comparison).
                z_loop = relative_pose(self.last_node_pose, self.start_pose)

            self.graph.add_edge(
                self.last_node_id,
                0,
                z_loop,
                information=self.loop_information,
                kind="loop",
            )

            # Snapshot poses + edges BEFORE optimization runs so we can
            # visualize the before/after side-by-side.
            self.pre_optim_poses = self.graph.poses().copy()
            self.pre_optim_edges = [(e.i, e.j, e.kind) for e in self.graph.edges]

            self.last_chi2 = self.graph.optimize(
                iterations=self.optimize_iterations,
                lm_lambda_init=self.lm_lambda_init,
                robust_kernel=self.robust_kernel,
                huber_delta=self.huber_delta,
            )
            self.optimize_result = self.graph.last_result
            self.closed = True
            return True

        return False
