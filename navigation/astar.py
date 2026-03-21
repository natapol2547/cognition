# A* algorithm implementation for navigation
# From https://gist.github.com/WolfpackWilson/9e12a21859b50e309394fdfc5aba67d9#file-astar-py


from __future__ import annotations

import heapq

import numpy as np


class Node:
    """A Node object represent a point in a 2D array."""

    def __init__(self, point: tuple, *, parent: Node | None = None):
        """Initialize the Node object using a position tuple.

        Parameters
        ----------
        point: tuple
            The coordinate of the Node in a 2D array
        parent: Node = None
            A connection to the parent Node.
        """
        self.pos = point
        self.x, self.y = point

        self.g = 0  # distance between current node and start node
        self.h = 0  # heuristic distance between current node and end node
        self.f = 0  # total cost of utilizing the node

        self.parent = parent

    def __repr__(self):
        return '<Node %s, f=%s>' % (str(self.pos), str(round(self.f, 2)))

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def __le__(self, other):
        return self.f <= other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __eq__(self, other):
        return self.f == other.f

    def __ne__(self, other):
        return self.f != other.f

    def manhattan_dist(self, node: Node) -> int:
        """Return the Manhattan distance given another node."""
        return abs(self.x - node.x) + abs(self.y - node.y)

    def euclidean_dist(self, node: Node) -> float:
        """Return the Euclidean distance given another node."""
        return ((self.x - node.x) ** 2 + (self.y - node.y) ** 2) ** 0.5

    def to_coord(self, vector: tuple, parent: bool = True) -> Node:
        """Return a new Node object from a direction vector."""
        return Node((self.x + vector[0], self.y + vector[1]), parent=self if parent else None)

    def pos_in_bdry(self, boundary: tuple) -> bool:
        """Return whether the Node is in the boundary."""
        x, y = boundary
        return (0 <= self.x < x) and (0 <= self.y < y)

    @staticmethod
    def trace_path(node: Node):
        """Find the path from a node to the start."""
        path = [node.pos]

        while node.parent:
            path.append(node.parent.pos)
            node = node.parent

        path.reverse()
        return path


def astar(graph: np.ndarray, start: Node, target: Node) -> Node:
    """Find the shortest path from a starting Node to a target Node.

    Parameters
    ----------
    graph: np.ndarray
        A 2D matrix of 1s (blocked) and other ints
    start: Node
        The starting node.
    target: Node
        The node to reach.
    Returns
    -------
    Node: the target node
    """
    vectors = [(1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]

    start.g = 0
    start.h = start.f = start.manhattan_dist(target)

    open_nodes = [start]
    heapq.heapify(open_nodes)
    closed_nodes = graph.copy()

    while open_nodes:
        curr_node = heapq.heappop(open_nodes)

        # exit if the target has been reached
        if curr_node.pos == target.pos:
            return curr_node

        # travel to neighboring nodes
        for vector in vectors:
            child = curr_node.to_coord(vector)

            # check if outside the graph
            if not child.pos_in_bdry(graph.shape):
                continue

            # check if traversed or if a barrier exists
            if closed_nodes[child.pos] == 1:
                continue

            # update node parameters
            child.g = child.parent.g + child.euclidean_dist(curr_node) # type: ignore
            child.h = child.manhattan_dist(target)
            child.f = child.g + child.h

            heapq.heappush(open_nodes, child)

        # close the current node
        closed_nodes[curr_node.pos] = 1

    # exit if no target found
    raise Warning('No path found.')


if __name__ == '__main__':
    s = Node((0, 0))
    t = Node((4, 4))

    mtx = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])

    end = astar(mtx, s, t)
    print(end.g)
    print(Node.trace_path(end))