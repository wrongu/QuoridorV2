import heapq
from collections import deque


class PathGraph(object):
    """Helper class that tracks and efficiently updates shortest paths from all nodes to sinks.

       This class takes ownership of the given adjacency graph (i.e. modifies it in cut() and
       uncut()). Adjacency graph format is a dict mapping from nodes to sets of nodes. Initial
       graph must be connected.
    """

    def __init__(self, init_graph, sinks):
        # Graph is a dict mapping from each node to all its neighbors. All connections are
        # bidirectional (or, equivalently, undirected).
        self._graph = init_graph

        self._sinks = set(sinks)

        # Downhill is a dict mapping each node to a tuple of (dist, next) in a shortest path to a
        # sink. When a node is 'severed', downhill[node] is None.
        self._downhill = {node: None for node in self._graph.keys()}
        for sink in sinks:
            self._downhill[sink] = (0, None)

        # Uphill is the inverse of _downhill, but maps to sets of nodes since _downhill may be
        # many-to-one.
        self._uphill = {node: set() for node in init_graph.keys()}

        # Connect everything and populate _uphill.
        self._reconnect_path(node for node in self._graph.keys() if node not in sinks)

    def get_distance(self, node):
        """Get distance from node to nearest sink (number of steps to get there), or -1 if
           unreachable.
        """
        path_info = self._downhill[node]
        return path_info[0] if path_info is not None else -1

    def has_path(self, node):
        """Return True iff there exists a path from node to a sink.
        """
        return self._downhill[node] is not None

    def cut(self, pair):
        """Given pair of adjacent nodes, cuts connections between them from the graph.
        """
        # TODO - optional cache for uncut.
        self._graph[pair[0]].discard(pair[1])
        self._graph[pair[1]].discard(pair[0])

        # Check if cut is on some downhill path. If so, sever all connections upstream of it and
        # recompute paths for those nodes.
        if self._downhill[pair[0]] is not None and self._downhill[pair[0]][1] == pair[1]:
            self._uphill[pair[1]].discard(pair[0])
            self._reconnect_path(self._sever(pair[0]))
        elif self._downhill[pair[1]] is not None and self._downhill[pair[1]][1] == pair[0]:
            self._uphill[pair[0]].discard(pair[1])
            self._reconnect_path(self._sever(pair[1]))

    def uncut(self, pair):
        """Opposite of cut().
        """
        nodeA, nodeB = pair
        self._graph[nodeA].add(nodeB)
        self._graph[nodeB].add(nodeA)

        # Re-attach nodes that had been completely cut off.
        if self._downhill[nodeA] is None or self._downhill[nodeB] is None:
            sev_node = nodeA if self._downhill[nodeA] is None else nodeB
            severed_nodes = set([sev_node])
            fringe = deque([sev_node])
            while len(fringe) > 0:
                node = fringe.pop()
                for neighbor in self._graph[node]:
                    if self._downhill[neighbor] is None and neighbor not in severed_nodes:
                        severed_nodes.add(neighbor)
                        fringe.append(neighbor)
            self._reconnect_path(severed_nodes)

        # Check if a shorter path now exists for nodeA through nodeB (or vice versa) and update
        # paths if so. Need to search in both directions. Downhill paths might be reversed, and
        # uphill nodes will need their distances updated.
        else:
            distA, distB = self._downhill[nodeA][0], self._downhill[nodeB][0]
            if abs(distA - distB) > 1:
                closer, farther = (nodeA, nodeB) if distA < distB else (nodeB, nodeA)
                dist_closer = min(distA, distB)

                def update_uphill(parent, node, parent_dist):
                    """Recursively update upstream from node (where 'parent' is one step downhill
                       from node).
                    """
                    self._downhill[node] = (parent_dist + 1, parent)
                    for up in self._uphill[node]:
                        update_uphill(node, up, parent_dist + 1)

                def update_downhill(parent, node, parent_dist):
                    """Recursively reverse the downhill direction from parent to node, stopping
                       when we are no longer shortening paths by reversing them.
                    """
                    (node_dist, node_child) = self._downhill[node]
                    if node_dist > parent_dist + 1:
                        # Recurse.
                        update_downhill(node, node_child, parent_dist + 1)
                        # Route 'node' through 'parent'.
                        self._uphill[self._downhill[node][1]].discard(node)
                        self._downhill[node] = (parent_dist + 1, parent)
                        self._uphill[parent].add(node)
                        self._uphill[node].discard(parent)

                # Possibly reverse nodes that are downhill from 'farther'.
                update_downhill(closer, farther, dist_closer)
                # Update distance counts uphill from 'farther' (note: this must happen after
                # update_downhill in case any previously downhill nodes are now uphill from
                # 'farther').
                update_uphill(closer, farther, dist_closer)

    def _sever(self, node):
        """Walk upstream from the given node, 'severing' each from _downhill and _uphill.
           Returns the set of severed nodes.
        """
        severed_nodes = set()
        severed_nodes.add(node)
        # Recurse uphill.
        for parent in self._uphill[node]:
            # Recursively cut off upstream from here.
            severed_nodes |= self._sever(parent)
        # Cut off this node.
        self._downhill[node] = None if node not in self._sinks else (0, None)
        self._uphill[node] = set()
        return severed_nodes

    def _reconnect_path(self, severed_nodes):
        """Compute shortest paths for each node in the given iterable of connected 'severed' nodes
           (i.e. those for which the 'downhill' direction is unknown).
        """
        severed_nodes = set(severed_nodes)

        # heap (priority queue) of known-path nodes that are on the border of the set of severed
        # nodes.
        border_heap = []
        border = set()
        # Add all 'border' nodes to the heap.
        for node in severed_nodes:
            for neighbor in self._graph[node]:
                if neighbor not in severed_nodes and neighbor not in border:
                    border.add(neighbor)
                    # The heap will sort by the first item in the tuple then the second, so we put
                    # 'dist' in the first slot to sort by distance.
                    (dist, _) = self._downhill[neighbor]
                    heapq.heappush(border_heap, (dist, neighbor))

        # Build _downhill and _uphill from shortest to longest.
        while len(severed_nodes) > 0 and len(border_heap) > 0:
            (dist, border_node) = heapq.heappop(border_heap)
            for neighbor in self._graph[border_node]:
                if neighbor in severed_nodes:
                    severed_nodes.discard(neighbor)
                    self._downhill[neighbor] = (dist + 1, border_node)
                    self._uphill[border_node].add(neighbor)
                    # Having added 'neighbor' to '_downhill', it now becomes part of the border.
                    heapq.heappush(border_heap, (dist + 1, neighbor))
