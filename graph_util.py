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

        # Distance from each node to a sink along its 'downhill' path, or -1 if not connected.
        self._dist = {node: -1 for node in self._graph.keys()}

        # Downhill is a dict mapping each node to the next node in a shortest path to a
        # sink. When a node is 'severed', downhill[node] is None. Sinks point to themselves.
        self._downhill = {node: None for node in self._graph.keys()}
        for sink in sinks:
            self._downhill[sink] = sink
            self._dist[sink] = 0

        # Uphill is the inverse of _downhill, but maps to sets of nodes since _downhill may be
        # many-to-one.
        self._uphill = {node: set() for node in init_graph.keys()}

        # Connect everything and populate _uphill.
        self._reconnect_path(node for node in self._graph.keys() if node not in sinks)

    def get_distance(self, node):
        """Get distance from node to nearest sink (number of steps to get there), or -1 if
           unreachable.
        """
        return self._dist[node]

    def has_path(self, node):
        """Return True iff there exists a path from node to a sink.
        """
        return self._downhill[node] is not None

    def get_path(self, node):
        """Generator of shortest path from node to a sink, inclusive of the sink but not the node.
        """
        while node not in self._sinks:
            node = self._downhill[node]
            yield node

    def cut(self, pairs):
        """Given pairs of adjacent nodes, cuts connections between them from the graph.
        """
        # TODO - optional cache for uncut.

        severed_nodes = set()

        for pair in pairs:
            nodeA, nodeB = pair
            self._graph[nodeA].discard(nodeB)
            self._graph[nodeB].discard(nodeA)

            # Check if cut is on some downhill path. If so, sever all connections upstream of it
            # and recompute paths for those nodes.
            if self._downhill[nodeA] == nodeB:
                self._uphill[nodeB].discard(nodeA)
                severed_nodes |= self._sever(nodeA)
            elif self._downhill[nodeB] == nodeA:
                self._uphill[nodeA].discard(nodeB)
                severed_nodes |= self._sever(nodeB)
        self._reconnect_path(severed_nodes)

    def uncut(self, pairs):
        """Opposite of cut().
        """
        search_fringe = deque()
        search_visited = set()
        for pair in pairs:
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
                distA, distB = self._dist[nodeA], self._dist[nodeB]
                if abs(distA - distB) > 1:
                    closer, farther = (nodeA, nodeB) if distA < distB else (nodeB, nodeA)
                    # Route 'farther' through 'closer'.
                    self._reroute(farther, closer)
                    # Add 'farther' to the set of fringe to search from.
                    search_fringe.append(farther)
                    search_visited.add(closer)
        # Search out from updated nodes - update all paths that can be made shorter.
        while len(search_fringe) > 0:
            node = search_fringe.popleft()
            node_dist = self._dist[node]
            for neighbor in self._graph[node] - search_visited:
                search_visited.add(neighbor)
                if self._dist[neighbor] > node_dist + 1:
                    # Route 'neighbor' through 'node'
                    self._reroute(neighbor, node)
                    # Search further from neighbor.
                    search_fringe.append(neighbor)

    def _reroute(self, fro, to):
        """Helper function to route 'fro' through 'to' (assuming it's valid on the graph), updating
           instance variables as necessary.
        """
        self._dist[fro] = self._dist[to] + 1
        self._uphill[self._downhill[fro]].discard(fro)
        self._downhill[fro] = to
        self._uphill[to].add(fro)

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
        if node not in self._sinks:
            self._dist[node], self._downhill[node] = 0, None
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
                    heapq.heappush(border_heap, (self._dist[neighbor], neighbor))

        # Build _downhill and _uphill from shortest to longest.
        while len(severed_nodes) > 0 and len(border_heap) > 0:
            (dist, border_node) = heapq.heappop(border_heap)
            for neighbor in self._graph[border_node]:
                if neighbor in severed_nodes:
                    severed_nodes.discard(neighbor)
                    self._dist[neighbor] = dist + 1
                    self._downhill[neighbor] = border_node
                    self._uphill[border_node].add(neighbor)
                    # Having added 'neighbor' to '_downhill', it now becomes part of the border.
                    heapq.heappush(border_heap, (dist + 1, neighbor))

    def _sanity_check(self):
        err = False
        for (node, next) in self._downhill.items():
            if next is not None:
                dist = self._dist[node]
                if next is not None and next not in self._graph[node] and node not in self._sinks:
                    print "INCONSISTENCY: connectivity of graph and downhill.", node, next
                    err = True
                if next is not None and node not in self._graph[next] and next not in self._sinks:
                    print "INCONSISTENCY: reverse connectivity of graph and downhill."
                    err = True
                if node not in self._sinks and next is not None and node not in self._uphill[next]:
                    print "INCONSISTENCY:", node, "not in uphill[downhill[", node, "]]"
                    err = True
                if node not in self._sinks and next is not None and dist != self._dist[next] + 1:
                    print "INCONSISTENCY: path lengths", node, dist, "->", next, self._dist[next]
                    err = True
        for (node, parents) in self._uphill.items():
            for par in parents:
                if self._downhill[par] is None:
                    print "INCONSISTENCY: uphill exists but not reciprocated"
                    err = True
                elif self._downhill[par] != node:
                    print "INCONSISTENCY:", node, "not in downhill[uphill[", node, "]]"
                    err = True
        if err:
            import pdb
            pdb.set_trace()
