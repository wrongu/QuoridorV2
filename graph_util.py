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
        if self._downhill[pair[0]] == pair[1]:
            self._reconnect_path(self._sever_upsteram([pair[0]]))
        elif self._downhill[pair[1]] == pair[0]:
            self._reconnect_path(self._sever_upsteram([pair[1]]))

    def uncut(self, pair):
        """Opposite of cut().
        """
        self._graph[pair[0]].add(pair[1])
        self._graph[pair[1]].add(pair[0])

        # Check if a shorter path now exists for pair[0] through pair[1] (or vice versa) and
        # update upstream paths if so.
        # TODO

    def _sever_upsteram(self, nodes, rec_set=None):
        """Walk upstream from the given set of nodes, 'severing' each from _downhill and _uphill.
           Returns the set of severed nodes.
        """
        severed_nodes = rec_set or set()
        severed_nodes.update(nodes)
        for node in nodes:
            # Recursively cut off upstream from here.
            self._sever_upsteram(self._uphill[node], severed_nodes)
            # Cut off this node.
            self._uphill[self._downhill[node]].discard(node)
            self._downhill[node] = None
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


def _bfs_search(graph, fro, to_set):
    """Given an adjacency graph, starting location 'fro', and set of target locations, returns True
    iff at least one target in 'to_set' is reachable.
    """
    bfs_tree = {fro: None}
    deq = deque([fro])
    while len(deq) > 0:
        v = deq.popleft()
        if v in to_set:
            return bfs_tree
        for neighbor in graph[v]:
            if neighbor not in bfs_tree:
                deq.append(neighbor)
                bfs_tree[neighbor] = v
    return bfs_tree


def _bfs_path(bfs_tree, fro, to_set):
    """Given a bfs_tree (output from _bfs_search), returns the shortest path from 'fro' to an item
    in to_set.
    """
    path = []
    for target in to_set:
        if target in bfs_tree:
            path = [target]
            while path[0] != fro:
                path.insert(0, bfs_tree[path[0]])
    return path


def _cuts_path(path, cut):
    """Given a path (list of locations) and a cut (a pair of adjacent locations), returns True iff
       the given cut severs the path.
    """
    for i in range(len(path) - 1):
        if path[i] in cut and path[i + 1] in cut:
            return True
    return False


def _is_reachable(graph, fro, to_set):
    """Given an adjacency graph, starting location 'fro', and set of target locations, returns True
    iff at least one target in 'to_set' is reachable.
    """
    if fro in to_set:
        return True
    deq = deque()
    deq.append(fro)
    visited = set([fro])
    while len(deq) > 0:
        for neighbor in graph[deq.pop()]:
            if neighbor in to_set:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                deq.append(neighbor)
    return False
