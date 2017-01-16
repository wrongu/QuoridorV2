from collections import deque


def _bfs_search(graph, fro, to_set):
    """Given an adjacency graph, starting location 'fro', and set of target locations, returns True
    iff at least one target in 'to_set' is reachable.
    """
    bfs_tree = {fro: None}
    deq = deque()
    deq.append(fro)
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
