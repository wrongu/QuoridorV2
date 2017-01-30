import numpy as np
from collections import defaultdict
from operator import itemgetter
from quoridor import encode_loc
INFINITY = 1e9


def alphabeta_search(game, eval_fn, max_depth=4):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function.

    The evaluation function must take in (game, whose_perspective)
    as its arguments.

    Modified from http://aima.cs.berkeley.edu/python/games.html
    """
    player = game.current_player

    def cutoff_test(game, depth):
        return (depth > max_depth) or (game.get_winner() is not None)

    def max_value(game, alpha, beta, depth, visited):
        if cutoff_test(game, depth):
            return eval_fn(game, player)
        v = -INFINITY
        for mv in game.all_legal_moves():
            with game.temp_move(mv):
                hsh = hash(game)
                if hsh in visited:
                    continue
                else:
                    visited.add(hsh)
                v = max(v, min_value(game, alpha, beta, depth + 1, visited))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
        return v

    def min_value(game, alpha, beta, depth, visited):
        if cutoff_test(game, depth):
            return eval_fn(game, player)
        v = INFINITY
        for mv in game.all_legal_moves():
            with game.temp_move(mv):
                hsh = hash(game)
                if hsh in visited:
                    continue
                else:
                    visited.add(hsh)
                v = min(v, max_value(game, alpha, beta, depth + 1, visited))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v

    visited = set([hash(game)])

    # Body of alphabeta_search starts here:
    best = (-INFINITY, None)
    for mv in game.all_legal_moves():
        with game.temp_move(mv):
            hsh = hash(game)
            if hsh in visited:
                continue
            else:
                visited.add(hsh)
            v = min_value(game, -INFINITY, INFINITY, 0, visited)
        if v > best[0]:
            best = (v, mv)
    return mv


def monte_carlo_tree_search(game, eval_fn, policy_fn, max_depth=10, n_search=1000):
    """Monte Carlo Tree Search, where moves are selected according to policy_fn, playouts go to a
       depth of max_depth, at which point states are evaluated with eval_fn (as defined in
       alphabeta_search). policy_fn must take in a 'game' and return a list of (mv, prob) tuples.
    """
    player = game.current_player
    mv_scores = defaultdict(lambda: 0)
    n_visit = defaultdict(lambda: 0)

    # If all players are out of walls, simply step along the shortest path (careful about jumping
    # situations though).
    if sum(p[1] for p in game.players) == 0:
        shortest_path = game.get_graph().get_path()
        shortest_path_step = next(shortest_path)
        mv = encode_loc(shortest_path_step)
        if game.is_legal(mv):
            return mv

    def sample_move(game):
        moves, probabilities = zip(*policy_fn(game))
        probabilities = [p / sum(probabilities) for p in probabilities]
        choice_idx = np.random.choice(len(moves), p=probabilities)
        while not game.is_legal(moves[choice_idx]):
            probabilities[choice_idx] = 0
            probabilities = [p / sum(probabilities) for p in probabilities]
            choice_idx = np.random.choice(len(moves), p=probabilities)
        return moves[choice_idx]

    def recursive_search(game, remaining_depth):
        if (remaining_depth == 0) or (game.get_winner() is not None):
            return eval_fn(game, player)
        else:
            with game.temp_move(sample_move(game)):
                return recursive_search(game, remaining_depth - 1)

    for i in range(n_search):
        init_mv = sample_move(game)
        with game.temp_move(init_mv):
            score = recursive_search(game, max_depth)
            n = n_visit[init_mv]
            mv_scores[init_mv] = (n_visit[init_mv] * mv_scores[init_mv] + score) / (n + 1.0)
            n_visit[init_mv] += 1

    # Choose max value move.
    return max(mv_scores.items(), key=itemgetter(1))[0]
