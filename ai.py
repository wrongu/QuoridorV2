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
            with temp_move(game, mv):
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
            with temp_move(game, mv):
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
        with temp_move(game, mv):
            hsh = hash(game)
            if hsh in visited:
                continue
            else:
                visited.add(hsh)
            v = min_value(game, -INFINITY, INFINITY, 0, visited)
        if v > best[0]:
            best = (v, mv)
    return mv


class temp_move:
    """Class providing do/undo functionality in a with statement.

    For example:

        game = Quoridor()
        game.exec_move("b5")
        with temp_move(game, "h5"):
            print game.history[-1] # shows move to h5
        print game.history[-1] # shows move to b5
    """
    def __init__(self, game, mv):
        self.game = game
        self.mv = mv

    def __enter__(self):
        self.game.exec_move(self.mv, False)
        return self.game

    def __exit__(self, type, value, traceback):
        self.game.undo()
