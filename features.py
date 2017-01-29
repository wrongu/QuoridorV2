from quoridor import ALL_WALLS, ALL_POSITIONS, INTERSECTING_WALLS, TOUCHING_WALLS, WALL_CUTS, GOALS
from quoridor import parse_loc, encode_loc


def simple_value(game, player):
    paths = [g.get_distance(p[0]) for (g, p) in zip(game._pathgraphs, game.players)]
    walls = [p[1] for p in game.players]

    path_diff = paths[player] - min([p for (i, p) in enumerate(paths) if i != player])
    wall_diff = walls[player] - max([w for (i, w) in enumerate(walls) if i != player])

    score = wall_diff - 4 * path_diff

    if game.get_winner() == player:
        score += 1000

    return score


def simple_policy(game):
    return [(mv, 1.0) for mv in game.all_legal_moves(True)]


###############################
# VALUE AND FEATURE FUNCTIONS #
###############################

def is_shortest_path(game, mv):
    """Returns True iff 'mv' is moving towards the player's goal as quickly as possible.
    """
    if len(mv) != 2:
        return False
    graph = game.get_graph()
    player = game.get_player()
    loc = parse_loc(mv)
    return graph.get_distance(loc) == graph.get_distance(player) - 1


def self_detour(game, mv):
    if len(mv) != 3:
        return 0
    # TODO - compute how much longer my own path would be by placing 'mv'


def opponent_detour(game, mv):
    if len(mv) != 3:
        return 0
    # TODO - compute how much longer opponent's path would be by placing 'mv'


def wall_touches_self(game, mv):
    pass


def wall_touches_opponent(game, mv):
    pass


def wall_touches_last_wall(game, mv):
    pass
