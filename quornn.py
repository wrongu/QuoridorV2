import torch
from quoridor import encode_loc, parse_loc, Quoridor
from typing import Iterable, Union

# TODO - extend to 4-player games? All functions here currently assume 2-player
# TODO - make batch operations

STATE_SHAPE = (6, 9, 9)
POLICY_SHAPE = (3, 9, 9)

def flip_y_perspective(row:int, current_player:int, is_vwall:bool=False)->int:
    """Flip row coordinates for player 1 so that -- no matter who the 'current player' is -- the enemy's gate is down.

    Note that flip_y_perspective is its own inverse, so flip_y_perspective(flip_y_perspective(r, c, v), c, v) == r
    """
    # Since vertical walls are labeled by their 'top' coordinate, we need a slightly different rule. This is because
    # what was the 'bottom' coordinate at (row+1) will become the top coordinate.
    if current_player == 0:
        return row
    elif is_vwall:
        # Essentially 8-(row+1) to flip 'bottom' of wall to 'top'
        return 7-row
    else:
        return 8-row

def encode_state_to_planes(game:Quoridor, add_batch=False, out:torch.Tensor=None) -> torch.Tensor:
    """Encode an instance of a Quoridor object into feature planes for input into a neural net.

    The feature planes directly encode the state:
    - plane [0] is a one 1-hot plane for the location of the current player, oriented with goals in the last row
    - plane [1] is flooded with all the same value, set equal to the number of walls the current player has remaining
    - planes [2] and [3] are the same, but for the opponent. The board is oriented the same as the 0th plane.
    - plane [4] contains two '1' entries everywhere a horizontal wall has been placed
    - plane [5] similarly contains all the vertical walls
    wall locations, and one for vertical wall locations. The 'current' player is always on plane 0 with goals at the
    last row, and the 'other' player is on plane 1 with goals at the 0th row.
    """

    if out is None:
        out = torch.zeros(6, 9, 9)
    else:
        out.fill_(0.0)

    # Current player on out 0 and 1
    (cur_row, cur_col), cur_walls = game.players[game.current_player]
    out[0, flip_y_perspective(cur_row, game.current_player), cur_col] = 1
    out[1, :, :] = cur_walls

    # Other player(s) on planes 2 and 3
    (other_row, other_col), other_walls = game.players[1-game.current_player]
    out[2, flip_y_perspective(other_row, game.current_player), other_col] = 1
    out[3, :, :] = other_walls

    # Horizontal walls on plane 4, vertical walls on plane 5, again oriented to the perspective of the current player
    for w in game.walls:
        (row, col) = parse_loc(w[:2])
        if w[2] == 'h':
            out[4, flip_y_perspective(row, game.current_player), col] = 1
            out[4, flip_y_perspective(row, game.current_player), col+1] = 1
        elif w[2] == 'v':
            out[5, flip_y_perspective(row, game.current_player, True), col] = 1
            out[5, flip_y_perspective(row, game.current_player, True)+1, col] = 1

    if add_batch:
        return out.unsqueeze(0)
    else:
        return out

def action_to_coordinate(action:str, current_player:int) -> tuple:
    """Given an action string (like 'b4' for pawn movement or 'd4h' for a wall), return the (plane, row, col) index into
    a policy tensor indexing that action. A policy tensor P has shape (3, 9, 9), where P[0] indicates movement to
    corresponding spaces on the board, P[1] indicates placement of a horizontal wall, and P[2] is vertical walls.

    Note that all of P[2:3, 8, :] and P[2:3, :, 8] (last row and col of walls) are all illegal moves.
    """
    if len(action) == 2:
        row, col = parse_loc(action)
        plane = 0
    elif len(action) == 3:
        # Wall action
        row, col = parse_loc(action[:2])
        plane = 1 if action[2] == 'h' else 2
    else:
        raise ValueError("Invalid action: {}".format(action))

    # If current player is 1, then all y coordinates (rows) are flipped.
    return (plane, flip_y_perspective(row, current_player, action[-1] == 'v'), col)

def encode_actions_to_planes(actions:Union[Iterable[str], str], current_player:int, out:torch.Tensor=None) -> torch.Tensor:
    """Given an action string (like 'b4' for pawn movement or 'd4h' for a wall), return the 1-hot encoding of it as a
    policy tensor. Given an iterable of actions, return the union of all such tensors.

    A policy tensor P has shape (3, 9, 9), where P[0] indicates movement to corresponding spaces on the board, P[1]
    indicates placement of a horizontal wall, and P[2] is vertical walls.

    Note that all of P[2:3, 8, :] and P[2:3, :, 8] (last row and col of walls) are all illegal moves.
    """
    # Make it iterable
    if isinstance(actions, str):
        actions = (actions,)

    # Encode each action in the list with a '1'
    if out is None:
        out = torch.zeros(3, 9, 9)
    else:
        out.fill_(0.0)
    for move in actions:
        out[action_to_coordinate(move, current_player)] = 1
    return out

def sample_action(policy_planes:torch.Tensor, current_player:int, temperature=1.0) -> str:
    """Sample an action from the given (3 x 9 x 9) policy. Behavior depends on the current_player because the policy is
    always from the perspective of the current player, while actions are in global board coordinates.

    Assuming 'policy_planes' contains all non-negative entries. This function performs no legality checks -- to enforce
    move legality, first multiply a policy by a mask that zeros out illegal moves.
    """
    def _idx_to_action(idx:int)->str:
        """Policy planes are [3 x 9 x 9] where plane [0] is movement, [1] is horizontal walls, and [2] is vertical walls

        This function takes a 1D flattened index in [0,243) and returns an action in appropriately transformed coords.
        """
        if idx < 0 or idx >= 3*9*9:
            raise ValueError("Action index out of bounds!")
        plane, row, col = idx // 81, (idx % 81) // 9, idx % 9

        # Note that flip_y_perspective is its own inverse, so it works both for translating from boards to tensors and
        # back again (here we're doing the back again part)
        if plane == 0:
            return encode_loc(flip_y_perspective(row, current_player), col)
        elif plane == 1:
            return encode_loc(flip_y_perspective(row, current_player), col)+"h"
        else:
            return encode_loc(flip_y_perspective(row, current_player, True), col)+"v"

    if temperature < 1e-6:
        # Do max operation instead of unstable low-temperature manipulations
        idx = torch.argmax(policy_planes)
    else:
        idx = torch.multinomial(policy_planes.flatten()**temperature, num_samples=1)
    return _idx_to_action(idx.item())

if __name__ == '__main__':
    # mini test
    q = Quoridor()

    legal_moves = q.all_legal_moves(partial_check=False)
    print("INITIAL STATE LEGAL MOVES ({} of them):".format(len(legal_moves)))
    print(legal_moves)

    for mv in legal_moves:
        planes = encode_actions_to_planes(mv, q.current_player)
        print("=========== {} ============".format(mv))
        print(planes)
        mv2 = sample_action(planes, 0)
        print(mv2)
        assert mv2 == mv, "Failed to encode/decode {}".format(mv)

    # Test that just sampling random moves leads to some illegal moves getting selected (this is expected)
    random_actions, masked_random_actions = ['']*100, ['']*100
    legal_mask = encode_actions_to_planes(legal_moves, q.current_player)
    for i in range(100):
        rand_policy = torch.rand(3, 9, 9)
        random_actions[i] = sample_action(rand_policy, 0)
        masked_random_actions[i] = sample_action(rand_policy * legal_mask, 0)

    print("RANDOM policy selected the following illegal actions:")
    print(set(random_actions) - set(legal_moves))
    print("RANDOM policy selected the following legal actions:")
    print(set(random_actions) & set(legal_moves))
    print("MASKED RANDOM policy selected the following illegal actions:")
    print(set(masked_random_actions) - set(legal_moves))
    print("MASKED RANDOM policy selected the following legal actions:")
    print(set(masked_random_actions) & set(legal_moves))

    q.exec_move('a4')
    q.exec_move('h5')
    q.exec_move('a1v')
    q.exec_move('d4h')
    q.exec_move('h3v')
    q.exec_move('h8v')
    planes0 = encode_state_to_planes(q)
    q.current_player = 1
    planes1 = encode_state_to_planes(q)

    print(planes0)
    print(planes1)

    # Test that plane 0 is 'current' player and flipped direction from perspective of other player
    assert torch.all(planes0[0] == torch.flipud(planes1[2]))
    assert torch.all(planes0[2] == torch.flipud(planes1[0]))
    # Test that walls are vertically flipped between two players' perspectives
    assert torch.all(planes0[4] == torch.flipud(planes1[4]))
    assert torch.all(planes0[5] == torch.flipud(planes1[5]))
