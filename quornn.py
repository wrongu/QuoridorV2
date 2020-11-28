import torch
from quoridor import encode_loc, parse_loc, Quoridor

# TODO - extend to 4-player games? All functions here currently assume 2-player

def encode_state_to_planes(quoridor_games) -> torch.Tensor:
    """Encode an instance or iterable instances of Quoridor objects into feature planes for input into a neural net.

    The feature planes directly encode the state: one 1-hot plane for each player, 1 plane for horizontal walls, and one
    for vertical wall locations. The 'current' player is always on plane 0 with goals at row [-1]
    """
    # Ensure iterable list
    if isinstance(quoridor_games, Quoridor):
        quoridor_games = [quoridor_games]

    def _game_to_planes(game):
        _planes = torch.zeros(4, 9, 9)

        # Current player on plane 0
        cur_row, cur_col = game.players[game.current_player][0]
        _planes[0, cur_row if game.current_player == 0 else 8-cur_row, cur_col] = 1

        # Other player(s) on plane 1
        for i in range(len(game.players)):
            if i == game.current_player: continue
            other_row, other_col = game.players[i][0]
            _planes[1, other_row if game.current_player == 0 else 8-other_row, other_col] = 1

        # Horizontal walls on plane 2, vertical walls on plane 3
        for w in game.walls:
            (row, col) = parse_loc(w[:2])
            if w[2] == 'h':
                _planes[2, row if game.current_player == 0 else 8-row, col] = 1
            elif w[2] == 'v':
                _planes[3, row if game.current_player == 0 else 7-row, col] = 1

        return _planes

    # Stack features along batch dimension
    return torch.stack([_game_to_planes(game) for game in quoridor_games], dim=0)

def action_to_coordinate(action:str, current_player:int) -> tuple:
    if len(action) == 2:
        row, col = parse_loc(action)
        plane = 0
    elif len(action) == 3:
        # Wall action
        row, col = parse_loc(action[:2])
        plane = 1 if action[2] == 'h' else 2
    else:
        raise ValueError("Invalid action: {}".format(action))
    return (plane, row if current_player == 0 else 8-row, col)

def encode_action_to_planes(actions, current_player:int) -> torch.Tensor:
    if isinstance(actions, str):
        actions = [actions]

    # Stack action planes along batch dimension
    planes = [torch.zeros(3, 9, 9) for _ in actions]
    for i, act in enumerate(actions):
        planes[i][action_to_coordinate(act, current_player)] = 1

    return torch.stack(planes, dim=0)

def decode_to_action(policy_planes:torch.Tensor, current_player:int, temperature=1.0, legal_moves=None):
    if len(policy_planes.size()) == 3:
        # Single batch - add a unit batch dimension in-place
        policy_planes.unsqueeze_(dim=0)

    # Enforce legality by creating a mask on the policy output.
    if legal_moves is None:
        legal_mask = torch.ones_like(policy_planes[0])
    else:
        legal_mask = torch.zeros_like(policy_planes[0])
        for mv in legal_moves:
            legal_mask[action_to_coordinate(mv, current_player)] = 1

    def _idx_to_action(idx:int):
        """Policy planes are [3 x 9 x 9] where plane [0] is movement, [1] is horizontal walls, and [2] is vertical walls
        """
        if idx < 0 or idx >= 3*9*9:
            raise ValueError("Action index out of bounds!")
        plane, row, col = idx // 81, (idx % 81) // 9, idx % 9
        if plane == 0:
            return encode_loc(row if current_player==0 else 8-row, col)
        elif plane == 1:
            return encode_loc(row if current_player==0 else 8-row, col)+"h"
        else:
            return encode_loc(row if current_player==0 else 7-row, col)+"v"

    num_batches = policy_planes.size()[0]
    actions = [None] * num_batches
    for b in range(num_batches):
        if temperature < 1e-6:
            # Do max operation instead of unstable low-temperature manipulations
            idx = torch.argmax(policy_planes[b])
        else:
            idx = torch.multinomial((policy_planes[b] * legal_mask).flatten()**temperature, num_samples=1)
        actions[b] = _idx_to_action(idx.item())
    return actions

if __name__ == '__main__':
    # mini test
    q = Quoridor()

    legal_moves = q.all_legal_moves(partial_check=False)
    print("INITIAL STATE LEGAL MOVES ({} of them):".format(len(legal_moves)))
    print(legal_moves)

    for mv in legal_moves:
        planes = encode_action_to_planes(mv, q.current_player)
        print("=========== {} ============".format(mv))
        print(planes[0])
        mv2 = decode_to_action(planes, 0)[0]
        print(mv2)
        assert mv2 == mv, "Failed to encode/decode {}".format(mv)

    # Test that just sampling random moves leads to some illegal moves getting selected (this is expected)
    rand_policy = torch.rand(100, 3, 9, 9)
    random_actions = decode_to_action(rand_policy, 0)
    masked_random_actions = decode_to_action(rand_policy, 0, legal_moves=legal_moves)

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
    planes0 = encode_state_to_planes(q)[0]
    q.current_player = 1
    planes1 = encode_state_to_planes(q)[0]

    print(planes0)
    print(planes1)

    # Test that plane 0 is 'current' player and flipped direction
    assert torch.all(planes0[0] == torch.flipud(planes1[1]))
    assert torch.all(planes0[1] == torch.flipud(planes1[0]))
    assert torch.all(planes0[2] == torch.flipud(planes1[2]))
    assert torch.all(planes0[3][:8, :] == torch.flipud(planes1[3][:8, :]))