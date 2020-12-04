from __future__ import annotations
import torch
from quoridor import Quoridor, IllegalMove
from quornn import encode_actions_to_planes, action_to_coordinate, sample_action


class TreeNode(object):
    def __init__(self, game_state:Quoridor, policy_output, value_output):
        # _counts is the number of times we've taken some action *from this state*. Initialized to all zeros. Stored
        # as a torch tensor over all possible actions, to be later masked with the set of legal actions
        self._counts = torch.zeros(3, 9, 9)
        self._total_reward = torch.zeros(3, 9, 9)
        self._policy = policy_output
        self._value = value_output
        self._legal_mask = encode_actions_to_planes(game_state.all_legal_moves(), game_state.current_player)
        self._player = game_state.current_player
        self._key = game_state.hash_key()
        self._children = {}
        self.__flagged = False

    @property
    def _avg_reward(self):
        return self._total_reward / (self._counts + 1e-6)

    def add_child(self, action:str, node:TreeNode):
        self._children[action] = node

    def upper_conf(self, c_puct):
        return self._avg_reward + c_puct * self._policy * torch.sqrt(self._counts.sum()) / (1. + self._counts)

    def policy_target(self, temperature=1.0):
        pol = self._counts**temperature
        return pol / pol.sum()

    def backup(self, action:str, value):
        action_ijk = action_to_coordinate(action, self._player)
        self._total_reward[action_ijk] += value
        self._counts[action_ijk] += 1

    def delete_unflagged_subtree(self):
        deleted_nodes = [self] if not self.__flagged else []
        for mv, node in self._children.items():
            if not node.__flagged:
                deleted_nodes += node.delete_unflagged_subtree()
                del self._children[mv]
        return deleted_nodes

    def subtree_flagged(self):
        return TreeNode.SubtreeFlagger(self)

    class SubtreeFlagger:
        def __init__(self, node):
            self.node = node

        def __enter__(self):
            def recursively_flag(node):
                node.__flagged = True
                for mv, chld in node._children.items():
                    recursively_flag(chld)
            recursively_flag(self.node)

        def __exit__(self):
            def recursively_unflag(node):
                node.__flagged = False
                for mv, chld in node._children.items():
                    recursively_unflag(chld)
            recursively_unflag(self.node)

class MonteCarloTreeSearch(object):
    def __init__(self, init_state:Quoridor, pol_val_fun):
        self.pol_val_fun = pol_val_fun
        self._root = TreeNode(init_state, *pol_val_fun(init_state))
        self._node_lookup = {init_state: self._root}
        self._state = init_state

    def search(self, c_puct=0.9, n_evals=1000):
        the_key = self._state.hash_key()
        if the_key != self._root._key:
            raise RuntimeError("Tree search precondition failed... the root should never deviate from the state object")

        for _ in range(n_evals):
            self._single_search(self._state, c_puct)
            if the_key != self._state.hash_key():
                raise RuntimeError("Consistency failure... calling _single_search modified the state!")

        # Return the TreeNode object corresponding to the root
        return self._node_lookup[self._state]

    def _single_search(self, game:Quoridor, c_puct) -> float:
        """Recursively run a single MCTS thread out from the given state using exploration parameter 'c_puct'.
        """
        node = self._node_lookup[game]
        action = sample_action(node.upper_conf(c_puct), node._player, temperature=0.0)
        with game.temp_move(action):
            winner = game.get_winner()
            if winner is not None:
                # Case 1: 'action' ended the game. Return +1 if a win from the perspective of whoever played the move
                backup_val = +1 if winner == node._player else -1
            elif game not in self._node_lookup:
                # Case 2: 'action' resulted in a state we've never seen before. Create a new node and return
                pol, val = self.pol_val_fun(game)
                new_node = TreeNode(game, pol, val)
                self._node_lookup[game] = new_node
                node.add_child(action, new_node)
                # "val" is from the perspective of "new_node" but we're evaluating "node". Flip sign for minmax.
                backup_val = -val
            else:
                # Case 3: we've seen this state before. But it's possible we're reaching it from a different history.
                # Ensure the parent/child relationship exists then recurse, flipping the sign of the child node's value.
                node.add_child(action, self._node_lookup[game])
                backup_val = -self._single_search(game, c_puct)

        # Apply backup
        node.backup(action, backup_val)
        return backup_val

    def step_and_prune(self, action):
        """Advance the tree by one move, fully discarding all un-taken branches of the tree
        """
        self._state.exec_move(action)

        new_root = self._node_lookup[self._state]
        with new_root.subtree_flagged():
            deleted_nodes = self._root.delete_unflagged_subtree()
            for node in deleted_nodes:
                del self._node_lookup[node._key]
        self._root = new_root