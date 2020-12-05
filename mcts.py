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

    def __str__(self):
        return "TreeNode[{}] --> [{}]".format(self._key, ",".join(self._children.keys()))

    def __repr__(self):
        return str(self)

    @property
    def _avg_reward(self):
        return self._total_reward / (self._counts + 1e-6)

    def add_child(self, action:str, node:TreeNode):
        self._children[action] = node

    def upper_conf(self, c_puct) -> torch.Tensor:
        u = self._avg_reward + c_puct * self._policy * torch.sqrt(self._counts.sum()) / (1. + self._counts)
        u.masked_fill_(self._legal_mask == 0, -float('inf'))
        return u

    def policy_target(self) -> torch.Tensor:
        return self._counts / self._counts.sum()

    def backup(self, action:str, value):
        action_ijk = action_to_coordinate(action, self._player)
        self._total_reward[action_ijk] += value.item()
        self._counts[action_ijk] += 1

    def delete_unflagged_subtree(self):
        deleted_nodes = {self} if not self.__flagged else set()
        to_delete = list(self._children.items())
        for mv, node in to_delete:
            if not node.__flagged:
                deleted_nodes |= node.delete_unflagged_subtree()
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

        def __exit__(self, type, value, traceback):
            def recursively_unflag(node):
                node.__flagged = False
                for mv, chld in node._children.items():
                    recursively_unflag(chld)
            recursively_unflag(self.node)

class MonteCarloTreeSearch(object):
    def __init__(self, init_state:Quoridor, pol_val_fun):
        self.pol_val_fun = pol_val_fun
        self._root = TreeNode(init_state, *pol_val_fun(init_state))
        self._node_lookup = {init_state.hash_key(): self._root}
        self._state = init_state

    @property
    def player(self):
        return self._root._player

    def search(self, c_puct=0.9, n_evals=1000, verbose=False) -> torch.Tensor:
        the_key = self._state.hash_key()
        if the_key != self._root._key:
            raise RuntimeError("Tree search precondition failed... the root should never deviate from the state object")

        for isearch in range(n_evals):
            if verbose:
                print("MCTS.search run", isearch+1, "of", n_evals)
                print("Root is", str(self._root), "in tree of size", len(self._node_lookup))
            self._single_search(self._state, c_puct, verbose=verbose)
            if the_key != self._state.hash_key():
                raise RuntimeError("Consistency failure... calling _single_search modified the state!")

        # Return estimated policy
        return self._node_lookup[self._state.hash_key()].policy_target()

    def _single_search(self, game:Quoridor, c_puct, verbose=False) -> float:
        """Recursively run a single MCTS thread out from the given state using exploration parameter 'c_puct'.
        """
        node = self._node_lookup[game.hash_key()]
        action = sample_action(node.upper_conf(c_puct), node._player, temperature=0.0)
        if verbose:
            print("\tsingle_search starting @", node, "\n\t\ttaking", action, end="")
        with game.temp_move(action):
            winner = game.get_winner()
            if winner is not None:
                if verbose:
                    print("--> winner is", winner)
                # Case 1: 'action' ended the game. Return +1 if a win from the perspective of whoever played the move
                backup_val = +1 if winner == node._player else -1
            elif game.hash_key() not in self._node_lookup:
                # Case 2: 'action' resulted in a state we've never seen before. Create a new node and return
                pol, val = self.pol_val_fun(game)
                new_node = TreeNode(game, pol, val)
                self._node_lookup[game.hash_key()] = new_node
                node.add_child(action, new_node)
                if verbose:
                    print("--> leaf <{}> with value".format(str(new_node)), val)
                # "val" is from the perspective of "new_node" but we're evaluating "node". Flip sign for minmax.
                backup_val = -val
            else:
                # Case 3: we've seen this state before. But it's possible we're reaching it from a different history.
                # Ensure the parent/child relationship exists then recurse, flipping the sign of the child node's value.
                if verbose:
                    print("--> recursing to node", self._node_lookup[game.hash_key()])
                node.add_child(action, self._node_lookup[game.hash_key()])
                backup_val = -self._single_search(game, c_puct, verbose=verbose)

        # Apply backup
        node.backup(action, backup_val)
        return backup_val

    def step_and_prune(self, action, verbose=False):
        """Advance the tree by one move, fully discarding all un-taken branches of the tree
        """
        if self._state.hash_key() != self._root._key:
            raise RuntimeError("Tree consistency failed... the root should never deviate from the state object")
        self._state.exec_move(action)

        new_root = self._node_lookup[self._state.hash_key()]
        with new_root.subtree_flagged():
            deleted_nodes = self._root.delete_unflagged_subtree()
            for node in deleted_nodes:
                del self._node_lookup[node._key]
        if verbose:
            import pprint
            print("-- PRUNING --")
            pprint.pprint(deleted_nodes)
        self._root = new_root


if __name__ == "__main__":
    import time
    mcts = MonteCarloTreeSearch(Quoridor(), lambda state: (torch.rand(3,9,9), 2*torch.rand(1)-1))

    tstart = time.time()
    the_pol = mcts.search(c_puct=2, n_evals=1000, verbose=False)
    tend = time.time()

    print("Completed", len(mcts._node_lookup), "searches in", tend-tstart, "seconds")

    the_act = sample_action(the_pol, mcts.player, temperature=0.0)
    mcts.step_and_prune(the_act, verbose=True)
