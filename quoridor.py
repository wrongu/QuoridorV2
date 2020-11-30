from graph_util import PathGraph


def parse_loc(loc_str):
    """Parse 'a1' into (0, 0).
    """
    row = ord(loc_str[0].lower()) - ord('a')
    col = int(loc_str[1]) - 1
    return (row, col)


def encode_loc(row, col):
    """Inverse of parse_loc.
    """
    return chr(row + ord('a')) + str(col + 1)


def cross(wall_str):
    return wall_str[0:2] + ('h' if wall_str[2] == 'v' else 'v')


BOARD_SIZE = 9
ALL_WALLS = set()
ALL_POSITIONS = set()
for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
        ALL_POSITIONS.add(encode_loc(row, col))
        if row < BOARD_SIZE-1 and col < BOARD_SIZE-1:
            ALL_WALLS.add(encode_loc(row, col) + 'h')
            ALL_WALLS.add(encode_loc(row, col) + 'v')

# Construct dict mapping from each wall to the set of walls that it physically rules out (including itself).
INTERSECTING_WALLS = {}
for wall in ALL_WALLS:
    INTERSECTING_WALLS[wall] = set([wall, cross(wall)])
    (row, col) = parse_loc(wall[0:2])
    if wall[2] == 'v':
        if row > 0:
            INTERSECTING_WALLS[wall].add(encode_loc(row - 1, col) + 'v')
        if row < BOARD_SIZE-2:
            INTERSECTING_WALLS[wall].add(encode_loc(row + 1, col) + 'v')
    elif wall[2] == 'h':
        if col > 0:
            INTERSECTING_WALLS[wall].add(encode_loc(row, col - 1) + 'h')
        if col < BOARD_SIZE-2:
            INTERSECTING_WALLS[wall].add(encode_loc(row, col + 1) + 'h')

# Construct mapping from each wall to the set of walls that it legally touches (colinear, L, or T)
TOUCHING_WALLS = {}
for wall in ALL_WALLS:
    TOUCHING_WALLS[wall] = set()
    (row, col) = parse_loc(wall[0:2])
    if wall[2] == 'v':
        # Colinear above
        if row >= 2:
            TOUCHING_WALLS[wall].add(encode_loc(row - 2, col) + 'v')
        # Colinear below
        if row <= BOARD_SIZE-3:
            TOUCHING_WALLS[wall].add(encode_loc(row + 2, col) + 'v')
        # 'T' above
        if row >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col) + 'h')
        # 'T' below
        if row <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col) + 'h')
        # 'T' left
        if col >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row, col - 1) + 'h')
        # 'T' right
        if col <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row, col + 1) + 'h')
        # 'L' above-left
        if row >= 1 and col >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col - 1) + 'h')
        # 'L' above-right
        if row >= 1 and col <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col + 1) + 'h')
        # 'L' below-left
        if row <= BOARD_SIZE-2 and col >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col - 1) + 'h')
        # 'L' below-right
        if row <= BOARD_SIZE-2 and col <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col + 1) + 'h')
    if wall[2] == 'h':
        # Colinear left
        if col >= 2:
            TOUCHING_WALLS[wall].add(encode_loc(row, col - 2) + 'h')
        # Colinear right
        if col <= BOARD_SIZE-3:
            TOUCHING_WALLS[wall].add(encode_loc(row, col + 2) + 'h')
        # 'T' left
        if col >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row, col - 1) + 'v')
        # 'T' right
        if col <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row, col + 1) + 'v')
        # 'T' above
        if row >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col) + 'v')
        # 'T' below
        if row <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col) + 'v')
        # 'L' left-above
        if col >= 1 and row >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col - 1) + 'v')
        # 'L' left-below
        if col >= 1 and row <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col - 1) + 'v')
        # 'L' right-above
        if col <= BOARD_SIZE-2 and row >= 1:
            TOUCHING_WALLS[wall].add(encode_loc(row - 1, col + 1) + 'v')
        # 'L' right-below
        if col <= BOARD_SIZE-2 and row <= BOARD_SIZE-2:
            TOUCHING_WALLS[wall].add(encode_loc(row + 1, col + 1) + 'v')

# Construct mapping from each wall to a list of the pairs of locations that it cuts off.
WALL_CUTS = {}
for wall in ALL_WALLS:
    (row, col) = parse_loc(wall[0:2])
    if wall[2] == 'v':
        WALL_CUTS[wall] = [[(row, col), (row, col + 1)],
                           [(row + 1, col), (row + 1, col + 1)]]
    elif wall[2] == 'h':
        WALL_CUTS[wall] = [[(row, col), (row + 1, col)],
                           [(row, col + 1), (row + 1, col + 1)]]

# Construct sets of goal positions. Player 0 begins at (0, 4) and tries to get to the last row. Player 1 is reversed.
GOALS = [set((BOARD_SIZE-1, col) for col in range(BOARD_SIZE)), set((0, col) for col in range(BOARD_SIZE))]


def create_adjacency_graph():
    adj = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            loc = (row, col)
            adj[loc] = set()
            if row > 0:
                adj[loc].add((row - 1, col))
            if row < BOARD_SIZE-1:
                adj[loc].add((row + 1, col))
            if col > 0:
                adj[loc].add((row, col - 1))
            if col < BOARD_SIZE-1:
                adj[loc].add((row, col + 1))
    return adj


class IllegalMove(Exception):
    pass


class Quoridor(object):
    """A class representing a single game of Quoridor.

    NOTATION:
        While grid locations are represented as a tuple of two zero-indexed numbers, the actual
        game notation uses letters for rows. 1-9 becomes a-i. a1 is the 'upper left' corner.
            (0,0) --> a1
            (1,6) --> g2
            (8,8) --> i9

        A *move* is denoted by the destination. Moving from b3 to c3 is just "c3"
        A *wall* is denoted with a 3-character string. The last character is "h" or "v" for
        Horizontal or Vertical walls
            -horizontal walls lie between rows and span 2 columns
            -vertical walls lie between columns and span 2 rows

            the other 2 characters specify the point of the top-left corner of the wall
            (highest row, lowest col)
            - must be between (0,0) and (7,7), or a1 and h8

        Here is a visual aid (imagine a1 is towards the upper-left):

            horizontal  vertical
            A  B        A || B
            ====          ||
            C  D        C || D

        For both of these, the point "A" will be used to denote the wall's location. For example, a
        fully notated wall might be 'd4h' for a horizontal wall that touches d4, d5, e4, and e5
    """

    def __init__(self):
        # Essential game properties.
        # Walls is a set of strings naming the walls that have been played.
        self.walls = set()
        # Each player represented by a list of (location, remaining walls).
        self.players = [[(0, BOARD_SIZE//2), 10],
                        [(BOARD_SIZE-1, BOARD_SIZE//2), 10]]
        # History is a list of tuples containing info useful for undoing moves. When moving pawns,
        # it is a tuple of (last_loc, new_loc). When placing walls, it contains the string.
        self.history = []
        self.current_player = 0

        # "Forward" history for redo.
        self.redo_stack = []

        # Efficiency helpers. There is one adjacency graph (a mesh connecting nodes to their neighbors), which is taken
        # over by the PathGraph objects. There are two PathGraph objects - one for each player. The job of the PathGraph
        # is to efficiently keep track of shortest paths from all positions on the board to the players' goal positions
        # as walls are added and removed that modify the graph.
        self._adjacency_graph = create_adjacency_graph()
        self._pathgraphs = [PathGraph(self._adjacency_graph, goals) for goals in GOALS]
        # "Open" walls are ones that can be played without physically overlapping previously played walls. The set of
        # legal wall placements is usually just this set of 'open' walls, but sometimes walls are additionally ruled out
        # as illegal if they cut off all of a player's paths to any goal.
        self._open_walls = set(ALL_WALLS)

    def __eq__(self, other):
        """Return True iff the current state of this Quoridor object matches another object (ignoring history)
        """
        return isinstance(other, Quoridor) and self.hash_key() == other.hash_key()

    def __hash__(self):
        """Provide a hash function that simply hashes the output of __key.
        """
        return hash(self.hash_key())

    ###########################
    # PUBLIC-FACING FUNCTIONS #
    ###########################

    def exec_move(self, mv, check_legal=True, is_redo=False):
        """Execute a move and update the state.

        The move mv is specified by a string - a position like "b5" for moving a pawn, and for walls a position plus "h"
        or "v" designation for horizontal and vertical walls, for instance "b5h"

        If check_legal is True, includes a call to self.is_legal and raises an IllegalMove exception if the move is not
        legal. If check_legal is False, we assume the move's legality is already verified. If an illegal move is played
        with check_legal False, behavior is undefined - it may crash, or break in other more subtle ways.

        If is_redo is False, the "redo" stack is reset since this move overwrites whatever history had been there.
        """
        if check_legal and not self.is_legal(mv):
            raise IllegalMove(mv)
        if len(mv) == 2:
            # Move the pawn.
            (row, col) = parse_loc(mv)
            # History includes both current position and next position so "undo" is just a matter of grabbing item [0]
            history_entry = (self.get_player()[0], (row, col))
            # Each player is stored as [(row, col), num_walls]. Update their position.
            self.get_player()[0] = (row, col)
        else:
            # Place a wall.
            self.walls.add(mv)
            # Each player is stored as [(row, col), num_walls]. Subtract 1 from count of their remaining walls.
            self.get_player()[1] -= 1
            # Cut the adjacency graph (call 'cut' on PathGraphs for each player)
            self._cut(mv)
            # Record just the wall string in history.
            history_entry = mv
            # Update list of 'open' wall spaces.
            self._open_walls -= INTERSECTING_WALLS[mv]

        self.history.append(history_entry)
        self.current_player = (self.current_player + 1) % len(self.players)
        if not is_redo:
            self.redo_stack = []

    def temp_move(self, mv):
        """Execute the given move in a `with` context, where it automatically undoes itself when
           the `with` is complete.
        """
        return Quoridor.TempMove(self, mv)

    def undo(self, allow_redo=True):
        """Undo the last move.
        """
        if len(self.history) > 0:
            prev_player = (len(self.players) + self.current_player - 1) % len(self.players)
            last_entry = self.history.pop()
            if type(last_entry) is tuple:
                self.players[prev_player][0] = last_entry[0]
                # Append move string to redo stack.
                if allow_redo:
                    self.redo_stack.append(encode_loc(*last_entry[1]))
            else:
                # Remove the wall.
                self.walls.discard(last_entry)
                # Add 1 back to count of remaining walls.
                self.players[prev_player][1] += 1
                # Repair the adjacency graph.
                self._uncut(last_entry)
                # Append wall string to redo stack.
                if allow_redo:
                    self.redo_stack.append(last_entry)
                # Add back in 'open' walls.
                for maybe_open in INTERSECTING_WALLS[last_entry]:
                    # Each of the walls touching 'last_entry' may be ruled out by some other played wall.
                    if all(w not in self.walls for w in INTERSECTING_WALLS[maybe_open]):
                        self._open_walls.add(maybe_open)
            self.current_player = prev_player

    def redo(self):
        """Play forward from series of calls to undo().
        """
        if len(self.redo_stack) > 0:
            self.exec_move(self.redo_stack.pop(), is_redo=True)

    def is_legal(self, mv):
        """Return True iff the given move is legal.
        """
        if type(mv) is not str:
            return False
        if len(mv) == 2:
            # This is pawn movement like 'b5' or 'h8'
            (row, col) = parse_loc(mv)
            # Check that move is on the board
            if row < 0 or col < 0 or row > BOARD_SIZE-1 or col > BOARD_SIZE-1:
                return False
            # Note: we cannot simply assert that the move is 1 space away due to possible jumping situations (at most 2
            # spaces away in a 2-player game)
            cur_loc = self.get_player()[0]
            other_player_locs = set([p[0] for p in self.players])
            adjacent_player_locs = other_player_locs & self._adjacency_graph[cur_loc]
            # Check that another player is not in the position.
            if (row, col) in adjacent_player_locs:
                return False
            # Check for jumping situation if there is another player adjacent to the current player
            if len(adjacent_player_locs) > 0:
                for (pr, pc) in adjacent_player_locs:
                    if pr == cur_loc[0]:
                        # Check horizontal jump (same row).
                        col_diff = pc - cur_loc[1]
                        one_further = (pr, pc + col_diff)
                        diagonals = [(pr - 1, pc), (pr + 1, pc)]
                        if one_further in self._adjacency_graph[(pr, pc)]:
                            if (row, col) == one_further and one_further not in other_player_locs:
                                return True
                        else:
                            # The simple jump is blocked. Check diagonals.
                            for d in diagonals:
                                if (row, col) == d and d in self._adjacency_graph[(pr, pc)] \
                                        and d not in other_player_locs:
                                    return True
                    elif pc == cur_loc[1]:
                        # Check vertical jump (same col).
                        row_diff = pr - cur_loc[0]
                        one_further = (pr + row_diff, pc)
                        diagonals = [(pr, pc - 1), (pr, pc + 1)]
                        if one_further in self._adjacency_graph[(pr, pc)]:
                            if (row, col) == one_further and one_further not in other_player_locs:
                                return True
                        else:
                            # The simple jump is blocked. Check diagonals.
                            for d in diagonals:
                                if (row, col) == d and d in self._adjacency_graph[(pr, pc)] \
                                        and d not in other_player_locs:
                                    return True
            # Having considered jumps, check that target is adjacent to current position, taking
            # into account walls.
            if (row, col) not in self._adjacency_graph[cur_loc]:
                return False
        elif len(mv) == 3:
            (row, col) = parse_loc(mv[0:2])
            # Check that player has a wall to spare.
            if not self.get_player()[1] > 0:
                return False
            # Check that wall is on the board.
            if row < 0 or col < 0 or row > BOARD_SIZE-2 or col > BOARD_SIZE-2:
                return False
            # Check that wall does not physically intersect with any wall that has been played.
            if mv not in self._open_walls:
                return False
            # (slow) check that wall does not cut off all paths to some goal for any player.
            # Efficiency note 1: only needs to be checked if 'mv' is 'touching' to some existing wall
            touching_wall, shortest_path_cut = False, False
            for wall in TOUCHING_WALLS[mv]:
                if wall in self.walls:
                    touching_wall = True
                    break
            # Efficiency note 2: we may skip checking this wall if it doesn't cut any player's shortest path.
            if touching_wall:
                cuts = WALL_CUTS[mv]
                for (player, graph) in zip(self.players, self._pathgraphs):
                    current = player[0]
                    for next in graph.get_path(current):
                        if [current, next] in cuts or [next, current] in cuts:
                            # The wall cuts this player's path..
                            shortest_path_cut = True
                            break
                        current = next
                    if shortest_path_cut:
                        break
            # After 2 tests, it's plausible that this wall cuts off a player. Do a full (slow) call to cut() to check.
            if touching_wall and shortest_path_cut:
                self._cut(mv)
                has_path = all(graph.has_path(player[0]) for (player, graph) in zip(self.players, self._pathgraphs))
                self._uncut(mv)
                return has_path
            return True
        else:
            return False
        return True

    def get_winner(self):
        """Return the index of the winning player, or None if nobody has won yet.
        """
        for i, p in enumerate(self.players):
            if p[0] in GOALS[i]:
                return i
        return None

    def all_legal_moves(self, partial_check=False):
        (row, col) = self.get_player()[0]
        legal_moves = []
        # Only check moves within +/- 2 spaces of the pawn (in case jump is legal)
        for r in range(max(0, row - 2), min(BOARD_SIZE-1, row + 2) + 1):
            for c in range(max(0, col - 2), min(BOARD_SIZE-1, col + 2) + 1):
                mv = encode_loc(r, c)
                if self.is_legal(mv):
                    legal_moves.append(mv)
        # If no walls available, just return legal moves
        if self.get_player()[1] == 0:
            return legal_moves
        # Only check legality of 'open' wall spaces. If partial_check is True, don't do a full graph-cutting check and
        # just assume all 'open' walls are legal.
        if partial_check:
            legal_walls = list(self._open_walls)
        else:
            legal_walls = [w for w in self._open_walls if self.is_legal(w)]
        return legal_moves + legal_walls

    def get_player(self, player_idx=None):
        """Return the [(row, col), num_walls] list of state for the current player (or the player at index player_idx).

        Note: the returned state is mutable! Calling get_player()[1] -= 1, for instance, subtracts 1 wall from the
        current player.
        """
        return self.players[self.current_player if player_idx is None else player_idx]

    def get_graph(self, player_idx=None):
        """Return the PathGraph object associated with the current player (or the player at index player_idx if given)
        """
        return self._pathgraphs[self.current_player if player_idx is None else player_idx]

    def hash_key(self):
        """Create a unique identifier for the present state of the game (history-free).

        A hashable key must be immutable, hence the use of tuples and frozen sets.
        """
        return (self.current_player, frozenset(self.walls), tuple(map(tuple, self.players)))

    def save(self, filename, header=""):
        """Save history of moves to a file.
        """
        with open(filename, "w") as f:
            if header:
                f.write("# " + header + "\n")
            f.write(str(len(self.players)) + "\n")
            for mv in self.history:
                if type(mv) is tuple:
                    mv = encode_loc(*mv[1])
                f.write(mv + "\n")

    @classmethod
    def load(cls, filename, undo_all=False):
        game = cls()
        with open(filename, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        # Drop header line
        if lines[0][0] == "#":
            lines = lines[1:]
        # First line is # players -- currently only supporting 2
        if int(lines[0]) != 2:
            raise ValueError("Only 2 players allowed.")
        # Execute all moves in the file
        for mv in lines[1:]:
            game.exec_move(mv)
        # If 'undo_all' is True, undo all the moves so they're sitting on the 'redo' stack but the game is at the start
        if undo_all:
            while len(game.history) > 0:
                game.undo(allow_redo=True)
        return game

    ####################
    # HELPER FUNCTIONS #
    ####################

    def _cut(self, wall):
        """Cut the adjacency graph with the given wall.
        """
        for graph in self._pathgraphs:
            graph.cut(WALL_CUTS[wall])

    def _uncut(self, wall):
        """Repair adjacency graph (undo `_cut(wall)`)
        """
        for graph in self._pathgraphs:
            graph.uncut(WALL_CUTS[wall])

    class TempMove:
        """Class providing do/undo functionality in a with statement.

        For example:

            game = Quoridor()
            game.exec_move("b5")
            with game.temp_move("h5"):
                print game.history[-1] # shows move to h5
            print game.history[-1] # shows move to b5
        """
        def __init__(self, game, mv):
            self.game = game
            self.mv = mv

        def __enter__(self, check_legal=False):
            # __enter__ is called when the with statement begins. Execute the move with is_redo set to True as a hack to
            # prevent overwriting the redo stack
            # TODO - cache
            self.game.exec_move(self.mv, check_legal=check_legal, is_redo=True)
            return self.game

        def __exit__(self, type, value, traceback):
            # __exit__ is called when the with statement ends. Undo the move without touching the redo stack.
            self.game.undo(allow_redo=False)
