from graph_util import _is_reachable


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

ALL_WALLS = set()
ALL_POSITIONS = set()
for row in range(9):
    for col in range(9):
        ALL_POSITIONS.add(encode_loc(row, col))
        if row < 8 and col < 8:
            ALL_WALLS.add(encode_loc(row, col) + 'h')
            ALL_WALLS.add(encode_loc(row, col) + 'v')

# Construct mapping from each wall to the set of walls that it physically rules out.
TOUCHING_WALLS = {}
for wall in ALL_WALLS:
    TOUCHING_WALLS[wall] = [wall, cross(wall)]
    (row, col) = parse_loc(wall[0:2])
    if wall[2] == 'v':
        if row > 0:
            TOUCHING_WALLS[wall].append(encode_loc(row - 1, col) + 'v')
        if row < 7 and wall[2]:
            TOUCHING_WALLS[wall].append(encode_loc(row + 1, col) + 'v')
    elif wall[2] == 'h':
        if col > 0:
            TOUCHING_WALLS[wall].append(encode_loc(row, col - 1) + 'h')
        if col < 7:
            TOUCHING_WALLS[wall].append(encode_loc(row, col + 1) + 'h')

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

# Construct sets of goal positions.
GOALS = [set([(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)]),
         set([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)])]


def create_adjacency_graph():
    adj = {}
    for row in range(9):
        for col in range(9):
            loc = (row, col)
            adj[loc] = set()
            if row > 0:
                adj[loc].add((row - 1, col))
            if row < 8:
                adj[loc].add((row + 1, col))
            if col > 0:
                adj[loc].add((row, col - 1))
            if col < 8:
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
        self.players = [[(0, 4), 10],
                        [(8, 4), 10]]
        # History is a list of tuples containing info useful for undoing moves. When moving pawns,
        # it is a tuple of (last_loc, new_loc). When placing walls, it contains the string.
        self.history = []
        self.current_player = 0

        # "Forward" history for redo.
        self.redo_stack = []

        # Efficiency helpers.
        self._adjacency_graph = create_adjacency_graph()

    def __eq__(self, other):
        return isinstance(other, Quoridor) and self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return (self.current_player, frozenset(self.walls), tuple(map(tuple, self.players)))

    ###########################
    # PUBLIC-FACING FUNCTIONS #
    ###########################

    def exec_move(self, mv, check_legal=True, is_redo=False):
        """Execute a move and update the state.
        """
        if check_legal and not self.is_legal(mv):
            raise IllegalMove(mv)
        if len(mv) == 2:
            # Move the pawn.
            (row, col) = parse_loc(mv)
            history_entry = (self.players[self.current_player][0], (row, col))
            self.players[self.current_player][0] = (row, col)
        else:
            # Place a wall.
            self.walls.add(mv)
            # Subtract 1 from count of remaining walls for this player.
            self.players[self.current_player][1] -= 1
            # Cut the adjacency graph.
            self._cut(mv)
            # Record just the wall string in history.
            history_entry = mv

        self.history.append(history_entry)
        self.current_player = (self.current_player + 1) % len(self.players)
        if not is_redo:
            self.redo_stack = []

    def undo(self):
        """Undo the last move.
        """
        if len(self.history) > 0:
            prev_player = (len(self.players) + self.current_player - 1) % len(self.players)
            last_entry = self.history.pop()
            if type(last_entry) is tuple:
                self.players[prev_player][0] = last_entry[0]
                # Append move string to redo stack.
                self.redo_stack.append(encode_loc(*last_entry[1]))
            else:
                # Remove the wall.
                self.walls.discard(last_entry)
                # Add 1 back to count of remaining walls.
                self.players[prev_player][1] += 1
                # Repair the adjacency graph.
                self._uncut(last_entry)
                # Append wall string to redo stack.
                self.redo_stack.append(last_entry)
            self.current_player = prev_player

    def redo(self):
        """Play forward from series of calls to undo().
        """
        if len(self.redo_stack) > 0:
            self.exec_move(self.redo_stack.pop(), is_redo=True)

    def is_legal(self, mv):
        """Return True iff the given move is legal.
        """
        if len(mv) == 2:
            (row, col) = parse_loc(mv)
            # Check that move is on the board
            if row < 0 or col < 0 or row > 8 or col > 8:
                return False
            cur_loc = self.players[self.current_player][0]
            other_player_locs = set([p[0] for p in self.players])
            adjacent_player_locs = other_player_locs & self._adjacency_graph[cur_loc]
            # Check that another player is not in the position.
            if (row, col) in adjacent_player_locs:
                return False
            # Check for jumping situation.
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
            if not self.players[self.current_player][1] > 0:
                return False
            # Check that wall is on the board.
            if row < 0 or col < 0 or row > 7 or col > 7:
                return False
            # Check that wall does not physically intersect with any wall that has been played.
            for touching in TOUCHING_WALLS[mv]:
                if touching in self.walls:
                    return False
            # Check that wall does not cut off all paths to some goal for any player.
            has_path = True
            self._cut(mv)
            for goals, player in zip(GOALS, self.players):
                if not _is_reachable(self._adjacency_graph, player[0], goals):
                    has_path = False
                    break
            self._uncut(mv)
            return has_path
        else:
            return False
        return True

    def get_winner(self):
        for i, p in enumerate(self.players):
            if p[0] in GOALS[i]:
                return i
        return None

    def all_legal_moves(self):
        return filter(self.is_legal, ALL_WALLS | ALL_POSITIONS)

    def save(self, filename):
        """Save moves to a file.
        """
        with open(filename, "w") as f:
            f.write(str(len(self.players)) + "\n")
            for mv in self.history:
                if type(mv) is tuple:
                    mv = encode_loc(*mv[1])
                f.write(mv + "\n")

    @classmethod
    def load(cls, filename):
        game = cls()
        with open(filename, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        if int(lines[0]) != 2:
            raise ValueError("Only 2 players allowed.")
        for mv in lines[1:]:
            game.exec_move(mv)
        return game

    ####################
    # HELPER FUNCTIONS #
    ####################

    def _cut(self, wall):
        """Cut the adjacency graph with the given wall.
        """
        for (loc1, loc2) in WALL_CUTS[wall]:
            self._adjacency_graph[loc1].discard(loc2)
            self._adjacency_graph[loc2].discard(loc1)

    def _uncut(self, wall):
        """Repair adjacency graph (undo `_cut(wall)`)
        """
        for (loc1, loc2) in WALL_CUTS[wall]:
            self._adjacency_graph[loc1].add(loc2)
            self._adjacency_graph[loc2].add(loc1)
