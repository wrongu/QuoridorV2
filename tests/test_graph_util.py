import unittest
from quoridor import create_adjacency_graph, WALL_CUTS
from graph_util import PathGraph


class TestPathGraph(unittest.TestCase):

    def setUp(self):
        self.graph = create_adjacency_graph()
        self.pg = PathGraph(self.graph, [(0, i) for i in range(9)])

    def testCutNoPathChange(self):
        init_paths = self.pg._downhill.items()
        # Make a horizontal cut
        self.pg.cut([[(3, 4), (3, 5)]])
        self.assertItemsEqual(init_paths, self.pg._downhill.items())

    def testCutSidestep(self):
        init_dist = self.pg._dist[(4, 4)]
        self.pg.cut([[(3, 4), (4, 4)]])
        self.assertIn(self.pg._downhill[(4, 4)], [(4, 3), (4, 5)])
        self.assertEqual(self.pg._dist[(4, 4)], init_dist + 1)

    def testFullCutoff(self):
        self.pg.cut([[(3, 4), (3, 3)]])
        self.pg.cut([[(3, 4), (3, 5)]])
        self.pg.cut([[(3, 4), (2, 4)]])
        self.pg.cut([[(3, 4), (4, 4)]])
        self.assertIsNone(self.pg._downhill[(3, 4)])

    def testCutWithinCut(self):
        self.pg.cut([[(3, 4), (3, 3)]])
        self.pg.cut([[(3, 4), (2, 4)]])
        self.pg.cut([[(3, 4), (4, 4)]])
        self.pg.cut([[(3, 5), (3, 6)]])
        self.pg.cut([[(3, 5), (2, 5)]])
        self.pg.cut([[(3, 5), (4, 5)]])
        # (3,4) and (3,5) are in a box disconnected from everything else.
        self.assertEqual(len(self.graph[(3, 4)]), 1)
        self.assertEqual(len(self.graph[(3, 5)]), 1)
        self.assertIn((3, 5), self.graph[(3, 4)])
        self.assertIn((3, 4), self.graph[(3, 5)])
        self.assertIsNone(self.pg._downhill[(3, 4)])
        self.assertIsNone(self.pg._downhill[(3, 5)])
        self.assertEqual(len(self.pg._uphill[(3, 4)]), 0)
        self.assertEqual(len(self.pg._uphill[(3, 5)]), 0)
        # Now cut between (3, 4) and (3, 5)
        self.pg.cut([[(3, 4), (3, 5)]])

    def testEncloseSink(self):
        self.pg.cut([[(0, 4), (0, 5)]])
        self.pg.cut([[(0, 5), (0, 6)]])
        self.pg.cut([[(1, 4), (1, 5)]])
        self.pg.cut([[(1, 5), (1, 6)]])
        self.pg.cut([[(1, 5), (2, 5)]])
        self.assertEqual(self.pg.get_distance((1, 5)), 1)

    def testSimpleUncut(self):
        init_paths = self.pg._downhill.items()
        self.pg.cut([[(3, 3), (4, 3)], [(3, 4), (4, 4)]])
        self.pg.uncut([[(3, 3), (4, 3)], [(3, 4), (4, 4)]])
        self.assertItemsEqual(init_paths, self.pg._downhill.items())

    def testUncutEnclosed(self):
        # Make a bunch of cuts including closing off a space, then undo and assert that the graph
        # is restored after each uncut.
        prev_downhills = []
        prev_uphills = []
        pairs = [[(3, 3), (3, 4)],
                 [(3, 3), (4, 3)],
                 [(3, 3), (2, 3)],
                 [(3, 3), (3, 2)]]
        for pair in pairs:
            prev_downhills.append(self.pg._downhill.items())
            prev_uphills.append(self.pg._downhill.items())
            self.pg.cut([pair])
        for pair in reversed(pairs):
            self.pg.uncut([pair])
            self.assertItemsEqual(prev_downhills.pop(), self.pg._downhill.items())
            self.assertItemsEqual(prev_uphills.pop(), self.pg._downhill.items())

if __name__ == '__main__':
    unittest.main()
