import unittest
from quoridor import create_adjacency_graph
from graph_util import PathGraph


class TestPathGraph(unittest.TestCase):

    def setUp(self):
        self.graph = create_adjacency_graph()
        self.pg = PathGraph(self.graph, [(0, i) for i in range(9)])

    def testCutNoPathChange(self):
        init_paths = self.pg._downhill.items()
        # Make a horizontal cut
        self.pg.cut([(3, 4), (3, 5)])
        self.assertItemsEqual(init_paths, self.pg._downhill.items())

    def testCutSidestep(self):
        init_dist = self.pg._downhill[(4, 4)][0]
        self.pg.cut([(3, 4), (4, 4)])
        self.assertIn(self.pg._downhill[(4, 4)][1], [(4, 3), (4, 5)])
        self.assertEqual(self.pg._downhill[(4, 4)][0], init_dist + 1)

    def testFullCutoff(self):
        self.pg.cut([(3, 4), (3, 3)])
        self.pg.cut([(3, 4), (3, 5)])
        self.pg.cut([(3, 4), (2, 4)])
        self.pg.cut([(3, 4), (4, 4)])
        self.assertIsNone(self.pg._downhill[(3, 4)])

    def testCutWithinCut(self):
        self.pg.cut([(3, 4), (3, 3)])
        self.pg.cut([(3, 4), (2, 4)])
        self.pg.cut([(3, 4), (4, 4)])
        self.pg.cut([(3, 5), (3, 6)])
        self.pg.cut([(3, 5), (2, 5)])
        self.pg.cut([(3, 5), (4, 5)])
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
        self.pg.cut([(3, 4), (3, 5)])

if __name__ == '__main__':
    unittest.main()
