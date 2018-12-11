import unittest
from quoridor import *


class TestQuoridor(unittest.TestCase):

    def setUp(self):
        self.game = Quoridor()

    def testError1(self):
        self.game.exec_move('h5h')
        self.game.exec_move('h4v')
        self.game.exec_move('a4')
        with self.assertRaises(IllegalMove) as context:
            self.game.exec_move('h6v')

if __name__ == '__main__':
    unittest.main()
