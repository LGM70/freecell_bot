import unittest
import numpy as np
from .freecell_env import FreeCellEnv
from .standard_spec import StandardSpec as Spec
# from .obs_space.compact_obs import CompactObsSpace as ObsSpace
from .obs_space.one_hot_obs import OneHotObsSpace as ObsSpace
from .action_space.tuple_action import TupleActionSpace as ActionSpace

class TestEnv(unittest.TestCase):
    """
    Unit test class for entire freecell environment
    
    Assert using TupleActionSpace
    """

    def setUp(self):
        # test using a smaller set of cards and game for convenience
        Spec.ranks = ['A', '2', '3', '4']
        Spec.num_cards = len(Spec.suits) * len(Spec.ranks)
        Spec.num_cascades = 4

        # setup the game
        self.env = FreeCellEnv(ObsSpace, ActionSpace)
        self.env.reset()
        # remove all cards
        for i in range(Spec.num_cascades):
            self.env._obs_space.remove_cards(self.env._obs, 'cascade', i,
                self.env._obs_space.get_cards(self.env._obs, 'cascade', i).shape[0]
            )
        # put cards on cascades
        cascade0 = np.zeros((2, ) + self.env._obs.shape[-1:], dtype=self.env._obs.dtype)
        cascade0[0] = self.env._obs_space.create_card(3, 2) # 4 of Hearts
        cascade0[1] = self.env._obs_space.create_card(2, 1) # 3 of Clubs
        self.env._obs_space.put_cards(self.env._obs, cascade0, 'cascade', 0)
        cascade1 = np.zeros((3, ) + self.env._obs.shape[-1:], dtype=self.env._obs.dtype)
        cascade1[0] = self.env._obs_space.create_card(3, 1) # 4 of Clubs
        cascade1[1] = self.env._obs_space.create_card(1, 2) # 2 of Hearts
        cascade1[2] = self.env._obs_space.create_card(0, 1) # Ace of Clubs
        self.env._obs_space.put_cards(self.env._obs, cascade1, 'cascade', 1)
        # cascade2 is empty
        cascade3 = np.zeros((3, ) + self.env._obs.shape[-1:], dtype=self.env._obs.dtype)
        cascade3[0] = self.env._obs_space.create_card(0, 2) # Ace of Hearts
        cascade3[1] = self.env._obs_space.create_card(2, 3) # 3 of Spades
        cascade3[2] = self.env._obs_space.create_card(1, 1) # 2 of Clubs
        self.env._obs_space.put_cards(self.env._obs, cascade3, 'cascade', 3)
        # put cards on cells
        self.env._obs_space.put_cards(
            self.env._obs,
            self.env._obs_space.create_card(2, 2)[np.newaxis, :], # 3 of Hearts
            'cell', 0
        )
        self.env._obs_space.put_cards(
            self.env._obs,
            self.env._obs_space.create_card(3, 3)[np.newaxis, :], # 4 of Spades
            'cell', 1
        )
        # put cards on foundations
        self.env._obs_space.put_cards(
            self.env._obs,
            self.env._obs_space.create_card(3, 0)[np.newaxis, :], # 4 of Diamonds
            'foundation', 0
        )
        self.env._obs_space.put_cards(
            self.env._obs,
            self.env._obs_space.create_card(1, 3)[np.newaxis, :], # 2 of Spades
            'foundation', 3
        )
        # print(self.env.render())
        # The game now looks like this
        # cells:        H3 S4 -- --
        # foundations:  D4 C- H- S2
        # cascades:     H4 C4 -- HA
        #               C3 H2    S3
        #                  CA    C2

    def test_everything_in_one_episode(self):
        self.assertEqual(self.env._max_move(), 6)
        # cascade and cascade
        _, _, _, _, result = self.env.step((1, 0))
        self.assertEqual(result['status'], 'success')
        _, _, _, _, result = self.env.step((3, 1))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((1, 2))
        self.assertEqual(result['status'], 'success')
        _, _, _, _, result = self.env.step((1, 2))
        self.assertEqual(result['status'], 'failure')
        # cascade and cell
        _, _, _, _, result = self.env.step((3, 7))
        self.assertEqual(result['status'], 'success')
        _, _, _, _, result = self.env.step((2, 5))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((1, 6))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((6, 0))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((5, 0))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((5, 1))
        self.assertEqual(result['status'], 'success')
        # cascade and foundation
        _, _, _, _, result = self.env.step((11, 3))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((0, 10))
        self.assertEqual(result['status'], 'failure')
        _, _, _, _, result = self.env.step((0, 9))
        self.assertEqual(result['status'], 'success')
        # cell and foundation
        _, _, _, _, result = self.env.step((8, 5))
        self.assertEqual(result['status'], 'success')
        _, _, _, _, result = self.env.step((7, 9))
        self.assertEqual(result['status'], 'success')
        # finish the game
        self.env.step((5, 8))
        self.env.step((3, 11))
        self.env.step((1, 11))
        self.env.step((3, 10))
        self.env.step((0, 10))
        self.env.step((0, 9))
        self.env.step((2, 9))
        self.env.step((4, 10))
        _, _, terminated, _, _ = self.env.step((0, 10))
        self.assertTrue(terminated)

if __name__ == '__main__':
    unittest.main()
