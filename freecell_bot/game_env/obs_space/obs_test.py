import unittest
# from .one_hot_obs import OneHotObsSpace as ObsSpace
from .compact_obs import CompactObsSpace as ObsSpace
from ..standard_spec import StandardSpec as Spec
import numpy as np

class TestObsSpace(unittest.TestCase):
    """
    Unit test class for observation space
    
    Import self-defined observation space as ObsSpace to test
    """

    def setUp(self):
        self.obs = ObsSpace.deal()
    
    def test_cards(self):
        random_ranks = np.random.randint(len(Spec.ranks), size=3)
        random_suits = np.random.randint(len(Spec.suits), size=3)
        cards = np.zeros((3,) + self.obs.shape[-1:], dtype=self.obs.dtype)
        for i in range(3):
            cards[i] = ObsSpace.create_card(random_ranks[i], random_suits[i])
        self.assertTrue(np.all(ObsSpace.calc_ranks(cards[0]) == random_ranks[0]), 'Wrong card rank')
        self.assertTrue(np.all(ObsSpace.calc_suits(cards[0]) == random_suits[0]), 'Wrong card suit')
        self.assertTrue(np.all(ObsSpace.calc_colors(cards[0]) == random_suits[0] % 2), 'Wrong card color')
        self.assertTrue(np.all(ObsSpace.calc_ranks(cards) == random_ranks), 'Wrong card rank')
        self.assertTrue(np.all(ObsSpace.calc_suits(cards) == random_suits), 'Wrong card suit')
        self.assertTrue(np.all(ObsSpace.calc_colors(cards) == random_suits % 2), 'Wrong card color')

    def test_get_cascade(self):
        for i in range(Spec.num_cascades):
            self.assertFalse(ObsSpace.has_empty_card(ObsSpace.get_cards(self.obs, 'cascade', i)))

    def test_deal(self):
        for i in range(Spec.num_cells):
            self.assertTrue(ObsSpace.has_empty_card(ObsSpace.get_cards(self.obs, 'cell', i)))
        for i in range(Spec.num_foundations):
            self.assertTrue(ObsSpace.has_empty_card(ObsSpace.get_cards(self.obs, 'foundation', i)))
        cards = np.array([])
        for i in range(Spec.num_cascades):
            cascade = ObsSpace.get_cards(self.obs, 'cascade', i)
            cascade = ObsSpace.calc_suits(cascade) * len(Spec.ranks) + ObsSpace.calc_ranks(cascade)
            cards = np.append(cards, cascade)
        self.assertEqual(cards.shape[0], Spec.num_cards, 'Wrong number of cards')
        self.assertEqual(np.unique(cards).shape[0], Spec.num_cards, 'Have duplicate cards')

    def test_put_cards(self):
        orig_len = ObsSpace.get_cards(self.obs, 'cascade', 0).shape[0]
        random_ranks = np.random.randint(len(Spec.ranks), size=3)
        random_suits = np.random.randint(len(Spec.suits), size=3)
        cards = np.zeros((3,) + self.obs.shape[-1:], dtype=self.obs.dtype)
        for i in range(3):
            cards[i] = ObsSpace.create_card(random_ranks[i], random_suits[i])
        ObsSpace.put_cards(self.obs, cards, 'cascade', 0)
        curr_len = ObsSpace.get_cards(self.obs, 'cascade', 0).shape[0]
        self.assertEqual(orig_len + 3, curr_len, 'Put wrong number of cards')

    def test_remove_cards(self):
        orig_len = ObsSpace.get_cards(self.obs, 'cascade', Spec.num_cascades - 1).shape[0]
        ObsSpace.remove_cards(self.obs, 'cascade', Spec.num_cascades - 1, 1)
        curr_len = ObsSpace.get_cards(self.obs, 'cascade', Spec.num_cascades - 1).shape[0]
        self.assertEqual(orig_len - 1, curr_len, 'Remove wrong number of cards')

if __name__ == '__main__':
    unittest.main()
