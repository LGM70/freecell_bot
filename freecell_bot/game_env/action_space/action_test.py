import unittest
from .tuple_action import TupleActionSpace as ActionSpace
from ..standard_spec import StandardSpec as Spec

class TestActionSpace(unittest.TestCase):
    """
    Unit test class for action space
    
    Import self-defined action space as ActionSpace to test
    """
    def test_random_action(self):
        random_action = ActionSpace.get_gym_space().sample()
        parsed_action = ActionSpace.parse_action(random_action)
        lengths = {
            'cascade': Spec.num_cascades,
            'cell': Spec.num_cells,
            'foundation': Spec.num_foundations
        }
        for i in range(2):
            loc_type, loc_idx = parsed_action[i * 2 : (i + 1) * 2]
            self.assertIn(loc_type, lengths.keys(), 'wrong location type')
            self.assertGreaterEqual(loc_idx, 0, 'negative location index')
            self.assertLess(loc_idx, lengths[loc_type], 'location index out of range')

if __name__ == '__main__':
    unittest.main()
