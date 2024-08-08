from typing import TypeAlias
from gymnasium import spaces
from .base_action import BaseActionSpace
from ..standard_spec import StandardSpec as Spec

class TupleActionSpace(BaseActionSpace):
    """
    (int, int) tuple as the freecell action space
    
    every tuple represents a move from the first component to the second component
    """

    _action_type: TypeAlias = tuple[int, int]

    @classmethod
    def get_gym_space(cls) -> spaces.Space:
        num_locations = Spec.num_cascades + Spec.num_cells + Spec.num_foundations
        gym_space = spaces.Tuple([
            spaces.Discrete(num_locations),
            spaces.Discrete(num_locations)
        ])
        return gym_space

    @classmethod
    def parse_action(cls, action: _action_type) -> tuple[Spec.loc_types, int, Spec.loc_types, int]:
        source, dest = action
        return TupleActionSpace._parse_location(source) + TupleActionSpace._parse_location(dest)

    @classmethod
    def _parse_location(cls, location: int) -> tuple[Spec.loc_types, int]:
        """
        Helper method for parsing action

        Raises:
            ValueError: when location is out of range

        Returns:
            Spec.loc_types: location type
            int: location index
        """
        if location < 0 or location > Spec.num_cascades + Spec.num_cells + Spec.num_foundations:
            raise ValueError(f'location cannot be {location}')
        if location < Spec.num_cascades:
            return 'cascade', location
        if location < Spec.num_cascades + Spec.num_cells:
            return 'cell', location - Spec.num_cascades
        return 'foundation', location - Spec.num_cascades - Spec.num_cells
