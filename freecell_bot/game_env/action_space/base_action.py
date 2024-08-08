from abc import ABC, abstractmethod
from typing import Any
from gymnasium import spaces

class BaseActionSpace(ABC):
    """Base abstract class for freecell action space"""

    @classmethod
    @abstractmethod
    def get_gym_space(cls) -> spaces.Space:
        """
        Generate a Gym space as the action space

        Returns:
            spaces.Space: action space
        """

    @classmethod
    @abstractmethod
    def parse_action(cls, action: Any) -> tuple[str, int, str, int]:
        """
        Parse action into a tuple of length 4, containing source and destination info
        
        Returns:
            str: source type
            int: source index
            str: destination type
            int: destination index
        """
