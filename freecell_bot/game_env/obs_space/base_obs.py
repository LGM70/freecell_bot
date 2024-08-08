from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces

class BaseObsSpace(ABC):
    """Base abstract class for freecell observation space"""

    @classmethod
    @abstractmethod
    def get_gym_space(cls) -> spaces.Space:
        """
        Generate a Gym space as the observation space

        Returns:
            spaces.Space: observation space
        """

    @classmethod
    @abstractmethod
    def deal(cls) -> np.ndarray:
        """
        Randomly generate a new freecell game

        Returns:
            np.ndarray: observation of the new game
        """

    @classmethod
    @abstractmethod
    def calc_ranks(cls, cards: np.ndarray) -> int | np.ndarray:
        """
        Calculate the ranks of a card or multiple cards

        Args:
            cards (np.ndarray): card or cards

        Returns:
            int | np.ndarray: rank or ranks
        """

    @classmethod
    @abstractmethod
    def calc_colors(cls, cards: np.ndarray) -> int | np.ndarray:
        """
        Calculate the colors of a card or multiple cards

        Args:
            cards (np.ndarray): card or cards

        Returns:
            int | np.ndarray: color or colors
        """

    @classmethod
    @abstractmethod
    def calc_suits(cls, cards: np.ndarray) -> int | np.ndarray:
        """
        Calculate the suits of a card or multiple cards

        Args:
            cards (np.ndarray): card or cards

        Returns:
            int | np.ndarray: suit or suits
        """

    @classmethod
    @abstractmethod
    def create_card(cls, rank: int, suit: int) -> np.ndarray:
        """
        Generate a card with given rank and suit

        Args:
            rank (int): rank
            suit (int): suit

        Returns:
            np.ndarray: generated card
        """

    @classmethod
    @abstractmethod
    def has_empty_card(cls, cards: np.ndarray) -> bool:
        """
        Check if any of given cards is empty

        Args:
            cards (np.ndarray): cards

        Returns:
            bool: True if has empty card
        """

    @classmethod
    @abstractmethod
    def get_cards(cls, obs: np.ndarray, loc_type: str, idx: int) -> np.ndarray:
        """
        Get cards at given location with no empty cards

        Args:
            obs (np.ndarray): observation
            loc_type (str): location type
            idx (int): location index

        Returns:
            np.ndarray: cards
        """

    @classmethod
    @abstractmethod
    def remove_cards(cls, obs: np.ndarray, loc_type: str, idx: int, cnt: int) -> None:
        """
        Remove cards at given location without verification

        Args:
            obs (np.ndarray): observation
            loc_type (str): location type
            idx (int): location index
            cnt (int): number of cards to be removed
        """

    @classmethod
    @abstractmethod
    def put_cards(cls, obs:np.ndarray, cards: np.ndarray, loc_type: str, idx: int) -> None:
        """
        Put cards at given location without verification

        Args:
            obs (np.ndarray): observation
            cards (np.ndarray): cards to put
            loc_type (str): location type
            idx (int): location index
        """
