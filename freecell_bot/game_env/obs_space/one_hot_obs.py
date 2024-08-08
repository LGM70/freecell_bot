from typing import TypeAlias
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from .base_obs import BaseObsSpace
from ..standard_spec import StandardSpec as Spec

class OneHotObsSpace(BaseObsSpace):
    """
    One-hot freecell observation space with shape (10, 19, 52)
    
    obs[:8], obs[-2], obs[-1] represents cascades, cells, and foundations, respectively
    2nd dimension has a length of 19 because it's the max possible length of cascade
    3nd dimension is one-hot representation of cards
    
    This observation space is designed for CNN-based agents
    """

    _obs_type: TypeAlias = npt.NDArray[np.int8]

    @classmethod
    def deal(cls) -> _obs_type:
        cards = np.arange(Spec.num_cards)
        np.random.shuffle(cards)

        obs = np.zeros((
            Spec.num_cascades + 2,
            Spec.num_cards // Spec.num_cascades + len(Spec.ranks),
            Spec.num_cards
        ), dtype=np.int8)

        cards_per_cascade = Spec.num_cards // Spec.num_cascades
        additional_cards = Spec.num_cards % Spec.num_cascades
        for i in range(Spec.num_cascades):
            if i < additional_cards:
                cards_in_cascade = cards[
                    i * (cards_per_cascade + 1):
                    (i + 1) * (cards_per_cascade + 1)
                ]
                cascade = obs[i, :cards_per_cascade + 1]
            else:
                cards_in_cascade = cards[
                    i * cards_per_cascade + additional_cards:
                    (i + 1) * cards_per_cascade + additional_cards
                ]
                cascade = obs[i, :cards_per_cascade]
            np.put_along_axis(cascade, cards_in_cascade[:, np.newaxis], 1, axis=1)

        return obs

    @classmethod
    def calc_ranks(cls, cards: _obs_type) -> int | _obs_type:
        if cls.has_empty_card(cards):
            raise ValueError('cards cannot be empty')
        return np.argmax(cards, axis=-1) // len(Spec.suits)

    @classmethod
    def calc_colors(cls, cards: _obs_type) -> int | _obs_type:
        return cls.calc_suits(cards) % len(Spec.colors)

    @classmethod
    def calc_suits(cls, cards: _obs_type) -> int | _obs_type:
        if cls.has_empty_card(cards):
            raise ValueError('cards cannot be empty')
        return np.argmax(cards, axis=-1) % len(Spec.suits)

    @classmethod
    def create_card(cls, rank: int, suit: int) -> _obs_type:
        if rank < 0 or rank >= len(Spec.ranks):
            raise ValueError(f'rank of card cannot be {rank}')
        if suit < 0 or suit >= Spec.num_foundations:
            raise ValueError(f'suit of card cannot be {suit}')
        card = np.zeros(Spec.num_cards, dtype=np.int8)
        card[rank * len(Spec.suits) + suit] = 1
        return card

    @classmethod
    def has_empty_card(cls, cards: _obs_type) -> bool:
        if len(cards.shape) > 2:
            raise ValueError(f'expect 1D or 2D array, but get {cards.shape}')
        return np.any(np.sum(cards, axis=-1) == 0)

    @classmethod
    def get_cards(cls, obs: _obs_type, loc_type: Spec.loc_types, idx: int) -> _obs_type:
        if loc_type == 'cascade':
            if idx < 0 or idx >= Spec.num_cascades:
                raise ValueError(f'cascade index cannot be {idx}')
            cascade = obs[idx]
            cascade_len = np.sum(cascade)
            return cascade[:cascade_len]
        if loc_type == 'cell':
            if idx < 0 or idx >= Spec.num_cells:
                raise ValueError(f'cell index cannot be {idx}')
            return obs[-2, idx][np.newaxis, :]
        # loc_type == 'foundation'
        if idx < 0 or idx >= Spec.num_foundations:
            raise ValueError(f'foundation index cannot be {idx}')
        return obs[-1, idx][np.newaxis, :]

    @classmethod
    def remove_cards(cls, obs: _obs_type, loc_type: Spec.loc_types, idx: int, cnt: int) -> None:
        if loc_type == 'cascade':
            if idx < 0 or idx >= Spec.num_cascades:
                raise ValueError(f'cascade index cannot be {idx}')
            cascade = obs[idx]
            cascade_len = np.sum(cascade)
            if cnt <= 0 or cnt > cascade_len:
                raise ValueError(f'cannot remove {cnt} cards')
            cascade[cascade_len - cnt : cascade_len] = np.zeros((cnt, Spec.num_cards), dtype=np.int8)
        elif loc_type == 'cell':
            if idx < 0 or idx >= Spec.num_cells:
                raise ValueError(f'cell index cannot be {idx}')
            if cnt != 1:
                raise ValueError(f'cannot remove {cnt} cards')
            obs[-2, idx] = np.zeros((Spec.num_cards), dtype=np.int8)
        else:
            # loc_type == 'foundation'
            if idx < 0 or idx >= Spec.num_foundations:
                raise ValueError(f'foundation index cannot be {idx}')
            if cnt != 1:
                raise ValueError(f'cannot remove {cnt} cards')
            obs[-1, idx] = np.zeros((Spec.num_cards), dtype=np.int8)

    @classmethod
    def put_cards(cls, obs: _obs_type, cards: _obs_type, loc_type: Spec.loc_types, idx: int) -> None:
        if len(cards.shape) != 2:
            raise ValueError(f'expect 2D array, but get {cards.shape}')
        if loc_type == 'cascade':
            if idx < 0 or idx >= Spec.num_cascades:
                raise ValueError(f'cascade index cannot be {idx}')
            cascade = obs[idx]
            cascade_len = np.sum(cascade)
            cascade_max_len = Spec.num_cards // Spec.num_cascades + len(Spec.ranks)
            if cascade_len + cards.shape[0] > cascade_max_len:
                raise ValueError('put too many cards')
            cascade[cascade_len : cascade_len + cards.shape[0]] = cards
        elif loc_type == 'cell':
            if idx < 0 or idx >= Spec.num_cells:
                raise ValueError(f'cell index cannot be {idx}')
            obs[-2, idx] = cards[0]
        else:
            # loc_type == 'foundation'
            if idx < 0 or idx >= Spec.num_foundations:
                raise ValueError(f'foundation index cannot be {idx}')
            obs[-1, idx] = cards[0]

    @classmethod
    def get_gym_space(cls) -> spaces.Space:
        return spaces.MultiDiscrete(np.full((
            Spec.num_cascades + 2,
            Spec.num_cards // Spec.num_cascades + len(Spec.ranks),
            Spec.num_cards
        ), 2, dtype=np.int8), dtype=np.int8)
