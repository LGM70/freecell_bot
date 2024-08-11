import os
import argparse
from typing import Any
import numpy as np
import gymnasium as gym
from termcolor import colored
from .obs_space.base_obs import BaseObsSpace
# from .obs_space.one_hot_obs import OneHotObsSpace as ObsSpace
from .obs_space.compact_obs import CompactObsSpace as ObsSpace
from .action_space.base_action import BaseActionSpace
from .action_space.tuple_action import TupleActionSpace as ActionSpace
from .standard_spec import StandardSpec as Spec

class FreeCellEnv(gym.Env):
    """Freecell game environment"""

    def __init__(self, obs_space_cls: BaseObsSpace, action_space_cls: BaseActionSpace, max_steps: int = 1000):
        self._obs_space = obs_space_cls
        self._action_space = action_space_cls
        self._obs = None
        self._step = 0
        self._max_steps = max_steps
        self.action_space = action_space_cls.get_gym_space()
        self.observation_space = obs_space_cls.get_gym_space()

    def step(self, action: Any) \
        -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._obs = np.copy(self._obs)
        self._step += 1
        source_type, source_idx, dest_type, dest_idx = self._action_space.parse_action(action)
        reward = -1.
        solved = False
        steps_exceed_limit = False
        result = {}
        if self._move(source_type, source_idx, dest_type, dest_idx):
            result['status'] = 'success'
            if source_type == 'foundation':
                reward -= 10.
            elif dest_type == 'foundation':
                reward += 10.
                if self._check_is_complete():
                    reward += 100.
                    solved = True
        else:
            reward -= 100.
            result['status'] = 'failure'
        if self._step >= self._max_steps:
            reward -= 200.
            steps_exceed_limit = True
        return self._obs, reward, solved, steps_exceed_limit, result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) \
        -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._obs = self._obs_space.deal()
        self._step = 0
        return self._obs, {}

    def render(self) -> str | list[str] | None:
        return self._visualize()

    def close(self) -> None:
        pass

    def _check_is_sequence(self, cards: Any) -> bool:
        consecutive_ranks = np.all(
            self._obs_space.calc_ranks(cards[:-1]) - self._obs_space.calc_ranks(cards[1:]) == 1
        )
        alternating_colors = np.all(
            self._obs_space.calc_colors(cards[:-1]) + self._obs_space.calc_colors(cards[1:]) == 1
        )
        return alternating_colors and consecutive_ranks

    def _count_sequence_len(self, cards: Any) -> int:
        for i in range(1, cards.shape[0]):
            curr, pre = cards[-i], cards[-i - 1]
            consecutive_ranks = self._obs_space.calc_ranks(pre) - self._obs_space.calc_ranks(curr) == 1
            alternating_colors = self._obs_space.calc_colors(curr) + self._obs_space.calc_colors(pre) == 1
            if not alternating_colors or not consecutive_ranks:
                return i
        return cards.shape[0]

    def _max_move(self) -> int:
        empty_cells = 0
        for i in range(Spec.num_cells):
            if self._obs_space.has_empty_card(self._obs_space.get_cards(self._obs, 'cell', i)):
                empty_cells += 1
        empty_cascades = 0
        for i in range(Spec.num_cascades):
            if self._obs_space.get_cards(self._obs, 'cascade', i).shape[0] == 0:
                empty_cascades += 1
        return 2 ** empty_cascades * (empty_cells + 1)

    def _move(self, source_type: Spec.loc_types, source_idx: int, dest_type: Spec.loc_types, dest_idx: int) -> bool:
        if source_type == dest_type and source_idx == dest_idx:
            return False
        if source_type == 'cascade' and dest_type == 'cascade':
            return self._move_multi_cards(source_idx, dest_idx)
        return self._move_one_card(source_type, source_idx, dest_type, dest_idx)

    def _move_one_card(self, source_type: Spec.loc_types, source_idx: int, dest_type: Spec.loc_types, dest_idx: int) -> bool:
        rank_req = None
        color_req = None
        suit_req = None

        if dest_type == 'cascade':
            dest_cascade = self._obs_space.get_cards(self._obs, dest_type, dest_idx)
            dest_cascade_len = dest_cascade.shape[0]
            if dest_cascade_len != 0:
                dest_last_card = dest_cascade[-1]
                rank_req = self._obs_space.calc_ranks(dest_last_card) - 1
                color_req = 1 - self._obs_space.calc_colors(dest_last_card)
        elif dest_type == 'cell':
            if not self._obs_space.has_empty_card(self._obs_space.get_cards(self._obs, dest_type, dest_idx)):
                return False
        else:
            # dest_type == 'foundation'
            suit_req = dest_idx
            dest_last_card = self._obs_space.get_cards(self._obs, dest_type, dest_idx)[-1]
            if self._obs_space.has_empty_card(dest_last_card):
                rank_req = 0
            else:
                rank_req = self._obs_space.calc_ranks(dest_last_card) + 1

        if rank_req is not None and (rank_req < 0 or rank_req > len(Spec.ranks)):
            return False

        if source_type == 'cascade':
            source_cascade = self._obs_space.get_cards(self._obs, source_type, source_idx)
            source_cascade_len = source_cascade.shape[0]
            if source_cascade_len == 0:
                return False
            source_last_card = source_cascade[-1:]
        else:
            # source_type in ['cell', 'foundation']
            source_last_card = self._obs_space.get_cards(self._obs, source_type, source_idx)
            if self._obs_space.has_empty_card(source_last_card):
                return False

        if (rank_req is not None and self._obs_space.calc_ranks(source_last_card) != rank_req) \
            or (color_req is not None and self._obs_space.calc_colors(source_last_card) != color_req) \
            or (suit_req is not None and self._obs_space.calc_suits(source_last_card) != suit_req):
            return False

        self._obs_space.put_cards(self._obs, source_last_card, dest_type, dest_idx)

        if source_type == 'foundation':
            source_last_card_rank = self._obs_space.calc_ranks(source_last_card).item()
        self._obs_space.remove_cards(self._obs, source_type, source_idx, 1)
        if source_type == 'foundation' and source_last_card_rank > 0:
            self._obs_space.put_cards(
                self._obs,
                self._obs_space.create_card(source_last_card_rank - 1, source_idx)[np.newaxis, :],
                source_type,
                source_idx
            )
        return True

    def _move_multi_cards(self, source_idx: int, dest_idx: int) -> bool:
        # from cascade to cascade
        dest_cascade = self._obs_space.get_cards(self._obs, 'cascade', dest_idx)
        dest_cascade_len = dest_cascade.shape[0]
        if dest_cascade_len != 0:
            dest_last_card = dest_cascade[-1]

        source_cascade = self._obs_space.get_cards(self._obs, 'cascade', source_idx)
        source_cascade_len = source_cascade.shape[0]
        if source_cascade_len == 0:
            return False
        source_last_card = source_cascade[-1]

        if dest_cascade_len != 0:
            cnt = self._obs_space.calc_ranks(dest_last_card) - self._obs_space.calc_ranks(source_last_card)
            if cnt <= 0 or cnt > min(self._max_move(), source_cascade_len) \
                or (
                    self._obs_space.calc_colors(dest_last_card) + self._obs_space.calc_colors(source_last_card) + cnt
                ) % 2 != 0 \
                or not self._check_is_sequence(source_cascade[-cnt:]):
                return False
        else:
            # move as many as possible
            cnt = min(self._max_move(), self._count_sequence_len(source_cascade))

        self._obs_space.put_cards(self._obs, source_cascade[-cnt:], 'cascade', dest_idx)
        self._obs_space.remove_cards(self._obs, 'cascade', source_idx, cnt)
        return True

    def _check_is_complete(self) -> bool:
        for i in range(Spec.num_cells):
            if not self._obs_space.has_empty_card(self._obs_space.get_cards(self._obs, 'cell', i)):
                return False
        for i in range(Spec.num_cascades):
            if self._obs_space.get_cards(self._obs, 'cascade', i).shape[0] != 0:
                return False
        return True

    def _visualize_card(self, card: np.ndarray, suit: int = None) -> str:
        suit_only = False
        if self._obs_space.has_empty_card(card):
            if suit is None:
                return '--'
            card = self._obs_space.create_card(0, suit)
            suit_only = True
        color = Spec.colors[self._obs_space.calc_colors(card)]
        suit = Spec.suits[self._obs_space.calc_suits(card)]
        rank = Spec.ranks[self._obs_space.calc_ranks(card)]
        if suit_only:
            return colored(suit[0], color) + '-'
        return colored(suit[0], color) + rank

    def _visualize(self) -> str:
        s1 = 'cells:\t\t '
        for i in range(Spec.num_cells):
            s1 += colored(str(Spec.num_cascades + i), 'blue') + '\t '
        s1 += '\n\t\t '
        for i in range(Spec.num_cells):
            s1 += self._visualize_card(self._obs_space.get_cards(self._obs, 'cell', i)[0]) + '\t '

        s2 = 'foundations:\t '
        for i in range(Spec.num_foundations):
            s2 += colored(str(Spec.num_cascades + Spec.num_cells + i), 'blue') + '\t '
        s2 += '\n\t\t '
        for i in range(Spec.num_foundations):
            s2 += self._visualize_card(self._obs_space.get_cards(self._obs, 'foundation', i)[0], suit=i) + '\t '

        s3 = 'cascades:\n'
        cascades = []
        max_cascade_len = 0
        for j in range(Spec.num_cascades):
            cascades.append(self._obs_space.get_cards(self._obs, 'cascade', j))
            max_cascade_len = max(max_cascade_len, cascades[-1].shape[0])
            s3 += colored(str(j), 'blue') + '\t '
        for i in range(max_cascade_len):
            s3 += '\n'
            for j in range(Spec.num_cascades):
                if cascades[j].shape[0] > i:
                    s3 += self._visualize_card(cascades[j][i]) + '\t '
                else:
                    s3 += '  \t '

        return s1 + '\n\n' + s2 + '\n\n' + s3

    def save(self, filename: str) -> None:
        if os.path.exists(filename):
            raise FileExistsError(f'{filename} already exists')
        np.save(filename, self._obs)

    def load(self, filename: str) -> None:
        if os.path.isfile(filename):
            raise FileNotFoundError(f'{filename} not found')
        self._obs = np.load(filename)
        print('WARNING: Please make sure the loading game has the same observation space')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    env = FreeCellEnv(ObsSpace, ActionSpace)

    if args.load:
        env.load(args.load)
    else:
        env.reset()

    moves = 0
    while True:
        try:
            os.system('clear')
            print(env.render() + '\n')
            source = int(input('Move cards from: '))
            dest = int(input('Move cards to: '))
            moves += 1
            _, _, terminated, truncated, info = env.step((source, dest))
            if info['status'] == 'failure':
                print('Invalid move, press enter to continue')
                input()
            elif terminated:
                print(f'Congrats! You solve this in {moves} steps')
                break
            if truncated:
                print(f'You failed to solve this in {moves} steps')
                break
        except KeyboardInterrupt as exc:
            if args.save:
                env.save(args.save)
            raise KeyboardInterrupt from exc
