from typing import Any
import argparse
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action, CardIndex, SuitIndex
from game.card import Card
from game.card_utils import CardEffect, Rank, Suit
from game.deck import Deck
from game.env import PrsiEnv
from game.game_state import GameState
from game.player import Player
from random import choice, randint
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--episodes", default=100_000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.2, type=float, help="Discount factor.")
parser.add_argument(
    "--hand_state_option", default="none", type=str, choices=["none", "simple", "full"]
)
parser.add_argument(
    "--played_subset",
    default="specials",
    type=str,
    choices=["sevens", "specials", "all"],
)
parser.add_argument(
    "--evaluate_for", default=200, type=int, help="Evaluation episodes."
)
parser.add_argument(
    "--model_path", default="mc-model.plk", type=str, help="Path to save model."
)
# TODO: load model
# TODO: add epsilon decay


# No hand state:
# Don't use hand in state at all

# Full hand state:
# bit array (converted -> int)

# Simple hand state:
# num of normal cards for each color + num of obers + num of aces + num of sevens
# maybe sort into buckets: 0, 1, 2+
# still probably bit array


"""
    hand_state: bit array -> u32 (but actual used size depends on hand state usage)
    opponent card count: u8
    top card: CardIndex (int)
    active suit: SuitIndex (int)
    CardEffect: IntEnum
    Effect strength: u8
    played cards subset: np.array, maybe flattened is better (?? TODO)
"""
State = tuple[
    np.uint32,
    np.uint8,
    CardIndex,
    SuitIndex,
    CardEffect,
    np.uint8,
    np.ndarray,
]


class MonteCarloAgent(BaseAgent):
    # estimate for actions in unseen states
    _default_actions = np.zeros(len(CARD_TO_INDEX)), np.zeros(len(SUIT_TO_INDEX))

    def __init__(
        self, opts: argparse.Namespace, player_info: Player | None = None
    ) -> None:
        super().__init__(player_info)
        # both indexed by state + action
        self.action_value_fn: dict[State, dict[Action, float]] = {}
        self.num_visits: dict[State, dict[Action, int]] = {}

        self.played_cards_subset: np.ndarray
        self.played_cards_subset_option: str
        self._init_played_subset(opts)

        self.train_episodes = opts.episodes
        self.epsilon = opts.epsilon
        self.gamma = opts.gamma
        self.hand_state_option = opts.hand_state_option

    def train(self, env: PrsiEnv, **opts: dict[str, Any]) -> None:
        for _ in range(self.train_episodes):
            ...
        raise NotImplementedError("TODO: Implement monte carlo training")

    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        for _ in range(episodes):
            ...

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for monte carlo agent")

    def save(self, path: str) -> None:
        raise NotImplementedError("TODO: save")

    def _init_played_subset(self, opts: argparse.Namespace) -> None:
        match opts.played_subset:
            case "specials":  # sevens + obers + aces
                self.played_cards_subset = np.zeros(3)
            case "sevens":
                self.played_cards_subset = np.zeros(1)
            case "all":
                self.played_cards_subset = np.zeros(32)
            case _:
                raise ValueError("Invalid argument.")
        self.played_cards_subset_option = opts.played_subset

    def _process_state(self, state: GameState, info: dict[str, Any]) -> State:
        """Correctly set played_cards_subset based on the state"""

        if state.top_card is None:
            raise ValueError("Can't process state without top card.")

        hand_state = self._get_hand_state()
        opponent_card_count = info.get("opponent_card_count", 0)
        top_card = CARD_TO_INDEX[state.top_card]
        active_suit = SUIT_TO_INDEX[state.actual_suit]
        card_effect = state.current_effect
        effect_strength = np.uint8(state.effect_strength)
        self._update_subset(state.top_card, info.get("deck_flipped_over", False))

        return (
            hand_state,
            opponent_card_count,
            top_card,
            active_suit,
            card_effect,
            effect_strength,
            self.played_cards_subset,
        )

    def _get_hand_state(self) -> np.uint32:
        match self.hand_state_option:
            case "none":
                return np.uint32(0)
            case "simple":
                raise NotImplementedError('TODO: _get_hand_state for "simple"')
            case "full":
                raise NotImplementedError('TODO: _get_hand_state for "full"')
            case _:
                raise ValueError("Invalid hand_state_option.")


    def _update_subset(self, card: Card, deck_flipped_over: bool) -> None:
        if deck_flipped_over:
            self.played_cards_subset = np.zeros_like(self.played_cards_subset)
        match self.played_cards_subset_option:
            case "all":
                idx = CARD_TO_INDEX[card] - 1 # None is 0
                self.played_cards_subset[idx] += 1
            case "specials":
                if (
                    card.rank != Rank.SEVEN
                    or card.rank != Rank.OBER
                    or card.rank != Rank.ACE
                ):
                    return
                if card.rank == Rank.SEVEN:
                    self.played_cards_subset[0] += 1
                if card.rank == Rank.OBER:
                    self.played_cards_subset[1] += 1
                if card.rank == Rank.ACE:
                    self.played_cards_subset[2] += 1
            case "sevens":
                if card.rank != Rank.SEVEN:
                    return
                self.played_cards_subset[0] += 1


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = PrsiEnv()
    agent = MonteCarloAgent(args, Player(0))
    agent.train(env)
    agent.evaluate(env, episodes=100)
    agent.save("mc_agent.pkl")
