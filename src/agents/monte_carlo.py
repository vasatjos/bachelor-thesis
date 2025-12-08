from typing import Any
import argparse
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action, CardIndex, SuitIndex
from game.card import Card
from game.card_utils import CardEffect, Rank, Suit
from game.env import PrsiEnv
from game.player import Player
from random import choice, randint
import numpy as np


# TODO: state: (tuple??)

# No hand state:
# Don't use hand in state at all

# Full hand state:
# bit array (converted -> int)

# Simple hand state:
# num of normal cards for each color + num of obers + num of aces + num of sevens
# maybe sort into buckets: 0, 1, 2+
# still probably bit array

#           hand: bit array -> u32
#           opponent card count: u8
#           top card: CardIndex (int)
#           active suit: SuitIndex (int)
#           CardEffect
#           Effect strength
#           played cards subset: np.array (3x4), maybe flattened is better (??)
#           state: u8 ((probably))

State = tuple[
    np.uint32 | None,
    np.uint8,
    CardIndex,
    SuitIndex,
    CardEffect,
    np.uint8,
    np.ndarray,
    np.uint8,
]


class MonteCarloAgent(BaseAgent):
    # estimate for actions in unseen states
    _default_actions = np.zeros(len(CARD_TO_INDEX)), np.zeros(len(SUIT_TO_INDEX))

    def __init__(
        self, opts: argparse.Namespace, player_info: Player | None = None
    ) -> None:
        super().__init__(player_info)
        # both indexed by state + action
        self.action_value_fn = {}
        self.num_visits = {}

        self.played_cards_subset = np.zeros((3, 4))  # obers + aces + sevens
        self.train_episodes = opts.episodes
        self.hand_state_variant = ...  # None vs Simple vs Full

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for monte carlo agent")

    def train(self, env: PrsiEnv, **opts: dict[str, Any]) -> None:
        for _ in range(self.train_episodes):
            ...
        raise NotImplementedError("TODO: Implement monte carlo training")

    def _process_state(self, state: Any) -> ...:
        """Correctly set played_cards_subset based on the state"""
        raise NotImplementedError("TODO: _process_state")
