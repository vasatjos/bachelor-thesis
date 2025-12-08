from typing import Any
import argparse
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank, Suit
from game.env import PrsiEnv
from game.player import Player
from random import choice, randint
import numpy as np


# TODO: state: (tuple??)
#           hand: bit array -> i32
#           opponent card count: u8
#           top card: CardIndex (int)
#           played cards subset: bit array -> i32
#           state: u8 ((probably))

# Simple hand state:
# num of normal cards for each color + num of obers + num of aces + num of sevens

# Full hand state:
# bit array (converted -> int)


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

        self.played_cards_subset = np.zeros(4 + 4 + 4) ->
        self.train_episodes = opts.episodes
        self.use_hand_state = opts.use_hand_state  # affects state count significantly
        self.hand_state_variant = ... # None vs Simple vs Full

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for monte carlo agent")

    def train(self, env: PrsiEnv, **opts: dict[str, Any]) -> None:
        for _ in range(self.train_episodes):
            ...
        raise NotImplementedError("TODO: Implement monte carlo training")

    def _process_state(self, state: Any) -> ...:
        raise NotImplementedError("TODO: _process_state")
