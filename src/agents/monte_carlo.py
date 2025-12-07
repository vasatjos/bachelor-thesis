from typing import Any
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


class MonteCarloAgent(BaseAgent):
    def __init__(self, player_info: Player | None = None) -> None:
        super().__init__(player_info)
        # both indexed by state + action
        self.action_value_fn = {}
        self.num_visits = {}

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for monte carlo agent")

    def train(self, episodes: int, env: PrsiEnv) -> None:
        raise NotImplementedError("TODO: Implement monte carlo training")
