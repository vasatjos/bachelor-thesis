from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank
from game.game_state import find_allowed_cards
from random import choice, randint


class RandomAgent(BaseAgent):
    def choose_action(self, state: Any, hand: set[Card]) -> Action:
        playable = tuple(find_allowed_cards(state) & hand)

        if not playable or randint(0, len(playable)) == len(playable):
            return (CARD_TO_INDEX[None], SUIT_TO_INDEX[None])  # draw

        card_choice = choice(playable)
        suit_index = (
            SUIT_TO_INDEX[card_choice.suit]
            if card_choice.rank != Rank.OBER
            else randint(1, 4)
        )

        return (CARD_TO_INDEX[card_choice], suit_index)

    def train(self) -> None:  # no training necessary, just chooses random action
        pass

    def save(self, path: str) -> None:
        pass

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")
