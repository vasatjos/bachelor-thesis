from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, DRAW_ACTION, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank
from game.game_state import find_allowed_cards
from random import choice, randint


class RandomAgent(BaseAgent):
    def choose_action(
        self, state: Any, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        playable = tuple(find_allowed_cards(state) & hand)

        playable_length = len(playable)
        if not playable or randint(0, playable_length) == playable_length:
            return DRAW_ACTION

        card_choice = choice(playable)
        suit_index = (
            SUIT_TO_INDEX[card_choice.suit]
            if card_choice.rank != Rank.OBER
            else randint(1, 4)
        )

        return (CARD_TO_INDEX[card_choice], suit_index)

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")

    # implementations for methods below not necessary
    def train(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
