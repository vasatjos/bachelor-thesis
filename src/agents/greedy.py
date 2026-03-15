from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, DRAW_ACTION, SUIT_TO_INDEX, Action
from prsi.card import Card
from prsi.card_utils import Rank
from prsi.game_state import GameState, find_allowed_cards
from random import choice, randint


class GreedyAgent(BaseAgent):
    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        playable = tuple(find_allowed_cards(state) & hand)

        if not playable:
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
