from typing import Any
from agents.base import BaseAgent
from agents.utils import DRAW_ACTION, INDEX_TO_SUIT, Action
from prsi.card import Card
from prsi.card_utils import Rank
from prsi.game_state import GameState, find_allowed_cards
from random import randrange


class GreedyAgent(BaseAgent):
    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        playable = tuple(find_allowed_cards(state) & hand)

        if not playable:
            return DRAW_ACTION

        chosen_card = playable[randrange(len(playable))]
        chosen_suit = (
            chosen_card.suit
            if chosen_card.rank != Rank.OBER
            else INDEX_TO_SUIT[randrange(4)]
        )

        return chosen_card, chosen_suit

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")
