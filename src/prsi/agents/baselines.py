from typing import Any
from prsi.agents.agent import Agent
from prsi.rl_utils import (
    Action,
    behave_randomly,
    find_allowed_cards,
    DRAW_ACTION,
    INDEX_TO_SUIT,
)
from prsi.card import Card
from prsi.card_utils import Rank
from prsi.game_state import GameState
from random import randrange


class RandomAgent(Agent):
    """
    A baseline opponent for prsi.

    Chooses a random action to perform (sometimes draws when not necessary).
    """

    def choose_action(
        self, state: GameState, hand: list[Card], info: dict[str, Any]
    ) -> Action:
        return behave_randomly(state, hand)

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")


class GreedyAgent(Agent):
    """
    A baseline opponent for prsi.

    Chooses a random card to play, only draws a card when no other option is available.
    """

    def choose_action(
        self, state: GameState, hand: list[Card], info: dict[str, Any]
    ) -> Action:
        allowed = find_allowed_cards(state)
        playable = [c for c in hand if c in allowed]
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
