from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank
from game.game_state import find_allowed_cards
from random import choice


class RandomAgent(BaseAgent):
    def choose_action(self, state: Any, hand: set[Card]) -> Action:
        valid_actions: list[Action] = []
        allowed_cards = find_allowed_cards(state)

        for card in hand:
            if card in allowed_cards:
                if card.rank == Rank.OBER:
                    for suit_idx in range(1, 4):
                        valid_actions.append((CARD_TO_INDEX[card], suit_idx))
                else:
                    valid_actions.append(
                        (CARD_TO_INDEX[card], SUIT_TO_INDEX[card.suit])
                    )

        # Can always draw
        valid_actions.append((0, 0))

        return choice(valid_actions)

    def train(self) -> None:  # no training necessary, just chooses random action
        pass

    def save(self, path: str) -> None:
        pass

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")
