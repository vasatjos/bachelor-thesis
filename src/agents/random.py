from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank
from random import choice, randint


class RandomAgent(BaseAgent):
    def choose_action(self, state: Any, hand: set[Card]) -> Action:
        hand_list: list[Card | None] = list(hand)
        hand_list.append(None)

        card: Card | None = choice(hand_list)
        if card is None:
            return CARD_TO_INDEX[card], -42  # suit will get discarded on drawing a card

        card_index = CARD_TO_INDEX[card]
        suit_index = SUIT_TO_INDEX[card.suit]
        if card.rank == Rank.OBER:
            suit_index = randint(1, 4)

        return card_index, suit_index

    def train(self) -> None:  # no training necessary, just chooses random action
        pass

    def save(self, path: str) -> None:
        pass

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")
