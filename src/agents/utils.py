from typing import Sequence

from random import randint, choice
from prsi.card import Card
from prsi.card_utils import Rank, Suit

import numpy as np

from prsi.game_state import GameState, find_allowed_cards

CardIndex = int
SuitIndex = int
Action = tuple[CardIndex, SuitIndex]
DRAW_ACTION = 0, 0

INDEX_TO_CARD: dict[CardIndex, Card | None] = {
    i: card
    for i, card in enumerate(
        (Card(suit, rank) for rank in Rank for suit in Suit), start=1
    )
}
INDEX_TO_CARD[0] = None  # Draw a card

CARD_TO_INDEX: dict[Card | None, CardIndex] = {
    card: i
    for i, card in enumerate(
        (Card(suit, rank) for rank in Rank for suit in Suit), start=1
    )
}
CARD_TO_INDEX[None] = 0  # Draw a card

INDEX_TO_SUIT: dict[SuitIndex, Suit | None] = {
    i: suit for i, suit in enumerate(Suit, start=1)
}
INDEX_TO_SUIT[0] = None

SUIT_TO_INDEX: dict[Suit | None, SuitIndex] = {
    suit: i for i, suit in enumerate(Suit, start=1)
}
SUIT_TO_INDEX[None] = 0


class ReplayBuffer:
    """Simple replay buffer with possibly limited capacity.

    Code from the package used in homework for the following course:
    https://ufal.mff.cuni.cz/courses/npfl139/2425-summer
    """

    def __init__(self, max_length: int | None = None):
        if max_length is not None:
            assert isinstance(max_length, int), (
                "The max_length argument must be an integer"
            )
            assert max_length > 0, "The max_length argument must be a positive integer"
        self._max_length = max_length
        self._data: list[object] = []
        self._offset = 0

    def __len__(self) -> int:
        return len(self._data)

    @property
    def max_length(self) -> int | None:
        return self._max_length

    def append(self, item: object) -> None:
        if self._max_length is not None and len(self._data) >= self._max_length:
            self._data[self._offset] = item
            self._offset = (self._offset + 1) % self._max_length
        else:
            self._data.append(item)

    def extend(self, items: Sequence[object]) -> None:
        if self._max_length is None:
            self._data.extend(items)
        else:
            for item in items:
                if len(self._data) >= self._max_length:
                    self._data[self._offset] = item
                    self._offset = (self._offset + 1) % self._max_length
                else:
                    self._data.append(item)

    def __getitem__(self, index: int) -> object:
        assert -len(self._data) <= index < len(self._data)
        return self._data[(self._offset + index) % len(self._data)]

    def sample(
        self, size: int, generator=np.random, replace: bool = True
    ) -> list[object]:
        # By default, the same element can be sampled multiple times. Making sure the samples
        # are unique is costly, and we do not mind the duplicites much during training.
        if replace:
            return [
                self._data[index]
                for index in generator.randint(len(self._data), size=size)
            ]
        else:
            return [
                self._data[index]
                for index in generator.choice(len(self._data), size=size, replace=False)
            ]


def behave_randomly(state: GameState, hand: set[Card]) -> Action:
    playable = tuple(find_allowed_cards(state) & hand)

    playable_length = len(playable)
    if playable_length == 0 or randint(0, playable_length) == playable_length:
        return DRAW_ACTION

    card = choice(playable)
    suit_idx = SUIT_TO_INDEX[card.suit] if card.rank != Rank.OBER else randint(1, 4)
    return CARD_TO_INDEX[card], suit_idx
