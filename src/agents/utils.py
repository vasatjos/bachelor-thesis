from typing import Sequence
from game.card import Card
from game.card_utils import Rank, Suit

import numpy as np

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


# Action index <-> (card_idx, suit_idx) mapping
def _build_action_maps() -> tuple[list[Action], dict[Action, int]]:
    """Return (index_to_action, action_to_index)."""
    actions: list[Action] = [DRAW_ACTION]
    for card_idx in range(1, 33):  # 32 cards
        for suit_idx in range(1, 5):  # 4 suits
            actions.append((card_idx, suit_idx))
    action_to_index = {a: i for i, a in enumerate(actions)}
    return actions, action_to_index


INDEX_TO_ACTION, ACTION_TO_INDEX = _build_action_maps()
NUM_ACTIONS = len(INDEX_TO_ACTION)  # 1 + 32*4 = 129


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
