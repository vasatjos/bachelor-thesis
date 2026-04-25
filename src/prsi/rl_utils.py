from typing import Sequence, Generic, TypeVar

from random import randrange
from prsi.card import Card
from prsi.card_utils import Rank, Suit

import numpy as np

from prsi.game_state import GameState, find_allowed_cards

CardIndex = int
SuitIndex = int
Action = tuple[Card, Suit] | None

DRAW_ACTION = None

INDEX_TO_CARD: list[Card] = [
    card for card in (Card(suit, rank) for rank in Rank for suit in Suit)
]

CARD_TO_INDEX: dict[Card, CardIndex] = {card: i for i, card in enumerate(INDEX_TO_CARD)}

INDEX_TO_SUIT: list[Suit] = [suit for suit in Suit]

SUIT_TO_INDEX: dict[Suit, SuitIndex] = {suit: i for i, suit in enumerate(Suit)}

INDEX_TO_ACTION: list[Action] = [DRAW_ACTION]
for card in INDEX_TO_CARD:
    if card.rank != Rank.OBER:
        INDEX_TO_ACTION.append((card, card.suit))
    else:
        for suit in Suit:
            INDEX_TO_ACTION.append((card, suit))

ACTION_TO_INDEX = {action: i for i, action in enumerate(INDEX_TO_ACTION)}


T = TypeVar("T")  # item type stored in the buffer


class ReplayBuffer(Generic[T]):
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
        self._data: list[T] = []
        self._offset = 0

    def __len__(self) -> int:
        return len(self._data)

    @property
    def max_length(self) -> int | None:
        return self._max_length

    def append(self, item: T) -> None:
        if self._max_length is not None and len(self._data) >= self._max_length:
            self._data[self._offset] = item
            self._offset = (self._offset + 1) % self._max_length
        else:
            self._data.append(item)

    def extend(self, items: Sequence[T]) -> None:
        if self._max_length is None:
            self._data.extend(items)
        else:
            for item in items:
                if len(self._data) >= self._max_length:
                    self._data[self._offset] = item
                    self._offset = (self._offset + 1) % self._max_length
                else:
                    self._data.append(item)

    def __getitem__(self, index: int) -> T:
        assert -len(self._data) <= index < len(self._data)
        return self._data[(self._offset + index) % len(self._data)]

    def sample(self, size: int, generator=np.random, replace: bool = True) -> list[T]:
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


def behave_randomly(state: GameState, hand: list[Card]) -> Action:
    """
    Select a random card to play from the ones available on hand (or draw).

    Note, this does not actually select a random action uniformly, since that would make
    playing an ober 4x more likely than other cards. Instead, selects a random
    card and if the card is on ober, selects a random suit as well.
    """
    allowed = find_allowed_cards(state)
    playable = [c for c in hand if c in allowed]

    playable_length = len(playable)
    random_idx = randrange(playable_length + 1)
    if playable_length == 0 or random_idx == playable_length:
        return DRAW_ACTION

    # using random index instead of random.choice for micro optimalization
    card = playable[random_idx]
    suit = card.suit if card.rank != Rank.OBER else INDEX_TO_SUIT[randrange(len(Suit))]
    return card, suit


def get_valid_actions(game_state: GameState, hand: list[Card]) -> list[Action]:
    valid_actions: list[Action] = []
    allowed_cards = find_allowed_cards(game_state)

    for card in hand:
        if card in allowed_cards:
            if card.rank == Rank.OBER:
                for suit in Suit:
                    valid_actions.append((card, suit))
            else:
                valid_actions.append((card, card.suit))

    valid_actions.append(DRAW_ACTION)
    return valid_actions


def get_valid_action_mask(game_state: GameState, hand: list[Card]) -> np.ndarray:
    """
    Return a boolean mask of valid actions, useful for neural network agents.
    """
    actions = get_valid_actions(game_state, hand)
    mask = np.zeros(len(ACTION_TO_INDEX), dtype=bool)
    for a in actions:
        mask[ACTION_TO_INDEX[a]] = True
    return mask
