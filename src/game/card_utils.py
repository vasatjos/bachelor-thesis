from enum import Enum, StrEnum, IntEnum
from game.card import Card

COLOR_RESET = "\033[0m"


class Suit(StrEnum):
    HEARTS = "\033[31m"
    LEAVES = "\033[32m"
    BELLS = "\033[34m"
    ACORNS = "\033[33m"


class Rank(IntEnum):
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    UNTER = 11
    OBER = 12
    KING = 13
    ACE = 14


class CardEffect(Enum):
    NONE = 0
    SKIP_TURN = 1
    DRAW_TWO = 2


def generate_suit(suit: Suit) -> set[Card]:
    return {Card(suit, rank) for rank in Rank}


def generate_rank(rank: Rank) -> set[Card]:
    return {Card(suit, rank) for suit in Suit}
