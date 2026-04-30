from enum import Enum, StrEnum, IntEnum

COLOR_RESET = "\033[0m"


class Suit(StrEnum):
    """
    Enum representing the suits in a german card deck.
    """

    HEARTS = "\033[31m"
    LEAVES = "\033[32m"
    BELLS = "\033[34m"
    ACORNS = "\033[33m"


class Rank(IntEnum):
    """
    Enum representing the ranks of cards in a german card deck.
    """

    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    UNTER = 11
    OBER = 12
    KING = 13
    ACE = 14


class CardEffect(Enum):
    """
    Enum representing the effect of cards in Prší.
    """

    NONE = 0
    SKIP_TURN = 1
    DRAW_TWO = 2
