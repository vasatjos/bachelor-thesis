from os import getenv
from functools import total_ordering

from prsi.card_utils import Suit, Rank, CardEffect, COLOR_RESET

USE_ICONS = getenv("PRSI_USE_ICONS", "true").lower() in ("true", "1", "yes", "t")

ICONS = {
    Suit.HEARTS: "󰋑",
    Suit.LEAVES: "󰌪",
    Suit.BELLS: "",
    Suit.ACORNS: "󰋟",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.UNTER: "",
    Rank.OBER: "",
    Rank.KING: "󰆥",
    Rank.ACE: "",
}


@total_ordering
class Card:
    """
    A simple Prší card representation.

    Every card has a suit, rank and effect (the effect can be null).
    """

    def __init__(self, suit: Suit, rank: Rank) -> None:
        """
        Initialize a card with a suit and rank.

        Args:
            suit: The suit of the card (Hearts, Leaves, Bells, Acorns).
            rank: The rank of the card (Seven, Eight, ..., Ace).
        """
        self.suit = suit
        self.rank = rank

        self.effect: CardEffect
        self._init_effect()

    def _init_effect(self) -> None:
        if self.rank is Rank.SEVEN:
            self.effect = CardEffect.DRAW_TWO
        elif self.rank is Rank.ACE:
            self.effect = CardEffect.SKIP_TURN
        else:
            self.effect = CardEffect.NONE

    def __str__(self) -> str:
        """
        Return a string representation of the card, optionally with icons and colors.
        """
        icon = f"{ICONS[self.suit]} {ICONS[self.rank]:<3}" if USE_ICONS else ""
        return (
            f"{self.suit.value}{icon}{self.rank.name} of {self.suit.name}{COLOR_RESET}"
        )

    def __repr__(self) -> str:
        """
        Return a formal string representation of the card.
        """
        return (
            f"Card(rank={self.rank.name}, suit={self.suit.name}, "
            + f"effect={self.effect.name}"
        )

    def __lt__(self, other) -> bool:
        """
        Compare two cards for ordering. Ordered primarily by suit and then by rank.
        """
        return (
            self.suit < other.suit or self.suit == other.suit and self.rank < other.rank
        )

    def __eq__(self, other) -> bool:
        """
        Check if two cards are equal (same suit and rank).
        """
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        """
        Return the hash of the card based on its suit and rank.
        """
        return hash((self.suit, self.rank))
