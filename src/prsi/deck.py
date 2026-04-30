from random import shuffle

from prsi.card_utils import Suit, Rank, CardEffect
from prsi.card import Card


def _generate_suit(suit: Suit) -> set[Card]:
    return {Card(suit, rank) for rank in Rank}


def _generate_rank(rank: Rank) -> set[Card]:
    return {Card(suit, rank) for suit in Suit}


# NOTE: get_x MUCH faster than _generate_x
def get_suit(suit: Suit) -> set[Card]:
    """
    Return all cards of the given suit.
    """
    match suit:
        case Suit.HEARTS:
            return Deck.HEARTS
        case Suit.BELLS:
            return Deck.BELLS
        case Suit.LEAVES:
            return Deck.LEAVES
        case Suit.ACORNS:
            return Deck.ACORNS


def get_rank(rank: Rank) -> set[Card]:
    """
    Return all cards of the rank.
    """
    match rank:
        case Rank.SEVEN:
            return Deck.SEVENS
        case Rank.EIGHT:
            return Deck.EIGHTS
        case Rank.NINE:
            return Deck.NINES
        case Rank.TEN:
            return Deck.TENS
        case Rank.UNTER:
            return Deck.UNTERS
        case Rank.OBER:
            return Deck.OBERS
        case Rank.KING:
            return Deck.KINGS
        case Rank.ACE:
            return Deck.ACES


class DeckEmptyError(Exception):
    pass


class Deck:
    """
    Deck of german playing cards.
    """

    CARD_COUNT = len(Suit) * len(Rank)

    ALL_CARDS = {Card(suit, rank) for suit in Suit for rank in Rank}
    SEVENS = _generate_rank(Rank.SEVEN)
    EIGHTS = _generate_rank(Rank.EIGHT)
    NINES = _generate_rank(Rank.NINE)
    TENS = _generate_rank(Rank.TEN)
    UNTERS = _generate_rank(Rank.UNTER)
    OBERS = _generate_rank(Rank.OBER)
    KINGS = _generate_rank(Rank.KING)
    ACES = _generate_rank(Rank.ACE)
    HEARTS = _generate_suit(Suit.HEARTS)
    BELLS = _generate_suit(Suit.BELLS)
    LEAVES = _generate_suit(Suit.LEAVES)
    ACORNS = _generate_suit(Suit.ACORNS)

    def __init__(self) -> None:
        self.discard_pile: list[Card]
        self.drawing_pile: list[Card]

        self.reset()

    def reset(self) -> None:
        """
        Shuffle the deck, place the top card on the discard pile (play the card).
        """
        self.discard_pile: list[Card] = []  # type: ignore[no-redef]
        self.drawing_pile: list[Card] = [  # type: ignore[no-redef]
            Card(suit, rank) for suit in Suit for rank in Rank
        ]
        shuffle(self.drawing_pile)

        top_card, _ = self.draw_card()
        if top_card is None:
            raise ValueError("Deck reset failed.")

        self.play_card(top_card)

    def draw_card(self) -> tuple[Card, bool]:
        """
        Draw a card from the drawing pile.

        If the drawing pile is empty, the discard pile gets flipped over
        and becomes the drawing pile.
        Whether the deck was flipped over or not is returned.

        Raises `DeckEmptyError` if there are now cards available.
        """

        if len(self.drawing_pile) > 0:
            return self.drawing_pile.pop(), False

        # Flip over playing pile
        playing_pile_top_card = self.discard_pile.pop()
        self.drawing_pile = list(reversed(self.discard_pile))
        self.discard_pile = [playing_pile_top_card]

        if len(self.drawing_pile) == 0:  # no cards even after flip, deck is empty
            msg = "No cards available to draw."
            raise DeckEmptyError(msg)

        return self.drawing_pile.pop(), True

    def play_card(self, card: Card) -> CardEffect:
        """
        Take a card and put it on top of the discard pile.

        Returns:
          The effect of the played card.
        """

        self.discard_pile.append(card)
        return card.effect

    def available_card_count(self) -> int:
        return len(self.drawing_pile) + len(self.discard_pile)
