from random import shuffle

from game.card_utils import Suit, Rank, CardEffect
from game.card import Card

def generate_suit(suit: Suit) -> set[Card]:
    return {Card(suit, rank) for rank in Rank}

def generate_rank(rank: Rank) -> set[Card]:
    return {Card(suit, rank) for suit in Suit}

class Deck:
    CARD_COUNT = len(Suit) * len(Rank)
    SEVENS = generate_rank(Rank.SEVEN)
    ACES = generate_rank(Rank.ACE)

    def __init__(self) -> None:
        self.discard_pile: list[Card]
        self.drawing_pile: list[Card]

        self.reset()

    def reset(self) -> None:
        self.discard_pile: list[Card] = []
        self.drawing_pile: list[Card] = [
            Card(suit, rank) for suit in Suit for rank in Rank
        ]
        shuffle(self.drawing_pile)

        top_card, _ = self.draw_card()
        if top_card is None:
            raise ValueError("Deck reset failed.")

        self.play_card(top_card)

    def draw_card(self) -> tuple[Card | None, bool]:
        """
        Draw a card from the drawing pile.

        If the drawing pile is empty, the discard pile gets flipped over
        and becomes the drawing pile.
        Whether the deck was flipped over or not is returned.
        """

        if len(self.drawing_pile) > 0:
            return self.drawing_pile.pop(), False

        # Flip over playing pile
        playing_pile_top_card = self.discard_pile.pop()
        self.drawing_pile = list(reversed(self.discard_pile))
        self.discard_pile = [playing_pile_top_card]

        if len(self.drawing_pile) == 0:
            print("Warning, no cards available to draw!")
            return None, True  # No cards available, shouldn't happen often

        return self.drawing_pile.pop(), True

    def play_card(self, card: Card) -> CardEffect:
        """
        Take a card and put it on top of the discard pile.

        Returns:
          The effect of the played card.
        """

        self.discard_pile.append(card)
        return card.effect
