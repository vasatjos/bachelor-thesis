from game.card import Card
from game.card_utils import Rank, Suit


# Action: a tuple of card + suit change, agents decides these 2 separately
# TODO: decide if game decides validity of change or agent decides validity of change

CardIndex = int
SuitIndex = int
Action = tuple[CardIndex, SuitIndex]

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

INDEX_TO_SUIT: dict[SuitIndex, Suit] = {
    i: suit for i, suit in enumerate(Suit)
}
SUIT_TO_INDEX: dict[Suit, SuitIndex] = {
    suit: i for i, suit in enumerate(Suit)
}
