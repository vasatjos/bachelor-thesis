from game.card import Card
from game.card_utils import Rank, Suit

CardIndex = int
SuitIndex = int
Action = tuple[CardIndex, SuitIndex]
DRAW_ACTION = 0, 0

ACTION_SPACE_SIZE = 1 + 32 + 1 + 4 # 2*None + cards + suits

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
