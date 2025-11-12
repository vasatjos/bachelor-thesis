from game.card import Card
from game.card_utils import Rank, Suit


# TODO: decide on this
class OberWrapper(Card):
    def __init__(self, suit: Suit, rank: Rank, change: Suit) -> None:
        super().__init__(suit, rank)
        self.changes_to = change

# Other option: Have actions be a tuple of card + change, agents decides these 2 separately
# Game decides validity of change vs. agent decides validity of change



# These only work with option 2
ACTION_TO_CARD = {
    i: card for i, card in enumerate((Card(suit, rank) for rank in Rank for suit in Suit))
}

CARD_TO_ACTION = {
    card: i for i, card in enumerate((Card(suit, rank) for rank in Rank for suit in Suit))
}

ACTIONS = ...
