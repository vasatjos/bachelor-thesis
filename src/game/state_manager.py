from game.card_utils import Rank, Suit, CardEffect
from game.card import Card
from game.deck import Deck


class GameStateManager:
    """
    Class which handles the stacking of card effects and other
    information not regarding players themselves.
    """

    def __init__(self) -> None:
        self.top_card: Card | None = None
        self.actual_suit: Suit | None = None
        self.current_effect: CardEffect | None = None
        self.effect_strength: int = 0  # values range from 0 to 4 (draw 8 cards)

    def update(self, card: Card | None, suit: Suit) -> None:
        if card is None:
            self.current_effect = None  # No card was played -> No effect
            self.effect_strength = 0

        elif card.rank is Rank.SEVEN:
            self.current_effect = CardEffect.DRAW_TWO
            self.effect_strength += 1
            self.top_card = card
            self.actual_suit = card.suit

        elif card.rank is Rank.ACE:
            self.current_effect = CardEffect.SKIP_TURN
            self.effect_strength = 1
            self.top_card = card
            self.actual_suit = card.suit

        elif card.rank is Rank.OBER:
            self.current_effect = None
            self.effect_strength = 0
            self.top_card = card
            self.actual_suit = suit

        else:
            self.current_effect = None
            self.effect_strength = 0
            self.top_card = card
            self.actual_suit = card.suit

    def find_allowed_cards(self) -> set[Card]:
        if self.current_effect is CardEffect.SKIP_TURN:
            return Deck.generate_rank(Rank.ACE)

        if self.current_effect is CardEffect.DRAW_TWO:
            return Deck.generate_rank(Rank.SEVEN)

        if self.top_card is None or self.actual_suit is None:
            raise NotImplementedError("find_allowed_cards called too soon.")

        current_suit_cards = Deck.generate_suit(self.actual_suit)
        current_rank_cards = Deck.generate_rank(self.top_card.rank)
        obers = Deck.generate_rank(Rank.OBER)

        return current_suit_cards | current_rank_cards | obers
