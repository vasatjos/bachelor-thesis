import bisect
from prsi.card import Card


class Player:
    """
    Player representation. Keeps track of which cards
    a player has on hand.
    """

    def __init__(self, player_id: int) -> None:
        self.hand: list[Card] = []
        self._id = player_id

    @property
    def id(self) -> int:
        return self._id

    def __eq__(self, other) -> bool:
        if not isinstance(other, Player):
            return False
        return self._id == other._id

    @property
    def card_count(self) -> int:
        return len(self.hand)

    def take_drawn_cards(self, drawn_cards: list[Card | None]) -> None:
        for card in drawn_cards:
            if card is not None:
                bisect.insort(self.hand, card)

    def play_card(self, card: Card) -> None:
        if card not in self.hand:
            raise KeyError("Selected card not found in player's hand.")
        self.hand.remove(card)
