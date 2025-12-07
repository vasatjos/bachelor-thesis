from game.card import Card


class Player:
    def __init__(self, player_id: int) -> None:
        self.hand_set: set[Card] = set()
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
        return len(self.hand_set)

    def take_drawn_cards(self, drawn_cards: list[Card]) -> None:
        self.hand_set.update(drawn_cards)

    def play_card(self, card: Card) -> None:
        if card not in self.hand_set:
            raise KeyError("Selected card not found in player's hand.")
        self.hand_set.remove(card)
