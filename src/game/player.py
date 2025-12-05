from game.card import Card
from game.card_utils import COLOR_RESET, Rank, Suit


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

    def card_count(self) -> int:
        return len(self.hand_set)

    def print_hand(self, cards: list[Card] | None = None) -> None:
        """
        Print given cards in a sorted order.
        """

        if cards is None:
            cards = list(self.hand_set)
        cards.sort()
        for i, card in enumerate(cards, start=1):
            print(f"{i:>3}. {card}")

    def select_card_to_play(self, allowed: set[Card]) -> tuple[Card | None, Suit | None]:
        """
        Select a card from the players hand which he will play.

        Parameters:
          allowed: A set of cards which can be played based on the state of the game
            and active effects.

        Returns:
          None if player chose to draw a card.
          Otherwise simply the card the player chose to play.
        """

        playable = list(self.hand_set & allowed)
        if len(playable) == 0:
            input("No cards available, press enter to draw/skip.")
            return None, None
        self.print_hand(playable)

        while True:
            choice_input = input(
                "Enter the number of the card you want to play, "
                + "don't enter anything to draw a card: "
            )
            if choice_input == "":
                return None, None

            try:
                choice = int(choice_input)
                if not 0 < choice <= len(playable):
                    print("Invalid input.")
                    continue
                break  # valid input
            except ValueError:
                print("Invalid input.")

        card_index = choice - 1
        chosen_card = playable[card_index]
        self.hand_set.remove(chosen_card)
        return chosen_card, self._get_suit_choice(chosen_card)

    @staticmethod
    def _get_suit_choice(card: Card) -> Suit:
        if card.rank != Rank.OBER:
            return card.suit

        suit_names = [
            f"{suit.value}({suit.name[0]}){suit.name[1:]}{COLOR_RESET}" for suit in Suit
        ]
        print(f"Available suits: {", ".join(suit_names)}")

        valid_suits = {"h", "l", "a", "b"}

        while True:
            choice = input("Please choose suit (first letter): ").strip().lower()
            if choice in valid_suits:
                break
            print("Please insert a valid suit letter.")

        if choice == "h":
            return Suit.HEARTS
        if choice == "l":
            return Suit.LEAVES
        if choice == "a":
            return Suit.ACORNS
        if choice == "b":
            return Suit.BELLS
        raise RuntimeError("Suit choice failed.")

    def take_drawn_cards(self, drawn_cards: list[Card]) -> None:
        self.hand_set.update(drawn_cards)
