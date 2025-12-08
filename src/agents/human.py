from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank, Suit, COLOR_RESET


class HumanAgent(BaseAgent):
    # NOTE: maybe could be used in final thesis evaluation against agents?
    def evaluate(self) -> None:
        raise NotImplementedError("Human player can't be evaluated")

    def save(self, path: str) -> None:
        raise NotImplementedError("Human player strategy can't be saved.")

    def train(self) -> None:
        raise NotImplementedError("Human player strategy can't be trained.")

    def choose_action(self, state: Any, hand: set[Card]) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for human agent")

    def _print_hand(
        self, cards: list[Card] | None = None, hand: set[Card] | None = None
    ) -> None:
        """
        Print given cards in a sorted order.
        """
        if cards is None:
            if hand is None:
                raise ValueError("Must provide either cards or hand")
            cards = list(hand)
        cards.sort()
        for i, card in enumerate(cards, start=1):
            print(f"{i:>3}. {card}")

    def _select_card_to_play(
        self, allowed: set[Card], hand: set[Card]
    ) -> tuple[Card | None, Suit | None]:
        """
        Select a card from the players hand which he will play.

        Parameters:
          allowed: A set of cards which can be played based on the state of the game
            and active effects.
          hand: The player's current hand.

        Returns:
          None if player chose to draw a card.
          Otherwise simply the card the player chose to play.
        """
        playable = list(hand & allowed)
        if len(playable) == 0:
            input("No cards available, press enter to draw/skip.")
            return None, None
        self._print_hand(playable)

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
        return chosen_card, self._get_suit_choice(chosen_card)

    @staticmethod
    def _get_suit_choice(card: Card) -> Suit:
        if card.rank != Rank.OBER:
            return card.suit

        suit_names = [
            f"{suit.value}({suit.name[0]}){suit.name[1:]}{COLOR_RESET}" for suit in Suit
        ]
        print(f"Available suits: {', '.join(suit_names)}")

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
