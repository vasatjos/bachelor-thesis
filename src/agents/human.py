from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank, Suit
from game.player import Player
from random import choice, randint


class HumanAgent(BaseAgent):
    def __init__(self, player_info: Player | None = None) -> None:
        super().__init__(player_info)

    # NOTE: maybe could be used in final thesis evaluation against agents?
    def evaluate(self) -> None:
        raise NotImplementedError("Human player can't be evaluated")

    def save(self, path: str) -> None:
        raise NotImplementedError("Human player strategy can't be saved.")

    def train(self) -> None:
        raise NotImplementedError("Human player strategy can't be trained.")

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for human agent")

    def _prompt_player_for_card_choice(
        self, player: Player
    ) -> tuple[Card | None, Suit | None]:
        raise NotImplementedError("TODO: Human agent TUI card choice")

        # self._print_game_state(player)
        # allowed = self._effect_manager.find_allowed_cards()
        # print("\nPlayable cards:")
        # player_choice = player.select_card_to_play(allowed)
        # print()
        #
        # return player_choice

    def _print_hand(self, cards: list[Card] | None = None) -> None:
        """
        Print given cards in a sorted order.
        """
        if self.player_info is None:
            raise RuntimeError("Player info not initialized")

        if cards is None:
            cards = list(self.player_info.hand_set)
        cards.sort()
        for i, card in enumerate(cards, start=1):
            print(f"{i:>3}. {card}")

    def _select_card_to_play(
        self, allowed: set[Card]
    ) -> tuple[Card | None, Suit | None]:
        """
        Select a card from the players hand which he will play.

        Parameters:
          allowed: A set of cards which can be played based on the state of the game
            and active effects.

        Returns:
          None if player chose to draw a card.
          Otherwise simply the card the player chose to play.
        """

        if self.player_info is None:
            raise RuntimeError("Player info not initialized")

        playable = list(self.player_info.hand_set & allowed)
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
        self.player_info.hand_set.remove(chosen_card)
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
