import os
from typing import Any
from agents.base import BaseAgent
from agents.greedy import GreedyAgent
from agents.monte_carlo import MonteCarloAgent
from agents.random import RandomAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank, Suit, COLOR_RESET
import argparse

from game.env import PrsiEnv
from game.game_state import GameState, find_allowed_cards

parser = argparse.ArgumentParser()

# TODO: fix seeding, doesn't work properly currently
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--evaluate_for", default=1, type=int, help="Evaluation episodes."
)
parser.add_argument(
    "--model_path", default="agent-strategies/mc_agent.pkl", type=str, help="Path to load model."
)
parser.add_argument("--load_model", action="store_true", help="Load model from disk.")
parser.add_argument("--opponent", default="greedy", type=str, choices=["random", "greedy", "monte_carlo"])


class HumanAgent(BaseAgent):
    def save(self, path: str) -> None:
        raise NotImplementedError("Human player strategy can't be saved.")

    def train(self) -> None:
        raise NotImplementedError("Human player strategy can't be trained.")

    def load(self, path: str) -> None:
        raise NotImplementedError("Human player strategy can't be loaded.")

    def choose_action(self, state: GameState, hand: set[Card], opponent_card_count: int) -> Action:
        top_card = state.top_card
        active_suit = state.actual_suit
        if top_card is None or active_suit is None:
            raise RuntimeError("Invalid game state values.")
        allowed = find_allowed_cards(state)
        os.system("clear")
        print(f"Top card: {top_card}")
        if top_card.rank is Rank.OBER:
            print(f"Suit: {active_suit.value}{active_suit.name}{COLOR_RESET}")
        print(f"Opponent card count: {opponent_card_count}")
        card, suit = self._select_card_to_play(allowed, hand)
        return CARD_TO_INDEX[card], SUIT_TO_INDEX[suit]


    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        wins = 0
        for _ in range(episodes):
            game_state, info = env.reset()
            hand = info["hand"]
            done = False

            reward = 0
            while not done:
                action = self.choose_action(game_state, hand, info["opponent_card_count"])
                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

            if reward > 0:
                wins += 1

        win_rate = wins / episodes
        print(f"Evaluation: {wins}/{episodes} wins ({win_rate:.2%})")


    def _print_hand(
        self, cards: list[Card] | set[Card]
    ) -> None:
        """
        Print given cards in a sorted order.
        """
        cards = list(cards)
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

        print("\nHand:")
        self._print_hand(hand)
        print("\nAvailable cards:")
        playable = sorted(hand & allowed)
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


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    match args.opponent:
        case "random":
            opponent = RandomAgent()
        case "greedy":
            opponent = GreedyAgent()
        case "monte_carlo":
            opponent = MonteCarloAgent(args) # TODO: default args here
        case _:
            raise ValueError("Invalid opponent")

    opponent.load(args.model_path)
    env = PrsiEnv(opponent)
    agent = HumanAgent()
    agent.evaluate(env, episodes=args.evaluate_for)
