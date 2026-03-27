import argparse
import os
from typing import Any
from prsi.agents.agent import Agent
from prsi.agents.baselines import GreedyAgent, RandomAgent
from prsi.rl_utils import Action
from prsi.card import Card, ICONS, USE_ICONS
from prsi.card_utils import Rank, Suit, COLOR_RESET
from prsi.env import PrsiEnv
from prsi.game_state import GameState, find_allowed_cards
from agents.monte_carlo import MonteCarloAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent

parser = argparse.ArgumentParser()

# TODO: fix seeding, doesn't work properly currently
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--evaluate_for", default=1, type=int, help="Evaluation episodes.")
parser.add_argument(
    "--model_path",
    default="",
    type=str,
    help="Path to load model.",
)
parser.add_argument(
    "--opponent",
    default="greedy",
    type=str,
    choices=["random", "greedy", "monte_carlo", "q_learning", "dqn"],
)


class HumanAgent(Agent):
    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        top_card = state.top_card
        active_suit = state.actual_suit
        if top_card is None or active_suit is None:
            raise RuntimeError("Invalid game state values.")

        allowed = find_allowed_cards(state)

        os.system("clear")
        print(f"Episode: {info['episode'] + 1}/{info['episodes']}")
        print(f"\nTop card: {top_card}")
        if top_card.rank is Rank.OBER:
            icon = f"{ICONS[active_suit]} " if USE_ICONS else ""
            print(f"Suit: {active_suit.value}{icon}{active_suit.name}{COLOR_RESET}")
        print(f"Opponent card count: {info.get('opponent_card_count', 0)}")

        return self._select_action(allowed, hand)

    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        wins = 0
        for i in range(episodes):
            game_state, info = env.reset()
            hand = info["hand"]
            done = False

            reward = 0.0

            info["episodes"] = episodes
            info["episode"] = i

            while not done:
                action = self.choose_action(game_state, hand, info)
                game_state, reward, done, info = env.step(action)
                info["episodes"] = episodes
                info["episode"] = i
                hand = info["hand"]

            if reward > 0:
                wins += 1

        win_rate = wins / episodes
        print(f"Evaluation: {wins}/{episodes} wins ({win_rate:.2%})")

    def _print_hand(
        self, cards: list[Card] | set[Card], show_numbers: bool = True
    ) -> None:
        """
        Print given cards in a sorted order.
        """
        cards = list(cards)
        cards.sort()
        for i, card in enumerate(cards, start=1):
            index = f"{i:>3}. " if show_numbers else ""
            print(f"{index}{card}")

    def _select_action(self, allowed: set[Card], hand: set[Card]) -> Action:
        """
        Select a card from the players hand which he will play.

        Parameters:
          allowed: A set of cards which can be played based on the state of the game
            and active effects.
          hand: The player's current hand.

        Returns:
          The action player chose to perform.
        """

        print("\nHand:")
        self._print_hand(hand, show_numbers=False)
        print("\nAvailable cards:")
        playable = sorted(hand & allowed)
        if len(playable) == 0:
            input("No cards available, press enter to draw/skip.")
            return None
        self._print_hand(playable)

        while True:
            choice_input = input(
                "Enter the number of the card you want to play, "
                + "don't enter anything to draw a card: "
            )
            if choice_input == "":
                return None

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

    opponent: Agent
    match args.opponent:
        case "random":
            opponent = RandomAgent()
        case "greedy":
            opponent = GreedyAgent()
        case "monte_carlo":
            opponent = MonteCarloAgent(path=args.model_path)
        case "q_learning":
            opponent = QLearningAgent(path=args.model_path)
        case "dqn":
            opponent = DQNAgent(path=args.model_path)
        case _:
            raise ValueError("Invalid opponent")

    env = PrsiEnv(opponent)
    agent = HumanAgent()
    agent.evaluate(env, episodes=args.evaluate_for)
