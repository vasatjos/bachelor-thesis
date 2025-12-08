from typing import Any
import argparse
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action, CardIndex, SuitIndex
from game.card import Card
from game.card_utils import CardEffect, Rank, Suit
from game.deck import Deck
from game.env import PrsiEnv
from game.game_state import GameState
from game.player import Player
from random import choice, randint
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--episodes", default=100_000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.2, type=float, help="Discount factor.")
parser.add_argument(
    "--hand_state_option", default="none", type=str, choices=["none", "simple", "full"]
)
parser.add_argument(
    "--played_subset",
    default="specials",
    type=str,
    choices=["sevens", "specials", "all"],
)
parser.add_argument(
    "--evaluate_for", default=200, type=int, help="Evaluation episodes."
)
parser.add_argument(
    "--model_path", default="mc-model.plk", type=str, help="Path to save model."
)
# TODO: load model
# TODO: add epsilon decay


# No hand state:
# Don't use hand in state at all

# Full hand state:
# bit array (converted -> int)

# Simple hand state:
# num of normal cards for each color + num of obers + num of aces + num of sevens
# maybe sort into buckets: 0, 1, 2+
# still probably bit array


"""
    hand_state: bit array -> u32, might be unused, depends on hyperparameter
    opponent card count: u8
    top card: CardIndex (int)
    active suit: SuitIndex (int)
    CardEffect: IntEnum
    Effect strength: u8
    played cards subset: list of ints, size depends on hyperparameter
"""
State = tuple[
    np.uint32,
    np.uint8,
    CardIndex,
    SuitIndex,
    CardEffect,
    np.uint8,
    tuple[np.uint8, ...]
]


class MonteCarloAgent(BaseAgent):
    # estimate for actions in unseen states
    _default_actions = np.zeros(len(CARD_TO_INDEX)), np.zeros(len(SUIT_TO_INDEX))

    def __init__(
        self, args: argparse.Namespace, player_info: Player | None = None
    ) -> None:
        super().__init__(player_info)
        # both indexed by state + action
        self.action_value_fn: dict[State, dict[Action, float]] = {}
        self.num_visits: dict[State, dict[Action, int]] = {}

        self.played_cards_subset: list[np.uint8]
        self.played_cards_subset_option: str
        self._init_played_subset(args)

        self.train_episodes = args.episodes
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.hand_state_option = args.hand_state_option

    def train(self, env: PrsiEnv, **opts: dict[str, Any]) -> None:
        for episode in range(self.train_episodes):
            # Reset environment and played cards tracking
            game_state = env.reset()
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False

            # Collect episode trajectory
            states: list[State] = []
            actions: list[Action] = []
            rewards: list[float] = []

            while not done:
                info = {"opponent_card_count": 0, "deck_flipped_over": False}
                state = self._process_state(game_state, info)
                action = self.choose_action(game_state)

                game_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

            # Compute returns from rewards
            G = 0.0
            returns: list[float] = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns.reverse()

            # Update Q-values (first-visit MC)
            visited: set[tuple[State, Action]] = set()
            for t in range(len(states)):
                state = states[t]
                action = actions[t]
                sa_pair = (state, action)

                if sa_pair not in visited:
                    visited.add(sa_pair)

                    # Initialize if not seen before
                    if state not in self.action_value_fn:
                        self.action_value_fn[state] = {}
                        self.num_visits[state] = {}
                    if action not in self.action_value_fn[state]:
                        self.action_value_fn[state][action] = 0.0
                        self.num_visits[state][action] = 0

                    # Incremental mean update
                    self.num_visits[state][action] += 1
                    self.action_value_fn[state][action] += (
                        returns[t] - self.action_value_fn[state][action]
                    ) / self.num_visits[state][action]

            if (episode + 1) % 10_000 == 0:
                print(
                    f"Episode {episode + 1}/{self. train_episodes}, "
                    f"States seen: {len(self.action_value_fn)}"
                )

    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # Greedy evaluation

        wins = 0
        for _ in range(episodes):
            game_state = env.reset()
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False

            while not done:
                action = self.choose_action(game_state)
                game_state, reward, done, _ = env.step(action)

            if reward > 0:  # type: ignore
                wins += 1

        self.epsilon = original_epsilon
        win_rate = wins / episodes
        print(f"Evaluation: {wins}/{episodes} wins ({win_rate:.2%})")

    def choose_action(self, state: Any) -> Action:
        if self.player_info is None:
            raise RuntimeError("Player info hasn't been set.  Use `set_player_info`.")

        valid_actions = self._get_valid_actions(state)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return choice(valid_actions)

        # Greedy: find best Q-value among valid actions
        info = {"opponent_card_count": 0, "deck_flipped_over": False}
        processed_state = self._process_state(state, info)

        best_action = valid_actions[0]
        best_value = self.action_value_fn.get(processed_state, {}).get(best_action, 0.0)

        for action in valid_actions[1:]:
            value = self.action_value_fn.get(processed_state, {}).get(action, 0.0)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def save(self, path: str) -> None:
        import pickle

        data = {
            "action_value_fn": self.action_value_fn,
            "num_visits": self.num_visits,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "hand_state_option": self.hand_state_option,
            "played_cards_subset_option": self.played_cards_subset_option,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def _init_played_subset(self, args: argparse.Namespace) -> None:
        match args.played_subset:
            case "specials":  # sevens + obers + aces
                self.played_cards_subset = [np.uint8(0)] * 3
            case "sevens":
                self.played_cards_subset = [np.uint8(0)]
            case "all":
                self.played_cards_subset = [np.uint8(0)] * 32
            case _:
                raise ValueError("Invalid argument.")
        self.played_cards_subset_option = args.played_subset

    def _process_state(self, state: GameState, info: dict[str, Any]) -> State:
        """Correctly set played_cards_subset based on the state"""

        if state.top_card is None:
            raise ValueError("Can't process state without top card.")

        hand_state = self._get_hand_state()
        opponent_card_count = info.get("opponent_card_count", 0)
        top_card = CARD_TO_INDEX[state.top_card]
        active_suit = SUIT_TO_INDEX[state.actual_suit]
        card_effect = state.current_effect
        effect_strength = np.uint8(state.effect_strength)
        self._update_subset(state.top_card, info.get("deck_flipped_over", False))

        return (
            hand_state,
            opponent_card_count,
            top_card,
            active_suit,
            card_effect,
            effect_strength,
            tuple(self.played_cards_subset),
        )

    def _get_hand_state(self) -> np.uint32:
        match self.hand_state_option:
            case "none":
                return np.uint32(0)
            case "simple":
                raise NotImplementedError('TODO: _get_hand_state for "simple"')
            case "full":
                raise NotImplementedError('TODO: _get_hand_state for "full"')
            case _:
                raise ValueError("Invalid hand_state_option.")

    def _update_subset(self, card: Card, deck_flipped_over: bool) -> None:
        if deck_flipped_over:
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
        match self.played_cards_subset_option:
            case "all":
                idx = CARD_TO_INDEX[card] - 1  # None is 0
                self.played_cards_subset[idx] += 1
            case "specials":
                if (
                    card.rank != Rank.SEVEN
                    or card.rank != Rank.OBER
                    or card.rank != Rank.ACE
                ):
                    return
                if card.rank == Rank.SEVEN:
                    self.played_cards_subset[0] += 1
                if card.rank == Rank.OBER:
                    self.played_cards_subset[1] += 1
                if card.rank == Rank.ACE:
                    self.played_cards_subset[2] += 1
            case "sevens":
                if card.rank != Rank.SEVEN:
                    return
                self.played_cards_subset[0] += 1

    def _get_valid_actions(self, game_state: GameState) -> list[Action]:
        """Get list of valid actions given current game state and hand."""
        if self.player_info is None:
            raise RuntimeError("Player info hasn't been set. Use `set_player_info`.")

        valid_actions: list[Action] = []
        allowed_cards = PrsiEnv.find_allowed_cards(game_state)

        for card in self.player_info.hand_set:
            if card in allowed_cards:
                if card.rank == Rank.OBER:
                    # Ober can change suit to any of the 4 suits
                    for suit_idx in range(1, 5):
                        valid_actions.append((CARD_TO_INDEX[card], suit_idx))
                else:
                    valid_actions.append(
                        (CARD_TO_INDEX[card], SUIT_TO_INDEX[card.suit])
                    )

        # Can always draw a card
        valid_actions.append((0, 0))

        return valid_actions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = PrsiEnv()
    agent = MonteCarloAgent(args, Player(0))
    agent.train(env)
    agent.evaluate(env, episodes=args.evaluate_for)
    agent.save("mc_agent.pkl")
