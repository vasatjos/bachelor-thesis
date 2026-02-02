import pickle
from typing import Any
import argparse
from agents.greedy import GreedyAgent
from agents.random import RandomAgent
from agents.trainable import TrainableAgent
from agents.utils import (
    CARD_TO_INDEX,
    DRAW_ACTION,
    SUIT_TO_INDEX,
    Action,
    CardIndex,
    SuitIndex,
)
from game.card import Card
from game.card_utils import CardEffect, Rank, Suit
from game.env import PrsiEnv
from game.game_state import GameState, find_allowed_cards
from random import choice, randint
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--episodes", default=1_000_000, type=int, help="Training episodes."
)
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument(
    "--epsilon_decay", default=1, type=float, help="Epsilon decay factor."
)
parser.add_argument("--min_epsilon", default=0.05, type=float, help="Minimum epsilon.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument(
    "--hand_state_option",
    default="count_truncated",
    type=str,
    choices=["count", "count_truncated", "simple", "full"],
)
parser.add_argument(
    "--truncated_hand_size",
    default=4,
    type=int,
    help="Max hand size for truncated count.",
)
parser.add_argument(
    "--played_subset",
    default="specials",
    type=str,
    choices=["sevens", "specials", "all"],
)
parser.add_argument(
    "--evaluate_for", default=500, type=int, help="Evaluation episodes."
)
parser.add_argument("--load_model", action="store_true", help="Load model from disk.")
parser.add_argument(
    "--model_path",
    default="agent-strategies/q-learning/model.pkl",
    type=str,
    help="Path to save/load model.",
)
parser.add_argument("--log_each", default=500, type=int, help="Log frequency.")
parser.add_argument(
    "--opponent", default="greedy", type=str, choices=["random", "greedy"]
)

State = tuple[
    np.uint32,
    np.uint8,
    CardIndex,
    SuitIndex,
    CardEffect,
    np.uint8,
    tuple[np.uint8, ...],
]


class QLearningAgent(TrainableAgent):
    SIMPLE_HAND_INDICES = {
        Suit.BELLS: 0,
        Suit.HEARTS: 1,
        Suit.LEAVES: 2,
        Suit.ACORNS: 3,
    }

    def __init__(
        self, args: argparse.Namespace | None = None, path: str | None = None
    ) -> None:
        if args is None and path is None:
            raise ValueError("Agent needs either args or a path to load them from.")

        # Q-value function indexed by state + action
        self.action_value_fn: dict[State, dict[Action, float]] = {}

        if args is None:  # agent is being used as opponent
            self.load(path)  # type: ignore
            return

        self.args = args
        self.epsilon = args.epsilon

        self.played_cards_subset: list[np.uint8]
        self._init_played_subset()

    def train(self, env: PrsiEnv) -> None:
        batch_wins = 0
        for episode in range(self.args.episodes):
            # Reset environment and played cards tracking
            game_state, info = env.reset()
            hand = info["hand"]
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False

            while not done:
                state = self._process_state(game_state, info, hand)
                action = self.choose_action(game_state, hand, info)

                next_game_state, reward, done, next_info = env.step(action)
                next_hand = next_info["hand"]

                # Q-Learning update
                if done:
                    # Terminal state has no future value
                    future_value = 0.0
                else:
                    next_state = self._process_state(
                        next_game_state, next_info, next_hand
                    )
                    future_value = self._get_max_q_value(
                        next_game_state, next_hand, next_state
                    )

                # Initialize Q-value if not seen before
                if state not in self.action_value_fn:
                    self.action_value_fn[state] = {}
                if action not in self.action_value_fn[state]:
                    self.action_value_fn[state][action] = 0.0

                # Q-learning update rule: Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))
                current_q = self.action_value_fn[state][action]
                update = self.args.alpha * (
                    reward + self.args.gamma * future_value - current_q
                )
                self.action_value_fn[state][action] += update

                # Move to next state
                game_state = next_game_state
                hand = next_hand
                info = next_info

                if done and reward == 1:
                    batch_wins += 1

            # Decay epsilon
            if self.epsilon > self.args.min_epsilon:
                self.epsilon *= self.args.epsilon_decay

            if (episode + 1) % self.args.log_each == 0:
                self.log(episode, batch_wins)
                batch_wins = 0

    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        env.reset(full=True)  # Agent starts first evaluation game
        wins = 0
        for _ in range(episodes):
            game_state, info = env.reset()
            hand = info["hand"]
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False

            reward = 0
            while not done:
                action = self.choose_action(game_state, hand, info)
                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

            if reward > 0:
                wins += 1

        self.epsilon = original_epsilon
        win_rate = wins / episodes
        print(f"Evaluation: {wins}/{episodes} wins ({win_rate:.2%})")

    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            playable = tuple(find_allowed_cards(state) & hand)

            playable_length = len(playable)
            if playable_length == 0 or randint(0, playable_length) == playable_length:
                return DRAW_ACTION

            card = choice(tuple(playable))
            suit_idx = (
                SUIT_TO_INDEX[card.suit] if card.rank != Rank.OBER else randint(1, 4)
            )
            return CARD_TO_INDEX[card], suit_idx

        processed_state = self._process_state(state, info, hand)
        valid_actions = self._get_valid_actions(state, hand)

        # Greedy: find best Q-value among valid actions
        best_action: Action = valid_actions[0]
        best_value = -np.inf

        for action in valid_actions:
            value = self.action_value_fn.get(processed_state, {}).get(action, 0.0)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _get_max_q_value(
        self, game_state: GameState, hand: set[Card], state: State
    ) -> float:
        """Get the maximum Q-value for the given state over all valid actions."""
        valid_actions = self._get_valid_actions(game_state, hand)

        max_value = -np.inf
        for action in valid_actions:
            value = self.action_value_fn.get(state, {}).get(action, 0.0)
            if value > max_value:
                max_value = value

        return max_value if max_value > -np.inf else 0.0

    def save(self, path: str) -> None:
        print(f"Saving model to {path}")
        data = {
            "action_value_fn": self.action_value_fn,
            "args": vars(self.args),
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Model saved successfully!")

    def load(self, path: str) -> None:
        print(f"Loading model from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.action_value_fn = data["action_value_fn"]
        args_dict = data.get("args", {})
        self.args = argparse.Namespace(**args_dict)
        self.epsilon = data.get("epsilon", self.args.epsilon)
        self._init_played_subset()
        print("Model loaded successfully!")

    def _init_played_subset(self) -> None:
        match self.args.played_subset:
            case "specials":  # sevens + obers + aces
                self.played_cards_subset = [np.uint8(0)] * 3
            case "sevens":
                self.played_cards_subset = [np.uint8(0)]
            case "all":
                self.played_cards_subset = [np.uint8(0)] * 32
            case _:
                raise ValueError("Invalid argument.")

    def _process_state(
        self, state: GameState, info: dict[str, Any], hand: set[Card]
    ) -> State:
        """Correctly set played_cards_subset based on the state"""

        if state.top_card is None:
            raise ValueError("Can't process state without top card.")

        hand_state = self._get_hand_state(hand)
        opponent_card_count = info.get("opponent_card_count", 0)
        if (
            self.args.hand_state_option == "count_truncated"
            and opponent_card_count > self.args.truncated_hand_size
        ):
            opponent_card_count = np.uint8(self.args.truncated_hand_size)
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

    def _get_hand_state(self, hand: set[Card]) -> np.uint32:
        match self.args.hand_state_option:
            case "count_truncated":
                hand_size = len(hand)
                if hand_size > self.args.truncated_hand_size:
                    hand_size = self.args.truncated_hand_size
                return np.uint32(hand_size)
            case "count":
                return np.uint32(len(hand))
            case "simple":
                state_array = np.zeros(7, dtype=np.uint8)
                for card in hand:
                    state_array[self.SIMPLE_HAND_INDICES[card.suit]] += 1
                    match card.rank:
                        case Rank.SEVEN:
                            state_array[4] += 1
                        case Rank.OBER:
                            state_array[5] += 1
                        case Rank.ACE:
                            state_array[6] += 1
                        case _:
                            pass

                packed = np.uint32(0)

                # Pack suits (4 bits each, max value 8)
                for i in range(4):
                    packed |= np.uint32(state_array[i] & 0xF) << (i * 4)

                # Pack specials (3 bits each, max value 4, starting at bit 16)
                for i in range(3):
                    packed |= np.uint32(state_array[i + 4] & 0x7) << (16 + i * 3)

                return packed

            case "full":
                packed = np.uint32(0)
                for card in hand:
                    packed |= np.uint32(1) << (CARD_TO_INDEX[card] - 1)
                return packed
            # TODO: option to act randomly on many cards
            case _:
                raise ValueError("Invalid hand_state_option.")

    def _update_subset(self, card: Card, deck_flipped_over: bool) -> None:
        if deck_flipped_over:
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
        match self.args.played_subset:
            case "all":
                idx = CARD_TO_INDEX[card] - 1  # None is 0, so indexing is 1 based
                self.played_cards_subset[idx] += 1
            case "specials":
                if (
                    card.rank != Rank.SEVEN
                    and card.rank != Rank.OBER
                    and card.rank != Rank.ACE
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

    def _get_valid_actions(
        self, game_state: GameState, hand: set[Card]
    ) -> list[Action]:
        """Get list of valid actions given current game state and hand."""
        valid_actions: list[Action] = []
        allowed_cards = find_allowed_cards(game_state)

        for card in hand:
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
        valid_actions.append(DRAW_ACTION)

        return valid_actions

    def log(self, episode: int, batch_wins: int) -> None:
        print(
            f"Episode {episode + 1:_}/{self.args.episodes:_}, "
            f"States seen: {len(self.action_value_fn):_}, "
            f"Batch win rate: {batch_wins / self.args.log_each:.2%}"
        )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    match args.opponent:
        case "random":
            opponent = RandomAgent()
        case "greedy":
            opponent = GreedyAgent()
        case _:
            raise ValueError("Invalid opponent")

    env = PrsiEnv(opponent)
    agent = QLearningAgent(args=args)
    if args.load_model:
        agent.load(args.model_path)
    else:
        agent.train(env)
        agent.save(args.model_path)

    agent.evaluate(env, episodes=args.evaluate_for)
