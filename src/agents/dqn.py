import argparse
from typing import Any
import collections

import numpy as np
import torch
import torch.nn as nn

from agents.base import BaseAgent
from agents.greedy import GreedyAgent
from agents.random import RandomAgent
from agents.trainable import TrainableAgent
from agents.utils import (
    CARD_TO_INDEX,
    DRAW_ACTION,
    SUIT_TO_INDEX,
    NUM_ACTIONS,
    ACTION_TO_INDEX,
    Action,
    CardIndex,
    SuitIndex,
    ReplayBuffer,
)
from game.card import Card
from game.card_utils import CardEffect, Rank, Suit
from game.env import PrsiEnv
from game.game_state import GameState, find_allowed_cards
from random import choice

parser = argparse.ArgumentParser()

# TODO: fix seeding, doesn't work properly currently

# OPTIONS
# ------------------------------
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--evaluate_for", default=10_000, type=int)
parser.add_argument("--load_model", action="store_true")
parser.add_argument("--model_path", default="agent-strategies/dqn/model.pth", type=str)
parser.add_argument("--log_each", default=10_000, type=int)

# HYPERPARAMETERS
# ------------------------------
parser.add_argument("--episodes", default=1_000_000, type=int)
parser.add_argument("--epsilon", default=0.2, type=float)
parser.add_argument("--epsilon_decay", default=1, type=float)
parser.add_argument("--min_epsilon", default=0.05, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--replay_buffer_size", default=100_000, type=int)
parser.add_argument("--target_update_freq", default=1_000, type=int)
parser.add_argument("--hidden_layer_size", default=1024, type=int)
parser.add_argument(
    "--hand_state_option",
    default="full",
    type=str,
    choices=["count", "count_truncated", "simple", "full", "full_simple"],
)
parser.add_argument("--truncated_hand_size", default=4, type=int)
parser.add_argument(
    "--played_subset",
    default="all",
    type=str,
    choices=["sevens", "specials", "all"],
)
parser.add_argument(
    "--opponent", default="greedy", type=str, choices=["random", "greedy"]
)


SIMPLE_HAND_INDICES = {
    Suit.BELLS: 0,
    Suit.HEARTS: 1,
    Suit.LEAVES: 2,
    Suit.ACORNS: 3,
}


def _state_to_vector(
    hand_state: np.uint32,
    opponent_card_count: int,
    top_card: CardIndex,
    active_suit: SuitIndex,
    card_effect: CardEffect,
    effect_strength: np.uint8,
    played_subset: list[np.uint8],
) -> np.ndarray:
    """Pack everything into a 1-D float32 array."""
    return np.array(
        [
            hand_state,
            opponent_card_count,
            top_card,
            active_suit,
            card_effect.value,
            effect_strength,
            *[v for v in played_subset],
        ],
        dtype=np.float32,
    )


class QNetwork(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = collections.namedtuple(
    "Transition", ["state", "action_idx", "reward", "done", "next_state"]
)


class DQNAgent(TrainableAgent):
    def __init__(
        self, args: argparse.Namespace | None = None, path: str | None = None
    ) -> None:
        if args is None and path is None:
            raise ValueError("Agent needs either args or a path to load them from.")

        self.played_cards_subset: list[np.uint8]

        if args is None:
            self.load(path)  # type: ignore
            return

        self.args = args
        self._init_played_subset()
        self._build_networks()

    def _build_networks(self) -> None:
        self.online_net = QNetwork(NUM_ACTIONS, self.args.hidden_layer_size).to(
            QNetwork.device
        )
        self.target_net = QNetwork(NUM_ACTIONS, self.args.hidden_layer_size).to(
            QNetwork.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.args.learning_rate
        )
        self.loss = nn.MSELoss()

    def _init_played_subset(self) -> None:
        match self.args.played_subset:
            case "specials":
                self.played_cards_subset = [np.uint8(0)] * 3
            case "sevens":
                self.played_cards_subset = [np.uint8(0)]
            case "all":
                self.played_cards_subset = [np.uint8(0)] * 32
            case _:
                raise ValueError("Invalid played_subset argument.")

    def train(self, env: PrsiEnv) -> None:
        replay_buffer = ReplayBuffer(max_length=self.args.replay_buffer_size)
        total_steps = 0
        batch_wins = 0

        for episode in range(self.args.episodes):
            game_state, info = env.reset()
            hand: set[Card] = info["hand"]
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False
            reward = 0.0

            while not done:
                state_vec = self._process_state(game_state, info, hand)
                action = self.choose_action(game_state, hand, info)
                action_idx = ACTION_TO_INDEX[action]

                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

                next_state_vec = self._process_state(game_state, info, hand)
                replay_buffer.append(
                    Transition(state_vec, action_idx, reward, done, next_state_vec)
                )
                total_steps += 1

                if len(replay_buffer) >= self.args.batch_size:
                    self._learn(replay_buffer)

                if total_steps % self.args.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

            if reward > 0:
                batch_wins += 1

            if self.args.epsilon > self.args.min_epsilon:
                self.args.epsilon *= self.args.epsilon_decay

            if (episode + 1) % self.args.log_each == 0:
                self.log(episode, batch_wins)
                batch_wins = 0

    def _learn(self, replay_buffer: ReplayBuffer) -> None:
        batch: list[Transition] = replay_buffer.sample(self.args.batch_size)  # type: ignore

        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32
        ).to(QNetwork.device)
        action_idxs = torch.tensor(
            np.array([t.action_idx for t in batch]), dtype=torch.long
        ).to(QNetwork.device)
        rewards = torch.tensor(
            np.array([t.reward for t in batch]), dtype=torch.float32
        ).to(QNetwork.device)
        dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float32).to(
            QNetwork.device
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32
        ).to(QNetwork.device)

        q_prediciton = (
            self.online_net(states).gather(1, action_idxs.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            target_q = rewards + self.args.gamma * (
                self.target_net(next_states).max(dim=1).values
            ) * (1.0 - dones)

        loss = self.loss(q_prediciton, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, env: PrsiEnv, episodes: int) -> None:
        original_epsilon = self.args.epsilon
        self.args.epsilon = 0.0

        env.reset(full=True)
        wins = 0
        for _ in range(episodes):
            game_state, info = env.reset()
            hand: set[Card] = info["hand"]
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
            done = False
            reward = 0.0

            while not done:
                action = self.choose_action(game_state, hand, info)
                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

            if reward > 0:
                wins += 1

        self.args.epsilon = original_epsilon
        print(f"Evaluation: {wins}/{episodes} wins ({wins / episodes:.2%})")

    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        """Return (env_action, action_index_for_replay_buffer)."""
        valid_actions = self._get_valid_actions(state, hand)

        # ε-greedy exploration
        if np.random.random() < self.args.epsilon:
            action = choice(valid_actions)
            return action

        # Greedy: pick valid action with highest Q-value
        state_vec = self._process_state(state, info, hand)
        state_tensor = torch.tensor(state_vec[np.newaxis], dtype=torch.float32).to(
            QNetwork.device
        )
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)[0].cpu().numpy()

        best_action = max(valid_actions, key=lambda a: q_values[ACTION_TO_INDEX[a]])
        return best_action

    def _process_state(
        self, state: GameState, info: dict[str, Any], hand: set[Card]
    ) -> np.ndarray:
        if state.top_card is None:
            raise ValueError("Can't process state without a top card.")

        hand_state = self._get_hand_state(hand)
        opponent_card_count = info.get("opponent_card_count", 0)
        if (
            self.args.hand_state_option == "count_truncated"
            and opponent_card_count > self.args.truncated_hand_size
        ):
            opponent_card_count = self.args.truncated_hand_size

        top_card = CARD_TO_INDEX[state.top_card]
        active_suit = SUIT_TO_INDEX[state.actual_suit]
        card_effect = state.current_effect
        effect_strength = np.uint8(state.effect_strength)
        self._update_subset(state.top_card, info.get("deck_flipped_over", False))

        return _state_to_vector(
            hand_state,
            opponent_card_count,
            top_card,
            active_suit,
            card_effect,
            effect_strength,
            self.played_cards_subset,
        )

    def _get_hand_state(self, hand: set[Card]) -> np.uint32:
        match self.args.hand_state_option:
            case "count_truncated":
                hand_size = min(len(hand), self.args.truncated_hand_size)
                return np.uint32(hand_size)
            case "count":
                return np.uint32(len(hand))
            case "simple":
                state_array = np.zeros(7, dtype=np.uint8)
                for card in hand:
                    state_array[SIMPLE_HAND_INDICES[card.suit]] += 1
                    match card.rank:
                        case Rank.SEVEN:
                            state_array[4] += 1
                        case Rank.OBER:
                            state_array[5] += 1
                        case Rank.ACE:
                            state_array[6] += 1
                packed = np.uint32(0)
                for i in range(4):
                    packed |= np.uint32(state_array[i] & 0xF) << (i * 4)
                for i in range(3):
                    packed |= np.uint32(state_array[i + 4] & 0x7) << (16 + i * 3)
                return packed
            case "full":
                packed = np.uint32(0)
                for card in hand:
                    packed |= np.uint32(1) << (CARD_TO_INDEX[card] - 1)
                return packed
            case "full_simple":
                if len(hand) > self.args.truncated_hand_size:
                    return np.uint32(0xFFFFFFFF)
                packed = np.uint32(0)
                for card in hand:
                    packed |= np.uint32(1) << (CARD_TO_INDEX[card] - 1)
                return packed
            case _:
                raise ValueError("Invalid hand_state_option.")

    def _update_subset(self, card: Card, deck_flipped_over: bool) -> None:
        if deck_flipped_over:
            self.played_cards_subset = [np.uint8(0)] * len(self.played_cards_subset)
        match self.args.played_subset:
            case "all":
                idx = CARD_TO_INDEX[card] - 1
                self.played_cards_subset[idx] = np.uint8(
                    self.played_cards_subset[idx] + 1
                )
            case "specials":
                if card.rank == Rank.SEVEN:
                    self.played_cards_subset[0] = np.uint8(
                        self.played_cards_subset[0] + 1
                    )
                elif card.rank == Rank.OBER:
                    self.played_cards_subset[1] = np.uint8(
                        self.played_cards_subset[1] + 1
                    )
                elif card.rank == Rank.ACE:
                    self.played_cards_subset[2] = np.uint8(
                        self.played_cards_subset[2] + 1
                    )
            case "sevens":
                if card.rank == Rank.SEVEN:
                    self.played_cards_subset[0] = np.uint8(
                        self.played_cards_subset[0] + 1
                    )

    def _get_valid_actions(
        self, game_state: GameState, hand: set[Card]
    ) -> list[Action]:
        valid_actions: list[Action] = []
        allowed_cards = find_allowed_cards(game_state)

        for card in hand:
            if card in allowed_cards:
                if card.rank == Rank.OBER:
                    for suit_idx in range(1, 5):
                        valid_actions.append((CARD_TO_INDEX[card], suit_idx))
                else:
                    valid_actions.append(
                        (CARD_TO_INDEX[card], SUIT_TO_INDEX[card.suit])
                    )

        valid_actions.append(DRAW_ACTION)
        return valid_actions

    def save(self, path: str) -> None:
        print(f"Saving model to {path}")
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "args": vars(self.args),
            },
            path,
        )
        print("Model saved successfully!")

    def load(self, path: str) -> None:
        print(f"Loading model from {path}")
        data = torch.load(path, map_location=QNetwork.device)
        args_dict = data.get("args", {})
        self.args = argparse.Namespace(**args_dict)
        self._init_played_subset()
        self._build_networks()
        self.online_net.load_state_dict(data["online_net"])
        self.target_net.load_state_dict(data["target_net"])
        print("Model loaded successfully!")

    def log(self, episode: int, batch_wins: int) -> None:
        print(
            f"Episode {episode + 1:_}/{self.args.episodes:_}, "
            f"Epsilon: {self.args.epsilon:.4f}, "
            f"Batch win rate: {batch_wins / self.args.log_each:.2%}"
        )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    opponent: BaseAgent
    match args.opponent:
        case "random":
            opponent = RandomAgent()
        case "greedy":
            opponent = GreedyAgent()
        case _:
            raise ValueError("Invalid opponent")

    env = PrsiEnv(opponent)
    agent = DQNAgent(args=args)

    if args.load_model:
        agent.load(args.model_path)
    else:
        agent.train(env)
        agent.save(args.model_path)

    agent.evaluate(env, episodes=args.evaluate_for)
