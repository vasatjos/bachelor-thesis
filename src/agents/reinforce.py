import os
import argparse
import random
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from prsi.agents.agent import Agent
from prsi.agents.baselines import GreedyAgent, RandomAgent
from prsi.rl_utils import (
    CARD_TO_INDEX,
    SUIT_TO_INDEX,
    INDEX_TO_ACTION,
    ACTION_TO_INDEX,
    Action,
    CardIndex,
    SuitIndex,
    get_valid_action_mask,
    DRAW_ACTION,
)
from prsi.card import Card
from prsi.card_utils import CardEffect, Rank
from prsi.env import PrsiEnv
from prsi.game_state import GameState
from agents.trainable import TrainableAgent
from agents.utils import Network

parser = argparse.ArgumentParser()

# OPTIONS
# ------------------------------
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--evaluate_for", default=10_000, type=int)
parser.add_argument("--load_model", action="store_true")
parser.add_argument(
    "--model_path",
    default="agent_strategies/reinforce/",
    type=str,
    help="Base path to save/load model. A subdirectory for the hyperparameters will be created here.",
)
parser.add_argument("--log_each", default=1_000, type=int)
parser.add_argument("--save_each", default=None, type=int)
parser.add_argument(
    "--disable_csv_logging",
    action="store_true",
    help="Disable saving logs to logs.csv.",
)

# HYPERPARAMETERS
# ------------------------------
parser.add_argument("--episodes", default=500_000, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--batch_size", default=32, type=int, help="Episodes per update.")
parser.add_argument("--hidden_layer_size", default=512, type=int)
parser.add_argument("--hidden_layer_count", default=2, type=int)
parser.add_argument("--entropy_regularization", default=0.01, type=float)
parser.add_argument("--baseline", action="store_true", help="Use value baseline.")
parser.add_argument(
    "--normalize_advantage", action="store_true", help="Normalize advantages per batch."
)
parser.add_argument(
    "--opponent",
    default="greedy",
    type=str,
    choices=["random", "greedy"],
)
parser.add_argument(
    "--hand_state_option",
    default="full",
    type=str,
    choices=["count", "count_truncated", "simple", "full"],
)
parser.add_argument("--truncated_hand_size", default=4, type=int)
parser.add_argument(
    "--played_subset",
    default="all",
    type=str,
    choices=["sevens", "specials", "all"],
)
parser.add_argument("--self_play", action="store_true", help="Train using self-play.")
parser.add_argument(
    "--self_play_update_freq",
    default=1_000,
    type=int,
    help="Frequency of self-play opponent update.",
)


class REINFORCEAgent(TrainableAgent):
    def __init__(
        self, args: argparse.Namespace | None = None, path: str | None = None
    ) -> None:
        if args is None and path is None:
            raise ValueError("Agent needs either args or a path to load them from.")

        self.log_data: list[dict[str, Any]] = []
        self.played_cards_subset: np.ndarray
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args is None:
            self.load(path)  # type: ignore
            return

        self.args = args
        self._init_played_subset()
        self.input_size = self._get_input_size()
        self._build_networks()

        hyper_str = self._get_hyperparameter_string()
        self.save_dir = os.path.join(self.args.model_path, hyper_str)
        self.full_model_path = os.path.join(self.save_dir, "model.pth")
        self.csv_path = os.path.join(self.save_dir, "logs.csv")

        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def device(self) -> torch.device:
        return self._device

    def train(self, env: PrsiEnv) -> None:
        batch_states: list[np.ndarray] = []
        batch_actions: list[int] = []
        batch_returns: list[float] = []
        batch_masks: list[np.ndarray] = []
        batch_wins = 0

        draw_actions = 0
        total_actions = 0

        for episode in range(self.args.episodes):
            if self.args.self_play and episode % self.args.self_play_update_freq == 0:
                game_state, info = env.reset(opponent=self.clone())
            else:
                game_state, info = env.reset()

            hand: list[Card] = info["hand"]
            self.played_cards_subset = np.zeros(
                len(self.played_cards_subset), dtype=np.uint8
            )

            self._update_subset(info.get("new_cards", []), False)

            episode_states: list[np.ndarray] = []
            episode_actions: list[int] = []
            episode_rewards: list[float] = []
            episode_masks: list[np.ndarray] = []

            done = False
            reward = 0.0

            while not done:
                state_vec = self._process_state(game_state, info, hand)
                action = self.choose_action(game_state, hand, info)
                if action == DRAW_ACTION:
                    draw_actions += 1
                total_actions += 1

                action_idx = ACTION_TO_INDEX[action]
                valid_mask = get_valid_action_mask(game_state, hand)

                episode_states.append(state_vec)
                episode_actions.append(action_idx)
                episode_masks.append(valid_mask)

                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

                self._update_subset(
                    info.get("new_cards", []), info.get("deck_flipped_over", False)
                )

                episode_rewards.append(float(reward))

            if reward > 0:
                batch_wins += 1

            # Compute returns from rewards
            G = 0.0
            returns: list[float] = []
            for r in reversed(episode_rewards):
                G = r + self.args.gamma * G
                returns.append(G)
            returns.reverse()

            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_returns.extend(returns)
            batch_masks.extend(episode_masks)

            if (episode + 1) % self.args.batch_size == 0:
                self._learn(batch_states, batch_actions, batch_returns, batch_masks)
                batch_states, batch_actions, batch_returns, batch_masks = [], [], [], []

            if (episode + 1) % self.args.log_each == 0:
                self.log(episode, batch_wins, draw_actions, total_actions)
                batch_wins = 0
                draw_actions = 0
                total_actions = 0

            if (
                self.args.save_each is not None
                and (episode + 1) % self.args.save_each == 0
            ):
                self.save(self.full_model_path)

    def evaluate(self, env: PrsiEnv, episodes: int, opponent: Agent) -> None:
        self.policy_net.eval()

        env.reset(full=True, opponent=opponent)
        wins = 0

        for _ in range(episodes):
            game_state, info = env.reset()
            hand: list[Card] = info["hand"]
            self.played_cards_subset = np.zeros(
                len(self.played_cards_subset), dtype=np.uint8
            )

            self._update_subset(info.get("new_cards", []), False)

            done = False
            reward = 0.0

            while not done:
                action = self.choose_action(game_state, hand, info)
                game_state, reward, done, info = env.step(action)
                hand = info["hand"]

                self._update_subset(
                    info.get("new_cards", []), info.get("deck_flipped_over", False)
                )

            if reward > 0:
                wins += 1

        print(f"Evaluation: {wins}/{episodes} wins ({wins / episodes:.2%})")

    def choose_action(
        self, state: GameState, hand: list[Card], info: dict[str, Any]
    ) -> Action:
        state_vec = self._process_state(state, info, hand)
        state_tensor = torch.tensor(state_vec[np.newaxis], dtype=torch.float32).to(
            self.device
        )

        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_tensor).squeeze(0)

        # Mask invalid actions before sampling
        action_mask = torch.tensor(
            get_valid_action_mask(state, hand), dtype=torch.bool, device=self.device
        )
        logits[~action_mask] = -1e9
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        action_idx = np.random.choice(len(probs), p=probs)
        return INDEX_TO_ACTION[action_idx]

    def save(self, path: str) -> None:
        print(f"Saving model to {path}")
        payload: dict = {
            "policy_net": self.policy_net.state_dict(),
            "args": vars(self.args),
        }
        if self.value_net is not None:
            payload["value_net"] = self.value_net.state_dict()
        torch.save(payload, path)

        if not self.args.disable_csv_logging and self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.csv_path, index=False)

        print("Model saved successfully!")

    def load(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "model.pth")

        print(f"Loading model from {path}")
        data = torch.load(path, map_location=self.device)
        self.args = argparse.Namespace(**data["args"])
        self._init_played_subset()
        self.input_size = self._get_input_size()
        self._build_networks()
        self.policy_net.load_state_dict(data["policy_net"])
        if self.value_net is not None and "value_net" in data:
            self.value_net.load_state_dict(data["value_net"])
        print("Model loaded successfully!")

    def clone(self) -> "REINFORCEAgent":
        cloned = REINFORCEAgent.__new__(REINFORCEAgent)
        cloned.args = self.args
        cloned._device = self._device
        cloned.save_dir = self.save_dir
        cloned.full_model_path = self.full_model_path
        cloned.csv_path = self.csv_path
        cloned.log_data = []
        cloned._init_played_subset()
        cloned.input_size = self.input_size
        cloned._build_networks()

        cloned.policy_net.load_state_dict(self.policy_net.state_dict())
        cloned.policy_net.eval()

        if self.value_net is not None and cloned.value_net is not None:
            cloned.value_net.load_state_dict(self.value_net.state_dict())
            cloned.value_net.eval()

        return cloned

    def log(
        self, episode: int, batch_wins: int, draw_actions: int, total_actions: int
    ) -> None:
        batch_win_rate = batch_wins / self.args.log_each
        draw_action_rate = draw_actions / total_actions if total_actions > 0 else 0.0

        print(
            f"Episode {episode + 1:_}/{self.args.episodes:_}, "
            f"Draw-action rate: {draw_action_rate:.2%}, "
            f"Batch win rate: {batch_win_rate:.2%}"
        )

        if self.args.disable_csv_logging:
            return

        self.log_data.append(
            {
                "episode": episode + 1,
                "draw_action_rate": draw_action_rate,
                "batch_win_rate": batch_win_rate,
            }
        )

    def _build_networks(self) -> None:
        self.policy_net = Network(
            self.input_size,
            self.args.hidden_layer_size,
            self.args.hidden_layer_count,
            PrsiEnv.ACTION_SPACE_SIZE,
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.args.learning_rate
        )

        self.value_net: Network | None = None
        self.value_optimizer: torch.optim.Adam | None = None
        if self.args.baseline:
            self.value_net = Network(
                self.input_size,
                self.args.hidden_layer_size,
                self.args.hidden_layer_count,
                1,
            ).to(self.device)
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=self.args.learning_rate
            )

    def _get_input_size(self) -> int:
        match self.args.hand_state_option:
            case "count_truncated" | "count":
                hand_size = 1
            case "simple":
                hand_size = 7
            case "full":
                hand_size = 32
            case _:
                raise ValueError("Invalid hand_state_option")

        fixed_part = 1 + 32 + 4 + 3 + 1

        match self.args.played_subset:
            case "specials":
                played_size = 3
            case "sevens":
                played_size = 1
            case "all":
                played_size = 32
            case _:
                raise ValueError("Invalid played_subset")

        return hand_size + fixed_part + played_size

    def _learn(
        self,
        batch_states: list[np.ndarray],
        batch_actions: list[int],
        batch_returns: list[float],
        batch_masks: list[np.ndarray],
    ) -> None:
        states = torch.tensor(
            np.array(batch_states), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
        masks = torch.tensor(
            np.array(batch_masks), dtype=torch.bool, device=self.device
        )

        if self.args.baseline:
            assert self.value_net is not None
            assert self.value_optimizer is not None

            self.value_net.train()
            predicted_values = self.value_net(states).squeeze(1)
            value_loss = nn.functional.mse_loss(predicted_values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            with torch.no_grad():
                advantage = returns - self.value_net(states).squeeze(1)
        else:
            advantage = returns

        if self.args.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self.policy_net.train()
        logits = self.policy_net(states)

        # Mask invalid actions before calculating probabilities and entropy
        logits[~masks] = -1e9

        dist = Categorical(logits=logits)
        taken_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(taken_log_probs * advantage).mean()
        policy_loss -= self.args.entropy_regularization * entropy

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def _init_played_subset(self) -> None:
        match self.args.played_subset:
            case "specials":
                self.played_cards_subset = np.zeros(3, dtype=np.uint8)
            case "sevens":
                self.played_cards_subset = np.zeros(1, dtype=np.uint8)
            case "all":
                self.played_cards_subset = np.zeros(32, dtype=np.uint8)
            case _:
                raise ValueError("Invalid played_subset argument.")

    def _process_state(
        self, state: GameState, info: dict[str, Any], hand: list[Card]
    ) -> np.ndarray:
        if state.top_card is None or state.actual_suit is None:
            raise ValueError("Can't process state without a top card.")

        hand_state = self._get_hand_state(hand)
        opponent_card_count = info.get("opponent_card_count", 0)
        if (
            self.args.hand_state_option == "count_truncated"
            and opponent_card_count > self.args.truncated_hand_size
        ):
            opponent_card_count = self.args.truncated_hand_size

        top_card = self._handle_top_card(state.top_card)
        active_suit = SUIT_TO_INDEX[state.actual_suit]
        card_effect = state.current_effect
        effect_strength = np.uint8(state.effect_strength)

        return self._state_to_vector(
            hand_state,
            opponent_card_count,
            top_card,
            active_suit,
            card_effect,
            effect_strength,
            self.played_cards_subset,
        )

    def _state_to_vector(
        self,
        hand_state: list[np.uint8] | np.ndarray,
        opponent_card_count: int,
        top_card: CardIndex,
        active_suit: SuitIndex,
        card_effect: CardEffect,
        effect_strength: np.uint8,
        played_subset: np.ndarray,
    ) -> np.ndarray:
        """Pack everything into a 1-D float32 array, normalized into 0-1."""
        hand: list[np.uint8] | list[float] | np.ndarray

        card_count_denominator = 31
        if self.args.hand_state_option == "count_truncated":
            card_count_denominator = self.args.truncated_hand_size
            hand = [hand_state[0] / self.args.truncated_hand_size]
        elif self.args.hand_state_option == "count":
            hand = [hand_state[0] / 31]
        elif self.args.hand_state_option == "simple":  # always numpy array here
            hand = hand_state / 4  # type: ignore
        else:  # one-hot, no normalization
            hand = hand_state

        if self.args.played_subset == "all":
            played = np.asarray(played_subset, dtype=np.float32)  # one-hot, already 0-1
        else:
            played = np.asarray(played_subset, dtype=np.float32) / 4.0

        top_card_1hot = np.zeros(32)
        top_card_1hot[top_card] = 1
        active_suit_1hot = np.zeros(4)
        active_suit_1hot[active_suit] = 1
        card_effect_1hot = np.zeros(3)
        card_effect_1hot[card_effect.value] = 1

        return np.array(
            [
                *hand,
                opponent_card_count / card_count_denominator,
                *top_card_1hot,
                *active_suit_1hot,
                *card_effect_1hot,
                effect_strength / 4,  # values 0-4
                *played,
            ],
            dtype=np.float32,
        )

    def _get_hand_state(self, hand: list[Card]) -> list[np.uint8] | np.ndarray:
        match self.args.hand_state_option:
            case "count_truncated":
                return [np.uint8(min(len(hand), self.args.truncated_hand_size))]
            case "count":
                return [np.uint8(len(hand))]
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
                return state_array
            case "full":
                state_array = np.zeros(32, dtype=np.uint8)
                for card in hand:
                    state_array[CARD_TO_INDEX[card]] = 1
                return state_array
            case _:
                raise ValueError("Invalid hand_state_option.")

    def _handle_top_card(self, top_card: Card) -> CardIndex:
        if not self.args.hand_state_option.startswith("full"):
            return 0
        return CARD_TO_INDEX[top_card]

    def _update_subset(self, new_cards: list[Card], deck_flipped_over: bool) -> None:
        if deck_flipped_over:
            self.played_cards_subset = np.zeros(
                len(self.played_cards_subset), dtype=np.uint8
            )
        for card in new_cards:
            match self.args.played_subset:
                case "all":
                    idx = CARD_TO_INDEX[card]
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

    def _get_hyperparameter_string(self) -> str:
        hyper_parts = []
        hyper_parts.append(f"seed{self.args.seed}")

        hyper_parts.append(f"gamma{self.args.gamma}")
        hyper_parts.append(f"lr{self.args.learning_rate}")
        hyper_parts.append(f"bs{self.args.batch_size}")
        hyper_parts.append(f"ent{self.args.entropy_regularization}")

        if self.args.baseline:
            hyper_parts.append("base")
        if self.args.normalize_advantage:
            hyper_parts.append("norm")

        hyper_parts.append(
            f"hid{self.args.hidden_layer_count}x{self.args.hidden_layer_size}"
        )

        hyper_parts.append(f"hand_{self.args.hand_state_option}")
        if self.args.hand_state_option == "count_truncated":
            hyper_parts.append(f"trunc{self.args.truncated_hand_size}")

        hyper_parts.append(f"sub_{self.args.played_subset}")

        if self.args.self_play:
            hyper_parts.append("selfplay")
            hyper_parts.append(f"spfreq{self.args.self_play_update_freq}")

        return "-".join(hyper_parts)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.seed is None:
        args.seed = int(time())
        print(f"Auto-generated seed: {args.seed}")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    opponent: Agent
    match args.opponent:
        case "random":
            opponent = RandomAgent()
        case "greedy":
            opponent = GreedyAgent()
        case _:
            raise ValueError("Invalid opponent")

    env = PrsiEnv(opponent)

    if args.load_model:
        agent = REINFORCEAgent(path=args.model_path)
    else:
        agent = REINFORCEAgent(args=args)
        agent.train(env)
        agent.save(agent.full_model_path)

    agent.evaluate(env, episodes=args.evaluate_for, opponent=opponent)
