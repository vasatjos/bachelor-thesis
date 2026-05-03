import argparse
import random
from time import time
import numpy as np
import torch
from agents.dqn import DQNAgent
from prsi.agents.agent import Agent
from prsi.agents.baselines import GreedyAgent, RandomAgent
from prsi.rl_utils import ReplayBuffer
from prsi.env import PrsiEnv

parser = argparse.ArgumentParser()

# OPTIONS
# ------------------------------
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--evaluate_for", default=10_000, type=int, help="Evaluation episodes."
)
parser.add_argument("--load_model", action="store_true", help="Load model from disk.")
parser.add_argument(
    "--model_path",
    default="agent_strategies/ddqn/",
    type=str,
    help="Base path to save/load model. A subdirectory for the hyperparameters will be created here.",
)
parser.add_argument("--log_each", default=10_000, type=int, help="Log frequency.")
parser.add_argument(
    "--save_each", default=None, type=int, help="Periodic saving frequency."
)
parser.add_argument(
    "--disable_csv_logging",
    action="store_true",
    help="Disable saving logs to logs.csv.",
)

# HYPERPARAMETERS
# ------------------------------
parser.add_argument(
    "--episodes", default=1_000_000, type=int, help="Training episodes."
)
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument(
    "--epsilon_decay", default=1, type=float, help="Epsilon decay factor."
)
parser.add_argument("--min_epsilon", default=0.001, type=float, help="Minimum epsilon.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=32, type=int, help="Mini batch size.")
parser.add_argument(
    "--replay_buffer_size", default=100_000, type=int, help="Size of the replay buffer."
)
parser.add_argument(
    "--target_update_freq",
    default=100,
    type=int,
    help="Frequency of target network weight copying.",
)
parser.add_argument(
    "--hidden_layer_size", default=512, type=int, help="Size of hidden NN layers."
)
parser.add_argument(
    "--hidden_layer_count", default=2, type=int, help="The amount of hidden NN layers."
)
parser.add_argument(
    "--hand_state_option",
    default="full",
    type=str,
    choices=["count", "count_truncated", "simple", "full"],
    help="Representation of cards on hand in the state.",
)
parser.add_argument(
    "--truncated_hand_size",
    default=4,
    type=int,
    help="Max hand size for truncated count.",
)
parser.add_argument(
    "--played_subset",
    default="all",
    type=str,
    choices=["sevens", "specials", "all"],
    help="Representation of already played cards in the state.",
)
parser.add_argument(
    "--opponent",
    default="greedy",
    type=str,
    choices=["random", "greedy"],
    help="Opponent the agent is learning against. Only used for evaluation if training via self-play.",
)
parser.add_argument("--self_play", action="store_true", help="Train using self-play.")
parser.add_argument(
    "--self_play_update_freq",
    default=1_000,
    type=int,
    help="Frequency of self-play opponent update.",
)


class DoubleDQNAgent(DQNAgent):
    def clone(self) -> "DoubleDQNAgent":
        cloned = DoubleDQNAgent.__new__(DoubleDQNAgent)
        cloned.args = self.args
        cloned.save_dir = self.save_dir
        cloned.full_model_path = self.full_model_path
        cloned.csv_path = self.csv_path
        cloned.log_data = []
        cloned._init_played_subset()
        cloned._build_networks()
        cloned.online_net.load_state_dict(self.online_net.state_dict())
        cloned.online_net.eval()
        return cloned

    def _learn(self, replay_buffer: ReplayBuffer) -> None:
        batch = replay_buffer.sample(self.args.batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        action_idxs = torch.tensor(
            np.array([t.action_idx for t in batch]),
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            np.array([t.reward for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.array([t.done for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_valid_masks = torch.tensor(
            np.array([t.next_valid_actions for t in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        # Q(s,a) from online network
        self.online_net.train()
        q_values = self.online_net(states)
        current_q = q_values.gather(1, action_idxs.unsqueeze(1)).squeeze(1)

        # Double DQN Target: r + gamma * Q_target(s', argmax_{a' valid} Q_online(s', a'))
        with torch.no_grad():
            # 1. Use online network to choose the best action for the next state
            next_q_online = self.online_net(next_states)
            next_q_online[~next_valid_masks] = -torch.inf  # Mask invalid actions
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # 2. Use target network to evaluate the chosen action
            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            # Calculate target
            target_q = rewards + self.args.gamma * max_next_q * (1.0 - dones)

        loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

    agent = DoubleDQNAgent(args=args)

    if args.load_model:
        agent.load(args.model_path)
    else:
        agent.train(env)
        agent.save(agent.full_model_path)

    agent.evaluate(env, episodes=args.evaluate_for, opponent=opponent)
