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

# TODO: fix seeding, doesn't work properly currently
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--episodes", default=1_000_000_000, type=int, help="Training episodes."
)
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument(
    "--epsilon_decay", default=1, type=float, help="Epsilon decay factor."
)
parser.add_argument("--min_epsilon", default=0.05, type=float, help="Minimum epsilon.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument(
    "--hand_state_option",
    default="count_truncated",
    type=str,
    choices=["count", "count_truncated", "simple", "full", "full_simple"],
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
    "--evaluate_for", default=10_000, type=int, help="Evaluation episodes."
)
parser.add_argument("--load_model", action="store_true", help="Load model from disk.")
parser.add_argument(
    "--model_path",
    default="agent-strategies/monte-carlo/model.pkl",
    type=str,
    help="Path to save/load model.",
)
parser.add_argument("--log_each", default=50_000, type=int, help="Log frequency.")
parser.add_argument(
    "--opponent", default="greedy", type=str, choices=["random", "greedy"]
)


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
    tuple[np.uint8, ...],
]


class DQNAgent(TrainableAgent): ...
