from typing import Any  # TODO: maybe state should be Hashable instead of Any
from abc import abstractmethod
from agents.base import BaseAgent
from agents.utils import Action
from game.card import Card


class TrainableAgent(BaseAgent):
    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be trained.")

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")
