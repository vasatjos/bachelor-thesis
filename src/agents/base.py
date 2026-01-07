from typing import Any  # TODO: maybe state should be Hashable instead of Any
from abc import ABC, abstractmethod
from agents.utils import Action
from copy import deepcopy
from game.card import Card


class BaseAgent(ABC):
    def clone(self) -> "BaseAgent":
        return deepcopy(self)

    @abstractmethod
    def choose_action(
        self, state: Any, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        raise NotImplementedError("Base class cannot choose action.")

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be trained.")

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be evaluated.")

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")
