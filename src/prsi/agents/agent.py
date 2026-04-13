from typing import Any  # TODO: maybe state should be Hashable instead of Any
from abc import ABC, abstractmethod
from prsi.rl_utils import Action
from copy import deepcopy
from prsi.card import Card


class Agent(ABC):
    """
    Abstract base class for agent implementation. Every Prsi agent must implement
    the abstract methods `choose_action` and `evaluate`.
    """

    def clone(self) -> "Agent":
        return deepcopy(self)

    @abstractmethod
    def choose_action(
        self, state: Any, hand: list[Card], info: dict[str, Any]
    ) -> Action:
        raise NotImplementedError("Base class cannot choose action.")

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be evaluated.")
