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
        """
        Return a deep copy of the agent. Useful for self-play.
        """
        return deepcopy(self)

    @abstractmethod
    def choose_action(
        self, state: Any, hand: list[Card], info: dict[str, Any]
    ) -> Action:
        """
        Returns the action the agent has chosen.
        """
        raise NotImplementedError("Base class cannot choose action.")

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """
        Evaluate the agent performance.
        """
        raise NotImplementedError("Base class cannot be evaluated.")
