from typing import Any  # TODO: maybe state should be Hashable instead of Any
from abc import ABC, abstractmethod
from game.player import Player
from agents.utils import Action


class BaseAgent(ABC):
    def __init__(self, player_info: Player | None = None) -> None:
        self.player_info: Player | None = player_info

    def set_player_info(self, player_info: Player) -> None:
        self.player_info = player_info

    @abstractmethod
    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("Base class cannot choose action.")

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be trained.")
