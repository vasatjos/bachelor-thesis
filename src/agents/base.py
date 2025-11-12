from typing import Any
from abc import ABC, abstractmethod
from game.player import Player


class BaseAgent(ABC):
    def __init__(self, player_info: Player | None = None) -> None:
        self.player_info: Player | None = player_info

    @abstractmethod
    def perform_action(self, state: Any) -> int:
        raise NotImplementedError("Base class cannot perform action.")

    def set_player_info(self, player_info: Player) -> None:
        self.player_info = player_info
