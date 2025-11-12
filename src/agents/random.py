from typing import Any
from agents.base import BaseAgent
from game.player import Player

class RandomAgent(BaseAgent):
    def __init__(self, player_info: Player | None = None) -> None:
        super().__init__(player_info)

    def perform_action(self, state: Any) -> int:
        raise NotImplementedError("TODO!")
