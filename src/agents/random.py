from typing import Any
from agents.base import BaseAgent
from agents.utils import Action, behave_randomly
from prsi.card import Card
from prsi.game_state import GameState


class RandomAgent(BaseAgent):
    def choose_action(
        self, state: GameState, hand: set[Card], info: dict[str, Any]
    ) -> Action:
        return behave_randomly(state, hand)

    def evaluate(self) -> None:
        raise NotImplementedError("TODO: should be simple though")
