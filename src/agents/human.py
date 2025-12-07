from typing import Any
from agents.base import BaseAgent
from agents.utils import CARD_TO_INDEX, SUIT_TO_INDEX, Action
from game.card import Card
from game.card_utils import Rank, Suit
from game.player import Player
from random import choice, randint

class HumanAgent(BaseAgent):
    def __init__(self, player_info: Player | None = None) -> None:
        super().__init__(player_info)

    def choose_action(self, state: Any) -> Action:
        raise NotImplementedError("TODO: Implement choose_action for human agent")

    # def prompt_player_for_card_choice(
    #     self, player: Player
    # ) -> tuple[Card | None, Suit | None]:
    #     ...
    #     self._print_game_state(player)
    #     allowed = self._effect_manager.find_allowed_cards()
    #     print("\nPlayable cards:")
    #     player_choice = player.select_card_to_play(allowed)
    #     print()
    #
    #     return player_choice

    def train(self) -> None:
        pass
