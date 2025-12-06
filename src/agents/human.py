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
        if self.player_info is None:
            msg = "Player info hasn't been set. Use `set_player_info`."
            raise RuntimeError(msg)

        hand_list: list[Card | None] = list(self.player_info.hand_set)
        hand_list.append(None)

        card: Card | None = choice(hand_list)
        if card is None:
            return CARD_TO_INDEX[card], -42 # suit will get discarded on drawing a card

        card_index = CARD_TO_INDEX[card]
        suit_index = SUIT_TO_INDEX[card.suit]
        if card.rank == Rank.OBER:
            suit_index = randint(1, 4)

        return card_index, suit_index

    def prompt_player_for_card_choice(
        self, player: Player
    ) -> tuple[Card | None, Suit | None]:
        ...
        self._print_game_state(player)
        allowed = self._effect_manager.find_allowed_cards()
        print("\nPlayable cards:")
        player_choice = player.select_card_to_play(allowed)
        print()

        return player_choice
