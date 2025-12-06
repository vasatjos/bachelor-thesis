import os

from game.card_utils import CardEffect, Rank, Suit
from game.deck import Deck
from game.card import Card
from game.player import Player
from game.state_manager import GameStateManager
from agents.utils import Action, CARD_TO_INDEX, SUIT_TO_INDEX
from agents.random import RandomAgent
from agents.base import BaseAgent


class PrsiEnv:
    STARTING_HAND_SIZE = 4
    PLAYER_COUNT = 2

    def __init__(self, opponent: BaseAgent = RandomAgent()) -> None:
        # TODO: maybe should also be agent, maybe shouldn't be private
        self._player: Player = Player(0)
        self._opponent: BaseAgent = opponent
        self._opponent.set_player_info(Player(1))
        self._deck: Deck = Deck()
        self._effect_manager: GameStateManager = GameStateManager()
        self._last_winner: BaseAgent | None = None


    def _deal(self) -> None:
        for _ in range(PrsiEnv.STARTING_HAND_SIZE):
            for player in self._players:
                player.take_drawn_cards([self._deck.draw_card()])

    def reset(self) -> None: ...

    # TODO: figure out how to represent state on the output
    def step(self, action: Action) -> ...:
        # perform action
        # update game state
        # have opponent select and perform action
        # update game state
        # return state ( + done, reward, etc.)
        ...

    def play(self) -> None:
        self._deck.reset()
        self._players = [Player(i) for i in range(self.PLAYER_COUNT)]
        self._effect_manager.update(
            self._deck.discard_pile[0], self._deck.discard_pile[0].suit
        )
        self._deal()
        self._game_loop()

    def _draw_cards(self) -> list[Card]:
        match self._effect_manager.current_effect:
            case CardEffect.DRAW_TWO:
                drawn = []
                for _ in range(self._effect_manager.effect_strength):
                    drawn.append(self._deck.draw_card())
                    drawn.append(self._deck.draw_card())
                return drawn
            case CardEffect.SKIP_TURN:
                return []
            case _:
                return [self._deck.draw_card()]

    def _take_turn(self, player: Player) -> bool:
        """
        Returns:
          Whether `player` won the game
        """
        if not player.card_count():
            seven_hearts = Card(Suit.HEARTS, Rank.SEVEN)
            if (
                self._effect_manager.top_card == seven_hearts
                and self._effect_manager.current_effect is not None
            ):
                pass
            else:
                self._last_winner = player
                self._effect_manager.update(None, self._effect_manager.top_card.suit)
                return True
        player_choice = self.prompt_player_for_card_choice(player)

        if player_choice is not None:
            self._deck.play_card(player_choice)
        else:
            drawn = self._draw_cards()
            player.take_drawn_cards(drawn)

        self._effect_manager.update(player_choice)
        return False

    def _game_loop(self) -> None:
        victory = False
        while not victory:
            for player in self._players:
                if self._last_winner is not None and player != self._last_winner:
                    continue  # start with last winner
                self._last_winner = None
                if victory := self._take_turn(player):
                    break

        self._end_game()
