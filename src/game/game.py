import os

from game.card_utils import CardEffect, COLOR_RESET, Rank, Suit
from game.deck import Deck
from game.card import Card
from game.player import Player
from game.state_manager import GameStateManager


class PrsiEnv:
    STARTING_HAND_SIZE = 4
    PLAYER_COUNT = 2

    def __init__(self, show_ui: bool = False) -> None:
        self._players: list[Player] = []
        self._deck: Deck = Deck()
        self._effect_manager: GameStateManager = GameStateManager()
        self._last_winner: Player | None = None
        self._show_ui = show_ui

    def _deal(self) -> None:
        for _ in range(PrsiEnv.STARTING_HAND_SIZE):
            for player in self._players:
                player.take_drawn_cards([self._deck.draw_card()])

    def _print_game_state(self, player: Player) -> None:
        player_id = player.id + 1  # print with one based index
        os.system("clear")
        input(f"Press enter to start player #{player_id} turn.")
        os.system("clear")

        for p in self._players:
            if p == player:
                print(f"Player #{player_id} currently playing.")
            else:
                print(f"Player #{p.id + 1} has {p.card_count()} cards")
        print(f"\nTop card: {self._effect_manager.top_card}")

        if (
            self._effect_manager.actual_suit is None
            or self._effect_manager.top_card is None
        ):
            raise RuntimeError("Manager not initialized.")

        if self._effect_manager.top_card.rank is Rank.OBER:
            print(
                f"Current suit: {self._effect_manager.actual_suit.value}"
                + f"{self._effect_manager.actual_suit.name}{COLOR_RESET}"
            )

        print("\nCards on hand:")
        player.print_hand()

    def _get_player_card_choice(self, player: Player) -> Card | None:
        self._print_game_state(player)
        allowed = self._effect_manager.find_allowed_cards()
        print("\nPlayable cards:")
        player_choice = player.select_card_to_play(allowed)
        print()

        return player_choice

    def play(self) -> None:
        self._deck.reset()
        self._players = [Player(i) for i in range(self.PLAYER_COUNT)]
        self._effect_manager.update(self._deck.discard_pile[0], first_card=True)
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

    def _print_order(self) -> None:
        os.system("clear")
        print("---GAME OVER---\n\nResults:\n")

        if self._last_winner is None:
            raise RuntimeError
        print(f"Winner: Player #{self._last_winner.id + 1}")
        input("\nPress Enter to continue.")

    def _end_game(self) -> None:
        self._print_order()

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
                self._effect_manager.update()
                return True
        player_choice = self._get_player_card_choice(player)

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
