from dataclasses import replace

from game.card_utils import CardEffect, Rank, Suit, generate_rank, generate_suit
from game.deck import Deck
from game.card import Card
from game.player import Player
from game.game_state import GameState
from agents.utils import Action, INDEX_TO_SUIT, INDEX_TO_CARD
from agents.random import RandomAgent
from agents.base import BaseAgent


class PrsiEnv:
    STARTING_HAND_SIZE = 4
    PLAYER_COUNT = 2

    def __init__(self, opponent: BaseAgent = RandomAgent()) -> None:
        self._player: Player = Player(0)
        self._opponent: BaseAgent = opponent
        self._opponent.set_player_info(Player(1))
        self._deck: Deck = Deck()
        self._state: GameState = GameState()
        self._done: bool = False
        self._player_won_last: bool = True  # winning player starts game

    @property
    def state(self) -> GameState:
        return self._state

    def reset(self, full: bool = False) -> GameState:
        """
        Reset the environment to initial state and return the starting state.

        Args:
            full: Controls whether who starts will be reset. If `True`, player starts.
                  If `False`, whoever won the last game starts.
        """
        self._deck.reset()
        self._player = Player(0)
        self._opponent.set_player_info(Player(1))
        self._done = False
        if full:
            self._player_won_last = True

        # Initialize effect with first card from discard pile
        first_card = self._deck.discard_pile[0]
        self._state = GameState(
            top_card=first_card,
            actual_suit=first_card.suit,
            current_effect=first_card.effect,
            effect_strength=1 if first_card.rank == Rank.SEVEN else 0,
        )

        self._deal()
        return self._state

    def step(self, action: Action) -> tuple[GameState, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take (card_index, suit_index)

        Returns:
            Tuple of (state, reward, done, info)
        """
        if self._opponent.player_info is None:
            raise RuntimeError("Opponent not initialized correctly.")

        if self._done:
            raise RuntimeError("Game is over. Call reset() to start a new game.")

        seven_of_hearts = Card(Suit.HEARTS, Rank.SEVEN)

        # Player's turn
        player_card = self._execute_action(self._player, action)
        if not self._opponent.player_info.card_count and player_card != seven_of_hearts:
            self._done = True
            return (
                self._state,
                -1.0,
                True,
                {"opponent_card_count": self._opponent.player_info.card_count},
            )
            # else: simply fall through to opponent's turn

        # Opponent's turn
        opponent_action = self._opponent.choose_action(self._state)
        opponent_card = self._execute_action(
            self._opponent.player_info, opponent_action
        )
        if not self._player.card_count and opponent_card != seven_of_hearts:
            self._done = True
            return (
                self._state,
                1.0,
                True,
                {"opponent_card_count": self._opponent.player_info.card_count},
            )

        return (
            self._state,
            0.0,
            False,
            {"opponent_card_count": self._opponent.player_info.card_count},
        )

    @staticmethod
    def find_allowed_cards(state: GameState) -> set[Card]:
        """Find all cards that can legally be played given current state."""
        if state.current_effect == CardEffect.SKIP_TURN:
            return generate_rank(Rank.ACE)

        if state.current_effect == CardEffect.DRAW_TWO:
            return generate_rank(Rank.SEVEN)

        if state.top_card is None or state.actual_suit is None:
            raise RuntimeError("Game state not initialized")

        return (
            generate_suit(state.actual_suit)
            | generate_rank(state.top_card.rank)
            | generate_rank(Rank.OBER)
        )

    def _execute_action(self, player: Player, action: Action) -> Card | None:
        card_idx, suit_idx = action
        card = INDEX_TO_CARD.get(card_idx)
        suit = INDEX_TO_SUIT.get(suit_idx)

        if card is not None:  # Playing a card
            player.play_card(card)
            self._deck.play_card(card)
            self._update_state(card, suit)
        else:  # Drawing cards
            drawn = self._draw_cards()
            player.take_drawn_cards(drawn)
            self._update_state(None)

        return card

    def _draw_cards(self) -> list[Card]:
        """Draw the appropriate number of cards based on current effect."""
        match self._state.current_effect:
            case CardEffect.DRAW_TWO:
                drawn = []
                for _ in range(self._state.effect_strength):
                    drawn.append(self._deck.draw_card())
                    drawn.append(self._deck.draw_card())
                return drawn
            case CardEffect.SKIP_TURN:
                return []
            case _:
                return [self._deck.draw_card()]

    def _deal(self) -> None:
        if self._opponent.player_info is None:
            raise TypeError("Can't deal to uninitialised player.")

        for _ in range(PrsiEnv.STARTING_HAND_SIZE):
            if self._player_won_last:
                self._player.take_drawn_cards([self._deck.draw_card()])
                self._opponent.player_info.take_drawn_cards([self._deck.draw_card()])
            else:
                self._opponent.player_info.take_drawn_cards([self._deck.draw_card()])
                self._player.take_drawn_cards([self._deck.draw_card()])

    def _update_state(self, card: Card | None, suit: Suit | None = None) -> None:
        if card is None:  # Player drew card(s) instead of playing
            self._state = replace(
                self._state,
                current_effect=None,
                effect_strength=0,
            )
        elif card.rank == Rank.SEVEN:
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=card.suit,
                current_effect=CardEffect.DRAW_TWO,
                effect_strength=self._state.effect_strength + 1,
            )
        elif card.rank == Rank.ACE:
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=card.suit,
                current_effect=CardEffect.SKIP_TURN,
                effect_strength=1,
            )
        elif card.rank == Rank.OBER:
            if suit is None:  # Ober requires suit selection
                raise ValueError("Suit must be provided when playing an Ober")
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=suit,
                current_effect=None,
                effect_strength=0,
            )
        else:
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=card.suit,
                current_effect=None,
                effect_strength=0,
            )
