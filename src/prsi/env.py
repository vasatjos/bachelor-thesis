from dataclasses import replace

from prsi.card_utils import CardEffect, Rank, Suit
from prsi.deck import Deck, DeckEmptyError
from prsi.card import Card
from prsi.player import Player
from prsi.game_state import GameState, find_allowed_cards
from prsi.rl_utils import DRAW_ACTION, Action, INDEX_TO_ACTION
from prsi.agents.baselines import GreedyAgent
from prsi.agents.agent import Agent


class PrsiEnv:
    """
    A reinforcement learning environment for the card game Prší
    in a 1 vs 1 setting.
    """

    STARTING_HAND_SIZE = 4
    PLAYER_COUNT = 2
    ACTION_SPACE_SIZE = len(INDEX_TO_ACTION)

    MAX_STEPS = 1000

    def __init__(self, opponent: Agent = GreedyAgent()) -> None:
        self._player_info: Player = Player(0)
        self._opponent: Agent = opponent
        self._opponent_player_info = Player(1)
        self._deck: Deck = Deck()
        self._state: GameState = GameState()
        self._done: bool = False
        self._player_won_last: bool = True  # winning player starts game
        self._ran_out_of_cards: bool = False
        self._steps: int = 0

    @property
    def state(self) -> GameState:
        return self._state

    def reset(
        self, full: bool = False, opponent: Agent | None = None
    ) -> tuple[GameState, dict]:
        """
        Reset the environment to initial state and return the starting state.

        Args:
            full: Controls whether who starts will be reset. If `True`, player starts.
                  If `False`, whoever won the last game starts.
            opponent: Change the opponent the environment uses. Useful for self-play.

        Returns:
            Tuple of (state, info) where info contains the player's hand.
        """
        self._steps = 0
        self._deck.reset()
        self._player_info = Player(0)
        self._opponent_player_info = Player(1)
        self._done = False
        self._ran_out_of_cards = False
        if full:
            self._player_won_last = True
        if opponent is not None:  # change opponent agent (for self-play purposes)
            self._opponent = opponent

        # Initialize effect with first card from discard pile
        first_card = self._deck.discard_pile[0]
        self._state = GameState(
            top_card=first_card,
            actual_suit=first_card.suit,
            current_effect=first_card.effect,
            effect_strength=1 if first_card.effect != CardEffect.NONE else 0,
        )

        self._deal()

        if self._player_won_last:  # player makes first move
            return self._state, {
                "hand": self._player_info.hand,
                "opponent_card_count": (self._opponent_player_info.card_count),
                "deck_flipped_over": False,
            }

        # Opponent makes first move
        player_info = {
            "hand": self._opponent_player_info.hand,
            "opponent_card_count": self._player_info.card_count,
            "deck_flipped_over": False,
        }
        opponent_action = self._opponent.choose_action(
            self._state, self._opponent_player_info.hand, player_info
        )
        flipped = self._execute_action(self._opponent_player_info, opponent_action)

        return self._state, {
            "hand": self._player_info.hand,
            "opponent_card_count": self._opponent_player_info.card_count,
            "deck_flipped_over": flipped,
        }

    def step(self, action: Action) -> tuple[GameState, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take, either (Card, Suit) or None when drawing

        Returns:
            Tuple of (state, reward, done, info) where info contains the player's hand.
        """
        if self._done:
            raise RuntimeError("Game is over. Call reset() to start a new game.")

        self._steps += 1

        try:
            flipped_player = self._execute_action(self._player_info, action)
        except DeckEmptyError as e:  # lose when drawing from empty deck
            self._done = True
            self._player_won_last = False
            print(e)
            return (
                self._state,
                0.0,
                True,
                {
                    "hand": self._player_info.hand,
                    "opponent_card_count": self._opponent_player_info.card_count,
                    "deck_flipped_over": True,
                },
            )

        seven_of_hearts = Card(Suit.HEARTS, Rank.SEVEN)
        if not self._opponent_player_info.card_count and (
            action is None or action[0] != seven_of_hearts
        ):
            self._done = True
            self._player_won_last = False
            return (
                self._state,
                -1.0,
                True,
                {
                    "hand": self._player_info.hand,
                    "opponent_card_count": self._opponent_player_info.card_count,
                    "deck_flipped_over": flipped_player,
                },
            )
            # else: simply fall through to opponent's turn

        player_info = {  # player here meaning opponent's opponent
            "hand": self._opponent_player_info.hand,
            "opponent_card_count": self._player_info.card_count,
            "deck_flipped_over": flipped_player,
        }
        opponent_action = self._opponent.choose_action(
            self._state, self._opponent_player_info.hand, player_info
        )
        try:
            flipped_opponent = self._execute_action(
                self._opponent_player_info, opponent_action
            )
        except DeckEmptyError:
            self._done = True
            self._player_won_last = True
            return (
                self._state,
                0.0,
                True,
                {
                    "hand": self._player_info.hand,
                    "opponent_card_count": self._opponent_player_info.card_count,
                    "deck_flipped_over": True,
                },
            )

        if not self._player_info.card_count and (
            opponent_action is None or opponent_action[0] != seven_of_hearts
        ):
            self._done = True
            self._player_won_last = True
            return (
                self._state,
                1.0,
                True,
                {
                    "hand": self._player_info.hand,
                    "opponent_card_count": self._opponent_player_info.card_count,
                    "deck_flipped_over": flipped_opponent,
                },
            )

        self._done = self._steps > self.MAX_STEPS  # truncate episode after max steps

        return (
            self._state,
            0.0,
            self._done,
            {
                "hand": self._player_info.hand,
                "opponent_card_count": self._opponent_player_info.card_count,
                "deck_flipped_over": flipped_player or flipped_opponent,
            },
        )

    def _execute_action(self, player: Player, action: Action) -> bool:
        """
        Returns whether the deck was flipped over.
        """
        if action == DRAW_ACTION:
            drawn, flipped = self._draw_cards()
            player.take_drawn_cards(drawn)
            self._update_state(None)
            return flipped

        card, suit = action  # type: ignore

        flipped = False
        if card not in find_allowed_cards(self.state) or card not in player.hand:
            raise ValueError("Selected card not allowed!")
        player.play_card(card)
        self._deck.play_card(card)
        self._update_state(card, suit)

        return flipped

    def _draw_cards(self) -> tuple[list[Card | None], bool]:
        """Draw the appropriate number of cards based on current effect."""
        deck_flipped = False
        match self._state.current_effect:
            case CardEffect.DRAW_TWO:
                drawn: list[Card | None] = []
                for _ in range(self._state.effect_strength):
                    for _ in range(2):
                        card, flip = self._deck.draw_card()
                        deck_flipped = deck_flipped or flip
                        if card is not None:
                            drawn.append(card)
                return drawn, bool(deck_flipped)
            case CardEffect.SKIP_TURN:
                return [], False
            case _:
                drawn_card, flipped = self._deck.draw_card()
                result: list[Card | None] = []
                if drawn_card is not None:
                    result.append(drawn_card)
                return result, flipped

    def _deal(self) -> None:
        for _ in range(PrsiEnv.STARTING_HAND_SIZE):
            if self._player_won_last:
                self._player_info.take_drawn_cards([self._deck.draw_card()[0]])
                self._opponent_player_info.take_drawn_cards([self._deck.draw_card()[0]])
            else:
                self._opponent_player_info.take_drawn_cards([self._deck.draw_card()[0]])
                self._player_info.take_drawn_cards([self._deck.draw_card()[0]])

    def _update_state(self, card: Card | None, suit: Suit | None = None) -> None:
        if card is None:  # Player drew card(s)
            self._state = replace(
                self._state,
                current_effect=CardEffect.NONE,
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
            if suit is None:
                raise ValueError("Suit must be provided when playing an Ober")
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=suit,
                current_effect=CardEffect.NONE,
                effect_strength=0,
            )
        else:
            self._state = replace(
                self._state,
                top_card=card,
                actual_suit=card.suit,
                current_effect=CardEffect.NONE,
                effect_strength=0,
            )
