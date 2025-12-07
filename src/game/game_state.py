from dataclasses import dataclass

from game.card import Card
from game.card_utils import CardEffect, Suit


@dataclass(frozen=True)
class GameState:
    top_card: Card | None = None
    actual_suit: Suit | None = None
    current_effect: CardEffect | None = None
    effect_strength: int = 0
