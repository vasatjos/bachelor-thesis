from dataclasses import dataclass

from game.card import Card
from game.card_utils import CardEffect, Suit, Rank
from game.deck import get_suit, get_rank


@dataclass(frozen=True)
class GameState:
    top_card: Card | None = None
    actual_suit: Suit | None = None
    current_effect: CardEffect = CardEffect.NONE
    effect_strength: int = 0

def find_allowed_cards(state: GameState) -> set[Card]:
    """Find all cards that can legally be played given current state."""
    if state.top_card is None or state.actual_suit is None:
        raise RuntimeError("Game state not initialized")

    if state.current_effect == CardEffect.SKIP_TURN:
        return get_rank(Rank.ACE)
    if state.current_effect == CardEffect.DRAW_TWO:
        return get_rank(Rank.SEVEN)

    return (
        get_suit(state.actual_suit)
        | get_rank(state.top_card.rank)
        | get_rank(Rank.OBER)
    )
