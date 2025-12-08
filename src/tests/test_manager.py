from game.card import Card
from game.deck import Deck
from game.state_manager import GameStateManager
from game.card_utils import CardEffect, Rank, Suit


def test_get_suit_choice_success(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "h")
    selected = GameStateManager._get_suit_choice()
    assert selected is Suit.HEARTS

    monkeypatch.setattr("builtins.input", lambda _: "a")
    selected = GameStateManager._get_suit_choice()
    assert selected is Suit.ACORNS

    monkeypatch.setattr("builtins.input", lambda _: "B")
    selected = GameStateManager._get_suit_choice()
    assert selected is Suit.BELLS

    monkeypatch.setattr("builtins.input", lambda _: "L")
    selected = GameStateManager._get_suit_choice()
    assert selected is Suit.LEAVES


def test_get_suit_choice_retry(monkeypatch):
    inputs = iter(["", "xyz", "1", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    selected = GameStateManager._get_suit_choice()
    assert selected is Suit.ACORNS


def test_find_allowed_sevens():
    manager = GameStateManager()
    manager.actual_suit = Suit.BELLS
    manager.top_card = Card(Suit.BELLS, Rank.SEVEN)
    manager.current_effect = CardEffect.DRAW_TWO
    manager.effect_strength = 1

    should_be_allowed = Deck.generate_rank(Rank.SEVEN)
    allowed = manager.find_allowed_cards()
    assert should_be_allowed == allowed


def test_find_allowed_aces():
    manager = GameStateManager()
    manager.actual_suit = Suit.HEARTS
    manager.top_card = Card(Suit.HEARTS, Rank.ACE)
    manager.current_effect = CardEffect.SKIP_TURN
    manager.effect_strength = 1

    should_be_allowed = Deck.generate_rank(Rank.ACE)
    allowed = manager.find_allowed_cards()
    assert should_be_allowed == allowed


def test_find_allowed_actual_is_different_from_top():
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.HEARTS, Rank.OBER)
    manager.current_effect = CardEffect.NONE
    manager.effect_strength = 0

    should_be_allowed = Deck.generate_suit(Suit.LEAVES) | Deck.generate_rank(Rank.OBER)
    allowed = manager.find_allowed_cards()
    assert should_be_allowed == allowed


def test_set_card_effect_ace():
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.LEAVES, Rank.NINE)
    manager.current_effect = CardEffect.NONE
    manager.effect_strength = 0

    manager.update(Card(Suit.LEAVES, Rank.ACE))

    assert manager.current_effect is CardEffect.SKIP_TURN
    assert manager.effect_strength == 1


def test_set_card_effect_seven():
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.LEAVES, Rank.NINE)
    manager.current_effect = CardEffect.NONE
    manager.effect_strength = 0

    manager.update(Card(Suit.LEAVES, Rank.SEVEN))

    assert manager.current_effect is CardEffect.DRAW_TWO
    assert manager.effect_strength == 1


def test_set_card_effect_stacking():
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.LEAVES, Rank.SEVEN)
    manager.current_effect = CardEffect.SKIP_TURN
    manager.effect_strength = 2

    manager.update(Card(Suit.HEARTS, Rank.SEVEN))

    assert manager.current_effect is CardEffect.DRAW_TWO
    assert manager.effect_strength == 3


def test_set_card_effect_change_suit(monkeypatch):
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.LEAVES, Rank.SEVEN)
    manager.current_effect = CardEffect.NONE
    manager.effect_strength = 0

    monkeypatch.setattr("builtins.input", lambda _: "b")
    manager.update(Card(Suit.HEARTS, Rank.OBER))

    assert manager.current_effect == CardEffect.NONE
    assert manager.effect_strength == 0
    assert manager.top_card == Card(Suit.HEARTS, Rank.OBER)
    assert manager.actual_suit == Suit.BELLS


def test_set_card_effect_reset_on_draw():
    manager = GameStateManager()
    manager.actual_suit = Suit.LEAVES
    manager.top_card = Card(Suit.LEAVES, Rank.SEVEN)
    manager.current_effect = CardEffect.NONE
    manager.effect_strength = 0

    manager.update()

    assert manager.current_effect == CardEffect.NONE
    assert manager.effect_strength == 0
    assert manager.top_card == Card(Suit.LEAVES, Rank.SEVEN)
    assert manager.actual_suit == Suit.LEAVES
