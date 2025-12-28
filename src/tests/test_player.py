from game.card import Card
from game.card_utils import Rank, Suit
from game.deck import get_suit
from game.player import Player


def test_choose_playing_card_allowed(monkeypatch):
    allowed: set[Card] = set()
    for suit in Suit:
        cards = set(get_suit(suit))
        allowed |= cards

    player = Player(0)
    c1 = Card(Suit.HEARTS, Rank.TEN)
    c2 = Card(Suit.HEARTS, Rank.SEVEN)
    c3 = Card(Suit.LEAVES, Rank.ACE)
    c4 = Card(Suit.BELLS, Rank.KING)
    player.hand_set.update([c1, c2, c3, c4])

    # Simulate input: choose the first playable card (1-based index)
    monkeypatch.setattr("builtins.input", lambda _: "1")

    selected = player.select_card_to_play(allowed)

    assert len(allowed) == 32
    assert selected == c2
    assert selected not in player.hand_set  # card should be removed from hand
    assert player.hand_set == {c1, c3, c4}


def test_choose_playing_card_with_no_allowed(monkeypatch):
    player = Player(0)
    card = Card(Suit.HEARTS, Rank.TEN)
    player.hand_set.add(card)

    allowed = set()  # nothing allowed

    # Simulate pressing enter
    monkeypatch.setattr("builtins.input", lambda _: "")

    result = player.select_card_to_play(allowed)

    assert result is None
    assert card in player.hand_set  # card should remain in hand


def test_choose_card_with_invalid_input(monkeypatch):
    player = Player(0)
    c1 = Card(Suit.HEARTS, Rank.TEN)
    c2 = Card(Suit.HEARTS, Rank.SEVEN)
    player.hand_set.update([c1, c2])
    allowed = set([c1])

    # Simulate invalid input "abc", then valid input "1"
    inputs = iter(["abc", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    selected = player.select_card_to_play(allowed)

    assert selected in allowed
    assert selected not in player.hand_set
