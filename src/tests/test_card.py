from game.card import Card
from game.card_utils import Suit, Rank, CardEffect


def test_card_effects():
    assert Card(Suit.HEARTS, Rank.SEVEN).effect == CardEffect.DRAW_TWO
    assert Card(Suit.BELLS, Rank.ACE).effect == CardEffect.SKIP_TURN
    assert Card(Suit.LEAVES, Rank.KING).effect == CardEffect.NONE
    assert Card(Suit.HEARTS, Rank.EIGHT).effect == CardEffect.NONE
    assert Card(Suit.LEAVES, Rank.TEN).effect == CardEffect.NONE


def test_card_ordering():
    low = Card(Suit.HEARTS, Rank.SEVEN)
    high = Card(Suit.HEARTS, Rank.ACE)
    same_1 = Card(Suit.BELLS, Rank.OBER)
    same_2 = Card(Suit.BELLS, Rank.OBER)
    assert low < high
    assert not high < low
    assert high >= low
    assert same_1 == same_2
