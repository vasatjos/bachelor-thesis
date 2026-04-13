import pytest
from prsi.card import Card
from prsi.card_utils import Suit, Rank
from prsi.player import Player


@pytest.fixture
def player() -> Player:
    return Player(player_id=1)


@pytest.fixture
def sample_card() -> Card:
    return Card(Suit.HEARTS, Rank.ACE)


@pytest.fixture
def sample_hand() -> list[Card]:
    return [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.BELLS, Rank.KING),
        Card(Suit.LEAVES, Rank.SEVEN),
    ]


def test_player_id_is_readonly():
    player = Player(player_id=7)
    with pytest.raises(AttributeError):
        player.id = 99  # type: ignore


def test_player_equality_same_id():
    p1 = Player(player_id=1)
    p2 = Player(player_id=1)
    assert p1 == p2


def test_player_inequality_different_id():
    p1 = Player(player_id=1)
    p2 = Player(player_id=2)
    assert p1 != p2


def test_player_not_equal_to_non_player():
    player = Player(player_id=1)
    assert player != 1
    assert player != "player"
    assert player != None  # noqa: E711


def test_card_count_starts_at_zero(player: Player):
    assert player.card_count == 0


def test_card_count_after_taking_cards(player: Player, sample_hand: list[Card]):
    player.take_drawn_cards(sample_hand)  # type: ignore
    assert player.card_count == len(sample_hand)


def test_card_count_after_playing_card(player: Player, sample_card: Card):
    player.take_drawn_cards([sample_card])
    player.play_card(sample_card)
    assert player.card_count == 0


def test_take_drawn_cards_adds_to_hand(player: Player, sample_hand: list[Card]):
    player.take_drawn_cards(sample_hand)  # type: ignore
    assert player.hand == sorted(sample_hand)


def test_take_drawn_cards_all_none(player: Player):
    player.take_drawn_cards([None, None, None])
    assert player.card_count == 0


def test_take_drawn_cards_empty_list(player: Player):
    player.take_drawn_cards([])
    assert player.card_count == 0


def test_take_drawn_cards_accumulates(player: Player):
    card1 = Card(Suit.HEARTS, Rank.ACE)
    card2 = Card(Suit.BELLS, Rank.KING)
    player.take_drawn_cards([card1])
    player.take_drawn_cards([card2])
    assert player.card_count == 2
    assert sorted([card1, card2]) == player.hand


def test_play_card_removes_from_hand(player: Player, sample_card: Card):
    player.take_drawn_cards([sample_card])
    player.play_card(sample_card)
    assert sample_card not in player.hand


def test_play_card_raises_if_card_not_in_hand(player: Player, sample_card: Card):
    with pytest.raises(KeyError):
        player.play_card(sample_card)
