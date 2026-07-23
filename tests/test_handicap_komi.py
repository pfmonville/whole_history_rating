import pytest

from whr.whole_history_rating import WHR


def test_advantage_config_defaults():
    w = WHR()
    assert w.config["pinned_handicap"] == {}
    assert w.config["pinned_komi"] == {}
    assert w.config["estimate_handicap_zero"] is False


def test_handicap_zero_baseline_pinned_by_default():
    w = WHR()
    assert w.handicap_gamma[0] == 1.0
    assert 0 in w._pinned_handicap_keys


def test_estimate_handicap_zero_unpins_baseline():
    w = WHR(config={"estimate_handicap_zero": True})
    assert 0 not in w._pinned_handicap_keys


def test_pinned_values_converted_to_gamma_and_marked():
    w = WHR(config={"pinned_handicap": {2: 200.0}, "pinned_komi": {6.5: -30.0}})
    assert w.handicap_gamma[2] == pytest.approx(10 ** (200.0 / 400.0))
    assert w.komi_gamma[6.5] == pytest.approx(10 ** (-30.0 / 400.0))
    assert 2 in w._pinned_handicap_keys and 6.5 in w._pinned_komi_keys


def test_config_not_mutated():
    src = {"pinned_handicap": {2: 200.0}}
    WHR(config=src)
    assert src == {"pinned_handicap": {2: 200.0}}


def test_ensure_advantage_keys_grows_tables_to_one():
    w = WHR()
    w._ensure_advantage_keys(3, 7.5)
    assert w.handicap_gamma[3] == 1.0
    assert w.komi_gamma[7.5] == 1.0
    # does not overwrite an existing (pinned) key
    w2 = WHR(config={"pinned_handicap": {3: 100.0}})
    before = w2.handicap_gamma[3]
    w2._ensure_advantage_keys(3, 6.5)
    assert w2.handicap_gamma[3] == before
