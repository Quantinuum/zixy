import pytest

from zixy.fermion.operator import general


def test_raw_string_preserves_order():
    string = general.String(3, "F1 F0^ F2")

    assert str(string) == "F1 F0^ F2"
    assert string.get_ops() == [(1, False), (0, True), (2, False)]


def test_raw_string_access_and_modification():
    string = general.String(4, [(0, True), (2, False)])

    assert string.max_len == 2
    assert string.get_ops() == [(0, True), (2, False)]

    string.set("F3 F1^")
    assert string.get_ops() == [(3, False), (1, True)]
    assert str(string) == "F3 F1^"


def test_raw_string_rejects_products_longer_than_max_len():
    string = general.String(4, "F0")

    with pytest.raises(ValueError, match="longer than max_len"):
        string.set("F0 F1")


def test_raw_strings_from_str_and_slicing():
    strings = general.Strings.from_str("F0 F0^, F1^ F2, F2 F1^ F0", 3)

    assert len(strings) == 3
    assert strings.max_len == 3
    assert str(strings[1:]) == "F1^ F2, F2 F1^ F0"
    assert [string.get_ops() for string in strings] == [
        [(0, False), (0, True)],
        [(1, True), (2, False)],
        [(2, False), (1, True), (0, False)],
    ]


def test_raw_string_set_deduplicates_and_preserves_length_metadata():
    strings = general.Strings.from_str("F1 F0^, F1 F0^, F0^ F1", 2)
    string_set = general.StringSet.from_cmpnts(strings)

    assert len(string_set) == 2
    assert str(string_set.to_cmpnts()) == "F1 F0^, F0^ F1"
