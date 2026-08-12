import pytest

from zixy.fermion.state import String, Strings, StringSet


def test_string_access():
    string = String(4, [True, False, True, False])

    assert len(string.modes) == 4
    assert string.get_list() == [True, False, True, False]
    assert string.get_tuple() == (True, False, True, False)
    assert string.get_set() == {0, 2}
    assert string.hamming_weight() == 2
    assert not string.is_vacuum()


def test_string_modification():
    string = String(4, {1, 3})

    assert string[1]
    assert not string[2]

    string[1] = 0
    string[2] = 1

    assert string.get_set() == {2, 3}
    assert str(string) == "[0, 0, 1, 1]"

    with pytest.raises(ValueError):
        string[0] = 2
    with pytest.raises(TypeError):
        string[0] = "bad"  # type: ignore[assignment]


def test_string_from_str():
    string = String.from_str("[1, 0, 1]", 3)

    assert string.get_set() == {0, 2}
    assert String.from_str(str(string), 3) == string


def test_string_from_str_errors():
    with pytest.raises(ValueError, match="one state string"):
        String.from_str("[1], [0]", 1)


def test_string_scalar_mul():
    string = String(3, {0, 2})

    term = 2.0 * string

    assert term.string == string
    assert term.coeff == 2.0


def test_string_vdot():
    lhs = String(3, {0, 2})
    rhs = String(3, {1})

    assert lhs.vdot(lhs.clone()) == 1
    assert lhs.vdot(rhs) == 0


def test_string_vdot_rejects_different_modes():
    lhs = String(2, {0})
    rhs = String(3, {0})

    with pytest.raises(ValueError, match="different modes"):
        lhs.vdot(rhs)


def test_array_sizing():
    strings = Strings(4)
    assert len(strings) == 0

    strings.append()
    strings.append({0, 2})
    strings.append_n(2, [False, True, False, True])

    assert len(strings) == 4
    assert strings.get_sets() == (set(), {0, 2}, {1, 3}, {1, 3})
    assert str(strings) == "[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]"

    strings.resize(2)
    assert str(strings) == "[0, 0, 0, 0], [1, 0, 1, 0]"


def test_string_array_from_str():
    strings = Strings.from_str("[1, 0], [0, 1]", 2)

    assert len(strings) == 2
    assert strings.get_sets() == ({0}, {1})
    assert strings[1:].clone() == Strings.from_iterable(({1},), 2)


def test_string_set():
    strings = Strings.from_str("[1, 0], [0, 1], [1, 0]", 2)
    string_set = StringSet.from_cmpnts(strings)

    assert len(string_set) == 2
    assert String(2, {0}) in string_set
    assert String(2, {1}) in string_set


def test_mapped_insert():
    string_set = StringSet(4)
    specs = (set(), {0, 2}, {1, 3}, {0, 1, 2})

    for i, spec in enumerate(specs):
        assert string_set.lookup(spec) is None
        assert string_set.insert(spec) == i
        assert string_set.lookup(spec) == i
        assert string_set.contains(spec)

    assert string_set.insert(specs[1]) == 1
    assert len(string_set) == len(specs)

    wrong_modes = String(5, {0, 2})
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.insert(wrong_modes)
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.lookup(wrong_modes)
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.remove(wrong_modes)


def test_mapped_equal():
    specs = (set(), {0, 2}, {1, 3}, {0, 1, 2})

    strings = Strings.from_iterable(specs, 4)
    reversed_strings = Strings.from_iterable(reversed(specs), 4)

    assert strings != reversed_strings
    assert StringSet.from_iterable(strings, 4) == StringSet.from_iterable(reversed_strings, 4)


def test_mapped_remove():
    string_set = StringSet(4)
    specs = (set(), {0, 2}, {1, 3}, {0, 1, 2})

    string_set.insert_iterable(specs)

    assert string_set.remove(specs[1]) == 1
    assert not string_set.contains(specs[1])

    with pytest.raises(KeyError):
        string_set.remove({3})
