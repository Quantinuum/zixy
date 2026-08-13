import pytest

from zixy.qubit.state import String, Strings, StringSet


def test_from_tuple():
    s = String(6, (0, 1, 1, 0, 0, 1))
    assert str(s) == "[0, 1, 1, 0, 0, 1]"
    assert String.from_str(str(s), 6) == s
    assert s.hamming_weight() == s.count(True) == 3
    with pytest.raises(ValueError) as err:
        String.from_str("[1, 0], [0, 1]", 2)
    assert str(err.value) == "Source string should contain one state string, got 2."
    with pytest.raises(ValueError) as err:
        s[1] = 2
    assert str(err.value) == "Integer bit argument must be either 0 or 1"
    with pytest.raises(TypeError) as err:
        s[1] = "sad"
    assert str(err.value) == "Bit argument should be bool or 0 or 1"
    s[4] = 1
    assert str(s) == "[0, 1, 1, 0, 1, 1]"
    assert s.hamming_weight() == s.count(True) == 4
    assert s.get_tuple() == (0, 1, 1, 0, 1, 1)
    assert s.get_set() == {1, 2, 4, 5}
    s2 = String(7, (1, 1, 1, 0, 0, 0))
    assert s2.get_tuple() == (1, 1, 1, 0, 0, 0, 0)
    s2.set({1, 3})
    assert s2.get_tuple() == (0, 1, 0, 1, 0, 0, 0)
    with pytest.raises(ValueError) as err:
        s.set(s2)
    assert str(err.value) == "Qubits-based objects are based on different qubit spaces."
    with pytest.raises(IndexError) as err:
        s2 = String(5, (1, 1, 1, 0, 0, 0))
    assert str(err.value) == "Element index 5 is out-of-bounds for container of length 5."
    s2 = String(6, (1, 1, 1, 0, 0, 0))
    s.set(s2)
    assert s.get_tuple() == (1, 1, 1, 0, 0, 0)
    assert String(6).is_vacuum()
    assert not String(6, (1,)).is_vacuum()


def test_str():
    vacuum = String(6)
    assert vacuum.to_str() == "[0, 0, 0, 0, 0, 0]"
    assert String.from_str(vacuum.to_str(), 6) == vacuum
    assert String(6, "") == vacuum
    with pytest.raises(TypeError):
        String(6, None)
    with pytest.raises(TypeError):
        vacuum.set(None)

    single = String(1, (1,))
    assert single.to_str() == "[1]"
    assert String.from_str(single.to_str(), 1) == single

    strings = Strings(4)
    assert strings.to_str() == ""
    assert Strings.from_str(strings.to_str(), 4) == strings

    string_set = StringSet.from_iterable(({1, 3},), 4)
    assert string_set.to_str() == "[0, 1, 0, 1]"
    assert StringSet.from_str(string_set.to_str()) == string_set


def test_strings_from_iterable():
    s = Strings.from_iterable(
        (
            {1, 3, 4, 9},
            {1, 2, 6},
            (
                0,
                1,
            )
            * 5,
            (0, 1, 1, 1, 0) * 2,
        ),
        10,
    )
    assert (
        str(s)
        == "[0, 1, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]"
    )
    assert (
        str(s[2::-1])
        == "[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0, 0, 1]"
    )
    assert Strings.from_str(str(s), 10) == s
    string_set = StringSet.from_iterable(({1, 3, 4, 9}, {1, 2, 6}, {1, 3, 4, 9}), 10)
    assert StringSet.from_str(str(string_set)) == string_set
