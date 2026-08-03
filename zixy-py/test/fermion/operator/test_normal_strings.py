import pytest

from zixy.fermion.operator import normal


def test_string_array_sizing():
    strings = normal.Strings(4)
    assert len(strings.modes) == 4
    assert len(strings) == 0

    strings.append()
    strings.append("F0^ F1")
    strings.append_n(3, ([2], [3]))

    assert len(strings) == 5
    assert str(strings) == ", F0^ F1, F2^ F3, F2^ F3, F2^ F3"

    strings.resize(2)
    assert len(strings) == 2
    assert str(strings) == ", F0^ F1"


def test_string_access_and_modification():
    string = normal.String(4, ([0, 2], [1, 3]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert string.creations == [0, 2]
    assert string.annihilations == [1, 3]

    assert string["cre", 0]
    assert not string["ann", 0]
    string["cre", 0] = False
    string["ann", 0] = True

    assert string.get_sets() == ([2], [0, 1, 3])
    assert str(string) == "F2^ F0 F1 F3"


def test_string_bool_list_spec():
    string = normal.String(4, ([True, False, True], [False, True, False, True]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert str(string) == "F0^ F2^ F1 F3"


def test_string_from_str_normal_ordered():
    string = normal.String.from_str("F0^ F1", 4)

    assert str(string) == "F0^ F1"
    assert normal.String.from_str(str(string), 4) == string


def test_string_from_str_rejects_non_single_positive_component():
    with pytest.raises(ValueError, match="exactly one positive component"):
        normal.String.from_str("F1 F0^", 4)

    with pytest.raises(ValueError, match="exactly one positive component"):
        normal.String.from_str("F0 F0^", 4)


def test_strings_from_str_and_slicing():
    strings = normal.Strings.from_str("F0^ F1, F2^, F3", 4)

    assert len(strings) == 3
    assert str(strings[1:]) == "F2^, F3"
    assert [string.get_sets() for string in strings] == [([0], [1]), ([2], []), ([], [3])]


def test_dagger_swaps_creation_and_annihilation_parts():
    string = normal.String(4, ([0, 2], [1, 3]))
    adjoint = string.daggered()

    assert string.get_sets() == ([0, 2], [1, 3])
    assert adjoint.get_sets() == ([1, 3], [0, 2])

    string.dagger()
    assert string == adjoint


def test_string_set_deduplicates():
    strings = normal.Strings.from_str("F0^ F1, F2^, F0^ F1", 4)
    string_set = normal.StringSet.from_cmpnts(strings)

    assert len(string_set) == 2
    assert normal.String(4, "F0^ F1") in string_set
    assert normal.String(4, "F2^") in string_set
