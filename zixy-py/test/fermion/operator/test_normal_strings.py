import pytest

from zixy.fermion.operator.normal import RealTerm, RealTermSum, String, Strings, StringSet


def test_array_sizing():
    strings = Strings(4)
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


def test_string_access():
    strings = Strings(6)
    specs = (
        ([0, 2], [1, 3]),
        ([True, False, True], [False, True, False, True]),
        ([], []),
        ([5], []),
        ([], [4]),
    )

    strings.append_iterable(specs)

    assert len(strings) == len(specs)
    assert [string.get_sets() for string in strings] == [
        ([0, 2], [1, 3]),
        ([0, 2], [1, 3]),
        ([], []),
        ([5], []),
        ([], [4]),
    ]
    assert [string.get_ops() for string in strings] == [
        [(0, True), (2, True), (1, False), (3, False)],
        [(0, True), (2, True), (1, False), (3, False)],
        [],
        [(5, True)],
        [(4, False)],
    ]
    assert strings[1:4].clone() == Strings.from_iterable(specs[1:4], 6)


def test_string_modification():
    string = String(4, ([0, 2], [1, 3]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert string.creations == [0, 2]
    assert string.annihilations == [1, 3]

    assert string["cre", 0]
    assert not string["ann", 0]
    string["cre", 0] = False
    string["ann", 0] = True

    assert string.get_sets() == ([2], [0, 1, 3])
    assert str(string) == "F2^ F0 F1 F3"

    with pytest.raises(KeyError):
        string["bad", 0]
    with pytest.raises(KeyError):
        string["bad", 0] = True


def test_string_from_bool_list_spec():
    string = String(4, ([True, False, True], [False, True, False, True]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert str(string) == "F0^ F2^ F1 F3"


def test_string_from_str():
    string = String.from_str("F0^ F1", 4)

    assert str(string) == "F0^ F1"
    assert String.from_str(str(string), 4) == string


def test_string_scalar_mul():
    string = String(4, "F0^ F1")

    term = string * 2.0

    assert isinstance(term, RealTerm)
    assert term.string == string
    assert term.coeff == 2.0


def test_string_mul():
    lhs = String(2, "F0^")
    rhs = String(2, "F1")

    product = lhs * rhs

    assert isinstance(product, RealTermSum)
    assert str(product) == "(1.0, F0^ F1)"


def test_string_mul_with_contraction():
    lhs = String(1, "F0")
    rhs = String(1, "F0^")

    product = lhs * rhs

    assert isinstance(product, RealTermSum)
    assert str(product) == "(1.0, ), (-1.0, F0^ F0)"


def test_string_mul_with_zero_product():
    lhs = String(1, "F0^")
    rhs = String(1, "F0^")

    product = lhs * rhs

    assert isinstance(product, RealTermSum)
    assert len(product) == 0


def test_string_from_str_errors():
    with pytest.raises(ValueError, match="exactly one positive component"):
        String.from_str("F1 F0^", 4)

    with pytest.raises(ValueError, match="exactly one positive component"):
        String.from_str("F0 F0^", 4)


def test_string_array_from_str():
    strings = Strings.from_str("F0^ F1, F2^, F3", 4)

    assert len(strings) == 3
    assert str(strings[1:]) == "F2^, F3"
    assert [string.get_sets() for string in strings] == [([0], [1]), ([2], []), ([], [3])]


def test_array_modification():
    strings = Strings.from_iterable(
        (
            ([0], [1]),
            ([2], []),
            ([], [3]),
            ([0, 2], [1, 3]),
        ),
        4,
    )

    strings[1], strings[3] = strings[3].clone(), strings[1].clone()

    assert str(strings) == "F0^ F1, F0^ F2^ F1 F3, F3, F2^"
    assert strings.clone() == strings
    assert strings.clone() is not strings
    assert strings.reordered((3, 2, 1, 0)).to_str() == "F2^, F3, F0^ F2^ F1 F3, F0^ F1"


def test_append_n():
    strings = Strings(4)
    strings.append_n(3, ([0], [1]))
    strings.append()

    assert str(strings) == "F0^ F1, F0^ F1, F0^ F1, "
    assert str(strings.filter_unique()) == "F0^ F1, "
    assert str(strings.filter_populated()) == "F0^ F1, F0^ F1, F0^ F1"


def test_errors():
    strings = Strings(4)

    with pytest.raises(IndexError):
        strings[0]

    with pytest.raises(ValueError):
        String.from_str("bad", 4)


def test_in_place_products():
    string = String(4, ([0, 2], [1, 3]))
    adjoint = string.daggered()

    assert string.get_sets() == ([0, 2], [1, 3])
    assert adjoint.get_sets() == ([1, 3], [0, 2])

    string.dagger()
    assert string == adjoint


def test_string_set():
    strings = Strings.from_str("F0^ F1, F2^, F0^ F1", 4)
    string_set = StringSet.from_cmpnts(strings)

    assert len(string_set) == 2
    assert String(4, "F0^ F1") in string_set
    assert String(4, "F2^") in string_set


def test_mapped_insert():
    string_set = StringSet(4)
    specs = (
        ([0], [1]),
        ([2], []),
        ([], [3]),
        ([0, 2], [1, 3]),
    )

    for i, spec in enumerate(specs):
        assert string_set.lookup(spec) is None
        assert string_set.insert(spec) == i
        assert string_set.lookup(spec) == i
        assert string_set.contains(spec)

    assert string_set.insert(specs[1]) == 1
    assert len(string_set) == len(specs)


def test_mapped_equal():
    specs = (
        ([0], [1]),
        ([2], []),
        ([], [3]),
        ([0, 2], [1, 3]),
    )

    strings = Strings.from_iterable(specs, 4)
    reversed_strings = Strings.from_iterable(reversed(specs), 4)

    assert strings != reversed_strings
    assert StringSet.from_iterable(strings, 4) == StringSet.from_iterable(reversed_strings, 4)


def test_mapped_remove():
    string_set = StringSet(4)
    specs = (
        ([0], [1]),
        ([2], []),
        ([], [3]),
        ([0, 2], [1, 3]),
    )

    string_set.insert_iterable(specs)

    assert string_set.remove(specs[0]) == 0
    assert not string_set.contains(specs[0])

    with pytest.raises(KeyError):
        string_set.remove(([1], [0]))


def test_string_set_from_iterable():
    specs = (
        ([0], [1]),
        ([2], []),
        ([0], [1]),
        ([], []),
        ([], [3]),
    )

    string_set = StringSet.from_iterable(specs, 4)
    assert len(string_set) == 4
    assert StringSet.from_iterable(reversed(specs), 4) == string_set
    assert StringSet.from_cmpnts(string_set.to_cmpnts()).to_cmpnts() == string_set.to_cmpnts()
