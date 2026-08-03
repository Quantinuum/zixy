import pytest

from zixy.fermion.operator.general import RealTerm, String, Strings, StringSet


def test_string_access():
    string = String(3, "F1 F0^ F2")

    assert str(string) == "F1 F0^ F2"
    assert string.get_ops() == [(1, False), (0, True), (2, False)]


def test_string_modification():
    string = String(4, [(0, True), (2, False)])

    assert string.max_len == 2
    assert string.get_ops() == [(0, True), (2, False)]

    string.set("F3 F1^")
    assert string.get_ops() == [(3, False), (1, True)]
    assert str(string) == "F3 F1^"


def test_string_scalar_mul():
    string = String(4, "F0^ F1")

    term = string * 2.0

    assert isinstance(term, RealTerm)
    assert term.string == string
    assert term.coeff == 2.0


def test_string_mul():
    lhs = String(3, "F1 F0^")
    rhs = String(3, "F2")

    product = lhs * rhs

    assert isinstance(product, RealTerm)
    assert str(product) == "(1.0, F1 F0^ F2)"
    assert product.string.get_ops() == [(1, False), (0, True), (2, False)]


def test_string_mul_errors():
    lhs = String(2, "F0")
    rhs = String(3, "F0")

    with pytest.raises(ValueError, match="different modes"):
        lhs * rhs


def test_array_sizing():
    strings = Strings(5, max_len=3)
    specs = (
        [(0, True), (1, False)],
        [(2, False), (0, True), (4, False)],
        [],
        [(3, True)],
    )

    strings.append_iterable(specs)

    assert len(strings) == len(specs)
    assert strings.max_len == 3
    assert [string.get_ops() for string in strings] == [list(spec) for spec in specs]
    assert strings[1:3].clone() == Strings.from_iterable(specs[1:3], 5, max_len=3)


def test_string_from_str_errors():
    string = String(4, "F0")

    with pytest.raises(ValueError, match="longer than max_len"):
        string.set("F0 F1")


def test_string_array_from_str():
    strings = Strings.from_str("F0 F0^, F1^ F2, F2 F1^ F0", 3)

    assert len(strings) == 3
    assert strings.max_len == 3
    assert str(strings[1:]) == "F1^ F2, F2 F1^ F0"
    assert [string.get_ops() for string in strings] == [
        [(0, False), (0, True)],
        [(1, True), (2, False)],
        [(2, False), (1, True), (0, False)],
    ]


def test_array_modification():
    strings = Strings.from_iterable(
        (
            [(0, True), (1, False)],
            [(2, False)],
            [],
            [(3, True), (0, False)],
        ),
        4,
        max_len=2,
    )

    strings[0], strings[3] = strings[3].clone(), strings[0].clone()

    assert str(strings) == "F3^ F0, F2, , F0^ F1"
    assert strings.clone() == strings
    assert strings.reordered((3, 2, 1, 0)).to_str() == "F0^ F1, , F2, F3^ F0"
    assert str(strings.filter_unique()) == "F3^ F0, F2, , F0^ F1"
    assert str(strings.filter_populated()) == "F3^ F0, F2, F0^ F1"


def test_errors():
    strings = Strings(4, max_len=1)

    with pytest.raises(IndexError):
        strings[0]

    with pytest.raises(ValueError):
        String.from_str("bad", 4)


def test_string_set():
    strings = Strings.from_str("F1 F0^, F1 F0^, F0^ F1", 2)
    string_set = StringSet.from_cmpnts(strings)

    assert len(string_set) == 2
    assert str(string_set.to_cmpnts()) == "F1 F0^, F0^ F1"


def test_mapped_insert():
    string_set = StringSet(4, max_len=3)
    specs = (
        [(0, True), (1, False)],
        [(1, False), (0, True)],
        [],
        [(2, True), (3, False), (1, True)],
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
        [(0, True), (1, False)],
        [(1, False), (0, True)],
        [],
        [(2, True), (3, False), (1, True)],
    )

    strings = Strings.from_iterable(specs, 4, max_len=3)
    reversed_strings = Strings.from_iterable(reversed(specs), 4, max_len=3)

    assert strings != reversed_strings
    assert StringSet.from_iterable(strings, 4, max_len=3) == StringSet.from_iterable(
        reversed_strings, 4, max_len=3
    )


def test_mapped_remove():
    string_set = StringSet(4, max_len=3)
    specs = (
        [(0, True), (1, False)],
        [(1, False), (0, True)],
        [],
        [(2, True), (3, False), (1, True)],
    )

    string_set.insert_iterable(specs)

    assert string_set.remove(specs[0]) == 0
    assert not string_set.contains(specs[0])

    with pytest.raises(KeyError):
        string_set.remove([(3, False)])


def test_string_set_from_iterable():
    specs = (
        [(0, True), (1, False)],
        [(1, False), (0, True)],
        [(0, True), (1, False)],
        [],
    )

    string_set = StringSet.from_iterable(specs, 2, max_len=2)
    assert len(string_set) == 3
    assert StringSet.from_iterable(reversed(specs), 2, max_len=2) == string_set
    assert StringSet.from_cmpnts(string_set.to_cmpnts()).to_cmpnts() == string_set.to_cmpnts()
