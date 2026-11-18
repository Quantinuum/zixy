import pytest

import zixy.fermion.operator.normal as zfon
from zixy.fermion.operator.normal import Modes, NormalFermionOperatorArray


def test_array_sizing():
    strings = zfon.Strings(4)
    assert len(strings.modes) == 4
    assert len(strings) == 0

    strings.append("")
    strings.append("F0^ F1")
    strings.append_n(3, ([2], [3]))

    assert len(strings) == 5
    assert str(strings) == ", F0^ F1, F2^ F3, F2^ F3, F2^ F3"

    strings.resize(2)
    assert len(strings) == 2
    assert str(strings) == ", F0^ F1"


def test_string_access():
    strings = zfon.Strings(6)
    specs = (
        ([0, 2], [1, 3]),
        ([True, False, True], [False, True, False, True]),
        ([], []),
        ([5], []),
        ([], [4]),
    )

    strings.extend(specs)

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
    assert strings[1:4].clone() == zfon.Strings.from_iterable(specs[1:4], 6)


@pytest.mark.parametrize(
    ("constructor", "args"),
    (
        (zfon.String, (2, "F1 F0^")),
        (zfon.Strings.from_str, ("F1 F0^", 2)),
        (zfon.StringSet.from_str, ("F1 F0^", 2)),
    ),
)
def test_rejects_non_normal_ordered_inputs(constructor, args):
    with pytest.raises(ValueError, match="normal-order"):
        constructor(*args)


def test_string_modification():
    string = zfon.String(4, ([0, 2], [1, 3]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert string.creations == [0, 2]
    assert string.annihilations == [1, 3]

    assert string[0, True]
    assert not string[0, False]
    string[0, True] = False
    string[0, False] = True

    assert string.get_sets() == ([2], [0, 1, 3])
    assert str(string) == "F2^ F0 F1 F3"

    with pytest.raises(KeyError):
        string[0, "bad"]
    with pytest.raises(KeyError):
        string[0, "bad"] = True


def test_string_from_bool_list_spec():
    string = zfon.String(4, ([True, False, True], [False, True, False, True]))

    assert string.get_sets() == ([0, 2], [1, 3])
    assert str(string) == "F0^ F2^ F1 F3"


def test_string_set_validates_mode_bounds():
    with pytest.raises(IndexError, match="out-of-bounds"):
        zfon.String(2, ([2], []))

    with pytest.raises(IndexError, match="out-of-bounds"):
        zfon.String(2, ([True, False, True], []))


@pytest.mark.parametrize(
    ("string_type", "source", "modes", "expected"),
    (
        (zfon.String, "F0^ F1", None, "F0^ F1"),
        (zfon.String, "  F0^   F1  ", None, "F0^ F1"),
        (zfon.String, "F0^ F1", 3, "F0^ F1"),
        (zfon.String, "", 3, ""),
        (zfon.Strings, "F0^ F1, F1^ F0, F0^ F1", None, "F0^ F1, F1^ F0, F0^ F1"),
        (
            zfon.Strings,
            " F0^ F1 ,   F1^ F0 , F0^ F1 ",
            None,
            "F0^ F1, F1^ F0, F0^ F1",
        ),
        (zfon.Strings, "F0^ F1, F1^ F0, F0^ F1", 3, "F0^ F1, F1^ F0, F0^ F1"),
        (zfon.Strings, "", 3, ""),
        (zfon.StringSet, "F0^ F1, F1^ F0, F0^ F1", None, "F0^ F1, F1^ F0"),
        (
            zfon.StringSet,
            " F0^ F1 ,   F1^ F0 , F0^ F1 ",
            None,
            "F0^ F1, F1^ F0",
        ),
        (zfon.StringSet, "F0^ F1, F1^ F0, F0^ F1", 3, "F0^ F1, F1^ F0"),
        (zfon.StringSet, "", 3, ""),
    ),
)
def test_from_str(string_type, source, modes, expected):
    args = () if modes is None else (modes,)
    parsed = string_type.from_str(source, *args)
    assert str(parsed) == expected

    if string_type is zfon.StringSet:
        assert parsed == zfon.StringSet.from_strings(zfon.Strings.from_str(source, *args))

    with pytest.raises(ValueError, match="mode index out of bounds"):
        zfon.String.from_str("F2", 2)


def test_from_ladder_product_validates_mode_bounds():
    modes = Modes.from_count(2)

    with pytest.raises(ValueError, match="mode index out of bounds"):
        NormalFermionOperatorArray.from_ladder_product(modes, [(2, True)])


def test_string_scalar_mul():
    string = zfon.String(4, "F0^ F1")

    term = string * 2.0

    assert isinstance(term, zfon.RealTerm)
    assert term.string == string
    assert term.coeff == 2.0


def test_string_mul():
    lhs = zfon.String(2, "F0^")
    rhs = zfon.String(2, "F1")

    product = lhs * rhs

    assert isinstance(product, zfon.RealTermSum)
    assert str(product) == "(1.0, F0^ F1)"


def test_string_mul_with_contraction():
    lhs = zfon.String(1, "F0")
    rhs = zfon.String(1, "F0^")

    product = lhs * rhs

    assert isinstance(product, zfon.RealTermSum)
    assert str(product) == "(1.0, ), (-1.0, F0^ F0)"


def test_string_mul_with_zero_product():
    lhs = zfon.String(1, "F0^")
    rhs = zfon.String(1, "F0^")

    product = lhs * rhs

    assert isinstance(product, zfon.RealTermSum)
    assert len(product) == 0


def test_string_from_str_errors():
    with pytest.raises(ValueError, match="does not normal-order to exactly one positive"):
        zfon.String.from_str("F1 F0^", 4)

    with pytest.raises(ValueError, match="does not normal-order to exactly one positive"):
        zfon.String.from_str("F0 F0^", 4)


def test_string_array_from_str():
    strings = zfon.Strings.from_str("F0^ F1, F2^, F3", 4)

    assert len(strings) == 3
    assert str(strings[1:]) == "F2^, F3"
    assert [string.get_sets() for string in strings] == [([0], [1]), ([2], []), ([], [3])]


def test_array_modification():
    strings = zfon.Strings.from_iterable(
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
    strings = zfon.Strings(4)
    strings.append_n(3, ([0], [1]))
    strings.append("")

    assert str(strings) == "F0^ F1, F0^ F1, F0^ F1, "
    assert str(strings.filter_unique()) == "F0^ F1, "
    assert str(strings.filter_populated()) == "F0^ F1, F0^ F1, F0^ F1"


def test_errors():
    strings = zfon.Strings(4)

    with pytest.raises(IndexError):
        strings[0]

    with pytest.raises(ValueError):
        zfon.String.from_str("bad", 4)


def test_in_place_products():
    string = zfon.String(4, ([0, 2], [1, 3]))
    adjoint = string.daggered()

    assert string.get_sets() == ([0, 2], [1, 3])
    assert adjoint.get_sets() == ([1, 3], [0, 2])

    string.dagger()
    assert string == adjoint


def test_string_set():
    strings = zfon.Strings.from_str("F0^ F1, F2^, F0^ F1", 4)
    string_set = zfon.StringSet.from_strings(strings)

    assert len(string_set) == 2
    assert zfon.String(4, "F0^ F1") in string_set
    assert zfon.String(4, "F2^") in string_set


def test_mapped_insert():
    string_set = zfon.StringSet(4)
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
        assert spec in string_set

    assert string_set.insert(specs[1]) == 1
    assert len(string_set) == len(specs)

    wrong_modes = zfon.String(5, ([0], [1]))
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.insert(wrong_modes)
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.lookup(wrong_modes)
    with pytest.raises(ValueError, match="different .*spaces"):
        string_set.remove(wrong_modes)


def test_mapped_equal():
    specs = (
        ([0], [1]),
        ([2], []),
        ([], [3]),
        ([0, 2], [1, 3]),
    )

    strings = zfon.Strings.from_iterable(specs, 4)
    reversed_strings = zfon.Strings.from_iterable(reversed(specs), 4)

    assert strings != reversed_strings
    assert zfon.StringSet.from_iterable(strings, 4) == zfon.StringSet.from_iterable(
        reversed_strings, 4
    )


def test_mapped_remove():
    string_set = zfon.StringSet(4)
    specs = (
        ([0], [1]),
        ([2], []),
        ([], [3]),
        ([0, 2], [1, 3]),
    )

    string_set.insert_iterable(specs)

    assert string_set.remove(specs[0]) == 0
    assert specs[0] not in string_set

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

    string_set = zfon.StringSet.from_iterable(specs, 4)
    assert len(string_set) == 4
    assert zfon.StringSet.from_iterable(reversed(specs), 4) == string_set
    assert (
        zfon.StringSet.from_strings(string_set.to_strings()).to_strings() == string_set.to_strings()
    )
