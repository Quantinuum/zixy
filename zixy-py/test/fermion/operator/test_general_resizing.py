"""Storage growth must preserve strings, coefficients, views and lookup keys."""

import warnings

import pytest
from sympy import Symbol

from zixy.fermion.operator.general import (
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    GeneralFermionOperatorArray,
    Modes,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    String,
    Strings,
    StringSet,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSet,
    SymbolicTermSum,
)


def test_string_growth_and_no_shrinking():
    string = String(2)
    for length in (1, 2, 65):
        with pytest.warns(UserWarning, match=f"to {length} to fit") as caught:
            string.set([(0, False)] * length)
        assert len(caught) == 1
        assert string.max_string_len == length
        assert string.get_ops() == [(0, False)] * length
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        string.set("F1^")
        string.clear()
    assert string.max_string_len == 65


@pytest.mark.parametrize("cls", (String, Strings, StringSet))
def test_parsed_initial_capacity_is_silent(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cls.from_str("F1 F0^", 2, max_string_len=1).max_string_len == 2
        assert cls.from_str("F1 F0^", 2, max_string_len=80).max_string_len == 80
        assert cls.from_str("", 2, max_string_len=80).max_string_len == 80


@pytest.mark.parametrize("cls", (Strings, StringSet, RealTerms, RealTermSet, RealTermSum))
def test_sequence_preallocation_and_streaming(cls):
    source = ["F0", "F0 F1", "F1"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = cls.from_iterable(source, 2, max_string_len=1)
    assert value.max_string_len == 2
    consumed = []

    def stream():
        for item in source:
            consumed.append(item)
            yield item

    with pytest.warns(UserWarning) as caught:
        streamed = cls.from_iterable(stream(), 2)
    assert len(caught) == 2
    assert consumed == source
    assert streamed == value


@pytest.mark.parametrize("n_modes", (1, 3, 65, 257))
def test_views_survive_packed_width_growth(n_modes):
    strings = Strings.from_iterable(["F0", "", "F0^"], n_modes)
    element = strings[0]
    view = strings[::2]
    for length in (2, 11, 32, 64, 65, 129):
        ops = [(n_modes - 1, bool(i % 2)) for i in range(length)]
        with pytest.warns(UserWarning):
            strings[1] = ops
        assert strings[1].get_ops() == ops
        assert element.get_ops() == [(0, False)]
        assert [s.get_ops() for s in view] == [[(0, False)], [(0, True)]]
        assert strings.max_string_len == length
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        element.set("F0^")
        strings[1] = ""
    assert str(view) == "F0^, F0^"
    assert strings.max_string_len == 129


def test_keys_include_length_and_ignore_capacity():
    values = ["", "F0", "F0 F0", "F1", "F1 F0"]
    small = StringSet.from_iterable(values, 2)
    large = StringSet.from_iterable(values, 2, max_string_len=129)
    assert len(small) == 5
    assert small == large
    for index, spec in enumerate(values):
        source = String(2, spec, max_string_len=256)
        assert small.lookup(source) == index
        assert small.insert(source) == index
    with pytest.warns(UserWarning) as caught:
        small.insert([(1, True)] * 65)
    assert len(caught) == 1
    for index, spec in enumerate(values):
        assert small.lookup(spec) == index
    small.remove("F0")  # Pop-and-swap moves the newly added long string.
    assert small.lookup([(1, True)] * 65) == 1
    assert small.lookup("F0") is None
    assert small.lookup("") == 0
    assert small.lookup("F0 F0") == 2
    assert small.max_string_len == 65


@pytest.mark.parametrize(
    ("term_type", "array_type", "set_type", "sum_type", "coeff"),
    (
        (RealTerm, RealTerms, RealTermSet, RealTermSum, 2.5),
        (ComplexTerm, ComplexTerms, ComplexTermSet, ComplexTermSum, 2 + 3j),
        (SymbolicTerm, SymbolicTerms, SymbolicTermSet, SymbolicTermSum, Symbol("x")),
    ),
)
def test_term_growth(term_type, array_type, set_type, sum_type, coeff):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        term = term_type(2, ("F0 F1", coeff), max_string_len=1)
    assert term.string.max_string_len == 2
    with pytest.warns(UserWarning) as caught:
        term.set(("F0 F1 F0", coeff))
    assert len(caught) == 1
    assert term.coeff == coeff
    for cls in (array_type, set_type, sum_type):
        value = cls.from_iterable([("F0", coeff)], 2)
        original = value.strings[0]
        with pytest.warns(UserWarning) as caught:
            if cls is array_type:
                value.append(term)
            elif cls is sum_type:
                value += term
            else:
                value.insert(term)
        assert len(caught) == 1
        assert value.max_string_len == 3
        assert str(original) == "F0"
        assert list(value.coeffs) == [coeff, coeff]


@pytest.mark.parametrize("cls", (Strings, StringSet, RealTerms, RealTermSet, RealTermSum))
def test_warning_as_error_does_not_mutate(cls):
    value = cls.from_iterable(["F0"], 2)
    before = str(value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            if cls in (Strings, RealTerms):
                value.append("F0 F1")
            elif cls is RealTermSum:
                value += "F0 F1"
            else:
                value.insert("F0 F1")
    assert str(value) == before
    assert len(value) == 1
    assert value.max_string_len == 1


def test_invalid_assignment_and_warning_error_are_atomic():
    string = String(2, "F0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            string.set("F0 F1")
        with pytest.raises(ValueError, match="bounds"):
            string.set("F0 F2")
        with pytest.raises(ValueError, match="same length"):
            string._impl.cmpnt_set_from_ops(0, [0, 1], [False])
        with pytest.raises(IndexError):
            string._impl.cmpnt_set_from_ops(3, [0, 1], [False, True])
    assert str(string) == "F0"
    assert string.max_string_len == 1


def test_negative_capacity_and_removed_keyword():
    for cls in (String, Strings, StringSet, RealTerm, RealTerms, RealTermSet, RealTermSum):
        with pytest.raises((OverflowError, ValueError)):
            cls(2, max_string_len=-1)
        with pytest.raises(TypeError):
            cls(2, max_len=2)


def test_binding_growth_and_silent_initial_allocation():
    array = GeneralFermionOperatorArray(Modes.from_count(2))
    array.resize(2)
    with pytest.warns(UserWarning):
        array.cmpnt_set_from_ops(1, [0, 1], [False, True])
    assert array.max_string_len == 2
    assert array.cmpnt_get_ops(0) == ([], [])
    assert array.cmpnt_get_ops(1) == ([0, 1], [False, True])


def test_product_preallocation_is_silent():
    left = RealTermSum.from_iterable([("F0", 2.0), ("F1", 3.0)], 2)
    right = RealTermSum.from_iterable([("F1 F0", 4.0)], 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        product = left * right
    assert product.max_string_len == 3
    assert len(product) == 2


def test_sum_preallocation_is_silent_and_refreshes_keys():
    left = RealTermSum.from_iterable([("F0", 2.0)], 2)
    right = RealTermSum.from_iterable([("F0 F1", 4.0)], 2, max_string_len=129)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        added = left + right
        subtracted = left - right
    assert added.max_string_len == 129
    assert left.max_string_len == 1
    assert list(added.coeffs) == [2.0, 4.0]
    assert list(subtracted.coeffs) == [2.0, -4.0]
    assert added.lookup("F0") == (0, 2.0)
    assert added.lookup("F0 F1") == (1, 4.0)
