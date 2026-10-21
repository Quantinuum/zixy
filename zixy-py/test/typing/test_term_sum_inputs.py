from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pytest
from sympy import Expr, sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs

if TYPE_CHECKING:
    from zixy._zixy import (
        FermionStateArray,
        GeneralFermionOperatorArray,
        NormalFermionOperatorArray,
        QubitPauliArray,
        QubitStateArray,
    )
    from zixy.container.terms import TermSpecT


def test_zqo_sum_iterable_inputs() -> None:
    class DerivedSum(zqo.RealTermSum):
        pass

    spec = "X0"
    string = zqo.String(2, spec)
    source = zqo.RealTerm(2, (string, 3.0))
    items: list[TermSpecT[QubitPauliArray, zqo.StringSpec, float]] = [
        (spec, 2.0),
        source,
        string,
        spec,
        (spec, None),
    ]
    # Lists and one-shot generators must sum duplicates instead of overwriting them.
    total = zqo.RealTermSum.from_iterable(items, 2)
    streamed = zqo.RealTermSum.from_iterable((item for item in items), 2)
    assert_type(total, zqo.RealTermSum)
    assert_type(streamed, zqo.RealTermSum)
    assert len(total) == len(streamed) == 1
    assert total[string] == streamed[string] == 8.0
    appended = zqo.RealTermSum(2)
    appended.add_iterable(iter(items))
    assert appended == total
    assert source.coeff == 3.0
    assert source.string == string
    derived = DerivedSum.from_iterable(items, 2)
    assert_type(derived, DerivedSum)
    assert type(derived) is DerivedSum
    assert derived[string] == 8.0
    assert len(zqo.RealTermSum.from_iterable([], 2)) == 0
    total.add_iterable([])
    assert total[string] == 8.0

    # Existing terms of a different coefficient type use checked destination conversion.
    complex_source = zqo.ComplexTerm(2, (string, 2 + 0j))
    converted = zqo.RealTermSum.from_iterable([complex_source], 2)
    assert converted[string] == 2.0
    converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    assert complex_source.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    with pytest.raises(ValueError):
        zqo.RealTermSum.from_iterable([complex_source], 2)
    with pytest.raises(ValueError):
        converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    with pytest.raises(TypeError):
        zqo.RealTermSum.from_iterable([zqs.RealTerm(2)], 2)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        converted.add_iterable([zqs.RealTerm(2)])  # type: ignore[list-item]

    # Symbolic sums follow the streaming Python path rather than the numeric Rust path.
    symbolic_items: list[TermSpecT[QubitPauliArray, zqo.StringSpec, Expr]] = [
        (spec, sympify("x")),
        zqo.SymbolicTerm(2, (string, sympify(2))),
    ]
    symbolic = zqo.SymbolicTermSum.from_iterable(iter(symbolic_items), 2)
    assert_type(symbolic, zqo.SymbolicTermSum)
    assert len(symbolic) == 1
    assert symbolic[string] == sympify("x + 2")
    symbolic.add_iterable([(spec, sympify(3))])
    assert symbolic[string] == sympify("x + 5")
    promoted = zqo.ComplexTermSum.from_iterable([source, (spec, 1j)], 2)
    assert_type(promoted, zqo.ComplexTermSum)
    assert promoted[string] == 3 + 1j


def test_zqs_sum_iterable_inputs() -> None:
    class DerivedSum(zqs.RealTermSum):
        pass

    spec = "[1, 0]"
    string = zqs.String(2, spec)
    source = zqs.RealTerm(2, (string, 3.0))
    items: list[TermSpecT[QubitStateArray, zqs.StringSpec, float]] = [
        (spec, 2.0),
        source,
        string,
        spec,
        (spec, None),
    ]
    # Lists and one-shot generators must sum duplicates instead of overwriting them.
    total = zqs.RealTermSum.from_iterable(items, 2)
    streamed = zqs.RealTermSum.from_iterable((item for item in items), 2)
    assert_type(total, zqs.RealTermSum)
    assert_type(streamed, zqs.RealTermSum)
    assert len(total) == len(streamed) == 1
    assert total[string] == streamed[string] == 8.0
    appended = zqs.RealTermSum(2)
    appended.add_iterable(iter(items))
    assert appended == total
    assert source.coeff == 3.0
    assert source.string == string
    derived = DerivedSum.from_iterable(items, 2)
    assert_type(derived, DerivedSum)
    assert type(derived) is DerivedSum
    assert derived[string] == 8.0
    assert len(zqs.RealTermSum.from_iterable([], 2)) == 0
    total.add_iterable([])
    assert total[string] == 8.0

    # Existing terms of a different coefficient type use checked destination conversion.
    complex_source = zqs.ComplexTerm(2, (string, 2 + 0j))
    converted = zqs.RealTermSum.from_iterable([complex_source], 2)
    assert converted[string] == 2.0
    converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    assert complex_source.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    with pytest.raises(ValueError):
        zqs.RealTermSum.from_iterable([complex_source], 2)
    with pytest.raises(ValueError):
        converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    with pytest.raises(TypeError):
        zqs.RealTermSum.from_iterable([zqo.RealTerm(2)], 2)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        converted.add_iterable([zqo.RealTerm(2)])  # type: ignore[list-item]

    # Symbolic sums follow the streaming Python path rather than the numeric Rust path.
    symbolic_items: list[TermSpecT[QubitStateArray, zqs.StringSpec, Expr]] = [
        (spec, sympify("x")),
        zqs.SymbolicTerm(2, (string, sympify(2))),
    ]
    symbolic = zqs.SymbolicTermSum.from_iterable(iter(symbolic_items), 2)
    assert_type(symbolic, zqs.SymbolicTermSum)
    assert len(symbolic) == 1
    assert symbolic[string] == sympify("x + 2")
    symbolic.add_iterable([(spec, sympify(3))])
    assert symbolic[string] == sympify("x + 5")
    promoted = zqs.ComplexTermSum.from_iterable([source, (spec, 1j)], 2)
    assert_type(promoted, zqs.ComplexTermSum)
    assert promoted[string] == 3 + 1j


def test_zfog_sum_iterable_inputs() -> None:
    class DerivedSum(zfog.RealTermSum):
        pass

    spec = "F0^ F1"
    string = zfog.String(2, spec, max_string_len=2)
    source = zfog.RealTerm(2, (string, 3.0), max_string_len=2)
    items: list[TermSpecT[GeneralFermionOperatorArray, zfog.StringSpec, float]] = [
        (spec, 2.0),
        source,
        string,
        spec,
        (spec, None),
    ]
    # Lists and one-shot generators must sum duplicates instead of overwriting them.
    total = zfog.RealTermSum.from_iterable(items, 2, max_string_len=2)
    streamed = zfog.RealTermSum.from_iterable((item for item in items), 2, max_string_len=2)
    assert_type(total, zfog.RealTermSum)
    assert_type(streamed, zfog.RealTermSum)
    assert len(total) == len(streamed) == 1
    assert total[string] == streamed[string] == 8.0
    appended = zfog.RealTermSum(2, max_string_len=2)
    appended.add_iterable(iter(items))
    assert appended == total
    assert source.coeff == 3.0
    assert source.string == string
    derived = DerivedSum.from_iterable(items, 2, max_string_len=2)
    assert_type(derived, DerivedSum)
    assert type(derived) is DerivedSum
    assert derived[string] == 8.0
    assert len(zfog.RealTermSum.from_iterable([], 2, max_string_len=2)) == 0
    total.add_iterable([])
    assert total[string] == 8.0

    # Existing terms of a different coefficient type use checked destination conversion.
    complex_source = zfog.ComplexTerm(2, (string, 2 + 0j), max_string_len=2)
    converted = zfog.RealTermSum.from_iterable([complex_source], 2, max_string_len=2)
    assert converted[string] == 2.0
    converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    assert complex_source.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    with pytest.raises(ValueError):
        zfog.RealTermSum.from_iterable([complex_source], 2, max_string_len=2)
    with pytest.raises(ValueError):
        converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    with pytest.raises(TypeError):
        zfog.RealTermSum.from_iterable([zfon.RealTerm(2)], 2, max_string_len=2)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        converted.add_iterable([zfon.RealTerm(2)])  # type: ignore[list-item]

    # Symbolic sums follow the streaming Python path rather than the numeric Rust path.
    symbolic_items: list[TermSpecT[GeneralFermionOperatorArray, zfog.StringSpec, Expr]] = [
        (spec, sympify("x")),
        zfog.SymbolicTerm(2, (string, sympify(2)), max_string_len=2),
    ]
    symbolic = zfog.SymbolicTermSum.from_iterable(iter(symbolic_items), 2, max_string_len=2)
    assert_type(symbolic, zfog.SymbolicTermSum)
    assert len(symbolic) == 1
    assert symbolic[string] == sympify("x + 2")
    symbolic.add_iterable([(spec, sympify(3))])
    assert symbolic[string] == sympify("x + 5")
    promoted = zfog.ComplexTermSum.from_iterable([source, (spec, 1j)], 2, max_string_len=2)
    assert_type(promoted, zfog.ComplexTermSum)
    assert promoted[string] == 3 + 1j


def test_zfon_sum_iterable_inputs() -> None:
    class DerivedSum(zfon.RealTermSum):
        pass

    spec = "F0^ F1"
    string = zfon.String(2, spec)
    source = zfon.RealTerm(2, (string, 3.0))
    items: list[TermSpecT[NormalFermionOperatorArray, zfon.StringSpec, float]] = [
        (spec, 2.0),
        source,
        string,
        spec,
        (spec, None),
    ]
    # Lists and one-shot generators must sum duplicates instead of overwriting them.
    total = zfon.RealTermSum.from_iterable(items, 2)
    streamed = zfon.RealTermSum.from_iterable((item for item in items), 2)
    assert_type(total, zfon.RealTermSum)
    assert_type(streamed, zfon.RealTermSum)
    assert len(total) == len(streamed) == 1
    assert total[string] == streamed[string] == 8.0
    appended = zfon.RealTermSum(2)
    appended.add_iterable(iter(items))
    assert appended == total
    assert source.coeff == 3.0
    assert source.string == string
    derived = DerivedSum.from_iterable(items, 2)
    assert_type(derived, DerivedSum)
    assert type(derived) is DerivedSum
    assert derived[string] == 8.0
    assert len(zfon.RealTermSum.from_iterable([], 2)) == 0
    total.add_iterable([])
    assert total[string] == 8.0

    # Existing terms of a different coefficient type use checked destination conversion.
    complex_source = zfon.ComplexTerm(2, (string, 2 + 0j))
    converted = zfon.RealTermSum.from_iterable([complex_source], 2)
    assert converted[string] == 2.0
    converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    assert complex_source.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    with pytest.raises(ValueError):
        zfon.RealTermSum.from_iterable([complex_source], 2)
    with pytest.raises(ValueError):
        converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    with pytest.raises(TypeError):
        zfon.RealTermSum.from_iterable([zfog.RealTerm(2)], 2)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        converted.add_iterable([zfog.RealTerm(2)])  # type: ignore[list-item]

    # Symbolic sums follow the streaming Python path rather than the numeric Rust path.
    symbolic_items: list[TermSpecT[NormalFermionOperatorArray, zfon.StringSpec, Expr]] = [
        (spec, sympify("x")),
        zfon.SymbolicTerm(2, (string, sympify(2))),
    ]
    symbolic = zfon.SymbolicTermSum.from_iterable(iter(symbolic_items), 2)
    assert_type(symbolic, zfon.SymbolicTermSum)
    assert len(symbolic) == 1
    assert symbolic[string] == sympify("x + 2")
    symbolic.add_iterable([(spec, sympify(3))])
    assert symbolic[string] == sympify("x + 5")
    promoted = zfon.ComplexTermSum.from_iterable([source, (spec, 1j)], 2)
    assert_type(promoted, zfon.ComplexTermSum)
    assert promoted[string] == 3 + 1j


def test_zfs_sum_iterable_inputs() -> None:
    class DerivedSum(zfs.RealTermSum):
        pass

    spec = "[1, 0]"
    string = zfs.String(2, spec)
    source = zfs.RealTerm(2, (string, 3.0))
    items: list[TermSpecT[FermionStateArray, zfs.StringSpec, float]] = [
        (spec, 2.0),
        source,
        string,
        spec,
        (spec, None),
    ]
    # Lists and one-shot generators must sum duplicates instead of overwriting them.
    total = zfs.RealTermSum.from_iterable(items, 2)
    streamed = zfs.RealTermSum.from_iterable((item for item in items), 2)
    assert_type(total, zfs.RealTermSum)
    assert_type(streamed, zfs.RealTermSum)
    assert len(total) == len(streamed) == 1
    assert total[string] == streamed[string] == 8.0
    appended = zfs.RealTermSum(2)
    appended.add_iterable(iter(items))
    assert appended == total
    assert source.coeff == 3.0
    assert source.string == string
    derived = DerivedSum.from_iterable(items, 2)
    assert_type(derived, DerivedSum)
    assert type(derived) is DerivedSum
    assert derived[string] == 8.0
    assert len(zfs.RealTermSum.from_iterable([], 2)) == 0
    total.add_iterable([])
    assert total[string] == 8.0

    # Existing terms of a different coefficient type use checked destination conversion.
    complex_source = zfs.ComplexTerm(2, (string, 2 + 0j))
    converted = zfs.RealTermSum.from_iterable([complex_source], 2)
    assert converted[string] == 2.0
    converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    assert complex_source.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    with pytest.raises(ValueError):
        zfs.RealTermSum.from_iterable([complex_source], 2)
    with pytest.raises(ValueError):
        converted.add_iterable([complex_source])
    assert converted[string] == 4.0
    with pytest.raises(TypeError):
        zfs.RealTermSum.from_iterable([zqs.RealTerm(2)], 2)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        converted.add_iterable([zqs.RealTerm(2)])  # type: ignore[list-item]

    # Symbolic sums follow the streaming Python path rather than the numeric Rust path.
    symbolic_items: list[TermSpecT[FermionStateArray, zfs.StringSpec, Expr]] = [
        (spec, sympify("x")),
        zfs.SymbolicTerm(2, (string, sympify(2))),
    ]
    symbolic = zfs.SymbolicTermSum.from_iterable(iter(symbolic_items), 2)
    assert_type(symbolic, zfs.SymbolicTermSum)
    assert len(symbolic) == 1
    assert symbolic[string] == sympify("x + 2")
    symbolic.add_iterable([(spec, sympify(3))])
    assert symbolic[string] == sympify("x + 5")
    promoted = zfs.ComplexTermSum.from_iterable([source, (spec, 1j)], 2)
    assert_type(promoted, zfs.ComplexTermSum)
    assert promoted[string] == 3 + 1j
