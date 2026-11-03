from __future__ import annotations

from typing import assert_type

import pytest
from sympy import Expr, I, Symbol, symbols

import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs


@pytest.mark.parametrize(
    ("operator_type", "operator_coeff", "state_type", "state_coeff", "result_type", "coeff"),
    (
        (zfon.RealTermSum, 2.0, zfs.RealTermSum, 3.0, zfs.RealTermSum, 6.0),
        (zfon.RealTermSum, 2.0, zfs.ComplexTermSum, 3j, zfs.ComplexTermSum, 6j),
        (
            zfon.RealTermSum,
            2.0,
            zfs.SymbolicTermSum,
            Symbol("b"),
            zfs.SymbolicTermSum,
            2 * Symbol("b"),
        ),
        (zfon.ComplexTermSum, 2j, zfs.RealTermSum, 3.0, zfs.ComplexTermSum, 6j),
        (zfon.ComplexTermSum, 2j, zfs.ComplexTermSum, 3j, zfs.ComplexTermSum, -6),
        (
            zfon.ComplexTermSum,
            2j,
            zfs.SymbolicTermSum,
            Symbol("b"),
            zfs.SymbolicTermSum,
            2 * I * Symbol("b"),
        ),
        (
            zfon.SymbolicTermSum,
            Symbol("a"),
            zfs.RealTermSum,
            3.0,
            zfs.SymbolicTermSum,
            3 * Symbol("a"),
        ),
        (
            zfon.SymbolicTermSum,
            Symbol("a"),
            zfs.ComplexTermSum,
            3j,
            zfs.SymbolicTermSum,
            3 * I * Symbol("a"),
        ),
        (
            zfon.SymbolicTermSum,
            Symbol("a"),
            zfs.SymbolicTermSum,
            Symbol("b"),
            zfs.SymbolicTermSum,
            Symbol("a") * Symbol("b"),
        ),
    ),
)
def test_apply_coefficient_combinations(
    operator_type, operator_coeff, state_type, state_coeff, result_type, coeff
):
    operator = operator_type.from_iterable([("F1^ F0", operator_coeff)], 2)
    state = state_type.from_iterable([("[1, 0]", state_coeff)], 2)

    result = operator.apply(state)

    assert type(result) is result_type
    assert result.lookup_coeff("[0, 1]") == coeff


@pytest.mark.parametrize(
    ("operator", "state", "result", "sign"),
    (
        ("F1^", "[1, 0]", "[1, 1]", -1),
        ("F1", "[1, 1]", "[1, 0]", -1),
        ("F1^ F1", "[0, 1]", "[0, 1]", 1),
        ("F2^ F0", "[1, 1, 0]", "[0, 1, 1]", -1),
    ),
)
def test_symbolic_apply_ladder_actions(operator, state, result, sign):
    modes = len(state.strip("[]").split(","))
    op = zfon.SymbolicTermSum.from_iterable([(operator, Symbol("a"))], modes)
    ket = zfs.SymbolicTermSum.from_iterable([(state, Symbol("b"))], modes)

    applied = op.apply(ket)

    assert applied.lookup_coeff(result) == sign * Symbol("a") * Symbol("b")


@pytest.mark.parametrize(
    ("operator", "state"),
    (
        ("F0^", "[1, 0]"),
        ("F1", "[1, 0]"),
    ),
)
def test_symbolic_apply_invalid_ladder_action_is_zero(operator, state):
    op = zfon.SymbolicTermSum.from_str(operator, 2)
    ket = zfs.SymbolicTermSum.from_str(state, 2)

    assert len(op.apply(ket)) == 0


@pytest.mark.parametrize(
    ("operator_type", "state_type"),
    (
        (zfon.RealTermSum, zfs.RealTermSum),
        (zfon.ComplexTermSum, zfs.ComplexTermSum),
        (zfon.SymbolicTermSum, zfs.SymbolicTermSum),
    ),
)
def test_apply_consolidates_and_cancels(operator_type, state_type):
    operator = operator_type.from_iterable([("", 1), ("F0^ F0", -1)], 1)
    state = state_type.from_iterable([("[1]", 1)], 1)

    assert len(operator.apply(state)) == 0


def test_symbolic_apply_consolidates_duplicate_outputs():
    operator = zfon.SymbolicTermSum.from_iterable([("", 1), ("F0^ F0", 1)], 1)
    state = zfs.SymbolicTermSum.from_iterable([("[1]", 1)], 1)

    result = operator.apply(state)

    assert len(result) == 1
    assert result.lookup_coeff("[1]") == 2


@pytest.mark.parametrize(
    ("operator", "state"),
    (
        (zfon.RealTermSum(2), zfs.RealTermSum.from_str("[1, 0]", 2)),
        (zfon.ComplexTermSum.from_str("F1^ F0", 2), zfs.ComplexTermSum(2)),
        (zfon.SymbolicTermSum(2), zfs.SymbolicTermSum.from_str("[1, 0]", 2)),
    ),
)
def test_apply_empty_inputs(operator, state):
    assert len(operator.apply(state)) == 0


def test_symbolic_apply_identity_on_empty_mode_space():
    operator = zfon.SymbolicTermSum.from_iterable([("", 2)], 0)
    state = zfs.SymbolicTermSum.from_iterable([("", 3)], 0)

    result = operator.apply(state)

    assert len(result) == 1
    assert result.lookup_coeff("") == 6


def test_symbolic_apply_rejects_different_modes():
    operator = zfon.SymbolicTermSum.from_str("F1^ F0", 2)
    state = zfs.SymbolicTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        operator.apply(state)


def test_apply_does_not_mutate_or_alias_inputs():
    operator = zfon.SymbolicTermSum.from_iterable([("", Symbol("a"))], 1)
    state = zfs.SymbolicTermSum.from_iterable([("[1]", Symbol("b"))], 1)
    operator_before = operator.clone()
    state_before = state.clone()

    result = operator.apply(state)

    assert operator == operator_before
    assert state == state_before
    result_term = next(iter(result))
    state_term = next(iter(state))
    assert not result_term.aliases(state_term)
    result_term.coeff = 0
    assert operator == operator_before
    assert state == state_before


def test_mixed_matrix_elements_and_expectation_values():
    a, b = symbols("a b")
    real_operator = zfon.RealTermSum.from_iterable([("F1^ F0", 2)], 2)
    complex_operator = zfon.ComplexTermSum.from_iterable([("F1^ F0", 2j)], 2)
    symbolic_operator = zfon.SymbolicTermSum.from_iterable([("F1^ F0", a)], 2)
    real_bra = zfs.RealTermSum.from_iterable([("[0, 1]", 1)], 2)
    real_ket = zfs.RealTermSum.from_iterable([("[1, 0]", 3)], 2)
    complex_ket = zfs.ComplexTermSum.from_iterable([("[1, 0]", 3j)], 2)
    symbolic_bra = zfs.SymbolicTermSum.from_iterable([("[0, 1]", b)], 2)
    occupied = zfs.SymbolicTermSum.from_iterable([("[1, 0]", b)], 2)
    number_operator = zfon.SymbolicTermSum.from_iterable([("F0^ F0", a)], 2)

    assert real_operator.mat_elem(real_bra, real_ket) == 6.0
    assert real_operator.mat_elem(real_bra, complex_ket) == 6j
    assert real_operator.mat_elem(symbolic_bra, complex_ket) == 6 * I * b.conjugate()
    assert complex_operator.mat_elem(real_bra, real_ket) == 6j
    assert symbolic_operator.mat_elem(real_bra, real_ket) == 3 * a
    assert number_operator.exp_val(occupied) == a * b * b.conjugate()
    assert isinstance(real_operator.mat_elem(symbolic_bra, complex_ket), Expr)


def test_operation_types():
    real_operator = zfon.RealTermSum(1)
    complex_operator = zfon.ComplexTermSum(1)
    symbolic_operator = zfon.SymbolicTermSum(1)
    real_state = zfs.RealTermSum(1)
    complex_state = zfs.ComplexTermSum(1)
    symbolic_state = zfs.SymbolicTermSum(1)

    assert_type(real_operator.apply(real_state), zfs.RealTermSum)
    assert_type(real_operator.apply(complex_state), zfs.ComplexTermSum)
    assert_type(real_operator.apply(symbolic_state), zfs.SymbolicTermSum)
    assert_type(complex_operator.apply(real_state), zfs.ComplexTermSum)
    assert_type(complex_operator.apply(complex_state), zfs.ComplexTermSum)
    assert_type(complex_operator.apply(symbolic_state), zfs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(real_state), zfs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(complex_state), zfs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(symbolic_state), zfs.SymbolicTermSum)
    assert_type(real_operator.mat_elem(real_state, real_state), float)
    assert_type(real_operator.mat_elem(real_state, complex_state), complex)
    assert_type(real_operator.mat_elem(symbolic_state, complex_state), Expr)
    assert_type(complex_operator.exp_val(real_state), complex)
    assert_type(complex_operator.exp_val(symbolic_state), Expr)
    assert_type(symbolic_operator.mat_elem(real_state, complex_state), Expr)
    assert_type(symbolic_operator.exp_val(real_state), Expr)
