from __future__ import annotations

from typing import assert_type

import pytest
from sympy import Expr, I, Symbol, symbols

import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs


@pytest.mark.parametrize(
    ("operator_type", "operator_coeff", "state_type", "state_coeff", "result_type", "coeff"),
    (
        (zqo.RealTermSum, 2.0, zqs.RealTermSum, 3.0, zqs.ComplexTermSum, 6j),
        (zqo.RealTermSum, 2.0, zqs.ComplexTermSum, 3j, zqs.ComplexTermSum, -6),
        (
            zqo.RealTermSum,
            2.0,
            zqs.SymbolicTermSum,
            Symbol("b"),
            zqs.SymbolicTermSum,
            2 * I * Symbol("b"),
        ),
        (zqo.ComplexTermSum, 2j, zqs.RealTermSum, 3.0, zqs.ComplexTermSum, -6),
        (zqo.ComplexTermSum, 2j, zqs.ComplexTermSum, 3j, zqs.ComplexTermSum, -6j),
        (
            zqo.ComplexTermSum,
            2j,
            zqs.SymbolicTermSum,
            Symbol("b"),
            zqs.SymbolicTermSum,
            -2 * Symbol("b"),
        ),
        (
            zqo.SymbolicTermSum,
            Symbol("a"),
            zqs.RealTermSum,
            3.0,
            zqs.SymbolicTermSum,
            3 * I * Symbol("a"),
        ),
        (
            zqo.SymbolicTermSum,
            Symbol("a"),
            zqs.ComplexTermSum,
            3j,
            zqs.SymbolicTermSum,
            -3 * Symbol("a"),
        ),
        (
            zqo.SymbolicTermSum,
            Symbol("a"),
            zqs.SymbolicTermSum,
            Symbol("b"),
            zqs.SymbolicTermSum,
            I * Symbol("a") * Symbol("b"),
        ),
    ),
)
def test_apply_coefficient_combinations(
    operator_type, operator_coeff, state_type, state_coeff, result_type, coeff
):
    operator = operator_type.from_iterable([("Y0", operator_coeff)], 1)
    state = state_type.from_iterable([("[0]", state_coeff)], 1)

    result = operator.apply(state)

    assert type(result) is result_type
    assert result.get("[1]") == coeff


@pytest.mark.parametrize(
    ("pauli", "ket", "result_string", "coeff"),
    (
        ("I0", "[1, 0]", "[1, 0]", 1),
        ("X0", "[0, 0]", "[1, 0]", 1),
        ("Y0", "[0, 0]", "[1, 0]", I),
        ("Z0", "[1, 0]", "[1, 0]", -1),
        ("X0 Z1", "[0, 1]", "[1, 1]", -1),
    ),
)
def test_symbolic_apply_pauli_actions(pauli, ket, result_string, coeff):
    operator = zqo.SymbolicTermSum.from_iterable([(pauli, Symbol("a"))], 2)
    state = zqs.SymbolicTermSum.from_iterable([(ket, Symbol("b"))], 2)

    result = operator.apply(state)

    assert result.get(result_string) == coeff * Symbol("a") * Symbol("b")


@pytest.mark.parametrize(
    ("operator_type", "state_type"),
    (
        (zqo.RealTermSum, zqs.RealTermSum),
        (zqo.ComplexTermSum, zqs.ComplexTermSum),
        (zqo.SymbolicTermSum, zqs.SymbolicTermSum),
    ),
)
def test_apply_consolidates_and_cancels(operator_type, state_type):
    operator = operator_type.from_iterable([("I0", 1), ("Z0", -1)], 1)
    state = state_type.from_iterable([("[0]", 1)], 1)

    assert len(operator.apply(state)) == 0


def test_apply_consolidates_duplicate_outputs():
    operator = zqo.SymbolicTermSum.from_iterable([("X0", 1), ("Z0", 1)], 1)
    state = zqs.SymbolicTermSum.from_iterable([("[0]", 1), ("[1]", 1)], 1)

    result = operator.apply(state)

    assert len(result) == 1
    assert result.get("[0]") == 2
    assert result.get("[1]") is None


@pytest.mark.parametrize(
    ("operator", "state"),
    (
        (zqo.RealTermSum(2), zqs.RealTermSum.from_str("[0, 0]", 2)),
        (zqo.ComplexTermSum.from_str("X0", 2), zqs.ComplexTermSum(2)),
        (zqo.SymbolicTermSum(2), zqs.SymbolicTermSum.from_str("[0, 0]", 2)),
    ),
)
def test_apply_empty_inputs(operator, state):
    assert len(operator.apply(state)) == 0


def test_symbolic_apply_rejects_different_qubits():
    operator = zqo.SymbolicTermSum.from_str("X0", 2)
    state = zqs.SymbolicTermSum.from_str("[0, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        operator.apply(state)


def test_apply_does_not_mutate_or_alias_inputs():
    operator = zqo.SymbolicTermSum.from_iterable([("I0", Symbol("a"))], 1)
    state = zqs.SymbolicTermSum.from_iterable([("[0]", Symbol("b"))], 1)
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
    real_operator = zqo.RealTermSum.from_iterable([("Y0", 2)], 1)
    symbolic_operator = zqo.SymbolicTermSum.from_iterable([("Y0", a)], 1)
    real_bra = zqs.RealTermSum.from_iterable([("[1]", 1)], 1)
    real_ket = zqs.RealTermSum.from_iterable([("[0]", 3)], 1)
    complex_ket = zqs.ComplexTermSum.from_iterable([("[0]", 3j)], 1)
    symbolic_bra = zqs.SymbolicTermSum.from_iterable([("[1]", b)], 1)
    symbolic_zero = zqs.SymbolicTermSum.from_iterable([("[0]", b)], 1)

    assert real_operator.mat_elem(real_bra, real_ket) == 6j
    assert real_operator.mat_elem(symbolic_bra, complex_ket) == -6 * b.conjugate()
    assert symbolic_operator.mat_elem(real_bra, real_ket) == 3 * I * a
    assert symbolic_operator.exp_val(symbolic_zero) == 0
    assert isinstance(real_operator.mat_elem(symbolic_bra, complex_ket), Expr)


def test_real_matrix_element_path():
    operator = zqo.RealTermSum.from_iterable([("X0", 2)], 1)
    bra = zqs.RealTermSum.from_iterable([("[1]", 1)], 1)
    ket = zqs.RealTermSum.from_iterable([("[0]", 3)], 1)

    result = operator.mat_elem(bra, ket, real=True)

    assert type(result) is float
    assert result == 6.0


def test_real_expectation_value_path():
    operator = zqo.RealTermSum.from_iterable([("Z0", 2)], 1)
    state = zqs.RealTermSum.from_iterable([("[0]", 3)], 1)

    result = operator.exp_val(state, real=True)

    assert type(result) is float
    assert result == 18.0


@pytest.mark.parametrize(
    ("bra", "ket"),
    (
        (zqs.ComplexTermSum.from_str("[1]", 1), zqs.RealTermSum.from_str("[0]", 1)),
        (zqs.RealTermSum.from_str("[1]", 1), zqs.ComplexTermSum.from_str("[0]", 1)),
        (zqs.SymbolicTermSum.from_str("[1]", 1), zqs.RealTermSum.from_str("[0]", 1)),
    ),
)
def test_real_matrix_element_path_requires_real_states(bra, ket):
    operator = zqo.RealTermSum.from_str("X0", 1)

    with pytest.raises(TypeError, match="real=True requires real bra and ket states"):
        operator.mat_elem(bra, ket, real=True)


def test_operation_types():
    real_operator = zqo.RealTermSum(1)
    complex_operator = zqo.ComplexTermSum(1)
    symbolic_operator = zqo.SymbolicTermSum(1)
    real_state = zqs.RealTermSum(1)
    complex_state = zqs.ComplexTermSum(1)
    symbolic_state = zqs.SymbolicTermSum(1)

    assert_type(real_operator.apply(real_state), zqs.ComplexTermSum)
    assert_type(real_operator.apply(complex_state), zqs.ComplexTermSum)
    assert_type(real_operator.apply(symbolic_state), zqs.SymbolicTermSum)
    assert_type(complex_operator.apply(real_state), zqs.ComplexTermSum)
    assert_type(complex_operator.apply(complex_state), zqs.ComplexTermSum)
    assert_type(complex_operator.apply(symbolic_state), zqs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(real_state), zqs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(complex_state), zqs.SymbolicTermSum)
    assert_type(symbolic_operator.apply(symbolic_state), zqs.SymbolicTermSum)
    assert_type(real_operator.mat_elem(real_state, complex_state), complex)
    assert_type(real_operator.mat_elem(real_state, real_state, real=True), float)
    assert_type(real_operator.mat_elem(symbolic_state, complex_state), Expr)
    assert_type(real_operator.exp_val(real_state, real=True), float)
    assert_type(complex_operator.exp_val(symbolic_state), Expr)
    assert_type(symbolic_operator.mat_elem(real_state, complex_state), Expr)
    assert_type(symbolic_operator.exp_val(real_state), Expr)
