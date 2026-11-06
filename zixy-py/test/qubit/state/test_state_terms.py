from typing import assert_type

import pytest
from sympy import Expr, I as imaginary_unit, S, symbols

import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import Sign
from zixy.qubit.operator import I, X


@pytest.mark.parametrize(
    ("term_type", "source", "qubits", "expected", "terms_type"),
    (
        (zqs.SignTerm, "(1, [1, 0])", 2, "(+1, [1, 0])", None),
        (
            zqs.SignTerms,
            "(1, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1]), (+1, [1, 0])",
            None,
        ),
        (
            zqs.SignTermSet,
            "(1, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1])",
            zqs.SignTerms,
        ),
        (zqs.ComplexSignTerm, "(1j, [1, 0])", 2, "(+i, [1, 0])", None),
        (
            zqs.ComplexSignTerms,
            "(1j, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+i, [1, 0]), (-1, [0, 1]), (+1, [1, 0])",
            None,
        ),
        (
            zqs.ComplexSignTermSet,
            "(1j, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1])",
            zqs.ComplexSignTerms,
        ),
        (zqs.RealTerm, " ( 2.5 , [1, 0] ) ", 2, "(2.5, [1, 0])", None),
        (
            zqs.RealTerms,
            " (2.0, [1, 0]) , (-0.5, [0, 1]) , (3.0, [1, 0]) ",
            2,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            None,
        ),
        (
            zqs.RealTermSet,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            2,
            "(3.0, [1, 0]), (-0.5, [0, 1])",
            zqs.RealTerms,
        ),
        (
            zqs.RealTermSum,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            2,
            "(5.0, [1, 0]), (-0.5, [0, 1])",
            None,
        ),
        (zqs.ComplexTerm, "(1j, [1, 0])", 2, "(1j, [1, 0])", None),
        (
            zqs.ComplexTerms,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "(1j, [1, 0]), ((2+0j), [0, 1]), ((3+0j), [1, 0])",
            None,
        ),
        (
            zqs.ComplexTermSet,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "((3+0j), [1, 0]), ((2+0j), [0, 1])",
            zqs.ComplexTerms,
        ),
        (
            zqs.ComplexTermSum,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "((3+1j), [1, 0]), ((2+0j), [0, 1])",
            None,
        ),
        (zqs.RealTerm, "(2, [1, 0, 1, 0])", 4, "(2.0, [1, 0, 1, 0])", None),
        (zqs.RealTerms, "\n(1, [1, 0]),\n(2, [0, 1])\n", 2, "(1.0, [1, 0]), (2.0, [0, 1])", None),
        (zqs.RealTermSum, "(2, [1, 0]), (-2, [1, 0])", 2, "(0.0, [1, 0])", None),
    ),
)
def test_from_str(term_type, source, qubits, expected, terms_type):
    parsed = term_type.from_str(source, qubits)
    assert str(parsed) == expected

    if terms_type is not None:
        assert parsed == term_type.from_terms(terms_type.from_str(source, qubits))


@pytest.mark.parametrize(
    ("term_type", "source", "qubits"),
    (
        (zqs.SignTerms, "", 2),
        (zqs.SignTermSet, "", 2),
        (zqs.ComplexSignTerms, "", 2),
        (zqs.ComplexSignTermSet, "", 2),
        (zqs.RealTerms, "", 2),
        (zqs.RealTermSet, "", 2),
        (zqs.RealTermSum, "", 2),
        (zqs.ComplexTerms, "", 2),
        (zqs.ComplexTermSet, "", 2),
        (zqs.ComplexTermSum, "", 2),
    ),
)
def test_from_str_empty_containers(term_type, source, qubits):
    assert str(term_type.from_str(source, qubits)) == ""


@pytest.mark.parametrize(
    ("term_type", "source"),
    (
        (zqs.RealTerm, "(2.5, [1, 0])"),
        (zqs.RealTerms, "(2.0, [1, 0]), (-0.5, [0, 1])"),
        (zqs.RealTermSet, "(2.0, [1, 0]), (-0.5, [0, 1])"),
        (zqs.RealTermSum, "(2.0, [1, 0]), (-0.5, [0, 1])"),
    ),
)
def test_from_str_round_trip(term_type, source):
    parsed = term_type.from_str(source, 2)
    assert term_type.from_str(str(parsed), 2) == parsed


@pytest.mark.parametrize(
    "term_type",
    (zqs.SymbolicTerm, zqs.SymbolicTerms, zqs.SymbolicTermSet, zqs.SymbolicTermSum),
)
def test_from_str_symbolic_not_implemented(term_type):
    with pytest.raises(NotImplementedError):
        term_type.from_str("(a, [1, 0])", 2)


@pytest.mark.parametrize(
    "source",
    ("1, [1, 0]", "(1 [1, 0])", "(1, [1, 0], extra)", "(, [1, 0])", "(1, )"),
)
def test_from_str_errors(source):
    with pytest.raises(ValueError):
        zqs.RealTerm.from_str(source, 2)

    with pytest.raises(IndexError):
        zqs.RealTerm.from_str("(1, [1, 0, 1])", 2)

    with pytest.raises(ValueError):
        zqs.SignTerm.from_str("(2, [1, 0])", 2)

    with pytest.raises(ValueError):
        zqs.ComplexSignTerm.from_str("(2, [1, 0])", 2)

    with pytest.raises(ValueError):
        zqs.RealTerm.from_str("(1j, [1, 0])", 2)


def test_term():
    term = zqs.SignTerm(6, "")
    assert term.coeff == Sign(False)
    with pytest.raises(TypeError):
        zqs.SignTerm(6, None)
    term.string.set((1, 0) * 3)
    assert str(term) == "(+1, [1, 0, 1, 0, 1, 0])"
    term *= -1
    assert str(term) == "(-1, [1, 0, 1, 0, 1, 0])"
    assert str(term * Sign(True)) == "(+1, [1, 0, 1, 0, 1, 0])"

    term = zqs.RealTerm.from_str("[1, 0, 1, 0, 1, 0]")
    assert str(term) == "(1.0, [1, 0, 1, 0, 1, 0])"
    assert term.coeff == 1.0


def test_term_real_sum():
    lc = zqs.RealTermSum(6)
    assert len(lc._impl._strings._impl.qubits) == 6
    assert len(lc) == 0
    assert str(lc) == ""
    lc += zqs.RealTerm(6, ((1, 0, 0, 0, 0, 1), 3.0))
    assert len(lc) == 1
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1])"
    lc += zqs.RealTerm(6, ((1, 1, 0, 0, 1, 1), 4.0))
    assert len(lc) == 2
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1]), (4.0, [1, 1, 0, 0, 1, 1])"
    assert lc.l1_norm == 3 + 4
    assert lc.l2_norm_square == 3**2 + 4**2
    assert zqs.RealTermSet.from_str(
        str(zqs.RealTermSet.from_terms(lc.to_terms()))
    ) == zqs.RealTermSet.from_terms(lc.to_terms())
    assert zqs.RealTermSum.from_str(str(lc)) == lc


def test_vdot_rejects_different_qubits():
    lhs = zqs.RealTermSum.from_str("[1, 0]", 2)
    rhs = zqs.RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        lhs.vdot(rhs)


def test_complex_vdot_rejects_different_qubits():
    lhs = zqs.ComplexTermSum.from_str("[1, 0]", 2)
    rhs = zqs.ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        lhs.vdot(rhs)


def test_symbolic_vdot():
    x, y, z = symbols("x y z", real=True)
    lhs = zqs.SymbolicTermSum.from_iterable(
        [
            ("[1, 0, 0]", x + imaginary_unit * y),
            ("[0, 1, 0]", 2 * x),
            ("[0, 0, 1]", z),
        ],
        3,
    )
    rhs = zqs.SymbolicTermSum.from_iterable(
        [("[1, 0, 0]", z), ("[0, 1, 0]", 3), ("[1, 1, 0]", y)], 3
    )
    lhs_before = lhs.clone()
    rhs_before = rhs.clone()

    result = lhs.vdot(rhs)

    assert_type(result, Expr)
    assert result == 6 * x + z * (x - imaginary_unit * y)
    assert result.subs(x, 2) == 12 + z * (2 - imaginary_unit * y)
    assert lhs == lhs_before
    assert rhs == rhs_before
    lhs["[1, 0, 0]"] = S.Zero
    rhs["[1, 0, 0]"] = S.Zero
    assert result == 6 * x + z * (x - imaginary_unit * y)


def test_symbolic_vdot_empty_and_disjoint():
    x = symbols("x")
    empty = zqs.SymbolicTermSum(2)
    lhs = zqs.SymbolicTermSum.from_iterable([("[1, 0]", x)], 2)
    rhs = zqs.SymbolicTermSum.from_iterable([("[0, 1]", x)], 2)

    assert empty.vdot(empty) is S.Zero
    assert empty.vdot(lhs) is S.Zero
    assert lhs.vdot(empty) is S.Zero
    assert lhs.vdot(rhs) is S.Zero


def test_symbolic_vdot_rejects_different_qubits():
    lhs = zqs.SymbolicTermSum.from_str("[1, 0]", 2)
    rhs = zqs.SymbolicTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        lhs.vdot(rhs)


def test_term_scalar_mul_preserves_viewed_string():
    terms = zqs.RealTerms.from_str("(1.0, [0, 0]), (2.0, [1, 1])", 2)

    right_scaled = terms[1] * 3.0
    left_scaled = 3.0 * terms[1]

    assert right_scaled.string == terms[1].string
    assert right_scaled.string != terms[0].string
    assert right_scaled.coeff == 6.0
    assert not right_scaled.string.aliases(terms[1].string)
    assert not right_scaled.string.aliases(terms[0].string)
    assert left_scaled.string == terms[1].string
    assert left_scaled.string != terms[0].string
    assert left_scaled.coeff == 6.0
    assert not left_scaled.string.aliases(terms[1].string)
    assert not left_scaled.string.aliases(terms[0].string)


def test_term_set_check_term():
    term_set = zqs.RealTermSet(3)
    term = zqs.RealTerm(3, ((1, 0, 0), 1.0))

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different qubits"):
        term_set._check_term(zqs.RealTerm(4, ((1, 0, 0, 0), 1.0)))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(zqo.RealTerm.from_string(zqo.String(3, (X, I, I)), 1.0))


def test_term_set_check_string():
    term_set = zqs.RealTermSet(3)
    string = zqs.String(3, (1, 0, 0))

    term_set._check_string(string)

    with pytest.raises(ValueError, match="different qubits"):
        term_set._check_string(zqs.String(4, (1, 0, 0, 0)))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_string(zqo.String(3, (X, I, I)))


def test_real_term_sum_to_dense_matrix_index_error():
    op = zqs.RealTermSum(65)

    with pytest.raises(IndexError):
        op.to_dense()


def test_complex_term_sum_to_dense_matrix_index_error():
    op = zqs.ComplexTermSum(65)

    with pytest.raises(IndexError):
        op.to_dense()
