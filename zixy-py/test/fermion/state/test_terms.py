import numpy as np
import pytest

from zixy.fermion.operator.normal import (
    ComplexTermSum as ComplexOperator,
    RealTerm as RealOperatorTerm,
    RealTermSum as RealOperator,
    String as OperatorString,
)
from zixy.fermion.state import (
    ComplexTerm,
    ComplexTerms,
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    String,
)


def test_term_from_str():
    term = RealTerm.from_str("(2.0, [1, 0, 1])", 3)

    assert term.string == String(3, {0, 2})
    assert term.coeff == 2.0


def test_terms_from_str():
    terms = RealTerms.from_str("(2.0, [1, 0]), (-3.0, [0, 1])", 2)

    assert len(terms) == 2
    assert [term.string.get_set() for term in terms] == [{0}, {1}]
    assert [term.coeff for term in terms] == [2.0, -3.0]


def test_term_sum_from_str():
    terms = RealTermSum.from_str("(2.0, [1, 0]), (-3.0, [0, 1])", 2)

    assert len(terms) == 2
    assert terms.strings.get_sets() == ({0}, {1})


def test_term_scalar_mul_preserves_viewed_string():
    terms = RealTerms.from_str("(1.0, [0, 0]), (2.0, [1, 1])", 2)

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
    term_set = RealTermSet(3)
    term = RealTerm.from_str("[1, 0, 0]", 3)

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_term(RealTerm.from_str("[1, 0, 0, 0]", 4))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(RealOperatorTerm.from_str("F0^ F1", 3))


def test_term_set_check_cmpnt():
    term_set = RealTermSet(3)
    string = String(3, {0})

    term_set._check_cmpnt(string)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_cmpnt(String(4, {0}))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_cmpnt(OperatorString(3, "F0^ F1"))


def test_real_dense_round_trip():
    state = RealTermSum.from_dense(2, [0.0, 1.0, 2.0, 0.0])

    assert state.strings.get_sets() == ({0}, {1})
    np.testing.assert_allclose(state.to_dense(), np.array([0.0, 1.0, 2.0, 0.0]))


def test_real_dense_big_endian():
    state = RealTermSum.from_dense(2, [0.0, 1.0, 0.0, 0.0], big_endian=True)

    assert state.strings.get_sets() == ({1},)
    np.testing.assert_allclose(state.to_dense(big_endian=True), np.array([0.0, 1.0, 0.0, 0.0]))


def test_complex_dense_round_trip():
    state = ComplexTermSum.from_dense(2, [0.0, 1.0 + 2.0j, 0.0, -1.0j])

    assert state.strings.get_sets() == ({0}, {0, 1})
    np.testing.assert_allclose(state.to_dense(), np.array([0.0, 1.0 + 2.0j, 0.0, -1.0j]))
    np.testing.assert_allclose(state.real_part.to_dense(), np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(state.imag_part.to_dense(), np.array([0.0, 2.0, 0.0, -1.0]))


def test_vdot():
    lhs = RealTermSum.from_str("(2.0, [1, 0]), (3.0, [0, 1])", 2)
    rhs = RealTermSum.from_str("(-1.0, [1, 0]), (4.0, [0, 1])", 2)

    assert lhs.vdot(rhs) == 10.0


def test_vdot_rejects_different_modes():
    lhs = RealTermSum.from_str("[1, 0]", 2)
    rhs = RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        lhs.vdot(rhs)


def test_complex_vdot():
    lhs = ComplexTermSum.from_str("((1+1j), [1, 0]), (2j, [0, 1])", 2)
    rhs = ComplexTermSum.from_str("(3j, [1, 0]), (1, [0, 1])", 2)

    assert lhs.vdot(rhs) == 3.0 + 1.0j


def test_complex_vdot_rejects_different_modes():
    lhs = ComplexTermSum.from_str("[1, 0]", 2)
    rhs = ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        lhs.vdot(rhs)


def test_real_operator_apply():
    op = RealOperator.from_str("F1^ F0", 2)
    state = RealTermSum.from_str("[1, 0]", 2)

    out = op.apply(state)

    assert isinstance(out, RealTermSum)
    assert str(out) == "(1.0, [0, 1])"


def test_real_operator_mat_elem():
    op = RealOperator.from_str("F1^ F0", 2)
    bra = RealTermSum.from_str("[0, 1]", 2)
    ket = RealTermSum.from_str("[1, 0]", 2)

    assert op.mat_elem(bra, ket) == 1.0
    assert op.exp_val(ket) == 0.0


def test_real_operator_apply_rejects_different_modes():
    op = RealOperator.from_str("F1^ F0", 2)
    state = RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        op.apply(state)


def test_real_operator_mat_elem_rejects_different_modes():
    op = RealOperator.from_str("F1^ F0", 2)
    bra = RealTermSum.from_str("[0, 1]", 2)
    ket = RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        op.mat_elem(bra, ket)


def test_complex_operator_apply():
    op = ComplexOperator.from_str("(1j, F1^ F0)", 2)
    state = ComplexTermSum.from_str("[1, 0]", 2)

    out = op.apply(state)

    assert isinstance(out, ComplexTermSum)
    assert str(out) == "(1j, [0, 1])"


def test_complex_operator_mat_elem():
    op = ComplexOperator.from_str("(1j, F1^ F0)", 2)
    bra = ComplexTermSum.from_str("[0, 1]", 2)
    ket = ComplexTermSum.from_str("[1, 0]", 2)

    assert op.mat_elem(bra, ket) == 1.0j
    assert op.exp_val(ket) == 0.0j


def test_complex_operator_apply_rejects_different_modes():
    op = ComplexOperator.from_str("(1j, F1^ F0)", 2)
    state = ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        op.apply(state)


def test_complex_operator_mat_elem_rejects_different_modes():
    op = ComplexOperator.from_str("(1j, F1^ F0)", 2)
    bra = ComplexTermSum.from_str("[0, 1]", 2)
    ket = ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different modes"):
        op.mat_elem(bra, ket)


def test_string_scalar_mul_promotes_to_term():
    term = 2.0 * String(2, {0})

    assert isinstance(term, RealTerm)
    assert term.string.get_set() == {0}
    assert term.coeff == 2.0


def test_complex_term_scalar_mul():
    term = ComplexTerm.from_str("(1j, [1, 0])", 2)

    out = term * 2.0

    assert isinstance(out, ComplexTerm)
    assert out.coeff == 2.0j


def test_terms_slice():
    terms = ComplexTerms.from_str("(1j, [1, 0]), (2j, [0, 1])", 2)

    assert len(terms[1:]) == 1
    assert terms[1].coeff == 2.0j
