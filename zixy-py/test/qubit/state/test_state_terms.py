import pytest

from zixy.container.coeffs import Sign
from zixy.qubit.pauli import I, RealTerm as PauliRealTerm, String as PauliString, X
from zixy.qubit.state import (
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    SignTerm,
    String,
)


def test_term():
    term = SignTerm(6, "")
    assert term.coeff == Sign(False)
    with pytest.raises(TypeError):
        SignTerm(6, None)
    term.string.set((1, 0) * 3)
    assert str(term) == "(+1, [1, 0, 1, 0, 1, 0])"
    term *= -1
    assert str(term) == "(-1, [1, 0, 1, 0, 1, 0])"
    assert str(term * Sign(True)) == "(+1, [1, 0, 1, 0, 1, 0])"

    term = RealTerm.from_str("[1, 0, 1, 0, 1, 0]")
    assert str(term) == "(1.0, [1, 0, 1, 0, 1, 0])"
    assert term.coeff == 1.0


def test_term_real_sum():
    lc = RealTermSum(6)
    assert len(lc._impl._cmpnts._impl.qubits) == 6
    assert len(lc) == 0
    assert str(lc) == ""
    lc += RealTerm(6, ((1, 0, 0, 0, 0, 1), 3.0))
    assert len(lc) == 1
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1])"
    lc += RealTerm(6, ((1, 1, 0, 0, 1, 1), 4.0))
    assert len(lc) == 2
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1]), (4.0, [1, 1, 0, 0, 1, 1])"
    assert lc.l1_norm == 3 + 4
    assert lc.l2_norm_square == 3**2 + 4**2
    assert RealTermSet.from_str(
        str(RealTermSet.from_terms(lc.to_terms()))
    ) == RealTermSet.from_terms(lc.to_terms())
    assert RealTermSum.from_str(str(lc)) == lc


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
    term = RealTerm(3, ((1, 0, 0), 1.0))

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different qubits"):
        term_set._check_term(RealTerm(4, ((1, 0, 0, 0), 1.0)))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(PauliRealTerm.from_cmpnt_coeff(PauliString(3, (X, I, I)), 1.0))


def test_term_set_check_cmpnt():
    term_set = RealTermSet(3)
    string = String(3, (1, 0, 0))

    term_set._check_cmpnt(string)

    with pytest.raises(ValueError, match="different qubits"):
        term_set._check_cmpnt(String(4, (1, 0, 0, 0)))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_cmpnt(PauliString(3, (X, I, I)))


def test_real_term_sum_to_dense_matrix_index_error():
    op = RealTermSum(65)

    with pytest.raises(IndexError):
        op.to_dense()


def test_complex_term_sum_to_dense_matrix_index_error():
    op = ComplexTermSum(65)

    with pytest.raises(IndexError):
        op.to_dense()
