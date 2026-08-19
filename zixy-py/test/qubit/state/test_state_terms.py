import pytest

from zixy.container.coeffs import Sign
from zixy.qubit.pauli import I, RealTerm as PauliRealTerm, String as PauliString, X
from zixy.qubit.state import (
    ComplexSignTerm,
    ComplexSignTerms,
    ComplexSignTermSet,
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    SignTerm,
    SignTerms,
    SignTermSet,
    String,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSet,
    SymbolicTermSum,
)


@pytest.mark.parametrize(
    ("term_type", "source", "qubits", "expected", "terms_type"),
    (
        (SignTerm, "(1, [1, 0])", 2, "(+1, [1, 0])", None),
        (
            SignTerms,
            "(1, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1]), (+1, [1, 0])",
            None,
        ),
        (
            SignTermSet,
            "(1, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1])",
            SignTerms,
        ),
        (ComplexSignTerm, "(1j, [1, 0])", 2, "(+i, [1, 0])", None),
        (
            ComplexSignTerms,
            "(1j, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+i, [1, 0]), (-1, [0, 1]), (+1, [1, 0])",
            None,
        ),
        (
            ComplexSignTermSet,
            "(1j, [1, 0]), (-1, [0, 1]), (1, [1, 0])",
            2,
            "(+1, [1, 0]), (-1, [0, 1])",
            ComplexSignTerms,
        ),
        (RealTerm, " ( 2.5 , [1, 0] ) ", 2, "(2.5, [1, 0])", None),
        (
            RealTerms,
            " (2.0, [1, 0]) , (-0.5, [0, 1]) , (3.0, [1, 0]) ",
            2,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            None,
        ),
        (
            RealTermSet,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            2,
            "(3.0, [1, 0]), (-0.5, [0, 1])",
            RealTerms,
        ),
        (
            RealTermSum,
            "(2.0, [1, 0]), (-0.5, [0, 1]), (3.0, [1, 0])",
            2,
            "(5.0, [1, 0]), (-0.5, [0, 1])",
            None,
        ),
        (ComplexTerm, "(1j, [1, 0])", 2, "(1j, [1, 0])", None),
        (
            ComplexTerms,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "(1j, [1, 0]), ((2+0j), [0, 1]), ((3+0j), [1, 0])",
            None,
        ),
        (
            ComplexTermSet,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "((3+0j), [1, 0]), ((2+0j), [0, 1])",
            ComplexTerms,
        ),
        (
            ComplexTermSum,
            "((1j), [1, 0]), ((2), [0, 1]), ((3), [1, 0])",
            2,
            "((3+1j), [1, 0]), ((2+0j), [0, 1])",
            None,
        ),
        (RealTerm, "(2, [1, 0, 1, 0])", 4, "(2.0, [1, 0, 1, 0])", None),
        (RealTerms, "\n(1, [1, 0]),\n(2, [0, 1])\n", 2, "(1.0, [1, 0]), (2.0, [0, 1])", None),
        (RealTermSum, "(2, [1, 0]), (-2, [1, 0])", 2, "(0.0, [1, 0])", None),
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
        (SignTerms, "", 2),
        (SignTermSet, "", 2),
        (ComplexSignTerms, "", 2),
        (ComplexSignTermSet, "", 2),
        (RealTerms, "", 2),
        (RealTermSet, "", 2),
        (RealTermSum, "", 2),
        (ComplexTerms, "", 2),
        (ComplexTermSet, "", 2),
        (ComplexTermSum, "", 2),
    ),
)
def test_from_str_empty_containers(term_type, source, qubits):
    assert str(term_type.from_str(source, qubits)) == ""


@pytest.mark.parametrize(
    ("term_type", "source"),
    (
        (RealTerm, "(2.5, [1, 0])"),
        (RealTerms, "(2.0, [1, 0]), (-0.5, [0, 1])"),
        (RealTermSet, "(2.0, [1, 0]), (-0.5, [0, 1])"),
        (RealTermSum, "(2.0, [1, 0]), (-0.5, [0, 1])"),
    ),
)
def test_from_str_round_trip(term_type, source):
    parsed = term_type.from_str(source, 2)
    assert term_type.from_str(str(parsed), 2) == parsed


@pytest.mark.parametrize(
    "term_type",
    (SymbolicTerm, SymbolicTerms, SymbolicTermSet, SymbolicTermSum),
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
        RealTerm.from_str(source, 2)

    with pytest.raises(IndexError):
        RealTerm.from_str("(1, [1, 0, 1])", 2)

    with pytest.raises(ValueError):
        SignTerm.from_str("(2, [1, 0])", 2)

    with pytest.raises(ValueError):
        ComplexSignTerm.from_str("(2, [1, 0])", 2)

    with pytest.raises(ValueError):
        RealTerm.from_str("(1j, [1, 0])", 2)


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


def test_vdot_rejects_different_qubits():
    lhs = RealTermSum.from_str("[1, 0]", 2)
    rhs = RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        lhs.vdot(rhs)


def test_complex_vdot_rejects_different_qubits():
    lhs = ComplexTermSum.from_str("[1, 0]", 2)
    rhs = ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        lhs.vdot(rhs)


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
