import pytest

from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.pauli import ComplexSignTerm, ComplexSignTerms, I, X, Y, Z


def test_term():
    term = ComplexSignTerm(6)
    assert len(term._impl) == 1
    assert len(term._impl._cmpnts) == 1
    assert len(term._impl._coeffs) == 1
    assert str(term.string) == ""
    assert str(term.coeff) == "+1"
    assert str(term) == "(+1, )"
    term.string.set(
        [
            X,
            Y,
        ]
        * 3
    )
    assert str(term) == "(+1, X0 Y1 X2 Y3 X4 Y5)"
    term *= Sign(1)
    assert str(term) == "(-1, X0 Y1 X2 Y3 X4 Y5)"
    term *= ComplexSign(1)
    assert str(term) == "(-i, X0 Y1 X2 Y3 X4 Y5)"
    term *= ComplexSign(1)
    assert str(term) == "(+1, X0 Y1 X2 Y3 X4 Y5)"
    term *= term
    assert term == ComplexSignTerm(6)
    assert str(term) == "(+1, )"

    other = ComplexSignTerm(7)
    with pytest.raises(ValueError) as err:
        term *= other
    assert str(err.value) == "Qubits-based objects are based on different qubit spaces."

    term.string.set((X, Y) * 3)
    other = ComplexSignTerm(6)
    other.string.set((X, Y, Z) * 2)
    term *= other
    assert term.string.get_tuple() == (I, I, Y, Z, Z, X)
    assert term.coeff == ComplexSign(0)
    # out-of-place mul should return a string equal to the original value of term
    assert (term * other).string.get_tuple() == (X, Y) * 3


def test_append_iterable():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (Y, Z, Z, I, X, Z),
    )
    a = ComplexSignTerms(n_qubit)
    a.append_iterable(tuples)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert len(a[:]) == len(a)
    assert len(a[0 : len(a) : 1]) == len(a)
    # view them in reverse order
    assert len(a[len(a) :: -1]) == len(a)
    assert len(a[::-1]) == len(a)
    assert a[::-1][0].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[::-1][1].string.get_tuple() == (X, Y, Z, Z, Y, X)
