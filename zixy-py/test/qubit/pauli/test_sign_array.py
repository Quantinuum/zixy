import numpy as np
import pytest

from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.pauli import ComplexSignTerm, I, SignTerm, SignTerms, SignTermSet, X, Y, Z


def test_term():
    term = SignTerm(6)
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
    term *= ComplexSign(0)
    assert str(term) == "(-1, X0 Y1 X2 Y3 X4 Y5)"
    term.string.imul_get_phase(term.string)
    assert str(term) == "(-1, )"

    other = SignTerm(7)
    with pytest.raises(ValueError) as err:
        term * other
    assert str(err.value) == "Qubits-based objects are based on different qubit spaces."
    with pytest.raises(ValueError) as err:
        term.string.imul_get_phase(other.string)
    assert str(err.value) == "Qubits-based objects are based on different qubit spaces."

    term.string.set((X, Y) * 3)
    other = SignTerm(6)
    other.string.set((X, Y, Z) * 2)
    assert term.string.imul_get_phase(other.string) == ComplexSign(0)
    assert term.string.get_tuple() == (I, I, Y, Z, Z, X)
    assert type(term * other) is ComplexSignTerm
    # out-of-place mul should return a string equal to the original value of term
    assert (term * other).string.get_tuple() == (X, Y) * 3


def test_sign_term_to_sparse_matrix():
    mat = SignTerm(1, (Y,)).to_sparse_matrix()
    assert np.allclose(mat.toarray(), np.array([[0, -1j], [1j, 0]]))

    mat = SignTerm(2, (Y, X)).to_sparse_matrix()
    assert np.allclose(
        mat.toarray(),
        np.array(
            [
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [1j, 0, 0, 0],
            ]
        ),
    )

    mat = SignTerm(4, (Z, I, X, Y)).to_sparse_matrix()
    mat_i = np.eye(2)
    mat_x = np.array([[0, 1], [1, 0]])
    mat_y = np.array([[0, -1j], [1j, 0]])
    mat_z = np.array([[1, 0], [0, -1]])
    expected = np.kron(np.kron(np.kron(mat_y, mat_x), mat_i), mat_z)
    assert np.allclose(mat.toarray(), expected)

    mat = SignTerm(4, ((Z, I, X, Y), -1)).to_sparse_matrix()
    assert np.allclose(mat.toarray(), -expected)


def test_append_iterable():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (Y, Z, Z, I, X, Z),
    )
    a = SignTerms(n_qubit)
    a.append_iterable(tuples)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert len(a[::1]) == len(a)
    # view them in reverse order
    assert len(a[len(a) :: -1]) == len(a)
    assert len(a[::-1]) == len(a)
    assert a[::-1][0].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[::-1][1].string.get_tuple() == (X, Y, Z, Z, Y, X)


def test_sign_term_set_cmpnts_only():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (Y, Z, Z, I, X, Z),
        (Y, X, X, I, X, Z),
        (X, X, I, X, Z, X),
    )
    a = SignTermSet(n_qubit)
    a.insert_iterable(tuples)
    for i in range(len(tuples)):
        assert a.insert(tuples[i]) == i
    a.remove(tuples[1])
    assert a.lookup(tuples[1]) is None
    assert a.lookup_index(tuples[0]) == 0
    assert a.lookup_index(tuples[2]) == 2
    assert a.lookup_index(tuples[3]) == 1


def test_sign_term_set():
    n_qubit = 6
    tuples = (
        ((X, Y, Z, Z, Y, X), Sign(True)),
        ((Y, Z, Z, I, X, Z), Sign(False)),
        ((Y, X, X, I, X, Z), Sign(False)),
        ((X, X, I, X, Z, X), Sign(True)),
    )
    a = SignTermSet(n_qubit)
    a.insert_iterable(tuples)
    assert len(a._impl._cmpnts) == len(a._impl._coeffs)
    assert len(a._impl._cmpnts) == len(tuples)
    for i in range(len(tuples)):
        assert a.contains(tuples[i][0])
    for i in range(len(tuples)):
        assert a.insert(tuples[i]) == i
    assert len(a._impl._cmpnts) == len(a._impl._coeffs)
    assert (
        str(a)
        == "(-1, X0 Y1 Z2 Z3 Y4 X5), (+1, Y0 Z1 Z2 X4 Z5), (+1, Y0 X1 X2 X4 Z5), (-1, X0 X1 X3 Z4 X5)"
    )
    a.remove(tuples[1][0])
    assert str(a) == "(-1, X0 Y1 Z2 Z3 Y4 X5), (-1, X0 X1 X3 Z4 X5), (+1, Y0 X1 X2 X4 Z5)"
    assert len(a._impl._cmpnts) == len(a._impl._coeffs)
    assert a.lookup_index(tuples[1][0]) is None
    assert a.lookup_index(tuples[0][0]) == 0
    assert a.lookup_index(tuples[2][0]) == 2
    assert a.lookup_index(tuples[3][0]) == 1

    assert a.lookup_coeff(tuples[1][0]) is None
    assert a.lookup_coeff(tuples[0][0]) == tuples[0][1]
    assert a.lookup_coeff(tuples[2][0]) == tuples[2][1]
    assert a.lookup_coeff(tuples[3][0]) == tuples[3][1]

    assert a.lookup(tuples[1][0]) is None
    assert a.lookup(tuples[0][0]) == (0, tuples[0][1])
    assert a.lookup(tuples[2][0]) == (2, tuples[2][1])
    assert a.lookup(tuples[3][0]) == (1, tuples[3][1])

    a[tuples[1][0]] = Sign(True)
    assert len(a) == len(tuples)
    assert (
        str(a)
        == "(-1, X0 Y1 Z2 Z3 Y4 X5), (-1, X0 X1 X3 Z4 X5), (+1, Y0 X1 X2 X4 Z5), (-1, Y0 Z1 Z2 X4 Z5)"
    )
