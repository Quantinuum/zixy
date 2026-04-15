import numpy as np
import pytest

from zixy.container.coeffs import ComplexSign
from zixy.qubit.pauli import I, String, Strings, StringSet, X, Y, Z


def test_array_sizing():
    a = Strings(0)
    assert len(a.qubits) == 0
    a = Strings(20)
    assert len(a.qubits) == 20
    # empty array
    assert len(a) == 0
    # append a trivial string
    a.append()
    assert len(a) == 1
    # and another
    a.append()
    assert len(a) == 2
    # append 8 trivial strings at once
    a.append_n(8)
    assert len(a) == 10
    a.resize(5)
    assert len(a) == 5


def test_string_access():
    n_qubit = 6
    a = Strings(n_qubit)
    valid_content = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X),
        (),
        (I,),
        {},
        {0: Y},
        {1: X, 3: Y, 5: Z},
    )
    for source in valid_content:
        a.append(source)
    assert len(a) == len(valid_content)

    padded = lambda tup: tup + (I,) * (n_qubit - len(tup))

    for i, source in enumerate(valid_content):
        assert type(a[i]) is String
        assert type(a[i]) is String
        if isinstance(source, tuple):
            # viewing access
            assert a[i].get_tuple() == padded(source)
            # copying access
            assert a[i].get_tuple() == padded(source)
        elif isinstance(source, dict):
            # viewing access
            assert a[i].get_dict() == source
            # copying access
            assert a[i].get_dict() == source
        else:
            raise RuntimeError("unreachable")

    tuples = a.get_tuples()
    s = slice(1, 6, 2)
    assert type(a[s]) is Strings, type(a[s])
    assert len(a[s]) == 3
    # getting all tuples from a sliced view should be the same as slicing
    # the sequence of all tuples from the full view.
    assert a[s].clone().get_tuples() == tuples[s]


def test_string_modification():
    n_qubit = 6
    a = String(n_qubit, (X, Y, Z, Z, Y, X))
    a = String(n_qubit, (X, Y, Z, Z, Y, X))
    a[4] = I
    assert a == String(n_qubit, (X, Y, Z, Z, I, X))
    a[2] = I
    a[-1] = Y
    assert tuple(a[i] for i in range(n_qubit)) == (X, Y, I, Z, I, Y)


def test_string_from_str():
    with pytest.raises(ValueError) as err:
        String.from_str("X0 Y2, Y3, X2 Y3, X3, Z0 Z1 Z2 Z3", 4)
    assert str(err.value) == "There should be exactly one Pauli string in the input, not 5."
    a = String.from_str("X0 Z1 Y2 Z3", 4)
    assert a.get_tuple() == (X, Z, Y, Z)


def test_string_to_sparse_matrix():
    mat = String(1, (Y,)).to_sparse_matrix()
    assert np.allclose(mat.toarray(), np.array([[0, -1j], [1j, 0]]))

    mat = String(2, (Y, X)).to_sparse_matrix()
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

    mat = String(4, (Z, I, X, Y)).to_sparse_matrix()
    mat_i = np.eye(2)
    mat_x = np.array([[0, 1], [1, 0]])
    mat_y = np.array([[0, -1j], [1j, 0]])
    mat_z = np.array([[1, 0], [0, -1]])
    expected = np.kron(np.kron(np.kron(mat_y, mat_x), mat_i), mat_z)
    assert np.allclose(mat.toarray(), expected)


def test_string_array_from_str():
    a = Strings.from_str("X0 Y2, Y3, X2 Y3, X3, Z0 Z1 Z2 Z3", 4)
    assert len(a) == 5
    with pytest.raises(ValueError) as err:
        Strings.from_str("X0 Y2, oops! Y3, X2 Y3, Z0 Z1 Z2 Z3", 4)
    assert str(err.value) == 'Bad parse: "oops!" is not a valid Pauli matrix in a sparse string.'
    with pytest.raises(IndexError) as err:
        Strings.from_str("X0 Y2, Y3, X2 Y3, Z0 Z1 Z2 Z6", 4)
    assert (
        str(err.value)
        == "Mode index 6 is out-of-bounds for component list with 4 modes per component."
    )


def test_array_modification():
    n_qubit = 6
    a = Strings(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    a.append_iterable(tuples)
    assert str(a) == "X0 Y1 Z2 Z3 Y4 X5, X0 Z1 Y2 X3, X0 Y1 Z2 Z3 X5, Y0 Z1 Z2 X4 Z5"
    # swap elements with inds 1 and 3
    a[1], a[3] = a[3].clone(), a[1].clone()
    tuples_swapped = tuple(tuples[i] for i in (0, 3, 2, 1))
    assert a.get_tuples() == tuples_swapped


def test_append_n():
    a = Strings(6)
    a.append_n(5, [I, X, Y, Z])
    assert str(a) == ", ".join(("X1 Y2 Z3",) * 5)


def test_errors():
    a = Strings(4)
    with pytest.raises(IndexError):
        a[0]


def test_pauli_counts():
    n_qubit = 6
    a = Strings(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (Y, Z, Y, X, I, I),
        (Y, Y, Y, Y, X, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
        (Z, I, Z, Z, Z, I),
    )
    a.append_iterable(tuples)
    for string, tup in zip(a, tuples, strict=False):
        for p in (I, X, Y, Z):
            assert string.count(p) == tup.count(p)
    # last one represents a diagonal matrix
    assert a[-1].is_diagonal()


def test_out_of_place_products():
    n_qubit = 6
    a = Strings(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    a.append_iterable(tuples)
    compat_mat = a.compatibility_matrix()
    assert str(a[0]) == "X0 Y1 Z2 Z3 Y4 X5"
    assert str(a[1]) == "X0 Z1 Y2 X3"
    assert (a[0] * a[1]).string.get_tuple() == (I, X, X, Y, Y, X)
    for i in range(len(tuples)):
        assert (a[i] * a[i]).string.is_identity()
        assert (a[i] * a[i]).coeff == ComplexSign(0)
        assert (a[i] * a[i]).string.is_identity()
        for j in range(len(tuples)):
            assert a[i].commutes_with(a[j]) == compat_mat[i, j]
            assert a[i].commutes_with(a[j]) == compat_mat[i, j]


def test_in_place_products():
    n_qubit = 6
    a = Strings(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (Y, Z, Z, I, X, X),
    )
    a.append_iterable(tuples)
    assert len(a) == len(tuples)
    assert a[0].get_tuple() == tuples[0]
    assert a[0].imul_get_phase(a[1]) == ComplexSign(1)
    assert a[0].get_tuple() == (Z, X, I, Z, Z, I)


def test_cmpnt_copy():
    n_qubit = 6
    a = Strings(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    a.append_iterable(tuples)
    assert len(a) == len(tuples)
    assert a.clone() is not a
    assert a.clone() == a


def test_mapped_insert():
    n_qubit = 6
    a = StringSet(n_qubit)
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    for i in range(len(tuples)):
        string = tuples[i]
        assert a.lookup(string) is None
        assert a.insert(string) == i
        assert a.lookup(string) == i
    for i in range(len(tuples)):
        string = tuples[i]
        assert a.lookup(string) == i
        assert a.insert(string) == i
        assert a.lookup(string) == i


def test_mapped_equal():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    a = Strings.from_iterable(tuples, n_qubit)
    b = Strings.from_iterable(tuples[::-1], n_qubit)
    assert a != b

    assert StringSet.from_iterable(a, n_qubit) == StringSet.from_iterable(b, n_qubit)
    assert StringSet.from_iterable(b, n_qubit) == StringSet.from_iterable(a, n_qubit)


def test_mapped_remove():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (X, Z, Y, X, I, I),
        (X, Y, Z, Z, I, X),
        (Y, Z, Z, I, X, Z),
    )
    a = StringSet.from_iterable(tuples, n_qubit)

    with pytest.raises(KeyError):
        a.remove(String(n_qubit))

    assert len(a) == len(tuples)
    # [0, 1, 2, 3]
    assert a.remove(String(n_qubit, tuples[0])) == 0
    assert len(a) == len(tuples) - 1
    # [3, 1, 2]
    assert a.remove(String(n_qubit, tuples[3])) == 0
    assert len(a) == len(tuples) - 2
    # [2, 1]
    assert a.remove(String(n_qubit, tuples[1])) == 1
    assert len(a) == len(tuples) - 3


def test_string_set():
    n_qubit = 4
    s = StringSet(n_qubit)
    assert len(s) == 0

    assert s.insert((X, Y, Z, I)) == 0
    assert len(s) == 1
    assert s.insert((X, X, Z, I)) == 1
    assert len(s) == 2
    assert s.insert((X, Y, Z, I)) == 0
    assert len(s) == 2
    assert s.insert((X, X, X, I)) == 2
    assert len(s) == 3
    assert s.insert((X, X, X, Y)) == 3
    assert len(s) == 4

    assert s.lookup((X, X, X, I)) == 2
    assert s.lookup((X, X, X, Z)) is None
    with pytest.raises(KeyError) as err:
        s.remove((X, X, X, Z))
    assert str(err.value) == "'Value X0 X1 X2 Z3 not found in StringSet.'"

    assert len(Strings.from_iterable(s, n_qubit)._impl) == len(s._impl)
    assert s.to_cmpnts() == Strings.from_iterable(s, n_qubit)
    assert Strings.from_iterable(s, n_qubit)._impl == s._impl
    strings = Strings.from_iterable(s, n_qubit)
    assert StringSet.from_cmpnts(strings).to_cmpnts() == strings


def test_string_set_from_iterable():
    s = StringSet.from_iterable(((X,), (Y,), (X,), (I,), (Z,), (Y,), (X,)), 1)
    assert len(s) == 4
