from zixy.container.coeffs import ComplexSign
from zixy.qubit.pauli import I, String as PauliString, X, Y
from zixy.qubit.state import String, StringSet


def test_imul():
    n_qubit = 6
    op = PauliString(n_qubit, (X, X, X, I, I, I))
    state = String(n_qubit, None)
    assert state.get_tuple() == (0,) * n_qubit
    assert state.imul_get_phase(op) == ComplexSign(0)
    assert state.get_tuple() == (1, 1, 1, 0, 0, 0)
    op = PauliString(n_qubit, (X, X, Y, I, I, I))
    assert state.imul_get_phase(op) == ComplexSign(3)
    assert state.get_tuple() == (0,) * n_qubit


def test_set():
    n_qubit = 6
    s = StringSet(n_qubit)
    assert s.insert(None) == 0
    assert len(s) == 1
    assert str(s) == "[0, 0, 0, 0, 0, 0]"
    assert s.insert((1,) * 6) == 1
    assert len(s) == 2
    assert str(s) == "[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]"
    assert s.insert(None) == 0
    assert len(s) == 2
    assert s.insert((1, 0) * 3) == 2
    assert len(s) == 3
    assert s.insert((1, 0, 0) * 2) == 3
    assert len(s) == 4
    assert (
        str(s) == "[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0]"
    )
    assert s.remove((1,) * 6) == 1
    assert not s.contains((1,) * 6)
    assert s.lookup(None) == 0
    assert s.lookup((1, 0) * 3) == 2
    assert s.lookup((1, 0, 0) * 2) == 1
    assert s.insert((1, 0, 0) * 2) == 1
