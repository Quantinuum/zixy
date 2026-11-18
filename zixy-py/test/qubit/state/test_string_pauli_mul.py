import pytest

import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import ComplexSign
from zixy.qubit.operator import I, X, Y


def test_imul():
    n_qubit = 6
    op = zqo.String(n_qubit, (X, X, X, I, I, I))
    state = zqs.String(n_qubit, "")
    assert state.get_tuple() == (0,) * n_qubit
    with pytest.raises(TypeError):
        zqs.String(n_qubit, None)
    assert state.imul_get_phase(op) == ComplexSign(0)
    assert state.get_tuple() == (1, 1, 1, 0, 0, 0)
    op = zqo.String(n_qubit, (X, X, Y, I, I, I))
    assert state.imul_get_phase(op) == ComplexSign(3)
    assert state.get_tuple() == (0,) * n_qubit


def test_set():
    n_qubit = 6
    s = zqs.StringSet(n_qubit)
    assert s.insert("") == 0
    assert len(s) == 1
    assert str(s) == "[0, 0, 0, 0, 0, 0]"
    assert s.insert((1,) * 6) == 1
    assert len(s) == 2
    assert str(s) == "[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]"
    assert s.insert("") == 0
    assert len(s) == 2
    assert s.insert((1, 0) * 3) == 2
    assert len(s) == 3
    assert s.insert((1, 0, 0) * 2) == 3
    assert len(s) == 4
    assert (
        str(s) == "[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0]"
    )
    assert s.remove((1,) * 6) == 1
    assert (1,) * 6 not in s
    assert s.lookup("") == 0
    with pytest.raises(TypeError):
        s.insert(None)
    with pytest.raises(TypeError):
        s.lookup(None)
    assert s.lookup((1, 0) * 3) == 2
    assert s.lookup((1, 0, 0) * 2) == 1
    assert s.insert((1, 0, 0) * 2) == 1
