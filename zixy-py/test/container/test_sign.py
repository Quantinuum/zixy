import numpy as np

from zixy.container.coeffs import Sign, SignCoeffs


def test_from_phases():
    phases = (0, 1, 0, 1, 1, 1, 0)
    v = SignCoeffs.from_phases(np.array([bool(i) for i in phases], dtype=bool))
    assert len(v) == 7
    assert str(v) == "[+1, -1, +1, -1, -1, -1, +1]"
    v = SignCoeffs.from_phases(phases)
    assert str(v) == "[+1, -1, +1, -1, -1, -1, +1]"


def test_scalar_conversions():
    assert int(Sign(False)) == 1
    assert int(Sign(True)) == -1


def test_scalar_mul():
    # only Sign * Sign results in Sign
    assert Sign(False) * Sign(False) == Sign(False)
    assert Sign(False) * Sign(True) == Sign(True)
    assert Sign(True) * Sign(False) == Sign(True)
    assert Sign(True) * Sign(True) == Sign(False)
    assert type(Sign(False) * -1) is int
    assert Sign(True) * -1 == 1


def test_element_mul():
    phases = (0, 1, 0, 1, 1, 1, 0)
    v = SignCoeffs.from_phases(phases)
    # mul each by -1
    for i in range(len(v)):
        v[i] *= Sign(True)
    assert v == SignCoeffs.from_phases(tuple((i + 1) % 2 for i in phases))


def test_extend():
    phases_first = (0, 1, 1, 0)
    phases_last = (1, 0, 0, 1, 1)
    v = SignCoeffs.from_phases(phases_first)
    v.extend(v)
    assert v == SignCoeffs.from_phases(phases_first * 2)
    v.extend(SignCoeffs.from_phases(phases_last))
    assert v == SignCoeffs.from_phases(phases_first * 2 + phases_last)


def test_unary_ops():
    assert -Sign(False) == Sign(True)
    assert -Sign(True) == Sign(False)
    assert +Sign(False) == Sign(False)
    assert +Sign(True) == Sign(True)
    v = SignCoeffs.from_phases((0, 1, 0, 1))
    assert -v == SignCoeffs.from_phases((1, 0, 1, 0))
    assert +v == v
