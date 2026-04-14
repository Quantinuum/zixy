import numpy as np

from zixy.container.coeffs import ComplexSign, ComplexSignCoeffs, Sign


def test_from_phases():
    v = ComplexSignCoeffs.from_phases(np.array((2, 3, 0, 1, 2, 1, 3, 0), dtype=np.uint8))
    assert len(v) == 8
    assert str(v) == "[-1, -i, +1, +i, -1, +i, -i, +1]"
    v = ComplexSignCoeffs.from_phases((2, 3, 0, 1, 2, 1, 3, 0))
    assert str(v) == "[-1, -i, +1, +i, -1, +i, -i, +1]"


def test_scalar_conversions():
    for i in range(4):
        assert complex(ComplexSign(i)) == 1j**i
        assert ComplexSign.from_complex(complex(ComplexSign(i))) == ComplexSign(i)


def test_scalar_mul():
    # ComplexSign * ComplexSign results in ComplexSign
    assert ComplexSign(3) * ComplexSign(2) == ComplexSign(1)
    # but so does ComplexSign * Sign
    assert ComplexSign(3) * Sign(True) == ComplexSign(1)
    # and Sign * ComplexSign
    assert Sign(True) * ComplexSign(3) == ComplexSign(1)
    assert type(ComplexSign(3) * -1) is complex
    assert ComplexSign(3) * -1 == 1j
    assert ComplexSign(1) * complex(0, 1) == -1


def test_element_mul():
    phases = (2, 3, 0, 1, 2, 2, 3, 0)
    v = ComplexSignCoeffs.from_phases(phases)
    # mul each by i
    for i in range(len(v)):
        v[i] *= ComplexSign(1)
    assert v == ComplexSignCoeffs.from_phases(tuple((i + 1) % 4 for i in phases))


def test_extend():
    phases_first = (2, 3, 0)
    phases_last = (1, 2, 2, 3, 0)
    v = ComplexSignCoeffs.from_phases(phases_first)
    v.extend(v)
    assert v == ComplexSignCoeffs.from_phases(phases_first * 2)
    v.extend(ComplexSignCoeffs.from_phases(phases_last))
    assert v == ComplexSignCoeffs.from_phases(phases_first * 2 + phases_last)


def test_unary_ops():
    assert -ComplexSign(0) == ComplexSign(2)
    assert -ComplexSign(1) == ComplexSign(3)
    assert -ComplexSign(2) == ComplexSign(0)
    assert -ComplexSign(3) == ComplexSign(1)
    assert +ComplexSign(0) == ComplexSign(0)
    assert +ComplexSign(1) == ComplexSign(1)
    assert +ComplexSign(2) == ComplexSign(2)
    assert +ComplexSign(3) == ComplexSign(3)
    v = ComplexSignCoeffs.from_phases((2, 3, 0, 1))
    assert -v == ComplexSignCoeffs.from_phases((0, 1, 2, 3))
    assert +v == v
