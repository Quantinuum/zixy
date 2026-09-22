"""Bulk numeric copying must preserve ownership, order and lookup maps."""

import pytest

from zixy.qubit.pauli import ComplexTermSet, ComplexTermSum, RealTermSet, RealTermSum


@pytest.mark.parametrize("cls", (RealTermSet, RealTermSum, ComplexTermSet, ComplexTermSum))
@pytest.mark.parametrize("values", ((), (0, 0, 0), (1, 2, 3), (0, 1e-20, 0)))
@pytest.mark.parametrize("nonzero", (False, True))
def test_copy_terms(cls, values, nonzero):
    source = cls.from_iterable([(f"X{i}", c) for i, c in enumerate(values)], 4)
    before = [(str(t.cmpnt), t.coeff) for t in source]

    copied = source.filter_nonzero() if nonzero else source.clone()

    expected = [(key, c) for key, c in before if not nonzero or c != 0]
    assert type(copied) is cls
    assert [(str(t.cmpnt), t.coeff) for t in copied] == expected
    for key, c in expected:
        assert copied.lookup_coeff(key) == c
    if expected:
        copied._data.coeffs[0] = 7
        assert source.lookup_coeff(expected[0][0]) == expected[0][1]
        copied.remove(expected[0][0])
        assert not copied.contains(expected[0][0])
    copied.insert(("Z3", 2))
    assert copied.lookup_coeff("Z3") == 2
    assert not source.contains("Z3")
    assert [(str(t.cmpnt), t.coeff) for t in source] == before


def test_filter_complex_imaginary_and_nonfinite():
    source = ComplexTermSum.from_iterable([("X0", 1e-20j), ("Y0", 1j), ("Z0", 0j)], 2)
    source._data.coeffs[1] = complex(float("inf"))
    result = source.filter_nonzero()
    assert result.lookup_coeff("X0") == 1e-20j
    assert result.lookup_coeff("Y0") == complex(float("inf"))
    assert not result.contains("Z0")
