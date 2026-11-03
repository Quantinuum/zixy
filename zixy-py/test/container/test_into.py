from contextlib import nullcontext

import pytest

import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSignCoeffs,
    RealCoeffs,
    SignCoeffs,
)
from zixy.qubit.operator import X, Y

PAULI_XY = (X, Y)
PAULI_YX = (Y, X)
STATE_10 = (1, 0)
STATE_01 = (0, 1)


@pytest.mark.parametrize(
    "source, target_type, exc_type",
    [
        (RealCoeffs.from_sequence((1.0, -2.5)), ComplexCoeffs, None),
        (RealCoeffs.from_sequence((1.0, -1.0)), SignCoeffs, None),
        (SignCoeffs.from_phases((False, True, False)), ComplexSignCoeffs, None),
        (ComplexSignCoeffs.from_phases((0, 2)), RealCoeffs, None),
        (ComplexSignCoeffs.from_phases((0, 1)), RealCoeffs, ValueError),
        (ComplexCoeffs.from_sequence((1.0 + 0j, -2.0 + 0j)), RealCoeffs, None),
        (ComplexCoeffs.from_sequence((1.0 + 1j,)), RealCoeffs, ValueError),
        (zqo.String(2, PAULI_XY), zqo.Strings, None),
        (zqo.Strings.from_iterable((PAULI_XY,), 2), zqo.String, None),
        (zqo.Strings.from_iterable((PAULI_XY, PAULI_YX), 2), zqo.String, ValueError),
        (zqo.String(2, PAULI_XY), zqo.RealTerm, None),
        (zqo.RealTerm(2, (PAULI_XY, 1.0)), zqo.String, None),
        (zqo.RealTerm(2, (PAULI_XY, 2.0)), zqo.String, ValueError),
        (zqo.RealTerms.from_iterable(((PAULI_XY, 1.0),), 2), zqo.RealTerm, None),
        (
            zqo.RealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2),
            zqo.RealTerm,
            ValueError,
        ),
        (zqo.RealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2), zqo.Strings, None),
        (
            zqo.RealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 2.0)), 2),
            zqo.Strings,
            ValueError,
        ),
        (zqo.ComplexTerms.from_iterable(((PAULI_XY, 1.0 + 0j),), 2), zqo.RealTerms, None),
        (zqo.ComplexTerms.from_iterable(((PAULI_XY, 1.0 + 1j),), 2), zqo.RealTerms, ValueError),
        (zqs.String(2, STATE_10), zqs.Strings, None),
        (zqs.Strings.from_iterable((STATE_10,), 2), zqs.String, None),
        (zqs.Strings.from_iterable((STATE_10, STATE_01), 2), zqs.String, ValueError),
        (zqs.RealTerm(2, (STATE_10, 1.0)), zqs.String, None),
        (zqs.RealTerm(2, (STATE_10, 2.0)), zqs.String, ValueError),
        (zqs.RealTerms.from_iterable(((STATE_10, 3.25),), 2), zqs.RealTerm, None),
        (zqs.RealTerms.from_iterable(((STATE_10, 1.0),), 2), zqs.String, None),
        (zqs.RealTerms.from_iterable(((STATE_10, 2.0),), 2), zqs.String, ValueError),
        (zqs.RealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 1.0)), 2), zqs.Strings, None),
        (
            zqs.RealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            zqs.Strings,
            ValueError,
        ),
        (zqs.ComplexTerms.from_iterable(((STATE_10, 1.0 + 0j),), 2), zqs.RealTerms, None),
        (zqs.ComplexTerms.from_iterable(((STATE_10, 1.0 + 1j),), 2), zqs.RealTerms, ValueError),
        (
            zqs.RealTermSet.from_terms(
                zqs.RealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2)
            ),
            zqs.RealTermSet,
            None,
        ),
        (
            zqs.RealTermSet.from_terms(
                zqs.RealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2)
            ),
            zqs.RealTermSum,
            None,
        ),
        (
            zqs.RealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            zqs.RealTermSet,
            None,
        ),
        (
            zqs.RealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            zqs.RealTermSum,
            None,
        ),
        (
            zqo.RealTermSet.from_terms(
                zqo.RealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2)
            ),
            zqo.ComplexTermSet,
            None,
        ),
        (
            zqo.RealTermSet.from_terms(zqo.RealTerms.from_iterable(((PAULI_XY, 1.0),), 2)),
            zqo.ComplexTermSum,
            None,
        ),
        (
            zqo.RealTermSum.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2),
            zqo.ComplexTermSum,
            None,
        ),
        (
            zqs.RealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            zqs.ComplexTermSum,
            None,
        ),
        (
            zqs.ComplexTermSum.from_iterable(((STATE_10, 1.0 + 1j),), 2),
            zqs.RealTermSet,
            ValueError,
        ),
        (zqo.String(2, PAULI_XY), zqs.String, TypeError),
    ],
)
def test_into_conversions(source, target_type, exc_type):
    ctx = nullcontext() if exc_type is None else pytest.raises(exc_type)
    with ctx:
        dest = source.into(target_type)
        assert isinstance(dest, target_type)
