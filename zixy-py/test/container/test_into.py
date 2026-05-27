from contextlib import nullcontext

import pytest

from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSignCoeffs,
    RealCoeffs,
    SignCoeffs,
)
from zixy.qubit.pauli import (
    ComplexTerms as PauliComplexTerms,
    ComplexTermSet as PauliComplexTermSet,
    ComplexTermSum as PauliComplexTermSum,
    RealTerm as PauliRealTerm,
    RealTerms as PauliRealTerms,
    RealTermSet as PauliRealTermSet,
    RealTermSum as PauliRealTermSum,
    String as PauliString,
    Strings as PauliStrings,
    X,
    Y,
)
from zixy.qubit.state import (
    ComplexTerms as StateComplexTerms,
    ComplexTermSum as StateComplexTermSum,
    RealTerm as StateRealTerm,
    RealTerms as StateRealTerms,
    RealTermSet as StateRealTermSet,
    RealTermSum as StateRealTermSum,
    String as StateString,
    Strings as StateStrings,
)

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
        (PauliString(2, PAULI_XY), PauliStrings, None),
        (PauliStrings.from_iterable((PAULI_XY,), 2), PauliString, None),
        (PauliStrings.from_iterable((PAULI_XY, PAULI_YX), 2), PauliString, ValueError),
        (PauliString(2, PAULI_XY), PauliRealTerm, None),
        (PauliRealTerm(2, (PAULI_XY, 1.0)), PauliString, None),
        (PauliRealTerm(2, (PAULI_XY, 2.0)), PauliString, ValueError),
        (PauliRealTerms.from_iterable(((PAULI_XY, 1.0),), 2), PauliRealTerm, None),
        (
            PauliRealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2),
            PauliRealTerm,
            ValueError,
        ),
        (PauliRealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2), PauliStrings, None),
        (
            PauliRealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 2.0)), 2),
            PauliStrings,
            ValueError,
        ),
        (PauliComplexTerms.from_iterable(((PAULI_XY, 1.0 + 0j),), 2), PauliRealTerms, None),
        (PauliComplexTerms.from_iterable(((PAULI_XY, 1.0 + 1j),), 2), PauliRealTerms, ValueError),
        (StateString(2, STATE_10), StateStrings, None),
        (StateStrings.from_iterable((STATE_10,), 2), StateString, None),
        (StateStrings.from_iterable((STATE_10, STATE_01), 2), StateString, ValueError),
        (StateRealTerm(2, (STATE_10, 1.0)), StateString, None),
        (StateRealTerm(2, (STATE_10, 2.0)), StateString, ValueError),
        (StateRealTerms.from_iterable(((STATE_10, 3.25),), 2), StateRealTerm, None),
        (StateRealTerms.from_iterable(((STATE_10, 1.0),), 2), StateString, None),
        (StateRealTerms.from_iterable(((STATE_10, 2.0),), 2), StateString, ValueError),
        (StateRealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 1.0)), 2), StateStrings, None),
        (
            StateRealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            StateStrings,
            ValueError,
        ),
        (StateComplexTerms.from_iterable(((STATE_10, 1.0 + 0j),), 2), StateRealTerms, None),
        (StateComplexTerms.from_iterable(((STATE_10, 1.0 + 1j),), 2), StateRealTerms, ValueError),
        (
            StateRealTermSet.from_terms(
                StateRealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2)
            ),
            StateRealTermSet,
            None,
        ),
        (
            StateRealTermSet.from_terms(
                StateRealTerms.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2)
            ),
            StateRealTermSum,
            None,
        ),
        (
            StateRealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            StateRealTermSet,
            None,
        ),
        (
            StateRealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            StateRealTermSum,
            None,
        ),
        (
            PauliRealTermSet.from_terms(
                PauliRealTerms.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2)
            ),
            PauliComplexTermSet,
            None,
        ),
        (
            PauliRealTermSet.from_terms(PauliRealTerms.from_iterable(((PAULI_XY, 1.0),), 2)),
            PauliComplexTermSum,
            None,
        ),
        (
            PauliRealTermSum.from_iterable(((PAULI_XY, 1.0), (PAULI_YX, 1.0)), 2),
            PauliComplexTermSum,
            None,
        ),
        (
            StateRealTermSum.from_iterable(((STATE_10, 1.0), (STATE_01, 2.0)), 2),
            StateComplexTermSum,
            None,
        ),
        (
            StateComplexTermSum.from_iterable(((STATE_10, 1.0 + 1j),), 2),
            StateRealTermSet,
            ValueError,
        ),
        (PauliString(2, PAULI_XY), StateString, TypeError),
    ],
)
def test_into_conversions(source, target_type, exc_type):
    ctx = nullcontext() if exc_type is None else pytest.raises(exc_type)
    with ctx:
        dest = source.into(target_type)
        assert isinstance(dest, target_type)
