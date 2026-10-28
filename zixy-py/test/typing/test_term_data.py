from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pytest
from sympy import Expr

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import (
    ComplexSign,
    RealCoeffs,
    Sign,
    SignCoeffs,
)
from zixy.container.data import TermData, _TermData

if TYPE_CHECKING:
    from zixy._zixy import ImplArray


def _first_coeff[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    data: _TermData[ImplT, SpecT, C],
) -> C:
    return data.coeffs[0]


def test_zqo_concrete_term_data() -> None:
    strings = zqo.Strings.from_iterable(["X0", ""], 2)
    coeffs = SignCoeffs.from_phases([False, True])
    data = TermData(strings, coeffs)
    assert_type(data, TermData[zqo.Strings, SignCoeffs])
    assert_type(data.strings, zqo.Strings)
    assert_type(data.strings[0], zqo.String)
    assert_type(data.coeffs, SignCoeffs)
    assert_type(data.coeffs[0], Sign)
    assert_type(data.strings_type, type[zqo.Strings])
    assert_type(data.string_type, type[zqo.String])
    assert_type(data.coeffs_type, type[SignCoeffs])
    assert_type(data.coeff_type, type[Sign])
    assert_type(data.coeffs_type.from_phases([False]), SignCoeffs)
    assert_type(_first_coeff(data), Sign)

    cloned = data.clone()
    reversed_data = data.clone(slice(None, None, -1))
    last = data.clone(-1)
    empty = data._empty_clone()
    assert_type(cloned, TermData[zqo.Strings, SignCoeffs])
    assert_type(reversed_data, TermData[zqo.Strings, SignCoeffs])
    assert_type(last, TermData[zqo.Strings, SignCoeffs])
    assert_type(empty, TermData[zqo.Strings, SignCoeffs])
    assert type(cloned.strings) is zqo.Strings
    assert type(cloned.coeffs) is SignCoeffs
    assert cloned == data
    assert str(reversed_data.strings) == str(strings[::-1])
    assert list(reversed_data.coeffs) == list(coeffs[::-1])
    assert last.strings[0] == strings[-1]
    assert last.coeffs[0] == coeffs[-1]
    assert len(empty) == 0
    empty.resize(2)
    assert len(empty.strings) == len(empty.coeffs) == 2

    # Views share both containers, while clones own independent data.
    view = TermData(strings[:1], coeffs[:1])
    view.strings[0].clear()
    view.coeffs[0] = Sign(True)
    assert strings[0] == zqo.String(2)
    assert coeffs[0] == Sign(True)
    assert cloned.strings[0] == zqo.String(2, "X0")
    assert cloned.coeffs[0] == Sign(False)

    class DerivedData(TermData[zqo.Strings, SignCoeffs]):
        pass

    derived = DerivedData(strings, coeffs)
    assert_type(derived.clone(), DerivedData)
    assert_type(derived._empty_clone(), DerivedData)
    assert type(derived.clone()) is DerivedData

    class DerivedCoeffs(SignCoeffs):
        pass

    specialised = TermData(strings, DerivedCoeffs.from_size(2))
    assert_type(specialised.clone(), TermData[zqo.Strings, DerivedCoeffs])
    assert type(specialised.clone().coeffs) is DerivedCoeffs

    # Existing term algorithms accept the projection without erasing scalar checks.
    real_data = TermData(strings, RealCoeffs.from_size(2))
    assert_type(real_data.coeffs[0], float)
    assert_type(real_data.coeff_type, type[float])
    assert_type(_first_coeff(real_data), float)
    term = zqo.RealTerm._create(real_data, 0)
    terms = zqo.RealTerms._create(real_data)
    assert_type(term, zqo.RealTerm)
    assert_type(terms, zqo.RealTerms)
    term.coeff = 3.0
    assert real_data.coeffs[0] == terms[0].coeff == 3.0
    with pytest.raises(TypeError):
        zqo.RealTerm._create(data, 0)  # type: ignore[arg-type]

    if TYPE_CHECKING:
        data.strings = strings  # type: ignore[misc]
        data.coeffs = coeffs  # type: ignore[misc]


def test_zqs_concrete_term_data() -> None:
    strings = zqs.Strings.from_iterable(["[1, 0]", ""], 2)
    coeffs = SignCoeffs.from_phases([False, True])
    data = TermData(strings, coeffs)
    assert_type(data, TermData[zqs.Strings, SignCoeffs])
    assert_type(data.strings, zqs.Strings)
    assert_type(data.strings[0], zqs.String)
    assert_type(data.coeffs, SignCoeffs)
    assert_type(data.coeffs[0], Sign)
    assert_type(data.strings_type, type[zqs.Strings])
    assert_type(data.string_type, type[zqs.String])
    assert_type(data.coeffs_type, type[SignCoeffs])
    assert_type(data.coeff_type, type[Sign])
    assert_type(data.coeffs_type.from_phases([False]), SignCoeffs)
    assert_type(_first_coeff(data), Sign)

    cloned = data.clone()
    reversed_data = data.clone(slice(None, None, -1))
    last = data.clone(-1)
    empty = data._empty_clone()
    assert_type(cloned, TermData[zqs.Strings, SignCoeffs])
    assert_type(reversed_data, TermData[zqs.Strings, SignCoeffs])
    assert_type(last, TermData[zqs.Strings, SignCoeffs])
    assert_type(empty, TermData[zqs.Strings, SignCoeffs])
    assert type(cloned.strings) is zqs.Strings
    assert type(cloned.coeffs) is SignCoeffs
    assert cloned == data
    assert str(reversed_data.strings) == str(strings[::-1])
    assert list(reversed_data.coeffs) == list(coeffs[::-1])
    assert last.strings[0] == strings[-1]
    assert last.coeffs[0] == coeffs[-1]
    assert len(empty) == 0
    empty.resize(2)
    assert len(empty.strings) == len(empty.coeffs) == 2

    # Views share both containers, while clones own independent data.
    view = TermData(strings[:1], coeffs[:1])
    view.strings[0].clear()
    view.coeffs[0] = Sign(True)
    assert strings[0] == zqs.String(2)
    assert coeffs[0] == Sign(True)
    assert cloned.strings[0] == zqs.String(2, "[1, 0]")
    assert cloned.coeffs[0] == Sign(False)

    class DerivedData(TermData[zqs.Strings, SignCoeffs]):
        pass

    derived = DerivedData(strings, coeffs)
    assert_type(derived.clone(), DerivedData)
    assert_type(derived._empty_clone(), DerivedData)
    assert type(derived.clone()) is DerivedData

    class DerivedCoeffs(SignCoeffs):
        pass

    specialised = TermData(strings, DerivedCoeffs.from_size(2))
    assert_type(specialised.clone(), TermData[zqs.Strings, DerivedCoeffs])
    assert type(specialised.clone().coeffs) is DerivedCoeffs

    # Existing term algorithms accept the projection without erasing scalar checks.
    real_data = TermData(strings, RealCoeffs.from_size(2))
    assert_type(real_data.coeffs[0], float)
    assert_type(real_data.coeff_type, type[float])
    assert_type(_first_coeff(real_data), float)
    term = zqs.RealTerm._create(real_data, 0)
    terms = zqs.RealTerms._create(real_data)
    assert_type(term, zqs.RealTerm)
    assert_type(terms, zqs.RealTerms)
    term.coeff = 3.0
    assert real_data.coeffs[0] == terms[0].coeff == 3.0
    with pytest.raises(TypeError):
        zqs.RealTerm._create(data, 0)  # type: ignore[arg-type]

    if TYPE_CHECKING:
        data.strings = strings  # type: ignore[misc]
        data.coeffs = coeffs  # type: ignore[misc]


def test_zfog_concrete_term_data() -> None:
    strings = zfog.Strings.from_iterable(["F0^", ""], 2)
    coeffs = SignCoeffs.from_phases([False, True])
    data = TermData(strings, coeffs)
    assert_type(data, TermData[zfog.Strings, SignCoeffs])
    assert_type(data.strings, zfog.Strings)
    assert_type(data.strings[0], zfog.String)
    assert_type(data.coeffs, SignCoeffs)
    assert_type(data.coeffs[0], Sign)
    assert_type(data.strings_type, type[zfog.Strings])
    assert_type(data.string_type, type[zfog.String])
    assert_type(data.coeffs_type, type[SignCoeffs])
    assert_type(data.coeff_type, type[Sign])
    assert_type(data.coeffs_type.from_phases([False]), SignCoeffs)
    assert_type(_first_coeff(data), Sign)

    cloned = data.clone()
    reversed_data = data.clone(slice(None, None, -1))
    last = data.clone(-1)
    empty = data._empty_clone()
    assert_type(cloned, TermData[zfog.Strings, SignCoeffs])
    assert_type(reversed_data, TermData[zfog.Strings, SignCoeffs])
    assert_type(last, TermData[zfog.Strings, SignCoeffs])
    assert_type(empty, TermData[zfog.Strings, SignCoeffs])
    assert type(cloned.strings) is zfog.Strings
    assert type(cloned.coeffs) is SignCoeffs
    assert cloned == data
    assert str(reversed_data.strings) == str(strings[::-1])
    assert list(reversed_data.coeffs) == list(coeffs[::-1])
    assert last.strings[0] == strings[-1]
    assert last.coeffs[0] == coeffs[-1]
    assert len(empty) == 0
    empty.resize(2)
    assert len(empty.strings) == len(empty.coeffs) == 2

    # Views share both containers, while clones own independent data.
    view = TermData(strings[:1], coeffs[:1])
    view.strings[0].clear()
    view.coeffs[0] = Sign(True)
    assert strings[0] == zfog.String(2)
    assert coeffs[0] == Sign(True)
    assert cloned.strings[0] == zfog.String(2, "F0^")
    assert cloned.coeffs[0] == Sign(False)

    class DerivedData(TermData[zfog.Strings, SignCoeffs]):
        pass

    derived = DerivedData(strings, coeffs)
    assert_type(derived.clone(), DerivedData)
    assert_type(derived._empty_clone(), DerivedData)
    assert type(derived.clone()) is DerivedData

    class DerivedCoeffs(SignCoeffs):
        pass

    specialised = TermData(strings, DerivedCoeffs.from_size(2))
    assert_type(specialised.clone(), TermData[zfog.Strings, DerivedCoeffs])
    assert type(specialised.clone().coeffs) is DerivedCoeffs

    # Existing term algorithms accept the projection without erasing scalar checks.
    real_data = TermData(strings, RealCoeffs.from_size(2))
    assert_type(real_data.coeffs[0], float)
    assert_type(real_data.coeff_type, type[float])
    assert_type(_first_coeff(real_data), float)
    term = zfog.RealTerm._create(real_data, 0)
    terms = zfog.RealTerms._create(real_data)
    assert_type(term, zfog.RealTerm)
    assert_type(terms, zfog.RealTerms)
    term.coeff = 3.0
    assert real_data.coeffs[0] == terms[0].coeff == 3.0
    with pytest.raises(TypeError):
        zfog.RealTerm._create(data, 0)  # type: ignore[arg-type]

    if TYPE_CHECKING:
        data.strings = strings  # type: ignore[misc]
        data.coeffs = coeffs  # type: ignore[misc]


def test_zfon_concrete_term_data() -> None:
    strings = zfon.Strings.from_iterable(["F0^", ""], 2)
    coeffs = SignCoeffs.from_phases([False, True])
    data = TermData(strings, coeffs)
    assert_type(data, TermData[zfon.Strings, SignCoeffs])
    assert_type(data.strings, zfon.Strings)
    assert_type(data.strings[0], zfon.String)
    assert_type(data.coeffs, SignCoeffs)
    assert_type(data.coeffs[0], Sign)
    assert_type(data.strings_type, type[zfon.Strings])
    assert_type(data.string_type, type[zfon.String])
    assert_type(data.coeffs_type, type[SignCoeffs])
    assert_type(data.coeff_type, type[Sign])
    assert_type(data.coeffs_type.from_phases([False]), SignCoeffs)
    assert_type(_first_coeff(data), Sign)

    cloned = data.clone()
    reversed_data = data.clone(slice(None, None, -1))
    last = data.clone(-1)
    empty = data._empty_clone()
    assert_type(cloned, TermData[zfon.Strings, SignCoeffs])
    assert_type(reversed_data, TermData[zfon.Strings, SignCoeffs])
    assert_type(last, TermData[zfon.Strings, SignCoeffs])
    assert_type(empty, TermData[zfon.Strings, SignCoeffs])
    assert type(cloned.strings) is zfon.Strings
    assert type(cloned.coeffs) is SignCoeffs
    assert cloned == data
    assert str(reversed_data.strings) == str(strings[::-1])
    assert list(reversed_data.coeffs) == list(coeffs[::-1])
    assert last.strings[0] == strings[-1]
    assert last.coeffs[0] == coeffs[-1]
    assert len(empty) == 0
    empty.resize(2)
    assert len(empty.strings) == len(empty.coeffs) == 2

    # Views share both containers, while clones own independent data.
    view = TermData(strings[:1], coeffs[:1])
    view.strings[0].clear()
    view.coeffs[0] = Sign(True)
    assert strings[0] == zfon.String(2)
    assert coeffs[0] == Sign(True)
    assert cloned.strings[0] == zfon.String(2, "F0^")
    assert cloned.coeffs[0] == Sign(False)

    class DerivedData(TermData[zfon.Strings, SignCoeffs]):
        pass

    derived = DerivedData(strings, coeffs)
    assert_type(derived.clone(), DerivedData)
    assert_type(derived._empty_clone(), DerivedData)
    assert type(derived.clone()) is DerivedData

    class DerivedCoeffs(SignCoeffs):
        pass

    specialised = TermData(strings, DerivedCoeffs.from_size(2))
    assert_type(specialised.clone(), TermData[zfon.Strings, DerivedCoeffs])
    assert type(specialised.clone().coeffs) is DerivedCoeffs

    # Existing term algorithms accept the projection without erasing scalar checks.
    real_data = TermData(strings, RealCoeffs.from_size(2))
    assert_type(real_data.coeffs[0], float)
    assert_type(real_data.coeff_type, type[float])
    assert_type(_first_coeff(real_data), float)
    term = zfon.RealTerm._create(real_data, 0)
    terms = zfon.RealTerms._create(real_data)
    assert_type(term, zfon.RealTerm)
    assert_type(terms, zfon.RealTerms)
    term.coeff = 3.0
    assert real_data.coeffs[0] == terms[0].coeff == 3.0
    with pytest.raises(TypeError):
        zfon.RealTerm._create(data, 0)  # type: ignore[arg-type]

    if TYPE_CHECKING:
        data.strings = strings  # type: ignore[misc]
        data.coeffs = coeffs  # type: ignore[misc]


def test_zfs_concrete_term_data() -> None:
    strings = zfs.Strings.from_iterable(["[1, 0]", ""], 2)
    coeffs = SignCoeffs.from_phases([False, True])
    data = TermData(strings, coeffs)
    assert_type(data, TermData[zfs.Strings, SignCoeffs])
    assert_type(data.strings, zfs.Strings)
    assert_type(data.strings[0], zfs.String)
    assert_type(data.coeffs, SignCoeffs)
    assert_type(data.coeffs[0], Sign)
    assert_type(data.strings_type, type[zfs.Strings])
    assert_type(data.string_type, type[zfs.String])
    assert_type(data.coeffs_type, type[SignCoeffs])
    assert_type(data.coeff_type, type[Sign])
    assert_type(data.coeffs_type.from_phases([False]), SignCoeffs)
    assert_type(_first_coeff(data), Sign)

    cloned = data.clone()
    reversed_data = data.clone(slice(None, None, -1))
    last = data.clone(-1)
    empty = data._empty_clone()
    assert_type(cloned, TermData[zfs.Strings, SignCoeffs])
    assert_type(reversed_data, TermData[zfs.Strings, SignCoeffs])
    assert_type(last, TermData[zfs.Strings, SignCoeffs])
    assert_type(empty, TermData[zfs.Strings, SignCoeffs])
    assert type(cloned.strings) is zfs.Strings
    assert type(cloned.coeffs) is SignCoeffs
    assert cloned == data
    assert str(reversed_data.strings) == str(strings[::-1])
    assert list(reversed_data.coeffs) == list(coeffs[::-1])
    assert last.strings[0] == strings[-1]
    assert last.coeffs[0] == coeffs[-1]
    assert len(empty) == 0
    empty.resize(2)
    assert len(empty.strings) == len(empty.coeffs) == 2

    # Views share both containers, while clones own independent data.
    view = TermData(strings[:1], coeffs[:1])
    view.strings[0].clear()
    view.coeffs[0] = Sign(True)
    assert strings[0] == zfs.String(2)
    assert coeffs[0] == Sign(True)
    assert cloned.strings[0] == zfs.String(2, "[1, 0]")
    assert cloned.coeffs[0] == Sign(False)

    class DerivedData(TermData[zfs.Strings, SignCoeffs]):
        pass

    derived = DerivedData(strings, coeffs)
    assert_type(derived.clone(), DerivedData)
    assert_type(derived._empty_clone(), DerivedData)
    assert type(derived.clone()) is DerivedData

    class DerivedCoeffs(SignCoeffs):
        pass

    specialised = TermData(strings, DerivedCoeffs.from_size(2))
    assert_type(specialised.clone(), TermData[zfs.Strings, DerivedCoeffs])
    assert type(specialised.clone().coeffs) is DerivedCoeffs

    # Existing term algorithms accept the projection without erasing scalar checks.
    real_data = TermData(strings, RealCoeffs.from_size(2))
    assert_type(real_data.coeffs[0], float)
    assert_type(real_data.coeff_type, type[float])
    assert_type(_first_coeff(real_data), float)
    term = zfs.RealTerm._create(real_data, 0)
    terms = zfs.RealTerms._create(real_data)
    assert_type(term, zfs.RealTerm)
    assert_type(terms, zfs.RealTerms)
    term.coeff = 3.0
    assert real_data.coeffs[0] == terms[0].coeff == 3.0
    with pytest.raises(TypeError):
        zfs.RealTerm._create(data, 0)  # type: ignore[arg-type]

    if TYPE_CHECKING:
        data.strings = strings  # type: ignore[misc]
        data.coeffs = coeffs  # type: ignore[misc]


def test_storage_rejects_foreign_string_family() -> None:
    data = TermData(zqs.Strings(2, 1), RealCoeffs.from_size(1))
    with pytest.raises(TypeError):
        zqo.RealTerm._create(data, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_strings,n_coeffs", [(0, 1), (1, 0), (1, 2), (2, 1)])
def test_term_data_rejects_mismatched_lengths(n_strings: int, n_coeffs: int) -> None:
    strings = zqo.Strings(2, n_strings)
    coeffs = RealCoeffs.from_size(n_coeffs)
    with pytest.raises(ValueError, match="strings and coefficients must have equal lengths"):
        TermData(strings, coeffs)


def test_term_data_checks_view_lengths() -> None:
    strings = zqo.Strings(2, 3)
    coeffs = RealCoeffs.from_size(4)
    data = TermData(strings[::2], coeffs[1::2])
    assert len(data) == 2
    assert len(TermData(strings[:0], coeffs[:0])) == 0
    with pytest.raises(ValueError, match="strings and coefficients must have equal lengths"):
        TermData(strings[::2], coeffs[:1])
