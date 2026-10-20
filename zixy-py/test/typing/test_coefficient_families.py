from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pytest
from sympy import Expr, sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import (
    Coeffs,
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
    convert_vec,
    get_coeffs_type,
)
from zixy.container.data import TermData
from zixy.container.terms import Term as TermBase, Terms as TermsBase, TermSet as TermSetBase

if TYPE_CHECKING:
    from zixy._zixy import ImplArray
    from zixy.container.terms import Terms, TermSet


def _array_coeffs[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    terms: Terms[ImplT, SpecT, C],
) -> Coeffs[C]:
    return terms.coeffs


def _set_coeffs[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    terms: TermSet[ImplT, SpecT, C],
) -> Coeffs[C]:
    return terms.coeffs


def _check_coefficient_views[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    terms: Terms[ImplT, SpecT, C],
    term_set: TermSet[ImplT, SpecT, C],
    expected: type[Coeffs[C]],
    value: C,
) -> None:
    assert type(terms.coeffs) is expected
    assert type(term_set.coeffs) is expected
    assert terms.coeffs_type is expected
    assert term_set.coeffs_type is expected
    assert terms[0].coeffs_type is expected
    terms[::-1].coeffs[0] = value
    assert terms[1].coeff == value
    term_set.coeffs[0] = value
    assert next(iter(term_set)).coeff == value
    assert len(terms[:0].coeffs) == 0
    cloned = terms.clone()
    assert type(cloned.coeffs) is expected
    assert list(cloned.coeffs) == list(terms.coeffs)
    assert type(term_set.clone().coeffs) is expected


def test_zqo_coefficient_relationships() -> None:
    signs = zqo.SignTerms(2, 2)
    signs_set = zqo.SignTermSet.from_terms(signs)
    assert_type(signs.coeffs, SignCoeffs)
    assert_type(signs.coeffs_type, type[SignCoeffs])
    assert_type(signs[0].coeffs_type, type[SignCoeffs])
    assert_type(signs.coeffs[0], Sign)
    assert_type(signs_set.coeffs, SignCoeffs)
    assert_type(signs_set.coeffs_type, type[SignCoeffs])
    assert_type(signs_set.coeffs[0], Sign)
    assert_type(_array_coeffs(signs), Coeffs[Sign])
    assert_type(_set_coeffs(signs_set), Coeffs[Sign])
    _check_coefficient_views(signs, signs_set, SignCoeffs, Sign(True))

    phases = zqo.ComplexSignTerms(2, 2)
    phases_set = zqo.ComplexSignTermSet.from_terms(phases)
    assert_type(phases.coeffs, ComplexSignCoeffs)
    assert_type(phases.coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases[0].coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases.coeffs[0], ComplexSign)
    assert_type(phases_set.coeffs, ComplexSignCoeffs)
    assert_type(phases_set.coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases_set.coeffs[0], ComplexSign)
    assert_type(_array_coeffs(phases), Coeffs[ComplexSign])
    assert_type(_set_coeffs(phases_set), Coeffs[ComplexSign])
    _check_coefficient_views(phases, phases_set, ComplexSignCoeffs, ComplexSign(1))

    reals = zqo.RealTerms(2, 2)
    reals_set = zqo.RealTermSet.from_terms(reals)
    assert_type(reals.coeffs, RealCoeffs)
    assert_type(reals.coeffs_type, type[RealCoeffs])
    assert_type(reals[0].coeffs_type, type[RealCoeffs])
    assert_type(reals.coeffs[0], float)
    assert_type(reals_set.coeffs, RealCoeffs)
    assert_type(reals_set.coeffs_type, type[RealCoeffs])
    assert_type(reals_set.coeffs[0], float)
    assert_type(_array_coeffs(reals), Coeffs[float])
    assert_type(_set_coeffs(reals_set), Coeffs[float])
    _check_coefficient_views(reals, reals_set, RealCoeffs, 2.0)

    complexes = zqo.ComplexTerms(2, 2)
    complexes_set = zqo.ComplexTermSet.from_terms(complexes)
    assert_type(complexes.coeffs, ComplexCoeffs)
    assert_type(complexes.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes[0].coeffs_type, type[ComplexCoeffs])
    assert_type(complexes.coeffs[0], complex)
    assert_type(complexes_set.coeffs, ComplexCoeffs)
    assert_type(complexes_set.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_set.coeffs[0], complex)
    assert_type(_array_coeffs(complexes), Coeffs[complex])
    assert_type(_set_coeffs(complexes_set), Coeffs[complex])
    _check_coefficient_views(complexes, complexes_set, ComplexCoeffs, 2j)

    symbols = zqo.SymbolicTerms(2, 2)
    symbols_set = zqo.SymbolicTermSet.from_terms(symbols)
    assert_type(symbols.coeffs, SymbolicCoeffs)
    assert_type(symbols.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols[0].coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols.coeffs[0], Expr)
    assert_type(symbols_set.coeffs, SymbolicCoeffs)
    assert_type(symbols_set.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_set.coeffs[0], Expr)
    assert_type(_array_coeffs(symbols), Coeffs[Expr])
    assert_type(_set_coeffs(symbols_set), Coeffs[Expr])
    _check_coefficient_views(symbols, symbols_set, SymbolicCoeffs, sympify("x"))

    reals_sum = zqo.RealTermSum.from_terms(reals)
    assert_type(reals_sum.coeffs, RealCoeffs)
    assert_type(reals_sum.coeffs_type, type[RealCoeffs])
    assert_type(reals_sum.coeffs[0], float)
    assert type(reals_sum.coeffs) is RealCoeffs
    reals_sum.coeffs[0] = 2.0
    assert next(iter(reals_sum)).coeff == 2.0

    complexes_sum = zqo.ComplexTermSum.from_terms(complexes)
    assert_type(complexes_sum.coeffs, ComplexCoeffs)
    assert_type(complexes_sum.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_sum.coeffs[0], complex)
    assert type(complexes_sum.coeffs) is ComplexCoeffs
    complexes_sum.coeffs[0] = 2j
    assert next(iter(complexes_sum)).coeff == 2j

    symbols_sum = zqo.SymbolicTermSum.from_terms(symbols)
    assert_type(symbols_sum.coeffs, SymbolicCoeffs)
    assert_type(symbols_sum.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_sum.coeffs[0], Expr)
    assert type(symbols_sum.coeffs) is SymbolicCoeffs
    symbols_sum.coeffs[0] = sympify("x")
    assert next(iter(symbols_sum)).coeff == sympify("x")


def test_zqs_coefficient_relationships() -> None:
    signs = zqs.SignTerms(2, 2)
    signs_set = zqs.SignTermSet.from_terms(signs)
    assert_type(signs.coeffs, SignCoeffs)
    assert_type(signs.coeffs_type, type[SignCoeffs])
    assert_type(signs[0].coeffs_type, type[SignCoeffs])
    assert_type(signs.coeffs[0], Sign)
    assert_type(signs_set.coeffs, SignCoeffs)
    assert_type(signs_set.coeffs_type, type[SignCoeffs])
    assert_type(signs_set.coeffs[0], Sign)
    assert_type(_array_coeffs(signs), Coeffs[Sign])
    assert_type(_set_coeffs(signs_set), Coeffs[Sign])
    _check_coefficient_views(signs, signs_set, SignCoeffs, Sign(True))

    phases = zqs.ComplexSignTerms(2, 2)
    phases_set = zqs.ComplexSignTermSet.from_terms(phases)
    assert_type(phases.coeffs, ComplexSignCoeffs)
    assert_type(phases.coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases[0].coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases.coeffs[0], ComplexSign)
    assert_type(phases_set.coeffs, ComplexSignCoeffs)
    assert_type(phases_set.coeffs_type, type[ComplexSignCoeffs])
    assert_type(phases_set.coeffs[0], ComplexSign)
    assert_type(_array_coeffs(phases), Coeffs[ComplexSign])
    assert_type(_set_coeffs(phases_set), Coeffs[ComplexSign])
    _check_coefficient_views(phases, phases_set, ComplexSignCoeffs, ComplexSign(1))

    reals = zqs.RealTerms(2, 2)
    reals_set = zqs.RealTermSet.from_terms(reals)
    assert_type(reals.coeffs, RealCoeffs)
    assert_type(reals.coeffs_type, type[RealCoeffs])
    assert_type(reals[0].coeffs_type, type[RealCoeffs])
    assert_type(reals.coeffs[0], float)
    assert_type(reals_set.coeffs, RealCoeffs)
    assert_type(reals_set.coeffs_type, type[RealCoeffs])
    assert_type(reals_set.coeffs[0], float)
    assert_type(_array_coeffs(reals), Coeffs[float])
    assert_type(_set_coeffs(reals_set), Coeffs[float])
    _check_coefficient_views(reals, reals_set, RealCoeffs, 2.0)

    complexes = zqs.ComplexTerms(2, 2)
    complexes_set = zqs.ComplexTermSet.from_terms(complexes)
    assert_type(complexes.coeffs, ComplexCoeffs)
    assert_type(complexes.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes[0].coeffs_type, type[ComplexCoeffs])
    assert_type(complexes.coeffs[0], complex)
    assert_type(complexes_set.coeffs, ComplexCoeffs)
    assert_type(complexes_set.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_set.coeffs[0], complex)
    assert_type(_array_coeffs(complexes), Coeffs[complex])
    assert_type(_set_coeffs(complexes_set), Coeffs[complex])
    _check_coefficient_views(complexes, complexes_set, ComplexCoeffs, 2j)

    symbols = zqs.SymbolicTerms(2, 2)
    symbols_set = zqs.SymbolicTermSet.from_terms(symbols)
    assert_type(symbols.coeffs, SymbolicCoeffs)
    assert_type(symbols.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols[0].coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols.coeffs[0], Expr)
    assert_type(symbols_set.coeffs, SymbolicCoeffs)
    assert_type(symbols_set.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_set.coeffs[0], Expr)
    assert_type(_array_coeffs(symbols), Coeffs[Expr])
    assert_type(_set_coeffs(symbols_set), Coeffs[Expr])
    _check_coefficient_views(symbols, symbols_set, SymbolicCoeffs, sympify("x"))

    reals_sum = zqs.RealTermSum.from_terms(reals)
    assert_type(reals_sum.coeffs, RealCoeffs)
    assert_type(reals_sum.coeffs_type, type[RealCoeffs])
    assert_type(reals_sum.coeffs[0], float)
    assert type(reals_sum.coeffs) is RealCoeffs
    reals_sum.coeffs[0] = 2.0
    assert next(iter(reals_sum)).coeff == 2.0

    complexes_sum = zqs.ComplexTermSum.from_terms(complexes)
    assert_type(complexes_sum.coeffs, ComplexCoeffs)
    assert_type(complexes_sum.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_sum.coeffs[0], complex)
    assert type(complexes_sum.coeffs) is ComplexCoeffs
    complexes_sum.coeffs[0] = 2j
    assert next(iter(complexes_sum)).coeff == 2j

    symbols_sum = zqs.SymbolicTermSum.from_terms(symbols)
    assert_type(symbols_sum.coeffs, SymbolicCoeffs)
    assert_type(symbols_sum.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_sum.coeffs[0], Expr)
    assert type(symbols_sum.coeffs) is SymbolicCoeffs
    symbols_sum.coeffs[0] = sympify("x")
    assert next(iter(symbols_sum)).coeff == sympify("x")


def test_zfog_coefficient_relationships() -> None:
    reals = zfog.RealTerms(2, 2)
    reals_set = zfog.RealTermSet.from_terms(reals)
    assert_type(reals.coeffs, RealCoeffs)
    assert_type(reals.coeffs_type, type[RealCoeffs])
    assert_type(reals[0].coeffs_type, type[RealCoeffs])
    assert_type(reals.coeffs[0], float)
    assert_type(reals_set.coeffs, RealCoeffs)
    assert_type(reals_set.coeffs_type, type[RealCoeffs])
    assert_type(reals_set.coeffs[0], float)
    assert_type(_array_coeffs(reals), Coeffs[float])
    assert_type(_set_coeffs(reals_set), Coeffs[float])
    _check_coefficient_views(reals, reals_set, RealCoeffs, 2.0)

    complexes = zfog.ComplexTerms(2, 2)
    complexes_set = zfog.ComplexTermSet.from_terms(complexes)
    assert_type(complexes.coeffs, ComplexCoeffs)
    assert_type(complexes.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes[0].coeffs_type, type[ComplexCoeffs])
    assert_type(complexes.coeffs[0], complex)
    assert_type(complexes_set.coeffs, ComplexCoeffs)
    assert_type(complexes_set.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_set.coeffs[0], complex)
    assert_type(_array_coeffs(complexes), Coeffs[complex])
    assert_type(_set_coeffs(complexes_set), Coeffs[complex])
    _check_coefficient_views(complexes, complexes_set, ComplexCoeffs, 2j)

    symbols = zfog.SymbolicTerms(2, 2)
    symbols_set = zfog.SymbolicTermSet.from_terms(symbols)
    assert_type(symbols.coeffs, SymbolicCoeffs)
    assert_type(symbols.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols[0].coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols.coeffs[0], Expr)
    assert_type(symbols_set.coeffs, SymbolicCoeffs)
    assert_type(symbols_set.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_set.coeffs[0], Expr)
    assert_type(_array_coeffs(symbols), Coeffs[Expr])
    assert_type(_set_coeffs(symbols_set), Coeffs[Expr])
    _check_coefficient_views(symbols, symbols_set, SymbolicCoeffs, sympify("x"))

    reals_sum = zfog.RealTermSum.from_terms(reals)
    assert_type(reals_sum.coeffs, RealCoeffs)
    assert_type(reals_sum.coeffs_type, type[RealCoeffs])
    assert_type(reals_sum.coeffs[0], float)
    assert type(reals_sum.coeffs) is RealCoeffs
    reals_sum.coeffs[0] = 2.0
    assert next(iter(reals_sum)).coeff == 2.0

    complexes_sum = zfog.ComplexTermSum.from_terms(complexes)
    assert_type(complexes_sum.coeffs, ComplexCoeffs)
    assert_type(complexes_sum.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_sum.coeffs[0], complex)
    assert type(complexes_sum.coeffs) is ComplexCoeffs
    complexes_sum.coeffs[0] = 2j
    assert next(iter(complexes_sum)).coeff == 2j

    symbols_sum = zfog.SymbolicTermSum.from_terms(symbols)
    assert_type(symbols_sum.coeffs, SymbolicCoeffs)
    assert_type(symbols_sum.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_sum.coeffs[0], Expr)
    assert type(symbols_sum.coeffs) is SymbolicCoeffs
    symbols_sum.coeffs[0] = sympify("x")
    assert next(iter(symbols_sum)).coeff == sympify("x")


def test_zfon_coefficient_relationships() -> None:
    signs = zfon.SignTerms(2, 2)
    signs_set = zfon.SignTermSet.from_terms(signs)
    assert_type(signs.coeffs, SignCoeffs)
    assert_type(signs.coeffs_type, type[SignCoeffs])
    assert_type(signs[0].coeffs_type, type[SignCoeffs])
    assert_type(signs.coeffs[0], Sign)
    assert_type(signs_set.coeffs, SignCoeffs)
    assert_type(signs_set.coeffs_type, type[SignCoeffs])
    assert_type(signs_set.coeffs[0], Sign)
    assert_type(_array_coeffs(signs), Coeffs[Sign])
    assert_type(_set_coeffs(signs_set), Coeffs[Sign])
    _check_coefficient_views(signs, signs_set, SignCoeffs, Sign(True))

    reals = zfon.RealTerms(2, 2)
    reals_set = zfon.RealTermSet.from_terms(reals)
    assert_type(reals.coeffs, RealCoeffs)
    assert_type(reals.coeffs_type, type[RealCoeffs])
    assert_type(reals[0].coeffs_type, type[RealCoeffs])
    assert_type(reals.coeffs[0], float)
    assert_type(reals_set.coeffs, RealCoeffs)
    assert_type(reals_set.coeffs_type, type[RealCoeffs])
    assert_type(reals_set.coeffs[0], float)
    assert_type(_array_coeffs(reals), Coeffs[float])
    assert_type(_set_coeffs(reals_set), Coeffs[float])
    _check_coefficient_views(reals, reals_set, RealCoeffs, 2.0)

    complexes = zfon.ComplexTerms(2, 2)
    complexes_set = zfon.ComplexTermSet.from_terms(complexes)
    assert_type(complexes.coeffs, ComplexCoeffs)
    assert_type(complexes.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes[0].coeffs_type, type[ComplexCoeffs])
    assert_type(complexes.coeffs[0], complex)
    assert_type(complexes_set.coeffs, ComplexCoeffs)
    assert_type(complexes_set.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_set.coeffs[0], complex)
    assert_type(_array_coeffs(complexes), Coeffs[complex])
    assert_type(_set_coeffs(complexes_set), Coeffs[complex])
    _check_coefficient_views(complexes, complexes_set, ComplexCoeffs, 2j)

    symbols = zfon.SymbolicTerms(2, 2)
    symbols_set = zfon.SymbolicTermSet.from_terms(symbols)
    assert_type(symbols.coeffs, SymbolicCoeffs)
    assert_type(symbols.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols[0].coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols.coeffs[0], Expr)
    assert_type(symbols_set.coeffs, SymbolicCoeffs)
    assert_type(symbols_set.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_set.coeffs[0], Expr)
    assert_type(_array_coeffs(symbols), Coeffs[Expr])
    assert_type(_set_coeffs(symbols_set), Coeffs[Expr])
    _check_coefficient_views(symbols, symbols_set, SymbolicCoeffs, sympify("x"))

    reals_sum = zfon.RealTermSum.from_terms(reals)
    assert_type(reals_sum.coeffs, RealCoeffs)
    assert_type(reals_sum.coeffs_type, type[RealCoeffs])
    assert_type(reals_sum.coeffs[0], float)
    assert type(reals_sum.coeffs) is RealCoeffs
    reals_sum.coeffs[0] = 2.0
    assert next(iter(reals_sum)).coeff == 2.0

    complexes_sum = zfon.ComplexTermSum.from_terms(complexes)
    assert_type(complexes_sum.coeffs, ComplexCoeffs)
    assert_type(complexes_sum.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_sum.coeffs[0], complex)
    assert type(complexes_sum.coeffs) is ComplexCoeffs
    complexes_sum.coeffs[0] = 2j
    assert next(iter(complexes_sum)).coeff == 2j

    symbols_sum = zfon.SymbolicTermSum.from_terms(symbols)
    assert_type(symbols_sum.coeffs, SymbolicCoeffs)
    assert_type(symbols_sum.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_sum.coeffs[0], Expr)
    assert type(symbols_sum.coeffs) is SymbolicCoeffs
    symbols_sum.coeffs[0] = sympify("x")
    assert next(iter(symbols_sum)).coeff == sympify("x")


def test_zfs_coefficient_relationships() -> None:
    signs = zfs.SignTerms(2, 2)
    signs_set = zfs.SignTermSet.from_terms(signs)
    assert_type(signs.coeffs, SignCoeffs)
    assert_type(signs.coeffs_type, type[SignCoeffs])
    assert_type(signs[0].coeffs_type, type[SignCoeffs])
    assert_type(signs.coeffs[0], Sign)
    assert_type(signs_set.coeffs, SignCoeffs)
    assert_type(signs_set.coeffs_type, type[SignCoeffs])
    assert_type(signs_set.coeffs[0], Sign)
    assert_type(_array_coeffs(signs), Coeffs[Sign])
    assert_type(_set_coeffs(signs_set), Coeffs[Sign])
    _check_coefficient_views(signs, signs_set, SignCoeffs, Sign(True))

    reals = zfs.RealTerms(2, 2)
    reals_set = zfs.RealTermSet.from_terms(reals)
    assert_type(reals.coeffs, RealCoeffs)
    assert_type(reals.coeffs_type, type[RealCoeffs])
    assert_type(reals[0].coeffs_type, type[RealCoeffs])
    assert_type(reals.coeffs[0], float)
    assert_type(reals_set.coeffs, RealCoeffs)
    assert_type(reals_set.coeffs_type, type[RealCoeffs])
    assert_type(reals_set.coeffs[0], float)
    assert_type(_array_coeffs(reals), Coeffs[float])
    assert_type(_set_coeffs(reals_set), Coeffs[float])
    _check_coefficient_views(reals, reals_set, RealCoeffs, 2.0)

    complexes = zfs.ComplexTerms(2, 2)
    complexes_set = zfs.ComplexTermSet.from_terms(complexes)
    assert_type(complexes.coeffs, ComplexCoeffs)
    assert_type(complexes.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes[0].coeffs_type, type[ComplexCoeffs])
    assert_type(complexes.coeffs[0], complex)
    assert_type(complexes_set.coeffs, ComplexCoeffs)
    assert_type(complexes_set.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_set.coeffs[0], complex)
    assert_type(_array_coeffs(complexes), Coeffs[complex])
    assert_type(_set_coeffs(complexes_set), Coeffs[complex])
    _check_coefficient_views(complexes, complexes_set, ComplexCoeffs, 2j)

    symbols = zfs.SymbolicTerms(2, 2)
    symbols_set = zfs.SymbolicTermSet.from_terms(symbols)
    assert_type(symbols.coeffs, SymbolicCoeffs)
    assert_type(symbols.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols[0].coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols.coeffs[0], Expr)
    assert_type(symbols_set.coeffs, SymbolicCoeffs)
    assert_type(symbols_set.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_set.coeffs[0], Expr)
    assert_type(_array_coeffs(symbols), Coeffs[Expr])
    assert_type(_set_coeffs(symbols_set), Coeffs[Expr])
    _check_coefficient_views(symbols, symbols_set, SymbolicCoeffs, sympify("x"))

    reals_sum = zfs.RealTermSum.from_terms(reals)
    assert_type(reals_sum.coeffs, RealCoeffs)
    assert_type(reals_sum.coeffs_type, type[RealCoeffs])
    assert_type(reals_sum.coeffs[0], float)
    assert type(reals_sum.coeffs) is RealCoeffs
    reals_sum.coeffs[0] = 2.0
    assert next(iter(reals_sum)).coeff == 2.0

    complexes_sum = zfs.ComplexTermSum.from_terms(complexes)
    assert_type(complexes_sum.coeffs, ComplexCoeffs)
    assert_type(complexes_sum.coeffs_type, type[ComplexCoeffs])
    assert_type(complexes_sum.coeffs[0], complex)
    assert type(complexes_sum.coeffs) is ComplexCoeffs
    complexes_sum.coeffs[0] = 2j
    assert next(iter(complexes_sum)).coeff == 2j

    symbols_sum = zfs.SymbolicTermSum.from_terms(symbols)
    assert_type(symbols_sum.coeffs, SymbolicCoeffs)
    assert_type(symbols_sum.coeffs_type, type[SymbolicCoeffs])
    assert_type(symbols_sum.coeffs[0], Expr)
    assert type(symbols_sum.coeffs) is SymbolicCoeffs
    symbols_sum.coeffs[0] = sympify("x")
    assert next(iter(symbols_sum)).coeff == sympify("x")


def test_coefficient_factories() -> None:
    assert_type(get_coeffs_type(float), type[RealCoeffs])
    assert_type(get_coeffs_type(complex), type[ComplexCoeffs])
    assert_type(get_coeffs_type(Sign), type[SignCoeffs])
    assert_type(get_coeffs_type(ComplexSign), type[ComplexSignCoeffs])
    assert_type(get_coeffs_type(Expr), type[SymbolicCoeffs])
    assert get_coeffs_type(float) is RealCoeffs
    assert get_coeffs_type(complex) is ComplexCoeffs
    assert get_coeffs_type(Sign) is SignCoeffs
    assert get_coeffs_type(ComplexSign) is ComplexSignCoeffs
    assert get_coeffs_type(Expr) is SymbolicCoeffs
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), ComplexCoeffs), ComplexCoeffs)


def test_zqo_custom_coefficient_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class CustomTerm(zqo.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zqo.RealTerms):
        term_type = CustomTerm

    class CustomSet(zqo.RealTermSet):
        terms_type = CustomTerms

    class CustomSum(zqo.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    term = CustomTerm(2)
    term.string.set("X0")
    terms = DerivedTerms(2, 2)
    terms[:] = term
    term_set = CustomSet.from_terms(terms)
    total = CustomSum.from_terms(terms)
    assert_type(term.coeffs_type, type[CustomCoeffs])
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(terms.coeffs_type, type[CustomCoeffs])
    assert_type(term_set.coeffs, CustomCoeffs)
    assert_type(total.coeffs, CustomCoeffs)
    assert_type(terms[::-1].coeffs, CustomCoeffs)
    assert_type(terms.clone().coeffs, CustomCoeffs)
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs), CustomCoeffs)
    assert type(term._data.coeffs) is CustomCoeffs
    assert type(terms.coeffs) is CustomCoeffs
    assert type(term_set.coeffs) is CustomCoeffs
    assert type(total.coeffs) is CustomCoeffs
    assert type(terms[::-1].coeffs) is CustomCoeffs
    assert type(terms.clone().coeffs) is CustomCoeffs
    assert type(term_set.clone().coeffs) is CustomCoeffs
    assert type(total.clone().coeffs) is CustomCoeffs
    assert type(CustomTerm.from_string_coeff(term.string, 2.0)._data.coeffs) is CustomCoeffs
    assert type(CustomTerm.from_str(str(term), 2)._data.coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str("", 2).coeffs) is CustomCoeffs
    assert type(CustomSum.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(zqo.RealTerm(2).into(CustomTerm)._data.coeffs) is CustomCoeffs
    assert type(zqo.RealTerms(2, 1).into(CustomTerms).coeffs) is CustomCoeffs
    assert type(zqo.RealTermSum(2).into(CustomSum).coeffs) is CustomCoeffs
    assert type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs)) is CustomCoeffs
    terms[1].coeff = 3.0
    terms[::-1].coeffs[0] = 4.0
    assert terms[1].coeff == 4.0
    cloned = terms.clone()
    cloned.coeffs[0] = 5.0
    assert terms[0].coeff != 5.0


def test_zqs_custom_coefficient_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class CustomTerm(zqs.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zqs.RealTerms):
        term_type = CustomTerm

    class CustomSet(zqs.RealTermSet):
        terms_type = CustomTerms

    class CustomSum(zqs.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    term = CustomTerm(2)
    term.string.set("[1, 0]")
    terms = DerivedTerms(2, 2)
    terms[:] = term
    term_set = CustomSet.from_terms(terms)
    total = CustomSum.from_terms(terms)
    assert_type(term.coeffs_type, type[CustomCoeffs])
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(terms.coeffs_type, type[CustomCoeffs])
    assert_type(term_set.coeffs, CustomCoeffs)
    assert_type(total.coeffs, CustomCoeffs)
    assert_type(terms[::-1].coeffs, CustomCoeffs)
    assert_type(terms.clone().coeffs, CustomCoeffs)
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs), CustomCoeffs)
    assert type(term._data.coeffs) is CustomCoeffs
    assert type(terms.coeffs) is CustomCoeffs
    assert type(term_set.coeffs) is CustomCoeffs
    assert type(total.coeffs) is CustomCoeffs
    assert type(terms[::-1].coeffs) is CustomCoeffs
    assert type(terms.clone().coeffs) is CustomCoeffs
    assert type(term_set.clone().coeffs) is CustomCoeffs
    assert type(total.clone().coeffs) is CustomCoeffs
    assert type(CustomTerm.from_string_coeff(term.string, 2.0)._data.coeffs) is CustomCoeffs
    assert type(CustomTerm.from_str(str(term), 2)._data.coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str("", 2).coeffs) is CustomCoeffs
    assert type(CustomSum.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(zqs.RealTerm(2).into(CustomTerm)._data.coeffs) is CustomCoeffs
    assert type(zqs.RealTerms(2, 1).into(CustomTerms).coeffs) is CustomCoeffs
    assert type(zqs.RealTermSum(2).into(CustomSum).coeffs) is CustomCoeffs
    assert type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs)) is CustomCoeffs
    terms[1].coeff = 3.0
    terms[::-1].coeffs[0] = 4.0
    assert terms[1].coeff == 4.0
    cloned = terms.clone()
    cloned.coeffs[0] = 5.0
    assert terms[0].coeff != 5.0


def test_zfog_custom_coefficient_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class CustomTerm(zfog.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zfog.RealTerms):
        term_type = CustomTerm

    class CustomSet(zfog.RealTermSet):
        terms_type = CustomTerms

    class CustomSum(zfog.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    term = CustomTerm(2, max_string_len=2)
    term.string.set("F0^ F1")
    terms = DerivedTerms(2, 2, max_string_len=2)
    terms[:] = term
    term_set = CustomSet.from_terms(terms)
    total = CustomSum.from_terms(terms)
    assert_type(term.coeffs_type, type[CustomCoeffs])
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(terms.coeffs_type, type[CustomCoeffs])
    assert_type(term_set.coeffs, CustomCoeffs)
    assert_type(total.coeffs, CustomCoeffs)
    assert_type(terms[::-1].coeffs, CustomCoeffs)
    assert_type(terms.clone().coeffs, CustomCoeffs)
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs), CustomCoeffs)
    assert type(term._data.coeffs) is CustomCoeffs
    assert type(terms.coeffs) is CustomCoeffs
    assert type(term_set.coeffs) is CustomCoeffs
    assert type(total.coeffs) is CustomCoeffs
    assert type(terms[::-1].coeffs) is CustomCoeffs
    assert type(terms.clone().coeffs) is CustomCoeffs
    assert type(term_set.clone().coeffs) is CustomCoeffs
    assert type(total.clone().coeffs) is CustomCoeffs
    assert type(CustomTerm.from_string_coeff(term.string, 2.0)._data.coeffs) is CustomCoeffs
    assert type(CustomTerm.from_str(str(term), 2)._data.coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str("", 2).coeffs) is CustomCoeffs
    assert type(CustomSum.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(zfog.RealTerm(2).into(CustomTerm)._data.coeffs) is CustomCoeffs
    assert type(zfog.RealTerms(2, 1).into(CustomTerms).coeffs) is CustomCoeffs
    assert type(zfog.RealTermSum(2).into(CustomSum).coeffs) is CustomCoeffs
    assert type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs)) is CustomCoeffs
    terms[1].coeff = 3.0
    terms[::-1].coeffs[0] = 4.0
    assert terms[1].coeff == 4.0
    cloned = terms.clone()
    cloned.coeffs[0] = 5.0
    assert terms[0].coeff != 5.0


def test_zfon_custom_coefficient_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class CustomTerm(zfon.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zfon.RealTerms):
        term_type = CustomTerm

    class CustomSet(zfon.RealTermSet):
        terms_type = CustomTerms

    class CustomSum(zfon.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    term = CustomTerm(2)
    term.string.set("F0^ F1")
    terms = DerivedTerms(2, 2)
    terms[:] = term
    term_set = CustomSet.from_terms(terms)
    total = CustomSum.from_terms(terms)
    assert_type(term.coeffs_type, type[CustomCoeffs])
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(terms.coeffs_type, type[CustomCoeffs])
    assert_type(term_set.coeffs, CustomCoeffs)
    assert_type(total.coeffs, CustomCoeffs)
    assert_type(terms[::-1].coeffs, CustomCoeffs)
    assert_type(terms.clone().coeffs, CustomCoeffs)
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs), CustomCoeffs)
    assert type(term._data.coeffs) is CustomCoeffs
    assert type(terms.coeffs) is CustomCoeffs
    assert type(term_set.coeffs) is CustomCoeffs
    assert type(total.coeffs) is CustomCoeffs
    assert type(terms[::-1].coeffs) is CustomCoeffs
    assert type(terms.clone().coeffs) is CustomCoeffs
    assert type(term_set.clone().coeffs) is CustomCoeffs
    assert type(total.clone().coeffs) is CustomCoeffs
    assert type(CustomTerm.from_string_coeff(term.string, 2.0)._data.coeffs) is CustomCoeffs
    assert type(CustomTerm.from_str(str(term), 2)._data.coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str("", 2).coeffs) is CustomCoeffs
    assert type(CustomSum.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(zfon.RealTerm(2).into(CustomTerm)._data.coeffs) is CustomCoeffs
    assert type(zfon.RealTerms(2, 1).into(CustomTerms).coeffs) is CustomCoeffs
    assert type(zfon.RealTermSum(2).into(CustomSum).coeffs) is CustomCoeffs
    assert type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs)) is CustomCoeffs
    terms[1].coeff = 3.0
    terms[::-1].coeffs[0] = 4.0
    assert terms[1].coeff == 4.0
    cloned = terms.clone()
    cloned.coeffs[0] = 5.0
    assert terms[0].coeff != 5.0


def test_zfs_custom_coefficient_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class CustomTerm(zfs.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zfs.RealTerms):
        term_type = CustomTerm

    class CustomSet(zfs.RealTermSet):
        terms_type = CustomTerms

    class CustomSum(zfs.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    term = CustomTerm(2)
    term.string.set("[1, 0]")
    terms = DerivedTerms(2, 2)
    terms[:] = term
    term_set = CustomSet.from_terms(terms)
    total = CustomSum.from_terms(terms)
    assert_type(term.coeffs_type, type[CustomCoeffs])
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(terms.coeffs_type, type[CustomCoeffs])
    assert_type(term_set.coeffs, CustomCoeffs)
    assert_type(total.coeffs, CustomCoeffs)
    assert_type(terms[::-1].coeffs, CustomCoeffs)
    assert_type(terms.clone().coeffs, CustomCoeffs)
    assert_type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs), CustomCoeffs)
    assert type(term._data.coeffs) is CustomCoeffs
    assert type(terms.coeffs) is CustomCoeffs
    assert type(term_set.coeffs) is CustomCoeffs
    assert type(total.coeffs) is CustomCoeffs
    assert type(terms[::-1].coeffs) is CustomCoeffs
    assert type(terms.clone().coeffs) is CustomCoeffs
    assert type(term_set.clone().coeffs) is CustomCoeffs
    assert type(total.clone().coeffs) is CustomCoeffs
    assert type(CustomTerm.from_string_coeff(term.string, 2.0)._data.coeffs) is CustomCoeffs
    assert type(CustomTerm.from_str(str(term), 2)._data.coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(CustomTerms.from_str("", 2).coeffs) is CustomCoeffs
    assert type(CustomSum.from_str(str(terms), 2).coeffs) is CustomCoeffs
    assert type(zfs.RealTerm(2).into(CustomTerm)._data.coeffs) is CustomCoeffs
    assert type(zfs.RealTerms(2, 1).into(CustomTerms).coeffs) is CustomCoeffs
    assert type(zfs.RealTermSum(2).into(CustomSum).coeffs) is CustomCoeffs
    assert type(convert_vec(RealCoeffs.from_sequence([1.0]), CustomCoeffs)) is CustomCoeffs
    terms[1].coeff = 3.0
    terms[::-1].coeffs[0] = 4.0
    assert terms[1].coeff == 4.0
    cloned = terms.clone()
    cloned.coeffs[0] = 5.0
    assert terms[0].coeff != 5.0


def test_coefficient_storage_matches_factory() -> None:
    class CustomCoeffs(RealCoeffs):
        pass

    class DerivedCoeffs(CustomCoeffs):
        pass

    class CustomTerm(zqo.RealTerm):
        coeffs_type = CustomCoeffs

    class CustomTerms(zqo.RealTerms):
        term_type = CustomTerm

    class CustomSet(zqo.RealTermSet):
        terms_type = CustomTerms

    # A stored subclass is valid, and coeffs_type must return that actual subclass.
    data = TermData(zqo.Strings(2, 1), DerivedCoeffs.from_size(1))
    terms = CustomTerms._create(data)
    term_set = CustomSet.from_terms(terms)
    assert_type(terms.coeffs, CustomCoeffs)
    assert_type(term_set.coeffs, CustomCoeffs)
    assert type(terms.coeffs) is DerivedCoeffs
    assert type(term_set.coeffs) is DerivedCoeffs
    assert terms.coeffs_type is DerivedCoeffs
    assert term_set.coeffs_type is DerivedCoeffs

    # Both normal initialization and internal construction reject inconsistent storage.
    mismatched = TermData(zqo.Strings(2, 1), RealCoeffs.from_size(1))
    ordinary_terms = zqo.RealTerms._create(mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        TermBase.__init__(CustomTerm.__new__(CustomTerm), mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        TermsBase.__init__(CustomTerms.__new__(CustomTerms), mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        TermSetBase.__init__(CustomSet.__new__(CustomSet), ordinary_terms)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        CustomTerm._create(mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        CustomTerms._create(mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        CustomSet.from_terms(ordinary_terms)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        CustomSet._create(mismatched)
    with pytest.raises(TypeError, match="Coefficient storage does not match its factory"):
        CustomSet._create(mismatched, term_set._string_set._map)

    adopted = CustomSet._create(data, term_set._string_set._map)
    assert_type(adopted.coeffs, CustomCoeffs)
    assert adopted.coeffs_type is DerivedCoeffs
    assert type(adopted.coeffs) is DerivedCoeffs
