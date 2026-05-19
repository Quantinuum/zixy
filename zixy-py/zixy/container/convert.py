# Copyright 2026 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion helpers between related container types."""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, TypeVar, cast

from zixy.container.base import ViewableBase
from zixy.container.cmpnts import Cmpnt, Cmpnts, ImplT, SpecT
from zixy.container.coeffs import (
    Coeffs,
    CoeffT,
    OtherCoeffT,
    convert,
    convert_vec,
    get_coeffs_type,
    unit,
)
from zixy.container.data import TermData
from zixy.container.terms import Term, Terms

OutT = TypeVar("OutT")
TargetSpecT = TypeVar("TargetSpecT")


def into(source: ViewableBase[Any, Any], t: type[OutT]) -> OutT:
    """Clone :param:`source` into a new related container of type :param:`t`.

    Args:
        source: Container to convert.
        t: Target container type.

    Returns:
        An owning clone represented as :param:`t`.

    Raises:
        TypeError: If the source-target pair is unsupported or has incompatible component types.
        ValueError: If a coefficient value cannot be represented by the target coefficient type.
    """
    if not isinstance(t, type):
        raise TypeError(f"Conversion target must be a type, got {type(t)}.")
    if issubclass(t, Cmpnt):
        return cast(OutT, _into_cmpnt(source, t))
    if issubclass(t, Cmpnts):
        return cast(OutT, _into_cmpnts(source, t))
    if issubclass(t, Term):
        return cast(OutT, _into_term(source, t))
    if issubclass(t, Terms):
        return cast(OutT, _into_terms(source, t))
    if issubclass(t, Coeffs):
        return cast(OutT, _into_coeffs(source, t))
    raise TypeError(f"Cannot convert {type(source)} into unsupported target type {t}.")


def _check_cmpnt_compatibility(
    source_type: type[Cmpnt[ImplT, SpecT]], target_type: type[Cmpnt[ImplT, TargetSpecT]]
) -> None:
    """Check that source and target component types are compatible for conversion."""
    if source_type.impl_type is not target_type.impl_type:
        raise TypeError(
            f"Cannot convert a container with component implementation type "
            f"{source_type.impl_type} into one with component implementation type "
            f"{target_type.impl_type}."
        )


def _require_single_item(source: Sized, t: type[Any]) -> None:
    """Check that :param:`source` contains exactly one item."""
    if len(source) != 1:
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: source must contain exactly one item."
        )


def _require_unit_coeff(source: Term[ImplT, SpecT, CoeffT], t: type[Any]) -> None:
    """Check that :param:`source` has the unit coefficient for its coefficient type."""
    coeff = source.coeff
    unit_coeff = unit(source.coeff_type)
    if coeff != unit_coeff:
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: coefficient {coeff} is not the unit "
            f"value {unit_coeff}."
        )


def _require_unit_coeffs(source: Terms[ImplT, SpecT, CoeffT], t: type[Any]) -> None:
    """Check that all coefficients in :param:`source` are unit values."""
    unit_coeff = unit(source.coeff_type)
    if any(coeff != unit_coeff for coeff in source.coeffs):
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: all coefficients must be the unit value "
            f"{unit_coeff}."
        )


def _clone_coeffs_as(
    source: Coeffs[CoeffT], target_type: type[Coeffs[OtherCoeffT]]
) -> Coeffs[OtherCoeffT]:
    """Clone coefficient data from :param:`source` as :param:`target_type`."""
    if type(source) is target_type:
        return source.clone()
    return convert_vec(source, target_type)


def _clone_cmpnts_as(
    source: Cmpnts[ImplT, SpecT], target_type: type[Cmpnts[ImplT, TargetSpecT]]
) -> Cmpnts[ImplT, TargetSpecT]:
    """Clone component data from :param:`source` as :param:`target_type`."""
    _check_cmpnt_compatibility(source.cmpnt_type, target_type.cmpnt_type)
    return target_type._create(source.clone()._impl)


def _clone_cmpnt_as(
    source: Cmpnt[ImplT, SpecT], target_type: type[Cmpnt[ImplT, TargetSpecT]]
) -> Cmpnt[ImplT, TargetSpecT]:
    """Clone component data from :param:`source` as :param:`target_type`."""
    _check_cmpnt_compatibility(type(source), target_type)
    return target_type._create(source.clone()._impl)


def _into_cmpnt(source: object, t: type[Cmpnt[ImplT, TargetSpecT]]) -> Cmpnt[ImplT, TargetSpecT]:
    """Convert to a single component."""
    if isinstance(source, Cmpnt):
        return _clone_cmpnt_as(source, t)
    if isinstance(source, Cmpnts):
        _require_single_item(source, t)
        return _into_cmpnt(source[0], t)
    if isinstance(source, Term):
        _check_cmpnt_compatibility(source.cmpnt_type, t)
        _require_unit_coeff(source, t)
        return _clone_cmpnt_as(source.cmpnt, t)
    if isinstance(source, Terms):
        _require_single_item(source, t)
        return _into_cmpnt(source[0], t)
    raise TypeError(f"Cannot convert {type(source)} into component type {t}.")


def _into_cmpnts(source: object, t: type[Cmpnts[ImplT, TargetSpecT]]) -> Cmpnts[ImplT, TargetSpecT]:
    """Convert to a component sequence."""
    if isinstance(source, Cmpnt):
        _check_cmpnt_compatibility(type(source), t.cmpnt_type)
        return t._create(source.clone()._impl)
    if isinstance(source, Cmpnts):
        return _clone_cmpnts_as(source, t)
    if isinstance(source, Term):
        _check_cmpnt_compatibility(source.cmpnt_type, t.cmpnt_type)
        _require_unit_coeff(source, t)
        return t._create(source.cmpnt.clone()._impl)
    if isinstance(source, Terms):
        _require_unit_coeffs(source, t)
        return _clone_cmpnts_as(source.cmpnts, t)
    raise TypeError(f"Cannot convert {type(source)} into component collection type {t}.")


def _into_term(
    source: object, t: type[Term[ImplT, TargetSpecT, CoeffT]]
) -> Term[ImplT, TargetSpecT, CoeffT]:
    """Convert to a single term."""
    if isinstance(source, Cmpnt):
        _check_cmpnt_compatibility(type(source), t.cmpnts_type.cmpnt_type)
        cmpnts = t.cmpnts_type._create(source.clone()._impl)
        coeffs = get_coeffs_type(t.coeff_type).from_scalar(unit(t.coeff_type))
        return t._create(TermData(cmpnts, coeffs))
    if isinstance(source, Cmpnts):
        _require_single_item(source, t)
        return _into_term(source[0], t)
    if isinstance(source, Term):
        _check_cmpnt_compatibility(source.cmpnt_type, t.cmpnts_type.cmpnt_type)
        cmpnts = t.cmpnts_type._create(source.cmpnt.clone()._impl)
        coeffs = get_coeffs_type(t.coeff_type).from_scalar(convert(source.coeff, t.coeff_type))
        return t._create(TermData(cmpnts, coeffs))
    if isinstance(source, Terms):
        _require_single_item(source, t)
        return _into_term(source[0], t)
    raise TypeError(f"Cannot convert {type(source)} into term type {t}.")


def _into_terms(
    source: object, t: type[Terms[ImplT, TargetSpecT, CoeffT]]
) -> Terms[ImplT, TargetSpecT, CoeffT]:
    """Convert to a term sequence."""
    cmpnts_type = t.term_type.cmpnts_type
    coeffs_type = get_coeffs_type(t.term_type.coeff_type)
    if isinstance(source, Cmpnt):
        _check_cmpnt_compatibility(type(source), cmpnts_type.cmpnt_type)
        cmpnts = cmpnts_type._create(source.clone()._impl)
        coeffs = coeffs_type.from_scalar(unit(t.term_type.coeff_type))
        return t._create(TermData(cmpnts, coeffs))
    if isinstance(source, Cmpnts):
        cmpnts = _clone_cmpnts_as(source, cmpnts_type)
        coeffs = coeffs_type.from_scalar(unit(t.term_type.coeff_type), len(source))
        return t._create(TermData(cmpnts, coeffs))
    if isinstance(source, Term):
        _check_cmpnt_compatibility(source.cmpnt_type, cmpnts_type.cmpnt_type)
        cmpnts = cmpnts_type._create(source.cmpnt.clone()._impl)
        coeffs = coeffs_type.from_scalar(convert(source.coeff, t.term_type.coeff_type))
        return t._create(TermData(cmpnts, coeffs))
    if isinstance(source, Terms):
        cmpnts = _clone_cmpnts_as(source.cmpnts, cmpnts_type)
        coeffs = convert_vec(source.coeffs, coeffs_type)
        return t._create(TermData(cmpnts, coeffs))
    raise TypeError(f"Cannot convert {type(source)} into term collection type {t}.")


def _into_coeffs(source: object, t: type[Coeffs[OtherCoeffT]]) -> Coeffs[OtherCoeffT]:
    """Convert to a coefficient vector."""
    if isinstance(source, Coeffs):
        return _clone_coeffs_as(source, t)
    raise TypeError(f"Cannot convert {type(source)} into coefficient type {t}.")
