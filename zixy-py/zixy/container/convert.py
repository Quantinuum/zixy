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

from typing import TYPE_CHECKING, Any, cast

from sympy import Expr

from zixy.container.coeffs import (
    Coeffs,
    ComplexSign,
    Sign,
    convert,
    convert_vec,
    get_coeffs_type,
    unit,
)
from zixy.container.data import TermData
from zixy.container.strings import String, Strings
from zixy.container.terms import Term, Terms

if TYPE_CHECKING:
    from collections.abc import Sized

    from zixy._zixy import ImplArray
    from zixy.container.base import ViewableBase


def into[T](source: ViewableBase[Any, Any], t: type[T]) -> T:
    """Clone ``source`` into a new related container of type ``t``.

    Args:
        source: Container to convert.
        t: Target container type.

    Returns:
        An owning clone represented as ``t``.

    Raises:
        TypeError: If the source-target pair is unsupported or has incompatible string types.
        ValueError: If a coefficient value cannot be represented by the target coefficient type.
    """
    if not isinstance(t, type):
        raise TypeError(f"Conversion target must be a type, got {type(t)}.")
    if issubclass(t, String):
        return cast("T", _into_string(source, t))
    if issubclass(t, Strings):
        return cast("T", _into_strings(source, t))
    if issubclass(t, Term):
        return cast("T", _into_term(source, t))
    if issubclass(t, Terms):
        return cast("T", _into_terms(source, t))
    if issubclass(t, Coeffs):
        return cast("T", _into_coeffs(source, t))
    raise TypeError(f"Cannot convert {type(source)} into unsupported target type {t}.")


def _check_string_compatibility[ImplT: ImplArray, SpecT, TargetSpecT](
    source_type: type[String[ImplT, SpecT]], target_type: type[String[ImplT, TargetSpecT]]
) -> None:
    """Check that source and target string types are compatible for conversion."""
    if source_type.impl_type is not target_type.impl_type:
        raise TypeError(
            f"Cannot convert a container with string implementation type "
            f"{source_type.impl_type} into one with string implementation type "
            f"{target_type.impl_type}."
        )


def _require_single_item(source: Sized, t: type[Any]) -> None:
    """Check that ``source`` contains exactly one item."""
    if len(source) != 1:
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: source must contain exactly one item."
        )


def _require_unit_coeff[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    source: Term[ImplT, SpecT, C], t: type[Any]
) -> None:
    """Check that ``source`` has the unit coefficient for its coefficient type."""
    coeff = source.coeff
    unit_coeff = unit(source.coeff_type)
    if coeff != unit_coeff:
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: coefficient {coeff} is not the unit "
            f"value {unit_coeff}."
        )


def _require_unit_coeffs[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    source: Terms[ImplT, SpecT, C], t: type[Any]
) -> None:
    """Check that all coefficients in ``source`` are unit values."""
    unit_coeff = unit(source.coeff_type)
    if any(coeff != unit_coeff for coeff in source.coeffs):
        raise ValueError(
            f"Cannot convert {type(source)} into {t}: all coefficients must be the unit value "
            f"{unit_coeff}."
        )


def _clone_coeffs_as[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](source: Coeffs[C], target_type: type[Coeffs[COther]]) -> Coeffs[COther]:
    """Clone coefficient data from ``source`` as ``target_type``."""
    if type(source) is target_type:
        return source.clone()
    return convert_vec(source, target_type)


def _clone_strings_as[ImplT: ImplArray, SpecT, TargetSpecT](
    source: Strings[ImplT, SpecT], target_type: type[Strings[ImplT, TargetSpecT]]
) -> Strings[ImplT, TargetSpecT]:
    """Clone string data from ``source`` as ``target_type``."""
    _check_string_compatibility(source.string_type, target_type.string_type)
    return target_type._create(source.clone()._impl)


def _clone_string_as[ImplT: ImplArray, SpecT, TargetSpecT](
    source: String[ImplT, SpecT], target_type: type[String[ImplT, TargetSpecT]]
) -> String[ImplT, TargetSpecT]:
    """Clone string data from ``source`` as ``target_type``."""
    _check_string_compatibility(type(source), target_type)
    return target_type._create(source.clone()._impl)


def _into_string[ImplT: ImplArray, SpecT](
    source: object, t: type[String[ImplT, SpecT]]
) -> String[ImplT, SpecT]:
    """Convert to a single string."""
    if isinstance(source, String):
        return _clone_string_as(source, t)
    if isinstance(source, Strings):
        _require_single_item(source, t)
        return _into_string(source[0], t)
    if isinstance(source, Term):
        _check_string_compatibility(source.string_type, t)
        _require_unit_coeff(source, t)
        return _clone_string_as(source.string, t)
    if isinstance(source, Terms):
        _require_single_item(source, t)
        return _into_string(source[0], t)
    raise TypeError(f"Cannot convert {type(source)} into string type {t}.")


def _into_strings[ImplT: ImplArray, SpecT](
    source: object, t: type[Strings[ImplT, SpecT]]
) -> Strings[ImplT, SpecT]:
    """Convert to a string sequence."""
    if isinstance(source, String):
        _check_string_compatibility(type(source), t.string_type)
        return t._create(source.clone()._impl)
    if isinstance(source, Strings):
        return _clone_strings_as(source, t)
    if isinstance(source, Term):
        _check_string_compatibility(source.string_type, t.string_type)
        _require_unit_coeff(source, t)
        return t._create(source.string.clone()._impl)
    if isinstance(source, Terms):
        _require_unit_coeffs(source, t)
        return _clone_strings_as(source.strings, t)
    raise TypeError(f"Cannot convert {type(source)} into string collection type {t}.")


def _into_term[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    source: object, t: type[Term[ImplT, SpecT, C]]
) -> Term[ImplT, SpecT, C]:
    """Convert to a single term."""
    if isinstance(source, String):
        _check_string_compatibility(type(source), t.strings_type.string_type)
        strings = t.strings_type._create(source.clone()._impl)
        coeffs = get_coeffs_type(t.coeff_type).from_scalar(unit(t.coeff_type))
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    if isinstance(source, Strings):
        _require_single_item(source, t)
        return _into_term(source[0], t)
    if isinstance(source, Term):
        _check_string_compatibility(source.string_type, t.strings_type.string_type)
        strings = t.strings_type._create(source.string.clone()._impl)
        coeffs = get_coeffs_type(t.coeff_type).from_scalar(convert(source.coeff, t.coeff_type))
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    if isinstance(source, Terms):
        _require_single_item(source, t)
        return _into_term(source[0], t)
    raise TypeError(f"Cannot convert {type(source)} into term type {t}.")


def _into_terms[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    source: object, t: type[Terms[ImplT, SpecT, C]]
) -> Terms[ImplT, SpecT, C]:
    """Convert to a term sequence."""
    strings_type = t.term_type.strings_type
    coeffs_type = get_coeffs_type(t.term_type.coeff_type)
    if isinstance(source, String):
        _check_string_compatibility(type(source), strings_type.string_type)
        strings = strings_type._create(source.clone()._impl)
        coeffs = coeffs_type.from_scalar(unit(t.term_type.coeff_type))
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    if isinstance(source, Strings):
        strings = _clone_strings_as(source, strings_type)
        coeffs = coeffs_type.from_scalar(unit(t.term_type.coeff_type), len(source))
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    if isinstance(source, Term):
        _check_string_compatibility(source.string_type, strings_type.string_type)
        strings = strings_type._create(source.string.clone()._impl)
        coeffs = coeffs_type.from_scalar(convert(source.coeff, t.term_type.coeff_type))
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    if isinstance(source, Terms):
        strings = _clone_strings_as(source.strings, strings_type)
        coeffs = convert_vec(source.coeffs, coeffs_type)
        term_data = TermData(strings, coeffs)
        return t._create(term_data)
    raise TypeError(f"Cannot convert {type(source)} into term collection type {t}.")


def _into_coeffs[C: (float, complex, Sign, ComplexSign, Expr)](
    source: object, t: type[Coeffs[C]]
) -> Coeffs[C]:
    """Convert to a coefficient vector."""
    if isinstance(source, Coeffs):
        return _clone_coeffs_as(source, t)
    raise TypeError(f"Cannot convert {type(source)} into coefficient type {t}.")
