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

"""Shared term parsing and arithmetic helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy.container.coeff import ComplexSign, Sign
from zixy.utils import split_top_level

if TYPE_CHECKING:
    from collections.abc import Callable

    from zixy.container.coeffs import Coeffs


def _parse_term_str(source: str) -> tuple[str | None, str]:
    """Parse a string representation of a term into its coefficient and string parts.

    Note:
        The string representation of terms is assumed to be a comma-separated parenthesis-enclosed
        pair of the coefficient and string. It may also be simply comma-separated strings, in
        which case the coefficient part is returned as ``None``.
    """
    # Remove any whitespace
    string = source.strip()

    # Remove any parentheses and whitespace from each element
    items = [item.strip() for item in split_top_level(string, ",")]

    # Separate the coefficient and string parts
    items_coeff: list[str] = []
    items_string: list[str] = []
    for item in items:
        if not item.startswith("(") or not item.endswith(")"):
            items_string.append(item)
            continue
        subitems = split_top_level(item[1:-1], ",", keep_empty=True)
        if len(subitems) == 1:
            items_string.append(subitems[0])
        elif len(subitems) == 2:
            items_coeff.append(subitems[0])
            items_string.append(subitems[1])
        else:
            raise ValueError(f"Invalid item '{item}' in string '{source}'.")

    # Join the coefficient and string parts into independently parsable strings
    source_string = ", ".join(items_string)
    if len(items_coeff) != 0 and len(items_coeff) != len(items_string):
        raise ValueError(f"Inconsistent coefficient and string counts in string '{source}'.")
    source_coeff = ", ".join(items_coeff) if items_coeff else None

    return source_coeff, source_string


def _coeffs_from_str[C: (float, complex, Sign, ComplexSign, Expr)](
    source_coeff: str | None, coeffs_type: type[Coeffs[C]], n_string: int
) -> Coeffs[C]:
    """Construct coefficient data from parsed coefficient text."""
    coeffs = (
        coeffs_type(size=n_string) if source_coeff is None else coeffs_type.from_str(source_coeff)
    )
    if len(coeffs) != n_string:
        raise ValueError(
            f"Length of coefficients ({len(coeffs)}) does not match length of strings ({n_string})."
        )
    return coeffs


def _positive_power[T](base: T, exponent: int, multiply: Callable[[T, T], T]) -> T:
    """Compute a validated positive power by repeated squaring."""
    if exponent < 0:
        raise ValueError(f"Term powers require a nonnegative integer exponent, got {exponent}.")
    result = base
    for bit in bin(exponent)[3:]:
        result = multiply(result, result)
        if bit == "1":
            result = multiply(result, base)
    return result
