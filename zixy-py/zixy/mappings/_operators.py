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

"""Shared mapping functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import zixy.qubit.operator as zqo
from zixy.container.coeffs import convert

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import zixy.fermion.operator.general as zfog
    import zixy.fermion.operator.normal as zfon
    from zixy._zixy import Qubits

    type _NumericOperator = (
        zfog.RealTerm
        | zfog.ComplexTerm
        | zfog.RealTermSum
        | zfog.ComplexTermSum
        | zfon.SignTerm
        | zfon.RealTerm
        | zfon.ComplexTerm
        | zfon.SignTermSum
        | zfon.RealTermSum
        | zfon.ComplexTermSum
    )


def _get_operator_ops(string: zfog.String | zfon.String, qubits: Qubits) -> list[tuple[int, bool]]:
    """Get an operator product whose indices fit the mapper's register.

    Args:
        string: General or normal-ordered fermionic operator string to map.
        qubits: Output register, whose size bounds the input mode indices.

    Returns:
        The ordered ``(mode, is_creation)`` pairs, preserving repeated operators.

    Raises:
        ValueError: An operator mode index is outside the mapper's register.
    """
    ops = string.get_ops()
    if any(mode < 0 or mode >= len(qubits) for mode, _ in ops):
        raise ValueError("Operator mode index is outside the mapper's qubit register.")
    return ops


@overload
def _apply_operator(
    value: _NumericOperator,
    map_string: Callable[[zfog.String | zfon.String], zqo.ComplexTermSum],
    qubits: Qubits,
    *,
    real: Literal[False] = False,
) -> zqo.ComplexTermSum: ...


@overload
def _apply_operator(
    value: _NumericOperator,
    map_string: Callable[[zfog.String | zfon.String], zqo.ComplexTermSum],
    qubits: Qubits,
    *,
    real: Literal[True],
) -> zqo.RealTermSum: ...


@overload
def _apply_operator(
    value: _NumericOperator,
    map_string: Callable[[zfog.String | zfon.String], zqo.ComplexTermSum],
    qubits: Qubits,
    *,
    real: bool,
) -> zqo.ComplexTermSum | zqo.RealTermSum: ...


def _apply_operator(
    value: _NumericOperator,
    map_string: Callable[[zfog.String | zfon.String], zqo.ComplexTermSum],
    qubits: Qubits,
    *,
    real: bool = False,
) -> zqo.ComplexTermSum | zqo.RealTermSum:
    """Extend an operator-string mapping linearly to a term or term sum.

    Args:
        value: General or normal-ordered fermionic operator term or term sum with numeric
            coefficients.
        map_string: Callable mapping a general or normal-ordered operator string to a new
            complex Pauli term sum.
        qubits: Output qubit register, including its labels and ordering.
        real: Whether to retain only the real Pauli coefficients after multiplying by the
            input coefficients.

    Returns:
        A new owning Pauli term sum, with complex coefficients by default or real coefficients
        when requested.
        Contributions to the same Pauli string are added together.

    Raises:
        TypeError: The input is not a numeric fermionic operator term or term sum, or a
            coefficient cannot be converted to the required output type.
        ValueError: An operator mode index is outside the mapper's register.

    Note:
        Inputs are read by view and are not modified. Empty sums preserve the output register.
        The string mapper must return independent output storage.
    """
    # The operator families import Mapper during package initialization.
    import zixy.fermion.operator.general as zfog  # noqa: PLC0415
    import zixy.fermion.operator.normal as zfon  # noqa: PLC0415

    if isinstance(
        value, zfog.SymbolicTerm | zfog.SymbolicTermSum | zfon.SymbolicTerm | zfon.SymbolicTermSum
    ):
        raise TypeError("Only numeric fermionic operator coefficients are supported.")
    term_types = (zfog.RealTerm, zfog.ComplexTerm, zfon.SignTerm, zfon.RealTerm, zfon.ComplexTerm)
    sum_types = (
        zfog.RealTermSum,
        zfog.ComplexTermSum,
        zfon.SignTermSum,
        zfon.RealTermSum,
        zfon.ComplexTermSum,
    )
    if not isinstance(value, term_types + sum_types):
        raise TypeError(f"Cannot map an instance of {type(value).__name__}.")
    terms = (value,) if isinstance(value, term_types) else value
    output = zqo.ComplexTermSum(qubits)
    for term in terms:
        mapped_sum = map_string(term.string)
        mapped_sum *= convert(term.coeff, complex)
        output += mapped_sum
    return output.real_part if real else output
