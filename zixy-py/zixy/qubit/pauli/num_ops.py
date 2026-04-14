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

"""Number operator utilities for terms containing Pauli strings."""

from collections.abc import Sequence
from typing import TypeVar

from zixy._zixy import Qubits
from zixy.container.coeffs import unit
from zixy.qubit.pauli import Z
from zixy.qubit.pauli._terms import ComplexTerm, RealTerm

TermT = TypeVar("TermT", RealTerm, ComplexTerm)


def create_num_op_general(t: type[TermT], qubits: int | Qubits, inds: Sequence[int]) -> TermT:
    """Create a number operator term.

    Args:
        t: The term type to create.
        qubits: The qubit register or qubit count.
        inds: The indices of the qubits to include in the number operator.

    Returns:
        The resulting term representing the number operator.
    """
    out = t(qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits))
    out += (None, len(inds) / 2)  # type: ignore[operator]
    for i in inds:
        out += ({i: Z}, -unit(out.coeff_type) / 2)
    return out  # type: ignore[no-any-return]


def create_num_op(t: type[TermT], qubits: int | Qubits) -> TermT:
    """Create a number operator term.

    Args:
        t: The term type to create.
        qubits: The qubit register or qubit count.

    Returns:
        The resulting term representing the number operator.
    """
    n = len(qubits) if isinstance(qubits, Qubits) else qubits
    return create_num_op_general(t, qubits, range(n))


def create_num_op_odd_bits(t: type[TermT], qubits: int | Qubits) -> TermT:
    """Create a number operator term that includes only the odd-indexed qubits.

    Args:
        t: The term type to create.
        qubits: The qubit register or qubit count.

    Returns:
        The resulting term representing the number operator.
    """
    n = len(qubits) if isinstance(qubits, Qubits) else qubits
    return create_num_op_general(t, qubits, range(1, n, 2))
