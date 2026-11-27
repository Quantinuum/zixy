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

"""Term sums for fermionic operators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

from sympy import Expr

import zixy.qubit.operator as zqo
from zixy._zixy import Qubits
from zixy.container.coeff import convert
from zixy.fermion._term_sum import TermSum as FermionTermSum
from zixy.fermion.operator._term_set import TermSet
from zixy.mappings.base import Mapper

if TYPE_CHECKING:
    from zixy._zixy import FermionArray


type MapperType = Callable[[int | Qubits], Mapper]


class TermSum[ImplT: FermionArray, SpecT, C: (float, complex, Expr), ElemT](
    FermionTermSum[ImplT, SpecT, C, ElemT],
    TermSet[ImplT, SpecT, C, ElemT],
):
    """A sum of terms consisting of fermionic ladder-operator strings and coefficients."""

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        pass

    def to_qubit(
        self,
        mapper: MapperType | None = None,
        qubits: int | Qubits | None = None,
    ) -> zqo.ComplexTermSum:
        """Map the term sum to a qubit Pauli term sum.

        Args:
            mapper: The mapper class to use. If ``None``, use
                :class:`~zixy.mappings.JordanWignerMapper`.
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the number of fermionic modes.

        Returns:
            The mapped Pauli term sum.

        Note:
            This function returns a term sum with complex coefficients. In cases where Hermitian
            operators guarantee a real mapped representation, users may wish to directly use the
            :class:`~zixy.fermion.mappings.Mapper` classes for finer control.
        """
        from zixy.mappings import JordanWignerMapper  # noqa: PLC0415

        mapper_type = JordanWignerMapper if mapper is None else mapper
        if qubits is None:
            qubits = Qubits.from_count(len(self.modes))
        elif isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        mapper_instance = mapper_type(qubits)
        out = zqo.ComplexTermSum(qubits)
        for term in self:
            string_type = self.terms_type.term_type.strings_type.string_type
            out += convert(term.coeff, complex) * mapper_instance.apply(
                term.string.into(string_type)
            )
        return out
