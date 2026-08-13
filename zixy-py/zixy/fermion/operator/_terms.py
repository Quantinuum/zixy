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

"""Base fermionic ladder-operator terms and collections of such terms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import Self

from zixy._zixy import Qubits
from zixy.container.cmpnts import SpecT
from zixy.container.coeffs import CoeffT, Sign, convert
from zixy.fermion._strings import ImplT
from zixy.fermion._terms import (
    Term as FermionTerm,
    Terms as FermionTerms,
    TermSet as FermionTermSet,
    TermSum as FermionTermSum,
)
from zixy.fermion.operator._strings import (
    ElemT,
    String as OperatorString,
    Strings as OperatorStrings,
)
from zixy.qubit.pauli._terms import ComplexTermSum as PauliComplexTermSum

if TYPE_CHECKING:
    from zixy.fermion.mappings import Mapper


def _product_sign(n_cre: int, n_ann: int) -> Sign:
    r"""Compute the sign of a product of fermionic ladder operators.

    Args:
        n_cre: The number of creation operators in the product.
        n_ann: The number of annihilation operators in the product.

    Returns:
        The sign of the product of ladder operators.
    """
    return Sign(((n_cre * (n_cre - 1) // 2) + (n_ann * (n_ann - 1) // 2)) & 1)


class Term(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTerm[ImplT, SpecT, CoeffT, ElemT]):
    """A term consisting of a fermionic ladder-operator string and a coefficient."""

    cmpnts_type: type[OperatorStrings[ImplT, SpecT, ElemT]]

    @property
    def string(self) -> OperatorString[ImplT, SpecT, ElemT]:
        """Get the string component of the term."""
        return cast(OperatorString[ImplT, SpecT, ElemT], self.cmpnt)

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        pass


class Terms(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTerms[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of terms consisting of fermionic ladder-operator strings and coefficients."""

    term_type: type[Term[ImplT, SpecT, CoeffT, ElemT]]

    @property
    def strings(self) -> OperatorStrings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(OperatorStrings[ImplT, SpecT, ElemT], self.cmpnts)


class TermSet(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTermSet[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of unique terms consisting of fermionic ladder-operator strings."""

    terms_type: type[Terms[ImplT, SpecT, CoeffT, ElemT]]

    @property
    def strings(self) -> OperatorStrings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(OperatorStrings[ImplT, SpecT, ElemT], self._impl._cmpnts)

    @property
    def coeffs(self) -> Any:
        """Get the coefficients of ``self``."""
        return self._impl._coeffs


class TermSum(
    Generic[ImplT, SpecT, CoeffT, ElemT],
    FermionTermSum[ImplT, SpecT, CoeffT, ElemT],
    TermSet[ImplT, SpecT, CoeffT, ElemT],
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
        mapper: type[Mapper] | None = None,
        qubits: int | Qubits | None = None,
    ) -> PauliComplexTermSum:
        """Map the term sum to a qubit Pauli term sum.

        Args:
            mapper: The mapper class to use. If ``None``, use
                :class:`~zixy.fermion.mappings.JordanWignerMapper`.
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the number of fermionic modes.

        Returns:
            The mapped Pauli term sum.

        Note:
            This function returns a term sum with complex coefficients. In cases where Hermitian
            operators guarantee a real mapped representation, users may wish to directly use the
            :class:`~zixy.fermion.mappings.Mapper` classes for finer control.
        """
        from zixy.fermion.mappings import JordanWignerMapper  # noqa: PLC0415

        mapper = JordanWignerMapper if mapper is None else mapper
        if qubits is None:
            qubits = Qubits.from_count(len(self.modes))
        elif isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        mapper_ = mapper(qubits)
        out = PauliComplexTermSum(qubits)
        for term in self:
            cmpnt_type = self.terms_type.term_type.cmpnts_type.cmpnt_type
            out += mapper_.encode(term.cmpnt.into(cmpnt_type), convert(term.coeff, complex))
        return out
