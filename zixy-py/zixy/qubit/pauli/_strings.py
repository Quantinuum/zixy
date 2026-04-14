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

"""Pauli string components and collections of such strings.

Pauli strings are components representing tensor products of single-qubit Pauli matrices, acting
on a register of qubits.

The structure of this module parallels that of :mod:`~zixy.container.cmpnts` and
:mod:`~zixy.qubit._strings`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from zixy._zixy import (
    PauliMatrix,
    PauliSprings,
    QubitPauliArray,
    Qubits,
)
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexSign,
    ComplexSignCoeffs,
    _imul_factor_error,
    unit,
)
from zixy.container.data import TermData
from zixy.qubit._strings import (
    String as StringBase,
    Strings as StringsBase,
    StringSet as StringSetBase,
    _default_qubits as _default_qubits_base,
)
from zixy.utils import slice_equal

if TYPE_CHECKING:
    from zixy.qubit.pauli._terms import Term, TermRegistry

StringSpec: TypeAlias = (
    None  # signifies the clear string
    | Sequence[PauliMatrix]
    | dict[int, PauliMatrix]
    | str
)
ElemT = PauliMatrix
SpecT = StringSpec
ImplT = QubitPauliArray


def _default_qubits(source: StringSpec = None) -> Qubits:
    """Construct the default qubits for a string specifier."""
    if isinstance(source, dict):
        if len(source) == 0:
            return Qubits.from_count(0)
        else:
            return Qubits.from_count(max(source.keys()) + 1)
    if isinstance(source, str):
        return Qubits.from_count(PauliSprings(source).default_n_qubit())
    else:
        return _default_qubits_base(source)


class String(StringBase[ImplT, SpecT, ElemT]):
    """A Pauli string.

    A single qubit-based Pauli string that may be an owning instance referencing a single element
    in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _term_registry: TermRegistry

    _springs_type = PauliSprings

    @staticmethod
    def _get_default_qubits(source: SpecT | None = None) -> Qubits:
        """Get the default qubit space for :param:`source`."""
        return _default_qubits(source)

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> String:
        """Create a new instance of :param:`cls` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.

        Returns:
            A new instance containing the Pauli string in the :param:`source`.
        """
        n = len(PauliSprings(source))
        if n != 1:
            raise ValueError(f"There should be exactly one Pauli string in the input, not {n}.")
        return cls(qubits, source)

    def __getitem__(self, i: int) -> PauliMatrix:
        """Return the element of the string at index :param:`i`."""
        return self._impl.cmpnt_get_pauli(self.index, i)

    def __setitem__(self, i: int, pauli: PauliMatrix | str) -> None:
        """Set the element of the string at index :param:`i`."""
        if isinstance(pauli, str):
            if pauli == "I":
                pauli = PauliMatrix.I
            elif pauli == "X":
                pauli = PauliMatrix.X
            elif pauli == "Y":
                pauli = PauliMatrix.Y
            elif pauli == "Z":
                pauli = PauliMatrix.Z
            else:
                raise ValueError(f'String "{pauli}" cannot be interpreted as a Pauli matrix.')
        self._impl.cmpnt_set_pauli(self.index, i, pauli)

    def get_dict(self) -> dict[int, PauliMatrix]:
        """Get the string as a dictionary of its elements."""
        return self._impl.cmpnt_get_dict(self.index)

    def set(self, source: SpecT | StringBase[ImplT, SpecT, ElemT] | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, dict):
            self._impl.cmpnt_set_from_dict(self.index, source)
        else:
            super().set(source)

    def is_identity(self) -> bool:
        """Check whether the string is identity (:math:`II...I`)."""
        return self.count(PauliMatrix.I) == len(self.qubits)

    def is_diagonal(self) -> bool:
        """Check whether the string is diagonal (only :math:`I` and :math:`Z`)."""
        return self.count(PauliMatrix.I) + self.count(PauliMatrix.Z) == len(self.qubits)

    @overload  # type: ignore[override]
    def __mul__(self, rhs: String) -> Term[ComplexSign]: ...

    @overload
    def __mul__(self, rhs: CoeffT) -> Term[CoeffT]: ...

    def __mul__(self, rhs: CoeffT | String) -> Term[CoeffT] | Term[ComplexSign]:
        """Multiplication of :param:`self` by :param:`rhs`."""
        if not isinstance(rhs, Coeff | String):
            return NotImplemented
        if isinstance(rhs, String):
            product, phase = self._impl.cmpnt_mul(self.index, rhs._impl, rhs.index)
            phases = ComplexSignCoeffs.from_scalar(ComplexSign(phase))
            term_type = self._term_registry[ComplexSign]
            return term_type._create(TermData(Strings._create(product), phases))
        return super().__mul__(rhs)  # type: ignore[return-value]

    def __imul__(self, rhs: String) -> Self:  # type: ignore
        """In-place multiplication of :param:`self` by :param:`rhs`."""
        if not isinstance(rhs, String):
            raise TypeError("String is only in-place multiplicable by String")
        phase = self.phase_of_mul(rhs)
        if phase != ComplexSign():
            raise _imul_factor_error(self, rhs, phase)
        self.imul_ignore_phase(rhs)
        return self

    def phase_of_mul(self, rhs: String) -> ComplexSign:
        """Get the phase resulting from the multiplication of :param:`self` and :param:`rhs`."""
        return ComplexSign(self._impl.cmpnt_phase_of_mul(self.index, rhs._impl, rhs.index))

    def imul_ignore_phase(self, rhs: String) -> None:
        """In-place multiplication of :param:`self` by :param:`rhs`, ignoring the scalar phase."""
        from zixy.qubit.pauli._terms import Term  # noqa: PLC0415

        if isinstance(rhs, Term):
            if rhs.coeff == unit(type(rhs.coeff)):
                rhs = rhs.string
            else:
                raise ValueError(
                    "Cannot right-multiply String in-place by term with non-unit coefficient"
                )
        elif not isinstance(rhs, String):
            raise TypeError(f"Cannot right-multiply String in-place by type {type(rhs)}")
        if self._impl.same_as(rhs._impl):
            self._impl.cmpnt_matrices_imul_internal(self.index, rhs.index)
        else:
            self._impl.cmpnt_matrices_imul_external(self.index, rhs._impl, rhs.index)

    def imul_get_phase(self, rhs: String) -> ComplexSign:
        """In-place multiplication of :param:`self` by :param:`rhs`, returning the phase.

        See Also:
            :meth:`~zixy.qubit.pauli._strings.String.imul_ignore_phase` for in-place
            multiplication that ignores the phase.
        """
        if type(rhs) is not String:
            raise TypeError(f"Cannot right-multiply String in-place by type {type(rhs)}")
        if self._impl.same_as(rhs._impl):
            phase = self._impl.cmpnt_imul_internal(self.index, rhs.index)
        else:
            phase = self._impl.cmpnt_imul_external(self.index, rhs._impl, rhs.index)
        return ComplexSign(phase)

    def set_mul(self, lhs: String, rhs: String) -> ComplexSign:
        """Set :param:`self` to the product of :param:`lhs` and :param:`rhs`, and return the phase.

        Args:
            lhs: Left hand side of the product.
            rhs: Right hand side of the product.

        Returns:
            The phase factor of the multiplication.
        """
        self._set_copy(lhs)
        return self.imul_get_phase(rhs)

    def commutes_with(self, rhs: String) -> bool:
        """Check whether :param:`self` commutes with :param:`rhs`."""
        return self._impl.cmpnt_commutes_with(self.index, rhs._impl, rhs.index)


class Strings(StringsBase[ImplT, SpecT, ElemT]):
    """A collection of Pauli strings.

    An array-like container of qubit-based Pauli strings that may be an owning instance referencing
    a contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    cmpnt_type = String

    _set_type: type[StringSet]

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Strings:
        """Create a new instance of :param:`cls` by parsing an input string.

        Args:
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.
            source: Input string to parse.

        Returns:
            A new instance containing the Pauli string in the :param:`source`.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        return cls._create(
            cls.cmpnt_type.impl_type(qubits if qubits is not None else None, PauliSprings(source))
        )

    @overload
    def __getitem__(self, indexer: int) -> String: ...

    @overload
    def __getitem__(self, indexer: slice) -> Strings: ...

    def __getitem__(self, indexer: int | slice) -> String | Strings:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        return super().__getitem__(indexer)  # type: ignore[return-value]

    def get_dicts(self) -> tuple[dict[int, PauliMatrix], ...]:
        """Get the strings as a list of dictionaries of their elements."""
        return tuple(self[i].get_dict() for i in range(len(self)))

    def compatibility_matrix(self) -> NDArray[np.uint8]:
        """Form the compatibility matrix for the strings in :param:`self`.

        Returns:
            Matrix with 1 where the strings represented by the row and column commute, 0 elsewhere.
        """
        if slice_equal(self.slice, slice(None, len(self._impl)), len(self._impl)):
            return self._impl.compatibility_matrix()
        else:
            # todo: have rust binding take inds to preclude this copy in this non-contiguous case.
            return self.clone()._impl.compatibility_matrix()

    def iter_filter_non_identity(self) -> Iterator[String]:
        """Lazy filter retaining only the strings that are different to identity."""
        return cast(Iterator[String], self.iter_filter_populated())

    def filter_non_identity(self) -> Strings:
        """Eagerly filter retaining only the strings that are different to identity."""
        return self.filter_populated()

    def centralizer_and_remainder(self) -> tuple[Strings, Strings]:
        """Get the centralizer and remainder of :param:`self`.

        The centralizer of a set :math:`S` of Pauli strings is the set of strings :math:`C` that
        commute with all others in :math:`S`. The remainder is the set of Pauli strings that are in
        :math:`S` but not in :math:`C`.

        Returns:
            The centralizer and remainder as a pair of
            :class:`~zixy.qubit.pauli._strings.Strings` instances.
        """
        c, r = self._impl.centralizer_and_remainder()
        return Strings._create(c), Strings._create(r)


class StringSet(StringSetBase[ImplT, SpecT, ElemT]):
    """A collection of unique Pauli strings.

    A set-like container of qubit-based Pauli strings that may be used to store unique components
    and perform set-like operations on them.
    """

    cmpnts_type = Strings


Strings._set_type = StringSet
