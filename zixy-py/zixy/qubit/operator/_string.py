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

"""Individual strings for qubit operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, overload

from sympy import Expr

from zixy._zixy import PauliMatrix, PauliSprings, QubitPauliArray, Qubits
from zixy.container.coeff import (
    Coeff,
    ComplexSign,
    Sign,
    _imul_factor_error,
    normalize_scalar,
    unit,
)
from zixy.container.coeffs import ComplexSignCoeffs
from zixy.container.data import TermData
from zixy.container.serialization import check_schema_version
from zixy.qubit._string import String as StringBase, _default_qubits as _default_qubits_base

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer
    from zixy.qubit.operator._term import ComplexSignTerm, Term

type StringSpec = Sequence[PauliMatrix] | dict[int, PauliMatrix] | str

ElemT = PauliMatrix
SpecT = StringSpec
ImplT = QubitPauliArray


def _default_qubits(source: StringSpec) -> Qubits:
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
    _clear_spec = ""
    _springs_type = PauliSprings

    @staticmethod
    def _get_default_qubits(source: SpecT) -> Qubits:
        """Get the default qubit space for ``source``."""
        return _default_qubits(source)

    def __getitem__(self, i: int) -> PauliMatrix:
        """Return the element of the string at index ``i``."""
        return self._impl.string_get_pauli(self.index, i)

    def __setitem__(self, i: int, pauli: PauliMatrix | str) -> None:
        """Set the element of the string at index ``i``."""
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
        self._impl.string_set_pauli(self.index, i, pauli)

    def get_dict(self) -> dict[int, PauliMatrix]:
        """Get the string as a dictionary of its elements."""
        return self._impl.string_get_dict(self.index)

    def _set_spec(self, source: SpecT) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, dict):
            self._impl.string_set_from_dict(self.index, source)
        else:
            super()._set_spec(source)

    def is_identity(self) -> bool:
        """Check whether the string is identity (:math:`II...I`)."""
        return self.count(PauliMatrix.I) == len(self.qubits)

    def is_diagonal(self) -> bool:
        """Check whether the string is diagonal (only :math:`I` and :math:`Z`)."""
        return self.count(PauliMatrix.I) + self.count(PauliMatrix.Z) == len(self.qubits)

    @overload
    def __mul__(self, rhs: String) -> ComplexSignTerm: ...
    @overload
    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[C]: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String
    ) -> Term[C] | ComplexSignTerm:
        """Multiplication of ``self`` by ``rhs``."""
        from zixy.qubit.operator._strings import Strings  # noqa: PLC0415

        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff | String):
            return NotImplemented
        from zixy.qubit.operator._term import ComplexSignTerm, get_term_type  # noqa: PLC0415

        if isinstance(rhs, String):
            product, phase = self._impl.string_mul(self.index, rhs._impl, rhs.index)
            phases = ComplexSignCoeffs.from_scalar(ComplexSign(phase))
            term_data = TermData(Strings._create(product), phases)
            return ComplexSignTerm._create(term_data)
        return get_term_type(type(rhs)).from_string(self, rhs)

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[C]:
        """Return the product of ``lhs`` and ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff):
            return NotImplemented
        from zixy.qubit.operator._term import get_term_type  # noqa: PLC0415

        term_type = get_term_type(type(lhs))
        return term_type.from_string(self, lhs)

    def __imul__(self, rhs: String) -> Self:  # type: ignore
        """In-place multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, String):
            raise TypeError("String is only in-place multiplicable by String")
        phase = self.phase_of_mul(rhs)
        if phase != ComplexSign():
            raise _imul_factor_error(self, rhs, phase)
        self.imul_ignore_phase(rhs)
        return self

    def phase_of_mul(self, rhs: String) -> ComplexSign:
        """Get the phase resulting from the multiplication of ``self`` and ``rhs``."""
        return ComplexSign(self._impl.string_phase_of_mul(self.index, rhs._impl, rhs.index))

    def set_mul(self, lhs: String, rhs: String) -> ComplexSign:
        """Set ``self`` to the product of ``lhs`` and ``rhs``, and return the phase.

        Args:
            lhs: Left hand side of the product.
            rhs: Right hand side of the product.

        Returns:
            The phase factor of the multiplication.
        """
        self._set_copy(lhs)
        return self.imul_get_phase(rhs)

    def imul_get_phase(self, rhs: String) -> ComplexSign:
        """In-place multiplication of ``self`` by ``rhs``, returning the phase.

        See Also:
            :meth:`~zixy.qubit.operator._string.String.imul_ignore_phase` for in-place
            multiplication that ignores the phase.
        """
        if type(rhs) is not String:
            raise TypeError(f"Cannot right-multiply String in-place by type {type(rhs)}")
        if self._impl.same_as(rhs._impl):
            phase = self._impl.string_imul_internal(self.index, rhs.index)
        else:
            phase = self._impl.string_imul_external(self.index, rhs._impl, rhs.index)
        return ComplexSign(phase)

    def imul_ignore_phase(self, rhs: String) -> None:
        """In-place multiplication of ``self`` by ``rhs``, ignoring the scalar phase."""
        from zixy.qubit.operator._term import Term  # noqa: PLC0415

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
            self._impl.string_matrices_imul_internal(self.index, rhs.index)
        else:
            self._impl.string_matrices_imul_external(self.index, rhs._impl, rhs.index)

    def commutes_with(self, rhs: String) -> bool:
        """Check whether ``self`` commutes with ``rhs``."""
        return self._impl.string_commutes_with(self.index, rhs._impl, rhs.index)

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> String:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the string specifier.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if not source.strip():
            return cls(qubits, "")
        n = len(PauliSprings(source))
        if n != 1:
            raise ValueError(f"Source string should contain one Pauli string, got {n}.")
        return cls(qubits, source)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the string, and the full qubit register.
        """
        data = super().to_dict()
        data["family"] = "qubit.operator"
        data["space"] = {"qubits": self.qubits.get_inds()}
        data["strings"] = [[str(p) for p in self.get_list()]]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            A new owning instance of ``cls`` parsed from ``source``, with storage independent of
            the source dictionary and any existing containers.
        """
        from zixy.qubit.operator._strings import Strings  # noqa: PLC0415

        check_schema_version(source)
        if len(source["strings"]) != 1:
            raise ValueError("A serialized String must contain exactly one string.")
        strings = Strings.from_dict(source)
        return cls._create(strings._impl)
