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

"""Individual strings for qubit states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy._zixy import BinarySprings, Qubits, QubitStateArray
from zixy.container.coeff import Coeff, ComplexSign, Sign, normalize_scalar
from zixy.container.serialization import check_schema_version
from zixy.qubit._string import (
    String as StringBase,
    _check_qubits_compatibility,
    _default_qubits as _default_qubits_base,
)

if TYPE_CHECKING:
    import zixy.qubit.operator as zqo
    from zixy.container.serialization import SerializedContainer
    from zixy.qubit.state._term import Term

type StringSpec = Sequence[bool] | set[int] | str

ElemT = bool
SpecT = StringSpec
ImplT = QubitStateArray


def _default_qubits(source: StringSpec) -> Qubits:
    """Construct the default qubits for a string specifier."""
    if isinstance(source, set):
        return Qubits.from_count(max(source) + 1)
    elif isinstance(source, str):
        return Qubits.from_count(BinarySprings(source).default_n_qubit())
    else:
        return _default_qubits_base(source)


class String(StringBase[ImplT, SpecT, ElemT]):
    """A state string.

    A single qubit-based state string that may be an owning instance referencing a single element
    in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT

    _clear_spec = ""
    _springs_type = BinarySprings

    @staticmethod
    def _get_default_qubits(source: SpecT) -> Qubits:
        """Get the default qubit space for ``source``."""
        return _default_qubits(source)

    def __getitem__(self, i: int) -> bool:
        """Return the bit value of the string at index ``i``."""
        return self._impl.string_get_bit(self.index, i)

    def __setitem__(self, i: int, bit: bool | int) -> None:
        """Set the bit value of the string at index ``i``."""
        if isinstance(bit, int):
            if not (bit == 0 or bit == 1):
                raise ValueError("Integer bit argument must be either 0 or 1")
            bit = bool(bit)
        elif not isinstance(bit, bool):
            raise TypeError("Bit argument should be bool or 0 or 1")
        self._impl.string_set_bit(self.index, i, bit)

    def get_set(self) -> set[int]:
        """Get the string as a set of the indices of bits with value 1."""
        return self._impl.string_get_set(self.index)

    def _set_spec(self, source: SpecT) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, set):
            self._impl.string_set_from_set(self.index, source)
        else:
            super()._set_spec(source)

    def is_vacuum(self) -> bool:
        r"""Check whether the string is the vacuum state (:math:`\left[0, 0, \ldots, 0\right]`)."""
        return self.count(True) == 0

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[C]:
        """Return the product of ``self`` and ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff):
            return NotImplemented
        from zixy.qubit.state._term import get_term_type  # noqa: PLC0415

        term_type = get_term_type(type(rhs))
        return term_type.from_string(self, rhs)

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[C]:
        """Return the product of ``lhs`` and ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff):
            return NotImplemented
        from zixy.qubit.state._term import get_term_type  # noqa: PLC0415

        term_type = get_term_type(type(lhs))
        return term_type.from_string(self, lhs)

    def imul_get_phase(self, op: zqo.String) -> ComplexSign:
        """In-place multiplication of ``self`` by ``op``, returning the phase."""
        return ComplexSign(self._impl.string_pauli_string_imul(self.index, op._impl, op.index))

    def vdot(self, other: String) -> int:
        """Compute the inner product of this string with another."""
        _check_qubits_compatibility(self.qubits, other.qubits)
        return int(self == other)

    def hamming_weight(self) -> int:
        r"""Get the Hamming weight of this bit string.

        The Hamming weight is defined as

        .. math:: \sum_i s_i

        where :math:`s_i` is the value of the bit at index :math:`i` in the string.
        """
        return self.count(True)

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
        n = len(BinarySprings(source))
        if n != 1:
            raise ValueError(f"Source string should contain one state string, got {n}.")
        return cls(qubits, source)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the string, and the full qubit register.
        """
        data = super().to_dict()
        data["family"] = "qubit.state"
        data["space"] = {"qubits": self.qubits.get_inds()}
        data["strings"] = [self.get_list()]
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
        from zixy.qubit.state._strings import Strings  # noqa: PLC0415

        check_schema_version(source)
        if len(source["strings"]) != 1:
            raise ValueError("A serialized String must contain exactly one string.")
        strings = Strings.from_dict(source)
        return cls._create(strings._impl)
