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

"""Individual strings for fermionic states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy._zixy import BinarySprings, FermionStateArray, Modes
from zixy.container.coeff import Coeff, ComplexSign, Sign, normalize_scalar
from zixy.container.serialization import check_schema_version
from zixy.fermion._string import (
    String as FermionString,
    _check_modes_compatibility,
    _serialize_modes,
)

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer
    from zixy.fermion.state._term import Term

type StringSpec = Sequence[bool] | set[int] | str

ElemT = bool
SpecT = StringSpec
ImplT = FermionStateArray


def _default_modes(source: SpecT) -> Modes:
    """Construct the default modes for a string specifier."""
    if isinstance(source, set):
        return Modes.from_count(max(source, default=-1) + 1)
    if isinstance(source, str):
        return Modes.from_count(BinarySprings(source).default_n_qubit())
    return Modes.from_count(len(source))


class String(FermionString[ImplT, SpecT, ElemT]):
    """A fermionic occupation-number state string.

    A single mode-based state string that may be an owning instance referencing a single element in
    a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _clear_spec = ""
    _get_default_modes = staticmethod(_default_modes)

    def __getitem__(self, i: int) -> bool:
        """Return the occupation value of the string at index ``i``."""
        return self._impl.string_get_bit(self.index, i)

    def __setitem__(self, i: int, bit: bool | int) -> None:
        """Set the occupation value of the string at index ``i``."""
        if isinstance(bit, int):
            if not (bit == 0 or bit == 1):
                raise ValueError("Integer bit argument must be either 0 or 1")
            bit = bool(bit)
        elif not isinstance(bit, bool):
            raise TypeError("Bit argument should be bool or 0 or 1")
        self._impl.string_set_bit(self.index, i, bit)

    def count(self, bit: bool) -> int:
        """Return the number of occurrences of ``bit`` in the string."""
        return self._impl.string_count(self.index, bit)

    def get_set(self) -> set[int]:
        """Get the string as a set of occupied mode indices."""
        return self._impl.string_get_set(self.index)

    def get_list(self) -> list[bool]:
        """Get the string as a list of occupation flags."""
        return self._impl.string_get_list(self.index)

    def get_tuple(self) -> tuple[bool, ...]:
        """Get the string as a tuple of occupation flags."""
        return tuple(self.get_list())

    def _set_spec(self, source: SpecT) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, set):
            self._impl.string_set_from_set(self.index, source)
        elif isinstance(source, Sequence) and not isinstance(source, str):
            self._impl.string_set_from_list(self.index, list(source))
        elif isinstance(source, str):
            if not source.strip():
                self._impl.string_clear(self.index)
                return
            springs = BinarySprings(source)
            if len(springs) == 0:
                self._impl.string_clear(self.index)
            elif len(springs) == 1:
                self._impl.string_set_from_spring(self.index, springs, 0)
            else:
                raise ValueError("Input represents more than one string.")
        else:
            self.raise_spec_type_error(source)

    def is_vacuum(self) -> bool:
        r"""Check whether the string is the vacuum state (:math:`\left[0, 0, \ldots, 0\right]`)."""
        return self.count(True) == 0

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[C]:
        """Return the product of ``self`` and ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff):
            return NotImplemented
        from zixy.fermion.state._term import get_term_type  # noqa: PLC0415

        term_type = get_term_type(type(rhs))
        return term_type.from_string(self, rhs)

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[C]:
        """Return the product of ``lhs`` and ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff):
            return NotImplemented
        from zixy.fermion.state._term import get_term_type  # noqa: PLC0415

        term_type = get_term_type(type(lhs))
        return term_type.from_string(self, lhs)

    def vdot(self, other: String) -> int:
        """Compute the inner product of this string with another."""
        _check_modes_compatibility(self.modes, other.modes)
        return int(self == other)

    def hamming_weight(self) -> int:
        r"""Get the Hamming weight of this bit string.

        The Hamming weight is defined as

        .. math:: \sum_i s_i

        where :math:`s_i` is the occupation value at index :math:`i` in the string.
        """
        return self.count(True)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if not source.strip():
            return cls(modes, "")
        n = len(BinarySprings(source))
        if n != 1:
            raise ValueError(f"Source string should contain one state string, got {n}.")
        return cls(modes, source)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the string, and the full mode space.
        """
        data = super().to_dict()
        data["family"] = "fermion.state"
        data["space"] = _serialize_modes(self.modes)
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
        from zixy.fermion.state._strings import Strings  # noqa: PLC0415

        check_schema_version(source)
        if len(source["strings"]) != 1:
            raise ValueError("A serialized String must contain exactly one string.")
        strings = Strings.from_dict(source)
        return cls._create(strings._impl)
