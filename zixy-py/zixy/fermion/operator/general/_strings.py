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

"""Raw fermionic ladder-operator string components and collections of such strings.

Raw fermionic strings are components representing ordered products of creation and annihilation
operators, acting on a register of fermionic modes.

The structure of this module parallels that of :mod:`~zixy.container.cmpnts`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from typing_extensions import Self

from zixy._zixy import GeneralFermionOperatorArray, Modes
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet
from zixy.container.coeffs import Coeff, CoeffT
from zixy.fermion.operator._strings import parse_ladder_product, parse_term_source

if TYPE_CHECKING:
    from zixy.fermion.operator.general._terms import TermRegistry

StringSpec: TypeAlias = None | str | Sequence[tuple[int, bool]]
ImplT = GeneralFermionOperatorArray


def _default_modes(source: StringSpec = None) -> Modes:
    """Construct the default modes for a string specifier."""
    if source is None:
        return Modes.from_count(0)
    ops = parse_ladder_product(source) if isinstance(source, str) else list(source)
    return Modes.from_count(max((i for i, _ in ops), default=-1) + 1)


class String(Cmpnt[ImplT, StringSpec]):
    """A raw fermionic ladder-operator string.

    A single mode-based raw fermionic string that may be an owning instance referencing a single
    element in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _term_registry: TermRegistry

    def __init__(self, modes: int | Modes | None = None, source: StringSpec = None):
        if modes is None:
            modes = _default_modes(source)
        elif isinstance(modes, int):
            modes = Modes.from_count(modes)
        ops = parse_ladder_product(source) if isinstance(source, str) else list(source or ())
        impl = self.impl_type(modes, len(ops))
        impl.resize(1)
        super().__init__(impl)
        if source is not None:
            self.set(ops)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

    @property
    def max_len(self) -> int:
        """Get the maximum operator-product length supported by the backing array."""
        return self._impl.max_len

    def __repr__(self) -> str:
        """Return a sparse-string representation of ``self``."""
        return self._impl.cmpnt_to_string(self.index)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` by parsing an input string.

        Args:
            source: String to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        return cls(modes, source)

    def set(self, source: StringSpec | String | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new raw ladder-operator product.

        Note:
            This method operates in-place and preserves the input operator order.
        """
        if source is None:
            self._impl.cmpnt_clear(self.index)
        elif isinstance(source, String):
            if self._impl.same_as(source._impl):
                self._impl.cmpnt_copy_internal(self.index, source.index)
            else:
                self._impl.cmpnt_copy_external(self.index, source._impl, source.index)
        else:
            ops = parse_ladder_product(source) if isinstance(source, str) else list(source)
            modes, adj = zip(*ops, strict=True) if ops else ((), ())
            self._impl.cmpnt_set_from_ops(self.index, list(modes), list(adj))

    def get_ops(self) -> list[tuple[int, bool]]:
        """Get the raw ladder-operator product as ``(mode, is_creation)`` pairs."""
        modes, adj = self._impl.cmpnt_get_ops(self.index)
        return list(zip(modes, adj, strict=True))

    @overload  # type: ignore[override]
    def __mul__(self, rhs: String) -> Any: ...

    @overload
    def __mul__(self, rhs: CoeffT) -> Any: ...

    def __mul__(self, rhs: String | CoeffT) -> Any:
        """Multiplication of ``self`` by ``rhs``.

        Multiplication by a scalar returns a term. Multiplication by another raw string returns a
        single real term whose string is the concatenated ladder-operator product.
        """
        if isinstance(rhs, Coeff):
            return super().__mul__(rhs)
        if not isinstance(rhs, String):
            return NotImplemented
        if self.modes != rhs.modes:
            raise ValueError("Cannot multiply strings defined over different modes.")
        term_type = self._term_registry[float]
        string = String(self.modes, self.get_ops() + rhs.get_ops())
        return term_type.from_cmpnt_coeff(string, 1.0)


class Strings(Cmpnts[ImplT, StringSpec]):
    """A collection of raw fermionic ladder-operator strings.

    An array-like container of mode-based raw fermionic strings that may be an owning instance
    referencing a contiguous Rust-bound data object, or a view on a slice of the elements in
    another collection.
    """

    cmpnt_type = String
    _set_type: type[StringSet]

    def __init__(self, modes: int | Modes = 0, n: int = 0, max_len: int = 0):
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        super().__init__(self.cmpnt_type.impl_type(modes, max_len))
        self.resize(n)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

    @property
    def max_len(self) -> int:
        """Get the maximum operator-product length supported by the backing array."""
        return self._impl.max_len

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` by parsing an input string.

        Args:
            source: String to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        parsed = parse_term_source(source)
        ops_by_term = [parse_ladder_product(cmpnt) for cmpnt, _ in parsed]
        if modes is None:
            modes = _default_modes([op for ops in ops_by_term for op in ops])
        out = cls(modes, max_len=max((len(ops) for ops in ops_by_term), default=0))
        out.append_iterable(ops_by_term)
        return out

    @overload
    def __getitem__(self, indexer: int) -> String: ...
    @overload
    def __getitem__(self, indexer: slice) -> Self: ...
    def __getitem__(self, indexer: int | slice) -> String | Self:
        return super().__getitem__(indexer)  # type: ignore[return-value]


class StringSet(CmpntSet[ImplT, StringSpec]):
    """A collection of unique raw fermionic ladder-operator strings.

    A set-like container of mode-based raw fermionic strings that may be used to store unique
    strings and perform set-like operations on them.
    """

    cmpnts_type = Strings

    def __init__(self, modes: int | Modes | ImplT = 0, max_len: int = 0):
        if isinstance(modes, self.cmpnts_type.cmpnt_type.impl_type):
            CmpntSet.__init__(self, modes)
            return
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        CmpntSet.__init__(self, self.cmpnts_type(modes, max_len=max_len)._impl)


Strings._set_type = StringSet
