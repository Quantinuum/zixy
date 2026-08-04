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
from typing import TYPE_CHECKING, TypeAlias, overload

from typing_extensions import Self

from zixy._zixy import GeneralFermionOperatorArray, Modes
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet
from zixy.container.coeffs import Coeff, CoeffT
from zixy.fermion._strings import _as_modes
from zixy.fermion.operator._strings import (
    LadderOp,
    String as OperatorString,
    Strings as OperatorStrings,
    StringSet as OperatorStringSet,
    parse_ladder_product,
    parse_term_source,
)

if TYPE_CHECKING:
    from zixy.fermion.operator.general._terms import Term

StringSpec: TypeAlias = None | str | Sequence[LadderOp]
ElemT = list[LadderOp]
SpecT = StringSpec
ImplT = GeneralFermionOperatorArray


def _default_modes(source: SpecT = None) -> Modes:
    """Construct the default modes for a string specifier."""
    if source is None:
        return Modes.from_count(0)
    ops = parse_ladder_product(source) if isinstance(source, str) else list(source)
    return Modes.from_count(max((i for i, _ in ops), default=-1) + 1)


class String(OperatorString[ImplT, SpecT, ElemT]):
    """A raw fermionic ladder-operator string.

    A single mode-based raw fermionic string that may be an owning instance referencing a single
    element in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT

    @staticmethod
    def _get_default_modes(source: SpecT | None = None) -> Modes:
        """Get the default modes for this string type based on a string specifier."""
        return _default_modes(source)

    def __init__(self, modes: int | Modes | None = None, source: SpecT = None):
        """Initialize the string.

        Args:
            modes: The mode space or number of modes.
            source: The string specifier to use for default modes and initial value.
        """
        if modes is None:
            modes = self._get_default_modes(source)
        modes = _as_modes(modes)
        ops = parse_ladder_product(source) if isinstance(source, str) else list(source or ())
        impl = self.impl_type(modes, len(ops))
        impl.resize(1)
        Cmpnt.__init__(self, impl)
        if source is not None:
            self.set(ops)

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
        return cls(modes, source)

    def set(self, source: SpecT | String | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new raw ladder-operator product.

        Note:
            This method operates in-place and preserves the input operator order.
        """
        if source is None:
            self._impl.cmpnt_clear(self.index)
        elif isinstance(source, String):
            self._set_copy(source)
        else:
            ops = parse_ladder_product(source) if isinstance(source, str) else list(source)
            modes, adj = zip(*ops, strict=True) if ops else ((), ())
            self._impl.cmpnt_set_from_ops(self.index, list(modes), list(adj))

    def get_ops(self) -> list[LadderOp]:
        """Get the raw ladder-operator product as ``(mode, is_creation)`` pairs."""
        modes, adj = self._impl.cmpnt_get_ops(self.index)
        return list(zip(modes, adj, strict=True))

    def get_sets(self) -> tuple[list[int], list[int]]:
        """Get the creation and annihilation mode lists.

        This is a grouped view of the ordered ladder-operator product and does not preserve
        operator order.
        """
        cre: list[int] = []
        ann: list[int] = []
        for mode, is_creation in self.get_ops():
            if is_creation:
                cre.append(mode)
            else:
                ann.append(mode)
        return cre, ann

    def __getitem__(self, item: LadderOp) -> bool:
        """Return whether an operator appears on a mode.

        Args:
            item: Pair of ``(mode, is_creation)``, where ``is_creation`` is ``True`` for a
                creation operator and ``False`` for an annihilation operator.

        Note:
            Raw fermionic strings store an ordered product rather than creation and annihilation
            bitsets. This check scans the operator product, so it scales linearly with the string
            length and only reports whether the requested operator appears at least once.
        """
        mode, is_creation = item
        if is_creation is not True and is_creation is not False:
            raise KeyError(is_creation)
        return any(
            op_mode == mode and op_is_creation is is_creation
            for op_mode, op_is_creation in self.get_ops()
        )

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, ignoring the scalar sign.

        The adjoint of a raw ladder-operator product reverses the operator order and swaps each
        creation operator with an annihilation operator.
        """
        self.set([(mode, not is_creation) for mode, is_creation in reversed(self.get_ops())])

    def daggered(self) -> Self:
        """Return the adjoint of ``self``, ignoring the scalar sign."""
        out = self.clone()
        out.dagger()
        return out

    @overload
    def __mul__(self, rhs: String) -> Term[float]: ...

    @overload
    def __mul__(self, rhs: CoeffT) -> Term[CoeffT]: ...

    def __mul__(self, rhs: String | CoeffT) -> Term[float] | Term[CoeffT]:
        """Multiplication of ``self`` by ``rhs``.

        Multiplication by a scalar returns a term. Multiplication by another raw string returns a
        single real term whose string is the concatenated ladder-operator product.
        """
        from zixy.fermion.operator.general._terms import RealTerm, get_term_type  # noqa: PLC0415

        if isinstance(rhs, Coeff):
            scalar_term_type = get_term_type(type(rhs))
            return scalar_term_type.from_cmpnt_coeff(self, rhs)
        if not isinstance(rhs, String):
            return NotImplemented
        if self.modes != rhs.modes:
            raise ValueError("Cannot multiply strings defined over different modes.")
        string = String(self.modes, self.get_ops() + rhs.get_ops())
        return RealTerm.from_cmpnt_coeff(string, 1.0)


class Strings(OperatorStrings[ImplT, SpecT, ElemT]):
    """A collection of raw fermionic ladder-operator strings.

    An array-like container of mode-based raw fermionic strings that may be an owning instance
    referencing a contiguous Rust-bound data object, or a view on a slice of the elements in
    another collection.
    """

    cmpnt_type = String
    _set_type: type[StringSet]

    def __init__(self, modes: int | Modes = 0, n: int = 0, max_len: int = 0):
        modes = _as_modes(modes)
        Cmpnts.__init__(self, self.cmpnt_type.impl_type(modes, max_len))
        self.resize(n)

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


class StringSet(OperatorStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique raw fermionic ladder-operator strings.

    A set-like container of mode-based raw fermionic strings that may be used to store unique
    strings and perform set-like operations on them.
    """

    cmpnts_type = Strings

    def __init__(self, modes: int | Modes = 0, max_len: int = 0):
        CmpntSet.__init__(self, self.cmpnts_type(modes, max_len=max_len)._impl)


Strings._set_type = StringSet
