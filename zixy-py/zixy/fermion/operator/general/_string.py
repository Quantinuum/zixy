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

"""Individual strings for general fermionic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, overload

from sympy import Expr

from zixy._zixy import FermionSprings, GeneralFermionOperatorArray, Modes
from zixy.container.coeff import Coeff, ComplexSign, Sign, normalize_scalar
from zixy.container.serialization import check_schema_version
from zixy.container.string import String as BaseString
from zixy.fermion._string import _check_modes_compatibility, _serialize_modes
from zixy.fermion.operator._string import LadderOp, String as OperatorString

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer
    from zixy.fermion.operator.general._term import Term

type StringSpec = str | Sequence[LadderOp]

ElemT = list[LadderOp]
SpecT = StringSpec
ImplT = GeneralFermionOperatorArray


def _string_length(source: BaseString[ImplT, SpecT] | SpecT) -> int:
    """Get the number of operators needed to store a string specifier."""
    if isinstance(source, BaseString):
        if not isinstance(source._impl, ImplT):
            raise TypeError("Source string must be a general fermion operator string.")
        return len(source._impl.string_get_ops(source.index)[0])
    if isinstance(source, str):
        if not source.strip():
            return 0
        springs = FermionSprings(source)
        return max((len(springs.get_ops(i)) for i in range(len(springs))), default=0)
    return len(source)


def _default_modes(source: SpecT) -> Modes:
    """Construct the default modes for a string specifier."""
    if isinstance(source, str):
        return Modes.from_count(
            0 if not source.strip() else FermionSprings(source).default_n_mode()
        )
    ops = list(source)
    return Modes.from_count(max((i for i, _ in ops), default=-1) + 1)


class String(OperatorString[ImplT, SpecT, ElemT]):
    """A raw fermionic ladder-operator string.

    A single mode-based raw fermionic string that may be an owning instance referencing a single
    element in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _clear_spec = ""

    def __init__(
        self,
        modes: int | Modes | None = None,
        source: SpecT | BaseString[ImplT, SpecT] = "",
        *,
        max_string_len: int = 0,
    ):
        """Create a string with a maximum length of ``max_string_len`` operators."""
        if isinstance(source, BaseString):
            if not isinstance(source._impl, self.impl_type):
                self.raise_spec_type_error(source)
            if modes is None:
                modes = source._impl.modes
        elif modes is None:
            modes = _default_modes(source)
        modes = Modes.from_count(modes) if isinstance(modes, int) else modes
        impl = self.impl_type(modes, max_string_len)
        impl._reserve_string_length(_string_length(source), warn=False)
        impl.resize(1)
        BaseString.__init__(self, impl)
        self.set(source)

    @staticmethod
    def _get_default_modes(source: SpecT) -> Modes:
        """Get the default modes for this string type based on a string specifier."""
        return _default_modes(source)

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self._impl.max_string_len

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
        if not isinstance(is_creation, bool):
            raise KeyError(is_creation)
        return any(
            op_mode == mode and op_is_creation is is_creation
            for op_mode, op_is_creation in self.get_ops()
        )

    def get_ops(self) -> list[LadderOp]:
        """Get the raw ladder-operator product as ``(mode, is_creation)`` pairs."""
        modes, adj = self._impl.string_get_ops(self.index)
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

    def _set_spec(self, source: SpecT) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new raw ladder-operator product.

        Note:
            This method operates in-place and preserves the input operator order.
        """
        if isinstance(source, str):
            if not source.strip():
                self._impl.string_clear(self.index)
                return
            springs = FermionSprings(source)
            if len(springs) == 0:
                self._impl.string_clear(self.index)
            elif len(springs) == 1:
                self._impl.string_set_from_spring(self.index, springs, 0)
            else:
                raise ValueError("Input represents more than one string.")
        else:
            ops = list(source)
            mode_inds, adj = zip(*ops, strict=True) if ops else ((), ())
            self._impl.string_set_from_ops(self.index, list(mode_inds), list(adj))

    @overload
    def __mul__(self, rhs: String) -> Term[float]: ...

    @overload
    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[C]: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: String | C
    ) -> Term[float] | Term[C]:
        """Multiplication of ``self`` by ``rhs``.

        Multiplication by a scalar returns a term. Multiplication by another raw string returns a
        single real term whose string is the concatenated ladder-operator product.
        """
        rhs = normalize_scalar(rhs)
        from zixy.fermion.operator.general._term import RealTerm, get_term_type  # noqa: PLC0415

        if isinstance(rhs, Coeff):
            scalar_term_type = get_term_type(type(rhs))
            return scalar_term_type.from_string(self, rhs)
        if not isinstance(rhs, String):
            return NotImplemented
        _check_modes_compatibility(self.modes, rhs.modes)
        string = String(self.modes, self.get_ops() + rhs.get_ops())
        return RealTerm.from_string(string)

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[C]:
        """Multiply a scalar by this operator string."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff):
            return NotImplemented
        return self * lhs

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

    @classmethod
    def from_str(
        cls, source: str, modes: int | Modes | None = None, *, max_string_len: int = 0
    ) -> Self:
        """Create an instance of ``cls`` by parsing an input string.

        Args:
            source: String to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.
            max_string_len: Maximum number of operators per string.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        return cls(modes, source, max_string_len=max_string_len)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the string, and the full mode space.
        """
        data = super().to_dict()
        data["family"] = "fermion.operator.general"
        data["space"] = _serialize_modes(self.modes)
        data["strings"] = [[list(op) for op in self.get_ops()]]
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
        from zixy.fermion.operator.general._strings import Strings  # noqa: PLC0415

        check_schema_version(source)
        if len(source["strings"]) != 1:
            raise ValueError("A serialized String must contain exactly one string.")
        strings = Strings.from_dict(source)
        return cls._create(strings._impl)
