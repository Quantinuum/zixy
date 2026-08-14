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

"""Normal-ordered fermionic ladder-operator string components and collections of such strings.

Normal-ordered fermionic strings are components representing products of creation operators
followed by annihilation operators, acting on a register of fermionic modes.

The structure of this module parallels that of :mod:`~zixy.container.cmpnts`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, overload

from typing_extensions import Self

from zixy._zixy import Modes, NormalFermionOperatorArray
from zixy.container.coeffs import Coeff, CoeffT, Sign
from zixy.fermion.operator._strings import (
    LadderOp,
    String as OperatorString,
    Strings as OperatorStrings,
    StringSet as OperatorStringSet,
    parse_ladder_product,
    parse_term_source,
)

if TYPE_CHECKING:
    from zixy.fermion.operator.normal._terms import RealTermSum, Term

StringSpec: TypeAlias = str | tuple[Sequence[int] | Sequence[bool], Sequence[int] | Sequence[bool]]
ElemT = tuple[list[int], list[int]]
SpecT = StringSpec
ImplT = NormalFermionOperatorArray


def _default_modes(source: SpecT) -> Modes:
    """Construct the default modes for a string specifier."""
    if isinstance(source, str):
        max_mode = -1
        for mode, _ in parse_ladder_product(source):
            max_mode = max(max_mode, mode)
        return Modes.from_count(max_mode + 1)
    cre, ann = source
    if not cre and not ann:
        return Modes.from_count(0)
    if all(isinstance(x, bool) for x in cre) and all(isinstance(x, bool) for x in ann):
        return Modes.from_count(max(len(cre), len(ann)))
    return Modes.from_count(max((*cre, *ann), default=-1) + 1)


class String(OperatorString[ImplT, SpecT, ElemT]):
    """A normal-ordered fermionic ladder-operator string.

    A single mode-based normal-ordered fermionic string that may be an owning instance referencing
    a single element in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _clear_spec = ""

    @staticmethod
    def _get_default_modes(source: SpecT) -> Modes:
        """Get the default modes for this string type based on a string specifier."""
        return _default_modes(source)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` by parsing an input string.

        Args:
            source: String to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        return cls(modes, source)

    def set(self, source: SpecT | String) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new normal-ordered string.

        Note:
            This method operates in-place.
        """
        if isinstance(source, String):
            self._set_copy(source)
        elif isinstance(source, str):
            if not source.strip():
                self._impl.cmpnt_clear(self.index)
                return
            cmpnts, signs = self._impl.from_ladder_product(self.modes, parse_ladder_product(source))
            if len(cmpnts) != 1 or signs[0] != 1:
                raise ValueError(
                    "Fermion string does not normal-order to exactly one positive component."
                )
            self._impl.cmpnt_copy_external(self.index, cmpnts, 0)
        elif isinstance(source, tuple) and len(source) == 2:
            cre, ann = source
            if all(isinstance(x, bool) for x in cre) and all(isinstance(x, bool) for x in ann):
                self._impl.cmpnt_set_from_lists(
                    self.index, [bool(x) for x in cre], [bool(x) for x in ann]
                )
            else:
                self._impl.cmpnt_set_from_sets(self.index, set(cre), set(ann))
        else:
            self.raise_spec_type_error(source)

    def get_ops(self) -> list[LadderOp]:
        """Get the normal-ordered ladder-operator product as ``(mode, is_creation)`` pairs."""
        cre, ann = self.get_sets()
        return [(i, True) for i in cre] + [(i, False) for i in ann]

    def get_sets(self) -> tuple[list[int], list[int]]:
        """Get the creation and annihilation mode sets."""
        return self._impl.cmpnt_get_sets(self.index)

    def __getitem__(self, item: LadderOp) -> bool:
        """Return whether an operator is present on a mode.

        Args:
            item: Pair of ``(mode, is_creation)``, where ``is_creation`` is ``True`` for a
                creation operator and ``False`` for an annihilation operator.
        """
        mode, is_creation = item
        if not isinstance(is_creation, bool):
            raise KeyError(is_creation)
        if is_creation:
            return self._impl.cmpnt_get_cre(self.index, mode)
        return self._impl.cmpnt_get_ann(self.index, mode)

    def __setitem__(self, item: LadderOp, value: bool) -> None:
        """Set whether an operator is present on a mode.

        Args:
            item: Pair of ``(mode, is_creation)``, where ``is_creation`` is ``True`` for a
                creation operator and ``False`` for an annihilation operator.
            value: Whether the operator is present.
        """
        mode, is_creation = item
        if not isinstance(is_creation, bool):
            raise KeyError(is_creation)
        if is_creation:
            self._impl.cmpnt_set_cre(self.index, mode, value)
        self._impl.cmpnt_set_ann(self.index, mode, value)

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, ignoring the scalar sign."""
        cre, ann = self.get_sets()
        self.set((ann, cre))

    def daggered(self) -> Self:
        """Return the adjoint of ``self``, ignoring the scalar sign."""
        out = self.clone()
        out.dagger()
        return out

    @overload
    def __mul__(self, rhs: String) -> RealTermSum: ...

    @overload
    def __mul__(self, rhs: CoeffT) -> Term[CoeffT]: ...

    def __mul__(self, rhs: String | CoeffT) -> RealTermSum | Term[CoeffT]:
        """Multiplication of ``self`` by ``rhs``.

        Multiplication by a scalar returns a term. Multiplication by another normal-ordered string
        returns a term sum, since normal-ordering can produce zero, one, or multiple terms.
        """
        from zixy.fermion.operator.normal._terms import RealTermSum, get_term_type  # noqa: PLC0415

        if isinstance(rhs, Coeff):
            scalar_term_type = get_term_type(type(rhs))
            return scalar_term_type.from_cmpnt_coeff(self, rhs)
        if not isinstance(rhs, String):
            return NotImplemented
        impl, signs = self._impl.cmpnt_mul(self.index, rhs._impl, rhs.index)
        out = RealTermSum(self.modes)
        for i in range(len(impl)):
            out += RealTermSum.terms_type.term_type.from_cmpnt_coeff(
                Strings._create(impl)[i], float(int(Sign(signs[i])))
            )
        return out


class Strings(OperatorStrings[ImplT, SpecT, ElemT]):
    """A collection of normal-ordered fermionic ladder-operator strings.

    An array-like container of mode-based normal-ordered fermionic strings that may be an owning
    instance referencing a contiguous Rust-bound data object, or a view on a slice of the elements
    in another collection.
    """

    cmpnt_type = String
    _set_type: type[StringSet]

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` by parsing an input string.

        Args:
            source: String to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        terms = parse_term_source(source)
        if modes is None:
            modes = _default_modes(" ".join(cmpnt for cmpnt, _ in terms))
        out = cls(modes)
        out.append_iterable(cmpnt for cmpnt, _ in terms)
        return out

    @overload
    def __getitem__(self, indexer: int) -> String: ...
    @overload
    def __getitem__(self, indexer: slice) -> Self: ...
    def __getitem__(self, indexer: int | slice) -> String | Self:
        """Get the element or elements selected by ``indexer``.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by ``indexer``.
        """
        return super().__getitem__(indexer)  # type: ignore[return-value]


class StringSet(OperatorStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique normal-ordered fermionic ladder-operator strings.

    A set-like container of mode-based normal-ordered fermionic strings that may be used to store
    unique strings and perform set-like operations on them.
    """

    cmpnts_type = Strings


Strings._set_type = StringSet
