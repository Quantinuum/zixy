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
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet

if TYPE_CHECKING:
    from zixy.fermion.operator._terms import TermRegistry

StringSpec: TypeAlias = (
    None | str | tuple[Sequence[int] | Sequence[bool], Sequence[int] | Sequence[bool]]
)
ImplT = NormalFermionOperatorArray


def _default_modes(source: StringSpec = None) -> Modes:
    """Construct the default modes for a string specifier."""
    if source is None:
        return Modes.from_count(0)
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


def parse_ladder_product(source: str) -> list[tuple[int, bool]]:
    """Parse a single fermionic ladder-operator product."""
    source = source.strip()
    if not source:
        return []
    out: list[tuple[int, bool]] = []
    for token in source.split():
        is_creation = token.endswith("^")
        digits = token[1:-1] if is_creation else token[1:]
        if not token.startswith("F") or not digits or not digits.isdecimal():
            raise ValueError(
                f'"{token}" is not a valid fermionic ladder operator in a sparse string.'
            )
        out.append((int(digits), is_creation))
    return out


def _split_top_level_commas(source: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for i, char in enumerate(source):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Mismatched brackets.")
        elif char == "," and depth == 0:
            parts.append(source[start:i].strip())
            start = i + 1
    if depth:
        raise ValueError("Mismatched brackets.")
    tail = source[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_term_source(source: str) -> list[tuple[str, str | None]]:
    """Parse comma-delimited term strings into component and optional coefficient text."""
    out: list[tuple[str, str | None]] = []
    for part in _split_top_level_commas(source):
        if part.startswith("(") and ")" in part:
            close = part.rfind(")")
            inner = part[1:close]
            if "," not in inner:
                raise ValueError(f'"{part}" is ill-formed')
            coeff, cmpnt = inner.split(",", 1)
            out.append((cmpnt.strip(), coeff.strip()))
        else:
            out.append((part, None))
    return out


class String(Cmpnt[ImplT, StringSpec]):
    """A normal-ordered fermionic ladder-operator string.

    A single mode-based normal-ordered fermionic string that may be an owning instance referencing
    a single element in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _term_registry: TermRegistry

    def __init__(self, modes: int | Modes | None = None, source: StringSpec = None):
        if modes is None:
            modes = _default_modes(source)
        elif isinstance(modes, int):
            modes = Modes.from_count(modes)
        impl = self.impl_type(modes)
        impl.resize(1)
        super().__init__(impl)
        if source is not None:
            self.set(source)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

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
            source: Specification for the new normal-ordered string.

        Note:
            This method operates in-place.
        """
        if source is None:
            self._impl.cmpnt_clear(self.index)
        elif isinstance(source, String):
            if self._impl.same_as(source._impl):
                self._impl.cmpnt_copy_internal(self.index, source.index)
            else:
                self._impl.cmpnt_copy_external(self.index, source._impl, source.index)
        elif isinstance(source, str):
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

    def get_sets(self) -> tuple[list[int], list[int]]:
        """Get the creation and annihilation mode sets."""
        return self._impl.cmpnt_get_sets(self.index)

    @property
    def creations(self) -> list[int]:
        """Get the creation mode indices."""
        return self.get_sets()[0]

    @property
    def annihilations(self) -> list[int]:
        """Get the annihilation mode indices."""
        return self.get_sets()[1]

    def __getitem__(self, item: tuple[str, int]) -> bool:
        """Return whether a creation or annihilation operator is present on a mode."""
        part, mode = item
        if part in {"cre", "^", "creation"}:
            return self._impl.cmpnt_get_cre(self.index, mode)
        if part in {"ann", "", "annihilation"}:
            return self._impl.cmpnt_get_ann(self.index, mode)
        raise KeyError(part)

    def __setitem__(self, item: tuple[str, int], value: bool) -> None:
        """Set whether a creation or annihilation operator is present on a mode."""
        part, mode = item
        if part in {"cre", "^", "creation"}:
            self._impl.cmpnt_set_cre(self.index, mode, value)
        elif part in {"ann", "", "annihilation"}:
            self._impl.cmpnt_set_ann(self.index, mode, value)
        else:
            raise KeyError(part)

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, ignoring the scalar sign."""
        cre, ann = self.get_sets()
        self.set((ann, cre))

    def daggered(self) -> Self:
        """Return the adjoint of ``self``, ignoring the scalar sign."""
        out = self.clone()
        out.dagger()
        return out


class Strings(Cmpnts[ImplT, StringSpec]):
    """A collection of normal-ordered fermionic ladder-operator strings.

    An array-like container of mode-based normal-ordered fermionic strings that may be an owning
    instance referencing a contiguous Rust-bound data object, or a view on a slice of the elements
    in another collection.
    """

    cmpnt_type = String
    _set_type: type[StringSet]

    def __init__(self, modes: int | Modes = 0, n: int = 0):
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        super().__init__(self.cmpnt_type.impl_type(modes))
        self.resize(n)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

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
        return super().__getitem__(indexer)  # type: ignore[return-value]


class StringSet(CmpntSet[ImplT, StringSpec]):
    """A collection of unique normal-ordered fermionic ladder-operator strings.

    A set-like container of mode-based normal-ordered fermionic strings that may be used to store
    unique strings and perform set-like operations on them.
    """

    cmpnts_type = Strings

    def __init__(self, modes: int | Modes | ImplT = 0):
        if isinstance(modes, self.cmpnts_type.cmpnt_type.impl_type):
            CmpntSet.__init__(self, modes)
            return
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        CmpntSet.__init__(self, self.cmpnts_type(modes)._impl)


Strings._set_type = StringSet
