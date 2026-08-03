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

"""Base fermionic ladder-operator string components and collections of such strings.

Ladder-operator strings are components representing products of creation and annihilation
operators, acting on a register of fermionic modes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeAlias, TypeVar

from typing_extensions import Self

from zixy.container.cmpnts import SpecT
from zixy.fermion._strings import (
    ImplT,
    String as FermionString,
    Strings as FermionStrings,
    StringSet as FermionStringSet,
)

LadderOp: TypeAlias = tuple[int, bool]
ElemT = TypeVar("ElemT")


def parse_ladder_product(source: str) -> list[LadderOp]:
    """Parse a single fermionic ladder-operator product."""
    source = source.strip()
    if not source:
        return []
    out: list[LadderOp] = []
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


class String(Generic[ImplT, SpecT, ElemT], FermionString[ImplT, SpecT, ElemT]):
    """A fermionic ladder-operator string."""

    def __repr__(self) -> str:
        """Return a sparse-string representation of ``self``."""
        return self._impl.cmpnt_to_string(self.index)

    @abstractmethod
    def get_ops(self) -> list[LadderOp]:
        """Get the ladder-operator product as ``(mode, is_creation)`` pairs."""
        pass

    @abstractmethod
    def get_sets(self) -> tuple[list[int], list[int]]:
        """Get the creation and annihilation mode lists."""
        pass

    @abstractmethod
    def __getitem__(self, item: LadderOp) -> bool:
        """Return whether an operator is present on a mode.

        Args:
            item: Pair of ``(mode, is_creation)``, where ``is_creation`` is ``True`` for a
                creation operator and ``False`` for an annihilation operator.
        """
        pass

    @property
    def creations(self) -> list[int]:
        """Get the creation mode indices."""
        return self.get_sets()[0]

    @property
    def annihilations(self) -> list[int]:
        """Get the annihilation mode indices."""
        return self.get_sets()[1]

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, ignoring the scalar sign."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``, ignoring the scalar sign."""
        pass


class Strings(Generic[ImplT, SpecT, ElemT], FermionStrings[ImplT, SpecT, ElemT]):
    """A collection of fermionic ladder-operator strings."""

    pass


class StringSet(Generic[ImplT, SpecT, ElemT], FermionStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique fermionic ladder-operator strings."""

    pass
