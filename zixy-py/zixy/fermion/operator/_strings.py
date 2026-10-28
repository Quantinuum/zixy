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

"""Base fermionic ladder-operator strings and collections of such strings.

Ladder-operator strings represent products of creation and annihilation
operators, acting on a register of fermionic modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from zixy._zixy import FermionArray

from abc import abstractmethod

from zixy.fermion._strings import (
    String as FermionString,
    Strings as FermionStrings,
    StringSet as FermionStringSet,
)

type LadderOp = tuple[int, bool]


class String[ImplT: FermionArray, SpecT, ElemT](FermionString[ImplT, SpecT, ElemT]):
    """A fermionic ladder-operator string."""

    def __repr__(self) -> str:
        """Return a sparse-string representation of ``self``."""
        return self._impl.string_to_string(self.index)

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
        """The creation mode indices."""
        return self.get_sets()[0]

    @property
    def annihilations(self) -> list[int]:
        """The annihilation mode indices."""
        return self.get_sets()[1]

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, ignoring the scalar sign."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``, ignoring the scalar sign."""
        pass


class Strings[ImplT: FermionArray, SpecT, ElemT](FermionStrings[ImplT, SpecT, ElemT]):
    """A collection of fermionic ladder-operator strings."""

    string_type: type[String[ImplT, SpecT, ElemT]]


class StringSet[ImplT: FermionArray, SpecT, ElemT](FermionStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique fermionic ladder-operator strings."""

    strings_type: type[Strings[ImplT, SpecT, ElemT]]
