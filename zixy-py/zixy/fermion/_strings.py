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

"""String sequences for fermionic systems."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from zixy._zixy import Modes
from zixy.container.strings import Strings as BaseStrings

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zixy._zixy import FermionArray
    from zixy.container.string import String as BaseString
    from zixy.fermion._string import String
    from zixy.fermion._string_set import StringSet


class Strings[ImplT: FermionArray, SpecT, ElemT](BaseStrings[ImplT, SpecT]):
    """A collection of fermionic strings.

    An array-like container of mode-based strings that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type: type[String[ImplT, SpecT, ElemT]]

    def __init__(self, modes: int | Modes = 0, *, size: int = 0):
        """Initialize the string array.

        Args:
            modes: The mode space or number of modes.
            size: Number of default elements with which to create the instance.
        """
        super().__init__(
            self.string_type.impl_type(Modes.from_count(modes) if isinstance(modes, int) else modes)
        )
        self.resize(size)

    @classmethod
    def from_iterable(
        cls, source: Iterable[SpecT | BaseString[ImplT, SpecT]], modes: int | Modes = 0
    ) -> Self:
        """Create an owning collection from an iterable of strings.

        Args:
            source: Iterable of compatible strings or string specifiers.
            modes: The mode space or number of modes.

        Returns:
            An independent owning instance of ``cls``.
        """
        return super().from_iterable(source, modes)

    @classmethod
    @abstractmethod
    def _set_type(cls) -> type[StringSet[ImplT, SpecT, ElemT]]:
        """Return the string set class paired with this collection."""

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self._impl.modes
