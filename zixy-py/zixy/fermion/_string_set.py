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

"""String sets for fermionic systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zixy._zixy import Modes
from zixy.container.string_set import StringSet as BaseStringSet

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zixy._zixy import FermionArray
    from zixy.container.string import String as BaseString
    from zixy.fermion._strings import Strings


class StringSet[ImplT: FermionArray, SpecT, ElemT](BaseStringSet[ImplT, SpecT]):
    """A collection of unique fermionic strings.

    A set-like container of mode-based strings that may be used to store unique strings and
    perform set-like operations on them.
    """

    strings_type: type[Strings[ImplT, SpecT, ElemT]]

    def __init__(self, modes: int | Modes = 0):
        """Initialize the string set.

        Args:
            modes: The mode space or number of modes.
        """
        BaseStringSet.__init__(
            self,
            self.strings_type(Modes.from_count(modes) if isinstance(modes, int) else modes)._impl,
        )

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
            Duplicate strings appear only once.
        """
        return super().from_iterable(source, modes)
