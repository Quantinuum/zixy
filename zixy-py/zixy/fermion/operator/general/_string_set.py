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

"""String sets for general fermionic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from zixy._zixy import Modes  # noqa: TC001 (runtime type hints)
from zixy.container.string import String as BaseString  # noqa: TC001 (runtime type hints)
from zixy.container.string_set import StringSet as BaseStringSet
from zixy.fermion._string import _serialize_modes
from zixy.fermion.operator._string_set import StringSet as OperatorStringSet
from zixy.fermion.operator.general._string import ElemT, ImplT, SpecT, _string_length
from zixy.fermion.operator.general._strings import Strings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zixy.container.serialization import SerializedContainer


class StringSet(OperatorStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique raw fermionic ladder-operator strings.

    A set-like container of mode-based raw fermionic strings that may be used to store unique
    strings and perform set-like operations on them.
    """

    strings_type = Strings

    def __init__(self, modes: int | Modes = 0, *, max_string_len: int = 0):
        """Initialize the string set.

        Args:
            modes: The mode space or number of modes.
            max_string_len: Maximum number of operators per string.
        """
        BaseStringSet.__init__(self, self.strings_type(modes, max_string_len=max_string_len)._impl)

    @classmethod
    def from_iterable(
        cls,
        source: Iterable[SpecT | BaseString[ImplT, SpecT]],
        modes: int | Modes = 0,
        *,
        max_string_len: int = 0,
    ) -> Self:
        """Create an owning collection from an iterable of strings.

        Args:
            source: Iterable of compatible strings or string specifiers.
            modes: The mode space or number of modes.
            max_string_len: Initial maximum number of operators per string. Known sequences
                reserve the required length before insertion; other iterables grow as needed.

        Returns:
            An independent owning instance of ``cls``.
            Duplicate strings appear only once.
        """
        out = cls(modes, max_string_len=max_string_len)
        if isinstance(source, Sequence):
            out._impl._reserve_string_length(
                max(map(_string_length, source), default=0), warn=False
            )
        out.insert_iterable(source)
        return out

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self._impl.max_string_len

    def _get_working_string(
        self, value: SpecT | BaseString[ImplT, SpecT]
    ) -> BaseString[ImplT, SpecT]:
        """Prepare a lookup or insertion string without warning about temporary storage."""
        self._working_string._impl._reserve_string_length(_string_length(value), warn=False)
        return super()._get_working_string(value)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the unique strings, and the full mode space.
        """
        data = super().to_dict()
        data["family"] = "fermion.operator.general"
        data["space"] = _serialize_modes(self._impl.modes)
        data["strings"] = [[list(op) for op in string.get_ops()] for string in self]
        return data
