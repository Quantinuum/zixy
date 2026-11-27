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

"""String sequences for qubit states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from zixy._zixy import Qubits
from zixy.container.serialization import check_schema_version
from zixy.qubit import state as zqs
from zixy.qubit._strings import Strings as StringsBase
from zixy.qubit.state._string import ElemT, ImplT, SpecT, String

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer


class Strings(StringsBase[ImplT, SpecT, ElemT]):
    """A collection of state strings.

    An array-like container of qubit-based state strings that may be an owning instance referencing
    a contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type = String

    @classmethod
    def _set_type(cls) -> type[zqs.StringSet]:
        """Return the string set class paired with this collection."""
        return zqs.StringSet

    def get_sets(self) -> tuple[set[int], ...]:
        """Get the strings as a list of sets of the indices of bits with value 1."""
        return tuple(self[i].get_set() for i in range(len(self)))

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the visible strings, and the full qubit
            register.
        """
        data = super().to_dict()
        data["family"] = "qubit.state"
        data["space"] = {"qubits": self.qubits.get_inds()}
        data["strings"] = [string.get_list() for string in self]
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
        check_schema_version(source)
        space = Qubits.from_inds(source["space"]["qubits"])
        rows = source["strings"]
        out = cls(space, size=len(rows))
        for i, row in enumerate(rows):
            if len(row) != len(space):
                raise ValueError("Serialized string length must match the space.")
            if row:
                out[i].set(row)
        return out
