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

"""String sequences for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from zixy._zixy import FermionSprings, Modes
from zixy.container.serialization import check_schema_version
from zixy.fermion._string import _deserialize_modes, _serialize_modes
from zixy.fermion.operator import normal as zfon
from zixy.fermion.operator._strings import Strings as OperatorStrings
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT, String

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer


class Strings(OperatorStrings[ImplT, SpecT, ElemT]):
    """A collection of normal-ordered fermionic ladder-operator strings.

    An array-like container of mode-based normal-ordered fermionic strings that may be an owning
    instance referencing a contiguous Rust-bound data object, or a view on a slice of the elements
    in another collection.
    """

    string_type = String

    @classmethod
    def _set_type(cls) -> type[zfon.StringSet]:
        """Return the string set class paired with this collection."""
        return zfon.StringSet

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
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        if not source.strip():
            out = cls(modes or 0)
            out.resize(0)
            return out
        return cls._create(cls.string_type.impl_type(modes, FermionSprings(source)))

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the visible strings, and the full mode
            space.
        """
        data = super().to_dict()
        data["family"] = "fermion.operator.normal"
        data["space"] = _serialize_modes(self.modes)
        data["strings"] = [[list(indices) for indices in string.get_sets()] for string in self]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            A new owning instance of ``cls`` parsed from ``source``, with storage independent of
            the source dictionary and any existing containers.

        Examples:
            Each row in ``strings`` contains two lists: creation indices followed by
            annihilation indices. The outer list contains one row per string. For
            ``F0^ F1`` over two unlabelled modes:

            >>> import zixy.fermion.operator.normal as zfon
            >>> from zixy import __version__
            >>> from zixy.container.base import __schema_version__
            >>> data = {
            ...     "schema_version": __schema_version__,
            ...     "package_version": __version__,
            ...     "family": "fermion.operator.normal",
            ...     "container": "Strings",
            ...     "space": {"modes": 2, "spin_ordering": None},
            ...     "strings": [[[0], [1]]],
            ... }
            >>> str(zfon.Strings.from_dict(data))
            'F0^ F1'
        """
        check_schema_version(source)
        space = _deserialize_modes(source["space"])
        rows = source["strings"]
        out = cls(space, size=len(rows))
        for i, (cre, ann) in enumerate(rows):
            if cre or ann:
                out[i].set((cre, ann))
        return out
