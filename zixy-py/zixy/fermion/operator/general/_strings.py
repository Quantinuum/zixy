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

"""String sequences for general fermionic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from zixy._zixy import FermionSprings, Modes
from zixy.container.base import requires_ownership
from zixy.container.serialization import check_schema_version
from zixy.container.string import String as BaseString  # noqa: TC001 (runtime type hints)
from zixy.container.strings import Strings as BaseStrings
from zixy.fermion._string import _deserialize_modes, _serialize_modes
from zixy.fermion.operator import general as zfog
from zixy.fermion.operator._strings import Strings as OperatorStrings
from zixy.fermion.operator.general._string import ElemT, ImplT, SpecT, String, _string_length

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zixy.container.serialization import SerializedContainer


class Strings(OperatorStrings[ImplT, SpecT, ElemT]):
    """A collection of raw fermionic ladder-operator strings.

    An array-like container of mode-based raw fermionic strings that may be an owning instance
    referencing a contiguous Rust-bound data object, or a view on a slice of the elements in
    another collection.
    """

    string_type = String

    def __init__(self, modes: int | Modes = 0, *, size: int = 0, max_string_len: int = 0):
        """Initialize the string array.

        Args:
            modes: The mode space or number of modes.
            size: Number of default elements with which to create the instance.
            max_string_len: Maximum number of operators per string.
        """
        modes = Modes.from_count(modes) if isinstance(modes, int) else modes
        BaseStrings.__init__(self, self.string_type.impl_type(modes, max_string_len))
        self.resize(size)

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
        """
        out = cls(modes, max_string_len=max_string_len)
        if isinstance(source, Sequence):
            out._impl._reserve_string_length(
                max(map(_string_length, source), default=0), warn=False
            )
        out.extend(source)
        return out

    @classmethod
    def _set_type(cls) -> type[zfog.StringSet]:
        """Return the string set class paired with this collection."""
        return zfog.StringSet

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self._impl.max_string_len

    @requires_ownership
    def append_n(self, n: int, source: SpecT | BaseString[ImplT, SpecT]) -> Self:
        """Append a string repeatedly."""
        if n < 0:
            raise ValueError("The number of strings to append must be non-negative.")
        if n == 0:
            return self
        value = self.string_type(self.modes, max_string_len=_string_length(source))
        value.set(source)
        self._impl._reserve_string_length(_string_length(value))
        return super().append_n(n, value)

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
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        if not source.strip():
            out = cls(modes or 0, max_string_len=max_string_len)
            out.resize(0)
            return out
        springs = FermionSprings(source)
        if modes is None:
            modes = Modes.from_count(springs.default_n_mode())
        return cls._create(cls.string_type.impl_type(modes, max_string_len, springs))

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the visible strings, and the full mode
            space.
        """
        data = super().to_dict()
        data["family"] = "fermion.operator.general"
        data["space"] = _serialize_modes(self.modes)
        data["strings"] = [[list(op) for op in string.get_ops()] for string in self]
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
            Each row in ``strings`` is an ordered list of ``[mode, creation]`` pairs,
            where ``True`` means creation and ``False`` means annihilation. Operator
            order and repeated operators are preserved. The outer list contains one
            row per string. For ``F0^ F1`` over two unlabelled modes:

            >>> import zixy.fermion.operator.general as zfog
            >>> from zixy import __version__
            >>> from zixy.container.base import __schema_version__
            >>> data = {
            ...     "schema_version": __schema_version__,
            ...     "package_version": __version__,
            ...     "family": "fermion.operator.general",
            ...     "container": "Strings",
            ...     "space": {"modes": 2, "spin_ordering": None},
            ...     "strings": [[[0, True], [1, False]]],
            ... }
            >>> str(zfog.Strings.from_dict(data))
            'F0^ F1'
        """
        check_schema_version(source)
        space = _deserialize_modes(source["space"])
        rows = [[(mode, creation) for mode, creation in row] for row in source["strings"]]
        return cls.from_iterable(rows, space)
