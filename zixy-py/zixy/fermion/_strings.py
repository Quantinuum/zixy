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

"""Base fermionic mode-based string components and collections of such strings."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from typing_extensions import Self

from zixy._zixy import Modes
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet, SpecT

if TYPE_CHECKING:
    from zixy._zixy import FermionArray

ElemT = TypeVar("ElemT")
ImplT = TypeVar("ImplT", bound="FermionArray")


def _as_modes(modes: int | Modes) -> Modes:
    """Convert a mode count or mode space to a mode space."""
    return Modes.from_count(modes) if isinstance(modes, int) else modes


class String(Generic[ImplT, SpecT, ElemT], Cmpnt[ImplT, SpecT]):
    """A fermionic string.

    A single mode-based string that may be an owning instance referencing a single element in a
    Rust-bound data object, or a view on an element in another collection.
    """

    @staticmethod
    @abstractmethod
    def _get_default_modes(source: SpecT | None = None) -> Modes:
        """Get the default modes for this string type based on a string specifier."""
        pass

    def __init__(self, modes: int | Modes | None = None, source: SpecT | None = None):
        """Initialize the string.

        Args:
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.
            source: The string specifier to use for default modes and initial value.
        """
        if modes is None:
            modes = self._get_default_modes(source)
        impl = self.impl_type(_as_modes(modes))
        impl.resize(1)
        super().__init__(impl)
        if source is not None:
            self.set(source)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

    def _set_copy(self, source: String[ImplT, SpecT, ElemT]) -> None:
        """Set the value of ``self`` to that of ``source``."""
        if self._impl.same_as(source._impl):
            self._impl.cmpnt_copy_internal(self.index, source.index)
        else:
            self._impl.cmpnt_copy_external(self.index, source._impl, source.index)

    @abstractmethod
    def set(self, source: SpecT | Self | None) -> None:
        """Set the value of the string."""
        pass


class Strings(Generic[ImplT, SpecT, ElemT], Cmpnts[ImplT, SpecT]):
    """A collection of fermionic strings.

    An array-like container of mode-based strings that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    cmpnt_type: type[String[ImplT, SpecT, ElemT]]
    _set_type: type[StringSet[ImplT, SpecT, ElemT]]

    def __init__(self, modes: int | Modes = 0, n: int = 0):
        """Initialize the string array.

        Args:
            modes: The mode space or number of modes.
            n: Number of default elements with which to create the instance.
        """
        super().__init__(self.cmpnt_type.impl_type(_as_modes(modes)))
        self.resize(n)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self._impl.modes

    @overload
    def __getitem__(self, indexer: int) -> String[ImplT, SpecT, ElemT]: ...
    @overload
    def __getitem__(self, indexer: slice) -> Self: ...
    def __getitem__(self, indexer: int | slice) -> String[ImplT, SpecT, ElemT] | Self:
        """Get the element or elements selected by ``indexer``."""
        return super().__getitem__(indexer)  # type: ignore[return-value]

    def filter_unique(self) -> Cmpnts[ImplT, SpecT]:
        """Get a new instance containing the unique strings of ``self``."""
        return self._set_type._create(self._impl).to_cmpnts()


class StringSet(Generic[ImplT, SpecT, ElemT], CmpntSet[ImplT, SpecT]):
    """A collection of unique fermionic strings."""

    cmpnts_type: type[Strings[ImplT, SpecT, ElemT]]

    def __init__(self, modes: int | Modes = 0):
        """Initialize the string set."""
        CmpntSet.__init__(self, self.cmpnts_type(_as_modes(modes))._impl)
