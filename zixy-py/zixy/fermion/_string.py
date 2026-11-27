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

"""Individual strings for fermionic systems."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from zixy._zixy import Modes
from zixy.container.string import String as BaseString

if TYPE_CHECKING:
    from typing import Any

    from zixy._zixy import FermionArray


def _serialize_modes(modes: Modes) -> dict[str, Any]:
    """Return a JSON-serializable dictionary representation of a mode space.

    Args:
        modes: Mode space to serialize.

    Returns:
        A new dictionary with the total mode count in ``modes`` and the spin-pair ordering
        in ``spin_ordering``: ``None`` for unlabelled modes, ``"major"`` for spin-major
        pairs, or ``"minor"`` for spin-minor pairs.
    """
    return {"modes": len(modes), "spin_ordering": modes.get_spin_ordering()}


def _deserialize_modes(source: dict[str, Any]) -> Modes:
    """Create a mode space from a JSON-serializable dictionary representation.

    Args:
        source: Dictionary containing the total ``modes`` count and ``spin_ordering``.
            The ordering is ``None``, ``"major"``, or ``"minor"``. Spin-paired spaces must
            contain an even number of modes.

    Returns:
        The mode space parsed from ``source``.
    """
    count = source["modes"]
    ordering = source["spin_ordering"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Mode count must be a non-negative integer.")
    if ordering is None:
        return Modes.from_count(count)
    if count % 2:
        raise ValueError("A spin-paired space must have an even number of modes.")
    if ordering == "major":
        return Modes.from_pair_count_spin_major(count // 2)
    if ordering == "minor":
        return Modes.from_pair_count_spin_minor(count // 2)
    raise ValueError(f"Unknown spin-pair ordering: {ordering!r}.")


def _check_modes_compatibility(*modes: Modes) -> None:
    """Check that all mode spaces are compatible."""
    if len(modes) > 1 and any(modes[0] != other for other in modes[1:]):
        raise ValueError("Objects are defined over different modes.")


class String[ImplT: FermionArray, SpecT, ElemT](BaseString[ImplT, SpecT]):
    """A fermionic string.

    A single mode-based string that may be an owning instance referencing a single element in a
    Rust-bound data object, or a view on an element in another collection.
    """

    def __init__(
        self,
        modes: int | Modes | None = None,
        source: SpecT | BaseString[ImplT, SpecT] = "",  # type: ignore[assignment]
    ):
        """Initialize the string.

        Args:
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.
            source: Specification or compatible string to copy, also used to infer default modes.
        """
        if isinstance(source, BaseString):
            if not isinstance(source._impl, self.impl_type):
                self.raise_spec_type_error(source)
            if modes is None:
                modes = source._impl.modes
        elif modes is None:
            modes = self._get_default_modes(source)
        impl = self.impl_type(Modes.from_count(modes) if isinstance(modes, int) else modes)
        impl.resize(1)
        super().__init__(impl)
        self.set(source)

    @staticmethod
    @abstractmethod
    def _get_default_modes(source: SpecT) -> Modes:
        """Get the default modes for this string type based on a string specifier."""
        pass

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self._impl.modes
