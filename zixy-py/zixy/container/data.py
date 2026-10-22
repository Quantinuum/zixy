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

"""Data for terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy.container.coeffs import Coeffs, ComplexSign, Sign
from zixy.container.strings import String, Strings
from zixy.utils import slice_single_item

if TYPE_CHECKING:
    from zixy._zixy import ImplArray


class TermData[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)]:
    """Data for terms, consisting of an array of strings and a vector of coefficients."""

    _coeffs: Coeffs[C]
    _strings: Strings[ImplT, SpecT]

    def __init__(self, strings: Strings[ImplT, SpecT], coeffs: Coeffs[C]):
        """Initialize the data.

        Args:
            strings: The string array.
            coeffs: The coefficient vector.
        """
        if not isinstance(strings, Strings):
            raise TypeError(f"strings must be a Strings, got {type(strings)}")
        if not isinstance(coeffs, Coeffs):
            raise TypeError(f"coeffs must be a Coeffs, got {type(coeffs)}")
        self._coeffs = coeffs
        self._strings = strings

    @property
    def strings(self) -> Strings[ImplT, SpecT]:
        """The strings of ``self``."""
        return self._strings

    @property
    def coeffs(self) -> Coeffs[C]:
        """The coefficients of ``self``."""
        return self._coeffs

    @property
    def strings_type(self) -> type[Strings[ImplT, SpecT]]:
        """The string container type of ``self``."""
        return type(self._strings)

    @property
    def string_type(self) -> type[String[ImplT, SpecT]]:
        """The string type of ``self``."""
        return self._strings.string_type

    @property
    def coeffs_type(self) -> type[Coeffs[C]]:
        """The coefficient container type of ``self``."""
        return type(self._coeffs)

    @property
    def coeff_type(self) -> type[C]:
        """The coefficient type of ``self``."""
        return self._coeffs.coeff_type

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        assert len(self.coeffs) == len(self.strings)
        return len(self.coeffs)

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.strings == other.strings and self.coeffs == other.coeffs

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return ", ".join(f"({c}, {s})" for s, c in zip(self.strings, self.coeffs, strict=False))

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of ``self``."""
        return type(self)(self.strings._empty_clone(), self.coeffs._empty_clone())

    def clone(self, s: int | slice = slice(None)) -> Self:
        """Return a deep copy of ``self``.

        Args:
            s: The slice of the containers to clone.

        Returns:
            A clone of ``self`` with the specified slice.
        """
        if isinstance(s, int):
            return self.clone(slice_single_item(slice(None), s, len(self)))
        cls = type(self)
        strings = self.strings[s].clone()
        coeffs = self.coeffs_type.from_view(self.coeffs[s])
        return cls(strings, coeffs)

    def resize(self, n: int) -> None:
        """Resize the underlying containers.

        Args:
            n: The new size of the containers.

        Raises:
            ValueError: If the container is a view.

        Note:
            This method operates in-place.
        """
        self.strings.resize(n)
        self.coeffs.resize(n)
