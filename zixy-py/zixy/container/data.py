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

from typing import TYPE_CHECKING, Any, Protocol, Self

from sympy import Expr

from zixy.container.coeffs import Coeffs, ComplexSign, Sign
from zixy.container.strings import String, Strings
from zixy.utils import slice_single_item

if TYPE_CHECKING:
    from zixy._zixy import ImplArray
    from zixy.container.base import _HasCoeffs, _HasCoeffType, _HasStrings, _HasStringType


class _TermData[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](Protocol):
    """A container for term data, consisting of an array of strings and a vector of coefficients."""

    @property
    def strings(self) -> Strings[ImplT, SpecT]: ...

    @property
    def coeffs(self) -> Coeffs[C]: ...

    def __len__(self) -> int: ...

    def _empty_clone(self) -> Self: ...

    def clone(self, s: int | slice = slice(None)) -> Self: ...

    def resize(self, n: int) -> None: ...


class TermData[StringsT: Strings[Any, Any], Cs: Coeffs[Any]]:
    """Data for terms, consisting of an array of strings and a vector of coefficients."""

    _coeffs: Cs
    _strings: StringsT

    def __init__(self, strings: StringsT, coeffs: Cs):
        """Initialize the data.

        Args:
            strings: The string array.
            coeffs: The coefficient vector.

        Raises:
            ValueError: If the string and coefficient containers have different lengths.
        """
        if not isinstance(strings, Strings):
            raise TypeError(f"strings must be a Strings, got {type(strings)}")
        if not isinstance(coeffs, Coeffs):
            raise TypeError(f"coeffs must be a Coeffs, got {type(coeffs)}")
        if len(strings) != len(coeffs):
            raise ValueError(
                f"strings and coefficients must have equal lengths: {len(strings)} != {len(coeffs)}"
            )
        self._coeffs = coeffs
        self._strings = strings

    @property
    def strings(self) -> StringsT:
        """The strings of ``self``."""
        return self._strings

    @property
    def coeffs(self) -> Cs:
        """The coefficients of ``self``."""
        return self._coeffs

    @property
    def strings_type(self) -> type[StringsT]:
        """The string container type of ``self``."""
        return type(self._strings)

    @property
    def string_type[StringT: String[Any, Any]](
        self: _HasStrings[_HasStringType[StringT]],
    ) -> type[StringT]:
        """The concrete string type of ``self``."""
        return self.strings.string_type

    @property
    def coeffs_type(self) -> type[Cs]:
        """The coefficient container type of ``self``."""
        return type(self._coeffs)

    @property
    def coeff_type[C: (float, complex, Sign, ComplexSign, Expr)](
        self: _HasCoeffs[_HasCoeffType[C]],
    ) -> type[C]:
        """The concrete coefficient type of ``self``."""
        return self.coeffs.coeff_type

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
