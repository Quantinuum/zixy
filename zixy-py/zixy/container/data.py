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

from typing import Generic

from typing_extensions import Self

from zixy.container.cmpnts import Cmpnt, Cmpnts, ImplT, SpecT
from zixy.container.coeffs import Coeffs, CoeffT
from zixy.utils import slice_single_item


class TermData(Generic[ImplT, SpecT, CoeffT]):
    """Data for terms, consisting of an array of components and a vector of coefficients."""

    _coeffs: Coeffs[CoeffT]
    _cmpnts: Cmpnts[ImplT, SpecT]

    def __init__(self, cmpnts: Cmpnts[ImplT, SpecT], coeffs: Coeffs[CoeffT]):
        """Initialize the data.

        Args:
            cmpnts: The component array.
            coeffs: The coefficient vector.
        """
        if not isinstance(cmpnts, Cmpnts):
            raise TypeError(f"cmpnts must be a Cmpnts, got {type(cmpnts)}")
        if not isinstance(coeffs, Coeffs):
            raise TypeError(f"coeffs must be a Coeffs, got {type(coeffs)}")
        self._coeffs = coeffs
        self._cmpnts = cmpnts

    @property
    def cmpnts(self) -> Cmpnts[ImplT, SpecT]:
        """Get the components of :param:`self`."""
        return self._cmpnts

    @property
    def coeffs(self) -> Coeffs[CoeffT]:
        """Get the coefficients of :param:`self`."""
        return self._coeffs

    @property
    def cmpnts_type(self) -> type[Cmpnts[ImplT, SpecT]]:
        """Get the component container type of :param:`self`."""
        return type(self._cmpnts)

    @property
    def cmpnt_type(self) -> type[Cmpnt[ImplT, SpecT]]:
        """Get the component type of :param:`self`."""
        return self._cmpnts.cmpnt_type

    @property
    def coeffs_type(self) -> type[Coeffs[CoeffT]]:
        """Get the coefficient container type of :param:`self`."""
        return type(self._coeffs)

    @property
    def coeff_type(self) -> type[CoeffT]:
        """Get the coefficient type of :param:`self`."""
        return self._coeffs.coeff_type

    def __len__(self) -> int:
        """Get the number of elements in :param:`self`."""
        assert len(self.coeffs) == len(self.cmpnts)
        return len(self.coeffs)

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.cmpnts == other.cmpnts and self.coeffs == other.coeffs

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return ", ".join(f"({c}, {s})" for s, c in zip(self.cmpnts, self.coeffs, strict=False))

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return type(self)(self.cmpnts._empty_clone(), self.coeffs._empty_clone())

    def clone(self, s: int | slice = slice(None)) -> Self:
        """Return a deep copy of :param:`self`.

        Args:
            s: The slice of the containers to clone.

        Returns:
            A clone of :param:`self` with the specified slice.
        """
        if isinstance(s, int):
            return self.clone(slice_single_item(slice(None), s, len(self)))
        cls = type(self)
        cmpnts = self.cmpnts[s].clone()
        coeffs = self.coeffs_type.from_view(self.coeffs[s])
        return cls(cmpnts, coeffs)

    def resize(self, n: int) -> None:
        """Resize the underlying containers.

        Args:
            n: The new size of the containers.

        Raises:
            ValueError: If the container is a view.

        Note:
            This method operates in-place.
        """
        self.cmpnts.resize(n)
        self.coeffs.resize(n)
