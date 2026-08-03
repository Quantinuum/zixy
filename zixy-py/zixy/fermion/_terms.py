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

"""Base fermionic mode-based terms and collections of such terms."""

from __future__ import annotations

from typing import Generic, cast

from zixy._zixy import Modes
from zixy.container.cmpnts import SpecT
from zixy.container.coeffs import CoeffT, get_coeffs_type
from zixy.container.data import TermData
from zixy.container.terms import (
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.fermion._strings import ElemT, ImplT, String, Strings


class Term(Generic[ImplT, SpecT, CoeffT, ElemT], TermBase[ImplT, SpecT, CoeffT]):
    """A term consisting of a fermionic string and a coefficient.

    A single mode-based term consisting of a string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    cmpnts_type: type[Strings[ImplT, SpecT, ElemT]]

    def __init__(self, modes: int | Modes = 0, source: SpecT | None = None):
        """Initialize the term.

        Args:
            modes: The mode space or number of modes.
            source: The term specifier to use for initial value.
        """
        cmpnts = self.cmpnts_type(modes, 1)
        coeffs = get_coeffs_type(self.coeff_type).from_size(1)
        data = TermData(cmpnts, coeffs)
        TermBase.__init__(self, data)
        self.set(source)

    @property
    def string(self) -> String[ImplT, SpecT, ElemT]:
        """Get the string component of the term."""
        return cast(String[ImplT, SpecT, ElemT], self.cmpnt)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.string.modes


class Terms(Generic[ImplT, SpecT, CoeffT, ElemT], TermsBase[ImplT, SpecT, CoeffT]):
    """A collection of terms consisting of fermionic strings and coefficients."""

    term_type: type[Term[ImplT, SpecT, CoeffT, ElemT]]

    def __init__(self, modes: int | Modes = 0, n: int = 0):
        """Initialize the term array.

        Args:
            modes: The mode space or number of modes.
            n: The number of items to initialize the array with.
        """
        cmpnts = self.term_type.cmpnts_type(modes, n)
        coeffs = get_coeffs_type(self.term_type.coeff_type).from_size(n)
        TermsBase.__init__(self, TermData(cmpnts, coeffs))

    @property
    def strings(self) -> Strings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(Strings[ImplT, SpecT, ElemT], self.cmpnts)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.strings.modes


class TermSet(Generic[ImplT, SpecT, CoeffT, ElemT], TermSetBase[ImplT, SpecT, CoeffT]):
    """A collection of unique terms consisting of fermionic strings and coefficients."""

    terms_type: type[Terms[ImplT, SpecT, CoeffT, ElemT]]

    def __init__(self, modes: int | Modes = 0):
        """Initialize the term set.

        Args:
            modes: The mode space or number of modes.
        """
        TermSetBase.__init__(self, self.terms_type(modes))

    @property
    def strings(self) -> Strings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(Strings[ImplT, SpecT, ElemT], self._impl._cmpnts)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.strings.modes


class TermSum(TermSet[ImplT, SpecT, CoeffT, ElemT], TermSumBase[ImplT, SpecT, CoeffT]):
    """A sum of terms consisting of fermionic strings and coefficients."""

    def __init__(self, modes: int | Modes = 0):
        """Initialize the term set.

        Args:
            modes: The mode space or number of modes.
        """
        TermSumBase.__init__(self, self.terms_type(modes))
