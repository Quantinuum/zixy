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

"""Individual terms and term specifications for fermionic systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import Modes  # noqa: TC001 (required for sphinx)
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.term import Term as TermBase

if TYPE_CHECKING:
    from zixy._zixy import FermionArray
    from zixy.container.term import TermSpecT
    from zixy.fermion._strings import Strings


class Term[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
    TermBase[ImplT, SpecT, C]
):
    """A term consisting of a fermionic string and a coefficient.

    A single mode-based term consisting of a string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    strings_type: type[Strings[ImplT, SpecT, ElemT]]

    def __init__(
        self,
        modes: int | Modes = 0,
        source: TermSpecT[ImplT, SpecT, C] = "",  # type: ignore[assignment]
    ):
        """Initialize the term.

        Args:
            modes: The mode space or number of modes.
            source: The term specifier to use for initial value.
        """
        strings = self.strings_type(modes, size=1)
        coeffs = self.coeffs_type(size=1)
        term_data = TermData(strings, coeffs)
        TermBase.__init__(self, term_data)
        self.set(source)

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self.string.modes
