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

"""Term sums for fermionic systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import Modes  # noqa: TC001 (required for sphinx)
from zixy.container.term_sum import TermSum as TermSumBase
from zixy.fermion._term_set import TermSet

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zixy._zixy import FermionArray
    from zixy.container.term import TermSpecT


class TermSum[ImplT: FermionArray, SpecT, C: (float, complex, Expr), ElemT](
    TermSet[ImplT, SpecT, C, ElemT], TermSumBase[ImplT, SpecT, C]
):
    """A sum of terms consisting of fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    def __init__(self, modes: int | Modes = 0):
        """Initialize the term set.

        Args:
            modes: The mode space or number of modes.
        """
        TermSumBase.__init__(self, self.terms_type(modes))

    @classmethod
    def from_iterable(
        cls, source: Iterable[TermSpecT[ImplT, SpecT, C]], modes: int | Modes = 0
    ) -> Self:
        """Create an owning collection from an iterable of terms.

        Args:
            source: Iterable of compatible terms or term specifiers.
            modes: The mode space or number of modes.

        Returns:
            An independent owning instance of ``cls``.
            Coefficients of duplicate strings are added together.
        """
        return cls.from_terms(cls.terms_type.from_iterable(source, modes))
