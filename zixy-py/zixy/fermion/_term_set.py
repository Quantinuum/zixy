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

"""Term sets for fermionic systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import Modes  # noqa: TC001 (required for sphinx)
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.term_set import TermSet as TermSetBase
from zixy.fermion._string import _check_modes_compatibility

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zixy._zixy import FermionArray
    from zixy.container.string import String as BaseString
    from zixy.container.term import Term as TermBase, TermSpecT
    from zixy.fermion._terms import Terms


class TermSet[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
    TermSetBase[ImplT, SpecT, C]
):
    """A collection of unique terms consisting of fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[ImplT, SpecT, C, ElemT]]

    def __init__(self, modes: int | Modes = 0):
        """Initialize the term set.

        Args:
            modes: The mode space or number of modes.
        """
        TermSetBase.__init__(self, self.terms_type(modes))

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
            The last coefficient is retained for duplicate strings.
        """
        out = cls(modes)
        out.insert_iterable(source)
        return out

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self._string_set._impl.modes

    def _check_string(self, value: BaseString[ImplT, SpecT]) -> None:
        """Hook to check whether a string is valid for insertion into ``self``."""
        if not isinstance(value, self.terms_type.term_type.strings_type.string_type):
            raise TypeError(
                f"Expected a {self.terms_type.term_type.strings_type.string_type.__name__} "
                "instance, "
                f"got {type(value).__name__}."
            )
        _check_modes_compatibility(self.modes, value.modes)

    def _check_term(self, value: TermBase[ImplT, SpecT, C]) -> None:
        """Hook to check whether a term is defined over the same modes as ``self``."""
        if not isinstance(value, self.terms_type.term_type):
            raise TypeError(
                f"Expected a {self.terms_type.term_type.__name__} instance, got "
                f"{type(value).__name__}."
            )
        _check_modes_compatibility(self.modes, value.modes)
