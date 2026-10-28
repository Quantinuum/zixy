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

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import Modes  # noqa: TC001 (required for sphinx)
from zixy.container.coeffs import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.terms import (
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.fermion._strings import _check_modes_compatibility

if TYPE_CHECKING:
    from zixy._zixy import FermionArray
    from zixy.container.strings import String as BaseString
    from zixy.container.terms import TermSpecT
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
        strings = self.strings_type(modes, 1)
        coeffs = self.coeffs_type.from_size(1)
        term_data = TermData(strings, coeffs)
        TermBase.__init__(self, term_data)
        self.set(source)

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self.string.modes


class Terms[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
    TermsBase[ImplT, SpecT, C]
):
    """A collection of terms consisting of fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of strings and coefficients that may
    be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a
    view on a slice of the elements in another collection.
    """

    term_type: type[Term[ImplT, SpecT, C, ElemT]]

    def __init__(self, modes: int | Modes = 0, n: int = 0):
        """Initialize the term array.

        Args:
            modes: The mode space or number of modes.
            n: The number of items to initialize the array with.
        """
        strings = self.term_type.strings_type(modes, n)
        coeffs = self.term_type.coeffs_type.from_size(n)
        term_data = TermData(strings, coeffs)
        TermsBase.__init__(self, term_data)

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self.strings.modes


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

    @property
    def modes(self) -> Modes:
        """The modes corresponding to ``self``."""
        return self.strings.modes

    def _check_term(self, value: TermBase[ImplT, SpecT, C]) -> None:
        """Hook to check whether a term is defined over the same modes as ``self``."""
        if not isinstance(value, self.terms_type.term_type):
            raise TypeError(
                f"Expected a {self.terms_type.term_type.__name__} instance, got "
                f"{type(value).__name__}."
            )
        _check_modes_compatibility(self.modes, value.modes)

    def _check_string(self, value: BaseString[ImplT, SpecT]) -> None:
        """Hook to check whether a string is valid for insertion into ``self``."""
        if not isinstance(value, self.terms_type.term_type.strings_type.string_type):
            raise TypeError(
                f"Expected a {self.terms_type.term_type.strings_type.string_type.__name__} "
                "instance, "
                f"got {type(value).__name__}."
            )
        _check_modes_compatibility(self.modes, value.modes)


class TermSum[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
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
