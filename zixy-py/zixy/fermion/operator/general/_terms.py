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

"""Term sequences for general fermionic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy._zixy import GeneralFermionOperatorArray
from zixy.container.base import requires_ownership
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms, Terms as TermsBase
from zixy.fermion.operator._terms import Terms as OperatorTerms
from zixy.fermion.operator.general._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.operator.general._term import (
    ComplexSignTerm,
    ComplexTerm,
    RealTerm,
    SignTerm,
    SymbolicTerm,
    _max_string_len_from_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sympy import Symbol

    from zixy._zixy import Modes
    from zixy.container.coeff import Number
    from zixy.container.term import Term as TermBase, TermSpecT
    from zixy.fermion.operator.general._term import Term


class Terms[C: (float, complex, Sign, ComplexSign, Expr)](OperatorTerms[ImplT, SpecT, C, ElemT]):
    """A collection of terms consisting of raw fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type: type[Term[C]]

    def __init__(self, modes: int | Modes = 0, *, size: int = 0, max_string_len: int = 0):
        """Initialize the term array.

        Args:
            modes: The mode space or number of modes.
            size: The number of items to initialize the array with.
            max_string_len: Maximum number of operators per string.
        """
        strings = self.term_type.strings_type(modes, size=size, max_string_len=max_string_len)
        coeffs = self.term_type.coeffs_type(size=size)
        term_data = TermData(strings, coeffs)
        TermsBase.__init__(self, term_data)

    @classmethod
    def from_iterable(
        cls,
        source: Iterable[TermSpecT[ImplT, SpecT, C]],
        modes: int | Modes = 0,
        *,
        max_string_len: int = 0,
    ) -> Self:
        """Create an owning collection from an iterable of terms.

        Args:
            source: Iterable of compatible terms or term specifiers.
            modes: The mode space or number of modes.
            max_string_len: Initial maximum number of operators per string. Known sequences
                reserve the required length before insertion; other iterables grow as needed.

        Returns:
            An independent owning instance of ``cls``.
        """
        out = cls(modes, max_string_len=max_string_len)
        if isinstance(source, Sequence):
            out.strings._impl._reserve_string_length(
                max(map(_max_string_len_from_source, source), default=0),
                warn=False,
            )
        out.extend(source)
        return out

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self.strings.max_string_len

    @requires_ownership
    def append_n(
        self,
        n: int,
        source: TermSpecT[ImplT, SpecT, C] = "",
    ) -> Self:
        """Append a term repeatedly."""
        if n < 0:
            raise ValueError("The number of terms to append must be non-negative.")
        if n == 0:
            return self
        value: TermBase[ImplT, SpecT, C] = self.term_type(
            self.modes, max_string_len=_max_string_len_from_source(source)
        )
        value.set(source)
        self.strings._impl._reserve_string_length(_max_string_len_from_source(value))
        return TermsBase[ImplT, SpecT, C].append_n(self, n, value)


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of raw fermionic strings and sign coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    :class:`~zixy.container.coeff.Sign` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SignTerm


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms consisting of raw fermionic strings and complex sign coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    :class:`~zixy.container.coeff.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class RealTerms(NumericTerms[GeneralFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of raw fermionic strings and real coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and ``float``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm


class ComplexTerms(NumericTerms[GeneralFermionOperatorArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of raw fermionic strings and complex coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    ``complex`` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of raw fermionic strings and symbolic coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.subs(values))
        return SymbolicTerms._create(term_data)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)
