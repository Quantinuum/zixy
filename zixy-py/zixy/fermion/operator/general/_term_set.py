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

"""Term sets for general fermionic operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy.container.coeff import ComplexSign, Sign
from zixy.container.string import String as BaseString  # noqa: TC001 (runtime type hints)
from zixy.container.term_set import TermSet as TermSetBase
from zixy.fermion.operator._term_set import TermSet as OperatorTermSet
from zixy.fermion.operator.general._string import ElemT, ImplT, SpecT, _string_length
from zixy.fermion.operator.general._term import _max_string_len_from_source
from zixy.fermion.operator.general._terms import (
    ComplexSignTerms,
    ComplexTerms,
    RealTerms,
    SignTerms,
    SymbolicTerms,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sympy import Symbol

    from zixy._zixy import Modes
    from zixy.container.coeff import Number
    from zixy.container.term import Term as TermBase, TermSpecT
    from zixy.fermion.operator.general._terms import Terms


class TermSet[C: (float, complex, Sign, ComplexSign, Expr)](
    OperatorTermSet[ImplT, SpecT, C, ElemT]
):
    """A collection of unique terms consisting of raw fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[C]]

    def __init__(self, modes: int | Modes = 0, *, max_string_len: int = 0):
        """Initialize the term set.

        Args:
            modes: The mode space or number of modes.
            max_string_len: Maximum number of operators per string.
        """
        TermSetBase.__init__(self, self.terms_type(modes, max_string_len=max_string_len))

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
            The last coefficient is retained for duplicate strings.
        """
        out = cls(modes, max_string_len=max_string_len)
        if isinstance(source, Sequence):
            out._string_set._impl._reserve_string_length(
                max(map(_max_string_len_from_source, source), default=0),
                warn=False,
            )
        out.insert_iterable(source)
        return out

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self._string_set._impl.max_string_len

    def _get_working_string(
        self, value: BaseString[ImplT, SpecT] | SpecT
    ) -> BaseString[ImplT, SpecT]:
        """Prepare a lookup string without warning about temporary storage."""
        self._working_term.string._impl._reserve_string_length(_string_length(value), warn=False)
        return super()._get_working_string(value)

    def _get_working_term(
        self,
        value: TermSpecT[ImplT, SpecT, C],
    ) -> TermBase[ImplT, SpecT, C]:
        """Prepare a term without warning about temporary storage."""
        self._working_term.string._impl._reserve_string_length(
            _max_string_len_from_source(value),
            warn=False,
        )
        return TermSetBase._get_working_term(self, value)


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms consisting of raw fermionic strings and sign coefficients.

    A set-like container of mode-based terms with :class:`~zixy.container.coeff.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class ComplexSignTermSet(TermSet[ComplexSign]):
    """Unique raw fermionic strings with complex sign coefficients.

    A set-like container of mode-based terms with :class:`~zixy.container.coeff.ComplexSign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexSignTerms


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of raw fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms with raw fermionic strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms with raw fermionic strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSet:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        out = self.clone()
        out.isubs(values)
        return out

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)
