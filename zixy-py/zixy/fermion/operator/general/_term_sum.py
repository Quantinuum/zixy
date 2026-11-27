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

"""Term sums for general fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, overload

from sympy import Expr

from zixy._zixy import GeneralFermionOperatorArray
from zixy.container.coeff import Coeff, _integer_exponent, normalize_scalar
from zixy.container.coeffs import Coeffs, ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.container.term_sum import (
    NumericTermSum,
    SymbolicTermSum as SymbolicTermSumBase,
    TermSum as BaseTermSum,
)
from zixy.container.utils import _positive_power
from zixy.fermion.operator._term_sum import TermSum as OperatorTermSum
from zixy.fermion.operator.general._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.operator.general._strings import Strings
from zixy.fermion.operator.general._term import _max_string_len_from_source
from zixy.fermion.operator.general._term_set import TermSet
from zixy.fermion.operator.general._terms import ComplexTerms, RealTerms, SymbolicTerms

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    import zixy.fermion.operator.normal as zfon
    from zixy._zixy import Modes
    from zixy.container.term import Term as TermBase, TermSpecT


class TermSum[C: (float, complex, Expr)](OperatorTermSum[ImplT, SpecT, C, ElemT], TermSet[C]):
    """A sum of terms consisting of raw fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    def __init__(self, modes: int | Modes = 0, *, max_string_len: int = 0):
        """Initialize the term sum.

        Args:
            modes: The mode space or number of modes.
            max_string_len: Maximum number of operators per string.
        """
        TermSet.__init__(self, modes, max_string_len=max_string_len)

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
            Coefficients of duplicate strings are added together.
        """
        return cls.from_terms(
            cls.terms_type.from_iterable(source, modes, max_string_len=max_string_len)
        )

    def __add__(self, rhs: TermBase[ImplT, SpecT, C] | Self) -> Self:
        """Add two operators, preallocating the result's string storage."""
        return self._added(rhs, 1)

    def _added(self, rhs: TermBase[ImplT, SpecT, C] | Self, scalar: int) -> Self:
        """Allocate a sum with enough string storage for both operands."""
        required = (
            rhs.max_string_len if isinstance(rhs, TermSum) else _max_string_len_from_source(rhs)
        )
        out = self.clone()
        if required > out.max_string_len:
            out._string_set._impl._reserve_string_length(required, warn=False)
            out._string_set._impl.refresh_map(out._string_set._map)
        BaseTermSum[ImplT, SpecT, C]._scaled_iadd(out, rhs, scalar)
        return out

    def __sub__(self, rhs: TermBase[ImplT, SpecT, C] | Self) -> Self:
        """Subtract two operators, preallocating the result's string storage."""
        return self._added(rhs, -1)

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        out = type(self)(self.modes, max_string_len=self.max_string_len)
        for term in self:
            out += term.into(self.terms_type.term_type).daggered()
        BaseTermSum.__init__(self, out.to_terms())

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class RealTermSum(NumericTermSum[GeneralFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of raw fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[float]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> RealTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[float] | Self) -> Self | RealTermSum:
        """Multiplication of ``self`` by ``rhs``.

        Term-sum multiplication in the general representation concatenates raw ladder-operator
        products without applying normal-ordering identities.
        """
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, RealTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(rhs._impl.coeffs, RealCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = rhs._impl.strings._impl
        rhs_coeffs = rhs._impl.coeffs._impl
        # TODO: support output by reference
        impl, coeffs, map_ = lhs_impl.lincomb_mul_real(lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs)
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data, map_=map_)

    def __pow__(self, exponent: int | np.integer[Any]) -> RealTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, RealTermSum.__mul__)

    def to_normal_ordered(self) -> zfon.RealTermSum:
        """Convert this raw general term sum to the normal-ordered representation."""
        import zixy.fermion.operator.normal as zfon  # noqa: PLC0415

        impl, coeffs, map_ = self._string_set._impl.lincomb_to_normal_order_real(
            self._string_set._impl, self.coeffs._impl
        )
        term_data = TermData(
            zfon.Strings._create(impl),
            RealCoeffs._create(coeffs),
        )
        return zfon.RealTermSum._create(term_data, map_=map_)


class ComplexTermSum(
    NumericTermSum[GeneralFermionOperatorArray, StringSpec, complex], TermSum[complex]
):
    """A sum of terms consisting of raw fermionic strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[complex]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> ComplexTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[complex] | Self) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``rhs``.

        Term-sum multiplication in the general representation concatenates raw ladder-operator
        products without applying normal-ordering identities.
        """
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, ComplexTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        assert isinstance(rhs._impl.coeffs, ComplexCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = rhs._impl.strings._impl
        rhs_coeffs = rhs._impl.coeffs._impl
        # TODO: support output by reference
        impl, coeffs, map_ = lhs_impl.lincomb_mul_complex(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data, map_=map_)

    def __pow__(self, exponent: int | np.integer[Any]) -> ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, ComplexTermSum.__mul__)

    def to_normal_ordered(self) -> zfon.ComplexTermSum:
        """Convert this raw general term sum to the normal-ordered representation."""
        import zixy.fermion.operator.normal as zfon  # noqa: PLC0415

        impl, coeffs, map_ = self._string_set._impl.lincomb_to_normal_order_complex(
            self._string_set._impl, self.coeffs._impl
        )
        term_data = TermData(
            zfon.Strings._create(impl),
            ComplexCoeffs._create(coeffs),
        )
        return zfon.ComplexTermSum._create(term_data, map_=map_)


class SymbolicTermSum(SymbolicTermSumBase[ImplT, SpecT], TermSum[Expr]):
    """A sum of terms consisting of raw fermionic strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms
