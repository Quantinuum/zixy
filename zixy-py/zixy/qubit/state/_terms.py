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

"""Terms containing state strings as components and collections of such terms.

The structure of this module parallels that of :mod:`~zixy.container.terms` and
:mod:`~zixy.qubit._terms`, but with components that are state strings, as defined in
:mod:`~zixy.qubit.state._strings`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol
from typing_extensions import Self

from zixy._zixy import BinarySprings, Qubits, QubitStateArray
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexCoeffs,
    ComplexSign,
    Number,
    OtherCoeffT,
    RealCoeffs,
    Sign,
    SymbolicCoeffs,
    _is_complex,
    _is_complex_sign,
    _is_expr,
    _is_float,
    _is_int,
    _is_sign,
    get_coeffs_type,
    typesafe_mul,
)
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms, NumericTermSum
from zixy.qubit._terms import (
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.qubit.state._strings import String, Strings, StringSpec
from zixy.utils import split_top_level

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None
SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _mul(lhs: Term[CoeffT], rhs: OtherCoeffT) -> Term[Any]:
    """Driver for multiplication of a term with a coefficient."""
    if not isinstance(rhs, Coeff):
        return NotImplemented
    scalar_product = lhs.coeff * rhs
    term_type = get_term_type(type(scalar_product))
    coeffs_type = get_coeffs_type(type(scalar_product))
    data = TermData(lhs._impl._cmpnts, coeffs_type.from_scalar(scalar_product))
    return term_type._create(data)


def _rmul(rhs: Term[CoeffT], lhs: OtherCoeffT) -> Term[Any]:
    """Driver for multiplication of a coefficient with a term."""
    return _mul(rhs, lhs)


class Term(TermBase[QubitStateArray, StringSpec, CoeffT, bool]):
    """A term consisting of a state string and a coefficient.

    A single qubit-based term consisting of a state string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the input
                string.

        Returns:
            A new instance containing the state string and coefficient in ``source``.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        impl = QubitStateArray(qubits, BinarySprings(source))
        cmpnts = cls.cmpnts_type._create(impl)
        coeffs_type = get_coeffs_type(cls.coeff_type)
        coeffs = (
            coeffs_type.from_str(source) if "(" in source else coeffs_type.from_size(len(cmpnts))
        )
        data = TermData(cmpnts, coeffs)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)

    @property
    def string(self) -> String:
        """Get the string component of the term."""
        return cast(String, self.cmpnt)

    def __imul__(self, rhs: Coeff | String | Term[CoeffT]) -> Self:
        """In-place multiplication of ``self`` by ``rhs``."""
        init_coeff = None
        different_qubits = True
        try:
            if isinstance(rhs, Coeff):
                different_qubits = False
                self.coeff = typesafe_mul(self.coeff, rhs)
            elif isinstance(rhs, String):
                raise NotImplementedError
                # phase = self.string.phase_of_mul(rhs)
                # different_qubits = False
                # self.coeff = typesafe_mul(self.coeff, phase)
                # self.string.imul_ignore_phase(rhs)
            else:
                raise NotImplementedError
                # phase = self.string.phase_of_mul(rhs.string)
                # different_qubits = False
                # init_coeff = self.coeff
                # self.coeff = typesafe_mul(self.coeff, phase)
                # self.coeff = typesafe_mul(self.coeff, rhs.coeff)
                # self.string.imul_ignore_phase(rhs.string)
        except ValueError as err:
            if different_qubits:
                raise err
            if init_coeff is not None:
                self.coeff = init_coeff
            extra = (
                f"Coefficient {rhs} is not {type(self.coeff)} representable."
                if isinstance(rhs, Coeff)
                else (
                    "Pauli string multiplication gives a ComplexSign factor "
                    f"that is not {type(self.coeff)} representable."
                )
            )
            raise ValueError(
                f"Cannot multiply {type(self)} {self} in-place by {type(rhs)} {rhs}. {extra}"
            )
        return self


class Terms(TermsBase[QubitStateArray, StringSpec, CoeffT, bool]):
    """A collection of terms consisting of state strings and coefficients.

    An array-like container of qubit-based terms consisting of state strings and coefficients that
    may be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or
    a view on a slice of the elements in another collection.
    """

    term_type: type[Term[CoeffT]]

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the input
                string.

        Returns:
            A new instance containing the state strings and coefficients in ``source``.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        term_strs = split_top_level(source)
        parsed_terms = [cls.term_type.from_str(term_str, qubits) for term_str in term_strs]
        out_qubits = (
            qubits
            if qubits is not None
            else (parsed_terms[0].qubits if parsed_terms else Qubits.from_count(0))
        )
        return cls.from_iterable(parsed_terms, out_qubits)

    @property
    def strings(self) -> Strings:
        """Get the string components of ``self``."""
        return cast(Strings, self.cmpnts)


class TermSet(TermSetBase[QubitStateArray, StringSpec, CoeffT, bool]):
    """A collection of unique terms consisting of state strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[CoeffT]]


class TermSum(TermSumBase[QubitStateArray, StringSpec, CoeffT, bool], TermSet[CoeffT]):
    """A sum of terms consisting of state strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the input
                string.

        Returns:
            A new instance containing the state strings and coefficients in ``source``.
        """
        terms = cls.terms_type.from_str(source, qubits)
        return cls.from_iterable(terms, terms.qubits)


class SignTerm(Term[Sign]):
    """A term consisting of a state string and a sign coefficient.

    A single qubit-based term consisting of a state string and a
    :class:`~zixy.container.coeffs.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.
    """

    coeff_type = Sign

    @overload
    def __mul__(self, other: Sign) -> SignTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: float) -> RealTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> SignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: float) -> RealTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of state strings and sign coefficients.

    An array-like container of qubit-based terms consisting of state strings and
    :class:`~zixy.container.coeffs.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms consisting of state strings and sign coefficients.

    A set-like container of qubit-based terms with :class:`~zixy.container.coeffs.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a state string and a complex sign coefficient.

    A single qubit-based term consisting of a state string and a
    :class:`~zixy.container.coeffs.ComplexSign` that may be an owning instance referencing a
    single element in a :class:`~zixy.container.data.TermData` instance, or a view on an
    element in another collection.
    """

    coeff_type = ComplexSign

    @overload
    def __mul__(self, other: Sign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms consisting of state strings and complex sign coefficients.

    An array-like container of qubit-based terms consisting of state strings and
    :class:`~zixy.container.coeffs.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class ComplexSignTermSet(TermSet[ComplexSign]):
    """A collection of unique terms consisting of state strings and complex sign coefficients.

    A set-like container of qubit-based terms with
    :class:`~zixy.container.coeffs.ComplexSign` coefficients that may be used to store unique
    terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexSignTerms


class RealTerm(Term[float]):
    """A term consisting of a state string and a real coefficient.

    A single qubit-based term consisting of a state string and a ``float`` coefficient that may be
    an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = float

    @overload
    def __mul__(self, other: Sign) -> RealTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: float) -> RealTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> RealTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: float) -> RealTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class RealTerms(NumericTerms[QubitStateArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of state strings and real coefficients.

    An array-like container of qubit-based terms consisting of state strings and ``float``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of state strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class RealTermSum(NumericTermSum[QubitStateArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of state strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    @classmethod
    def from_dense(
        cls,
        qubits: int | Qubits | None = None,
        source: Sequence[float] = tuple(),
        big_endian: bool = False,
    ) -> Self:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            qubits: The qubit register or qubit count.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The constructed instance.
        """
        if qubits is None:
            qubits = Qubits.from_hilbert_space_dim(len(source))
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        out = cls(qubits)
        assert isinstance(out._impl._coeffs, RealCoeffs)  # TODO: resolve
        out._impl._cmpnts._impl.from_dense_real(
            out._cmpnt_set._map, out._impl._coeffs._impl, list(source), big_endian
        )
        return out

    def to_dense(self, big_endian: bool = False) -> NDArray[np.float64]:
        """Convert ``self`` to a dense vector.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The dense vector.
        """
        assert isinstance(self._impl._coeffs, RealCoeffs)  # TODO: resolve
        return self._impl._cmpnts._impl.to_dense_real(self._impl._coeffs._impl, big_endian)

    def vdot(self, rhs: RealTermSum) -> float:
        """Compute the inner product of ``self`` with ``rhs``."""
        assert isinstance(self._impl._coeffs, RealCoeffs)  # TODO: resolve
        assert isinstance(rhs._impl._coeffs, RealCoeffs)  # TODO: resolve
        return float(
            self._impl._cmpnts._impl.vdot_real(
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._impl._cmpnts._impl,
                rhs._impl._coeffs._impl,
            )
        )


class ComplexTerm(Term[complex]):
    """A term consisting of a state string and a complex coefficient.

    A single qubit-based term consisting of a state string and a ``complex`` coefficient that may
    be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = complex

    @overload
    def __mul__(self, other: Sign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class ComplexTerms(NumericTerms[QubitStateArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of state strings and complex coefficients.

    An array-like container of qubit-based terms consisting of state strings and ``complex``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms consisting of state strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class ComplexTermSum(NumericTermSum[QubitStateArray, StringSpec, complex], TermSum[complex]):
    """A sum of terms consisting of state strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    @classmethod
    def from_dense(
        cls,
        qubits: int | Qubits | None = None,
        source: Sequence[float] = tuple(),
        big_endian: bool = False,
    ) -> ComplexTermSum:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            qubits: The qubit register or qubit count.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The constructed instance.
        """
        if qubits is None:
            qubits = Qubits.from_hilbert_space_dim(len(source))
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        out = cls(qubits)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)  # TODO: resolve
        out._impl._cmpnts._impl.from_dense_complex(
            out._cmpnt_set._map, out._impl._coeffs._impl, list(source), big_endian
        )
        return out

    @property
    def real_part(self) -> RealTermSum:
        """Return the real part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)  # TODO: resolve
        data = TermData(self._data.cmpnts.clone(), self._data.coeffs.real_part)
        return RealTermSum._create(data)

    @property
    def imag_part(self) -> RealTermSum:
        """Return the imaginary part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)  # TODO: resolve
        data = TermData(self._data.cmpnts.clone(), self._data.coeffs.imag_part)
        return RealTermSum._create(data)

    def to_dense(self, big_endian: bool = False) -> NDArray[np.complex128]:
        """Convert ``self`` to a dense vector.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The dense vector.
        """
        assert isinstance(self._impl._coeffs, ComplexCoeffs)  # TODO: resolve
        return self._impl._cmpnts._impl.to_dense_complex(self._impl._coeffs._impl, big_endian)

    def vdot(self, rhs: ComplexTermSum) -> complex:
        """Compute the inner product of ``self`` with ``rhs``."""
        assert isinstance(self._impl._coeffs, ComplexCoeffs)  # TODO: resolve
        assert isinstance(rhs._impl._coeffs, ComplexCoeffs)  # TODO: resolve
        return complex(
            self._impl._cmpnts._impl.vdot_complex(
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._impl._cmpnts._impl,
                rhs._impl._coeffs._impl,
            )
        )


class SymbolicTerm(Term[Expr]):
    """A term consisting of a state string and a symbolic coefficient.

    A single qubit-based term consisting of a state string and a :class:`~sympy.Expr` coefficient
    that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = Expr

    @overload
    def __mul__(self, other: Sign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: float) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: complex) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: float) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: complex) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.isubs`
        """
        self.coeff = self.coeff.subs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerm:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        out = self.clone()
        out.isubs(values)
        return out

    def try_to_real(self) -> RealTerm:
        """Try to evaluate ``self`` as a term containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.state._terms.RealTerm` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.try_to_real`
        """
        cmpnts = RealTerm.cmpnts_type.from_cmpnt(self.string.clone())
        coeffs = get_coeffs_type(RealTerm.coeff_type).from_scalar(float(self.coeff.evalf()))
        return RealTerm._create(TermData(cmpnts, coeffs))

    def try_to_complex(self) -> ComplexTerm:
        """Try to evaluate ``self`` as a term containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.state._terms.ComplexTerm` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.try_to_complex`
        """
        cmpnts = ComplexTerm.cmpnts_type.from_cmpnt(self.string.clone())
        coeffs = get_coeffs_type(ComplexTerm.coeff_type).from_scalar(complex(self.coeff.evalf()))
        return ComplexTerm._create(TermData(cmpnts, coeffs))


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of state strings and symbolic coefficients.

    An array-like container of qubit-based terms consisting of state strings and
    :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of ``self``."""
        return cast(SymbolicCoeffs, self._data.coeffs[self.slice])

    @property
    def free_symbols(self) -> set[Symbol]:
        """Get the set of free (unsubstituted) symbols in ``self``.

        Returns:
            Union of the sets of free symbols across all coefficients in ``self``.
        """
        return self.coeffs.free_symbols

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.subs(values)))

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to ``variable`` in-place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Note:
            This method operates in-place.
        """
        self.coeffs.idiff(variable)

    def diff(self, variable: Symbol | str) -> SymbolicTerms:
        """Differentiate partially with respect to ``variable`` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.diff(variable)))

    def try_to_real(self) -> RealTerms:
        """Try to evaluate ``self`` as terms containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.state._terms.RealTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.
        """
        return RealTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_real()))

    def try_to_complex(self) -> ComplexTerms:
        """Try to evaluate ``self`` as terms containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.state._terms.ComplexTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.
        """
        return ComplexTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_complex()))


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms consisting of state strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of state strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms


def get_term_type(coeff_type: type[CoeffT]) -> type[Term[CoeffT]]:
    """Get the term type corresponding to ``coeff_type``."""
    if _is_int(coeff_type) or _is_float(coeff_type):
        return cast(type[Term[CoeffT]], RealTerm)
    elif _is_complex(coeff_type):
        return cast(type[Term[CoeffT]], ComplexTerm)
    elif _is_sign(coeff_type):
        return cast(type[Term[CoeffT]], SignTerm)
    elif _is_complex_sign(coeff_type):
        return cast(type[Term[CoeffT]], ComplexSignTerm)
    elif _is_expr(coeff_type):
        return cast(type[Term[CoeffT]], SymbolicTerm)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type} for term type lookup.")
