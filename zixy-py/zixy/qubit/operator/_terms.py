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

"""Term sequences for qubit operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sympy import Expr

from zixy._zixy import PauliMatrix, QubitPauliArray
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.coeffs import ComplexCoeffs, ComplexSignCoeffs, RealCoeffs, SignCoeffs
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms
from zixy.qubit._terms import Terms as TermsBase
from zixy.qubit.operator._string import StringSpec
from zixy.qubit.operator._term import (
    ComplexSignTerm,
    ComplexTerm,
    RealTerm,
    SignTerm,
    SymbolicTerm,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy import Symbol

    from zixy._zixy import Qubits, SymplecticPart
    from zixy.container.coeff import Number
    from zixy.qubit.clifford import GateList
    from zixy.qubit.operator._term import Term


class Terms[C: (float, complex, Sign, ComplexSign, Expr)](
    TermsBase[QubitPauliArray, StringSpec, C, PauliMatrix]
):
    """A collection of terms consisting of Pauli strings and coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and coefficients that
    may be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or
    a view on a slice of the elements in another collection.
    """

    term_type: type[Term[C]]

    def lexicographic_sort(self, ascending: bool = False) -> None:
        """Lexicographically sort the terms in-place.

        Args:
            ascending: Whether to sort in ascending order.

        Note:
            This method operates in-place.
        """
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            self._data.strings._impl.lexicographic_sort_with_sign_vec(coeffs._impl, ascending)
        elif isinstance(coeffs, ComplexSignCoeffs):
            self._data.strings._impl.lexicographic_sort_with_complex_sign_vec(
                coeffs._impl, ascending
            )
        elif isinstance(coeffs, RealCoeffs):
            self._data.strings._impl.lexicographic_sort_with_real_vec(coeffs._impl, ascending)
        elif isinstance(coeffs, ComplexCoeffs):
            self._data.strings._impl.lexicographic_sort_with_complex_vec(coeffs._impl, ascending)
        else:
            raise TypeError(f"Sort not implemented for coefficient type {type(coeffs)}")

    def relabel(self, qubits: Qubits) -> Self:
        """Relabel the strings using a new register of qubits.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            ``self`` for chaining.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.operator._strings.Strings.relabel`
        """
        self.strings.relabel(qubits)
        return self

    def relabelled(self, qubits: Qubits) -> Self:
        """Relabel the strings using a new register of qubits.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            The resulting value.

        See Also:
            :meth:`~zixy.qubit.operator._strings.Strings.relabelled`
        """
        out = self.clone()
        out.relabel(qubits)
        return out

    def standardize(self, n_qubit: int) -> Self:
        """Standardize the string labels according to a given number of qubits.

        Args:
            n_qubit: The number of qubits.

        Returns:
            ``self`` for chaining.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.operator._strings.Strings.standardize`
        """
        self.strings.standardize(n_qubit)
        return self

    def standardized(self, n_qubit: int) -> Self:
        """Standardize the string labels according to a given number of qubits.

        Args:
            n_qubit: The number of qubits.

        Returns:
            The resulting value.

        See Also:
            :meth:`~zixy.qubit.operator._strings.Strings.standardized`
        """
        out = self.clone()
        out.strings.standardize(n_qubit)
        return out

    def canonicalize(
        self,
        mode_order: Sequence[tuple[int, SymplecticPart]],
        to_solve: Sequence[int],
        additional_reduces: Sequence[int],
    ) -> Sequence[tuple[int, int]]:
        """In place tableau canonicalization.

        Canonicalizes a given subset with respect to a given ordering of the binary entries in the
        symplectic form.

        Args:
            mode_order: The order of binary entries to try reducing to at most one non-zero entry.
            to_solve: The subset of the strings to canonicalize over (e.g. if some partial
                canonicalization has already been done, skip those strings).
            additional_reduces: Strings outside of ``to_solve`` to include in the reduction step
                (e.g. if some partial canonicalization has already been done, reduce the
                strings that already have leading entries).

        Returns:
            The sequence of imul operations as pairs ``(lhs_written, rhs_read)``.
        """
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            return self._data.strings._impl.canonicalize_sign(
                coeffs._impl, mode_order, to_solve, additional_reduces
            )
        elif isinstance(coeffs, ComplexSignCoeffs):
            return self._data.strings._impl.canonicalize_complex_sign(
                coeffs._impl, mode_order, to_solve, additional_reduces
            )
        else:
            raise TypeError(f"Canonicalization not valid for coefficient type {type(coeffs)}")

    def canonicalize_all(self) -> Sequence[tuple[int, int]]:
        """In place tableau canonicalization.

        Canonicalizes the entire Terms with respect to solving X parts first (in qubit order),
        then Z parts.

        Returns:
            The sequence of imul operations as pairs ``(lhs_written, rhs_read)``.
        """
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            return self._data.strings._impl.canonicalize_all_sign(coeffs._impl)
        elif isinstance(coeffs, ComplexSignCoeffs):
            return self._data.strings._impl.canonicalize_all_complex_sign(coeffs._impl)
        else:
            raise TypeError(f"Canonicalization not valid for coefficient type {type(coeffs)}")

    def conj_clifford_list(self, gates: GateList) -> None:
        """Conjugate ``self`` by a list of Clifford gates.

        Args:
            gates: The Clifford gates to conjugate by.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.operator._term.Term.conj_clifford_list`
        """
        signs = self._data.strings._impl.conj_clifford_list(gates._impl)
        coeffs = self._impl.coeffs
        coeffs *= SignCoeffs._create(signs)


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of Pauli strings and sign coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~zixy.container.coeff.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms consisting of Pauli strings and complex sign coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~zixy.container.coeff.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class RealTerms(NumericTerms[QubitPauliArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of Pauli strings and real coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and ``float``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm

    def numeric_sort(self, ascending: bool = False, by_magnitude: bool = True) -> None:
        """Sort the strings in-place according to their coefficients.

        Args:
            ascending: Whether to sort in ascending order.
            by_magnitude: Whether to sort by magnitude instead of value.

        Note:
            This method operates in-place.
        """
        array = self.coeffs.to_numpy()
        if by_magnitude:
            array = np.abs(array)
        inds = tuple(int(i) for i in np.argsort(array))
        if not ascending:
            inds = inds[::-1]
        tmp = self.reordered(inds)
        self._impl = tmp._impl


class ComplexTerms(NumericTerms[QubitPauliArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of Pauli strings and complex coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and ``complex``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm

    def numeric_sort(self, ascending: bool = False) -> None:
        """Sort the strings in-place according to the magnitude of their coefficients.

        Args:
            ascending: Whether to sort in ascending order.

        Note:
            This method operates in-place.
        """
        array = np.abs(self.coeffs.to_numpy())
        inds = tuple(int(i) for i in np.argsort(array))
        if not ascending:
            inds = inds[::-1]
        tmp = self.reordered(inds)
        self._impl = tmp._impl


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of Pauli strings and symbolic coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    @property
    def free_symbols(self) -> set[Symbol]:
        """The set of free (unsubstituted) symbols in ``self``.

        Returns:
            Union of the sets of free symbols across all coefficients in ``self``.
        """
        return self.coeffs.free_symbols

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

    def diff(self, variable: Symbol | str) -> SymbolicTerms:
        """Differentiate partially with respect to ``variable`` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.diff(variable))
        return SymbolicTerms._create(term_data)

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to ``variable`` in-place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Note:
            This method operates in-place.
        """
        self.coeffs.idiff(variable)

    def try_to_real(self) -> RealTerms:
        """Try to evaluate ``self`` as terms containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.operator._terms.RealTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.try_to_real())
        return RealTerms._create(term_data)

    def try_to_complex(self) -> ComplexTerms:
        """Try to evaluate ``self`` as terms containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.operator._terms.ComplexTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.try_to_complex())
        return ComplexTerms._create(term_data)
