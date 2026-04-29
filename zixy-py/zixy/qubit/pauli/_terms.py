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

"""Terms containing Pauli strings as components and collections of such terms.

The structure of this module parallels that of :mod:`~zixy.container.terms` and
:mod:`~zixy.qubit._terms`, but with components that are Pauli strings, as defined in
:mod:`~zixy.qubit.pauli._strings`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol
from typing_extensions import Self

from zixy._zixy import (
    PauliMatrix,
    PauliSprings,
    QubitPauliArray,
    Qubits,
    SymplecticPart,
)
from zixy.container import terms
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    Number,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
    get_coeffs_type,
    typesafe_mul,
)
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms, NumericTermSum
from zixy.fermion import mappings
from zixy.qubit._terms import (
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.qubit.clifford import GateList
from zixy.qubit.pauli._strings import String, Strings, StringSpec
from zixy.qubit.state._strings import Strings as StateStrings
from zixy.qubit.state._terms import (
    ComplexTermSum as ComplexState,
    RealTermSum as RealState,
)
from zixy.utils import DEFAULT_COMMUTES_ATOL

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None
SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


class Term(TermBase[QubitPauliArray, StringSpec, CoeffT, PauliMatrix]):
    """A term consisting of a Pauli string and a coefficient.

    A single qubit-based term consisting of a Pauli string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]

    @classmethod
    def term_data_from_str(
        cls, source: str, qubits: int | Qubits | None = None
    ) -> TermData[QubitPauliArray, StringSpec, CoeffT]:
        """Parse an input string and return the corresponding term data.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.

        Returns:
            The parsed term data.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        impl, phases = QubitPauliArray.with_phases(qubits, PauliSprings(source))
        cmpnts = cls.cmpnts_type._create(impl)
        coeffs_type = get_coeffs_type(cls.coeff_type)
        coeffs = coeffs_type.parse(source) if "(" in source else coeffs_type.from_size(len(phases))
        coeffs *= ComplexSignCoeffs._create(phases)
        return TermData(cmpnts, coeffs)

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of :param:`cls` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.

        Returns:
            A new instance containing the Pauli string and coefficient in the :param:`source`.
        """
        data = cls.term_data_from_str(source, qubits)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)

    @property
    def string(self) -> String:
        """Get the string component of the term."""
        return cast(String, self.cmpnt)

    def __imul__(self, rhs: Coeff | String | Term[CoeffT]) -> Self:  # type: ignore[misc,override]
        """In-place multiplication of :param:`self` by :param:`rhs`."""
        init_coeff = None
        different_qubits = True
        try:
            if isinstance(rhs, Coeff):
                different_qubits = False
                self.coeff = typesafe_mul(self.coeff, rhs)
            elif isinstance(rhs, String):
                phase = self.string.phase_of_mul(rhs)
                different_qubits = False
                self.coeff = typesafe_mul(self.coeff, phase)
                self.string.imul_ignore_phase(rhs)
            else:
                phase = self.string.phase_of_mul(rhs.string)
                different_qubits = False
                init_coeff = self.coeff
                self.coeff = typesafe_mul(self.coeff, phase)
                self.coeff = typesafe_mul(self.coeff, rhs.coeff)
                self.string.imul_ignore_phase(rhs.string)
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

    def conj_clifford_list(self, gates: GateList) -> None:
        """Conjugate :param:`self` by a list of Clifford gates.

        Args:
            gates: The Clifford gates to conjugate by.

        Note:
            This method operates in-place.
        """
        signs = self._data.cmpnts._impl.conj_clifford_list(gates._impl)
        self._impl._coeffs *= SignCoeffs._create(signs)


class Terms(TermsBase[QubitPauliArray, StringSpec, CoeffT, PauliMatrix]):
    """A collection of terms consisting of Pauli strings and coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and coefficients that
    may be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or
    a view on a slice of the elements in another collection.
    """

    term_type: type[Term[CoeffT]]

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of :param:`cls` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.

        Returns:
            A new instance containing the Pauli strings and coefficients in the :param:`source`.
        """
        return cls._create(cls.term_type.term_data_from_str(source, qubits))

    @property
    def strings(self) -> Strings:
        """Get the components of :param:`self`."""
        return cast(Strings, self.cmpnts)

    def lexicographic_sort(self, ascending: bool = False) -> None:
        """Lexicographically sort the terms in-place.

        Args:
            ascending: Whether to sort in ascending order.

        Note:
            This method operates in-place.
        """
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            self._data.cmpnts._impl.lexicographic_sort_with_sign_vec(coeffs._impl, ascending)
        elif isinstance(coeffs, ComplexSignCoeffs):
            self._data.cmpnts._impl.lexicographic_sort_with_complex_sign_vec(
                coeffs._impl, ascending
            )
        elif isinstance(coeffs, RealCoeffs):
            self._data.cmpnts._impl.lexicographic_sort_with_real_vec(coeffs._impl, ascending)
        elif isinstance(coeffs, ComplexCoeffs):
            self._data.cmpnts._impl.lexicographic_sort_with_complex_vec(coeffs._impl, ascending)
        else:
            raise TypeError(f"Sort not implemented for coefficient type {type(coeffs)}")

    def conj_clifford_list(self, gates: GateList) -> None:
        """Conjugate :param:`self` by a list of Clifford gates.

        Args:
            gates: The Clifford gates to conjugate by.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.pauli._terms.Term.conj_clifford_list`
        """
        signs = self._data.cmpnts._impl.conj_clifford_list(gates._impl)
        self._impl._coeffs *= SignCoeffs._create(signs)

    def relabel(self, qubits: Qubits) -> Self:
        """Relabel the strings using a new register of qubits.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            :param:`self` for chaining.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.pauli._strings.Strings.relabel`
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
            :meth:`~zixy.qubit.pauli._strings.Strings.relabelled`
        """
        out = self.clone()
        out.relabel(qubits)
        return out

    def standardize(self, n_qubit: int) -> Self:
        """Standardize the string labels according to a given number of qubits.

        Args:
            n_qubit: The number of qubits.

        Returns:
            :param:`self` for chaining.

        Note:
            This method operates in-place.

        See Also:
            :meth:`~zixy.qubit.pauli._strings.Strings.standardize`
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
            :meth:`~zixy.qubit.pauli._strings.Strings.standardized`
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
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            return self._data._cmpnts._impl.canonicalize_sign(
                coeffs._impl, mode_order, to_solve, additional_reduces
            )
        elif isinstance(coeffs, ComplexSignCoeffs):
            return self._data._cmpnts._impl.canonicalize_complex_sign(
                coeffs._impl, mode_order, to_solve, additional_reduces
            )
        else:
            raise TypeError(f"Canonicalization not valid for coefficient type {type(coeffs)}")

    def canonicalize_all(self) -> Sequence[tuple[int, int]]:
        coeffs = self._data.coeffs
        if isinstance(coeffs, SignCoeffs):
            return self._data._cmpnts._impl.canonicalize_all_sign(coeffs._impl)
        elif isinstance(coeffs, ComplexSignCoeffs):
            return self._data._cmpnts._impl.canonicalize_all_complex_sign(coeffs._impl)
        else:
            raise TypeError(f"Canonicalization not valid for coefficient type {type(coeffs)}")


class TermSet(TermSetBase[QubitPauliArray, StringSpec, CoeffT, PauliMatrix]):
    """A collection of unique terms consisting of Pauli strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[CoeffT]]


class TermSum(TermSumBase[QubitPauliArray, StringSpec, CoeffT, PauliMatrix], TermSet[CoeffT]):
    """A sum of terms consisting of Pauli strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create a new instance of :param:`cls` by parsing an input string.

        Args:
            source: Input string to parse.
            qubits: Space of qubits or a number of qubits. If ``None``, infer from the max qubit
                index in the input string.

        Returns:
            A new instance containing the Pauli strings and coefficients in the :param:`source`.
        """
        terms = cls.terms_type.from_str(source, qubits)
        return cls.from_iterable(terms, terms.qubits)

    def commutator(self, other: TermSum[CoeffT]) -> TermSum[Any]:
        """Compute the commutator of :param:`self` with :param:`other`.

        .. math::
            [A, B] = AB - BA.
        """
        out = cast(Any, self) * other - cast(Any, other) * self
        return cast(TermSum[Any], out)

    def commutes_with(self, other: TermSum[CoeffT], atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether :param:`self` commutes with :param:`other`.

        .. math::
            [A, B] = 0.

        Args:
            other: The other operand.
            atol: The absolute tolerance.

        Returns:
            Whether :param:`self` commutes with :param:`other`.

        See Also:
            :meth:`commutator`
        """
        coeffs = self.commutator(other)._data.coeffs
        if isinstance(coeffs, RealCoeffs | ComplexCoeffs):
            return not coeffs.any_significant(atol)
        return len(coeffs) == 0

    def conserves_hamming_weight(self, atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether :param:`self` conserves Hamming weight.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether :param:`self` conserves Hamming weight.

        See Also:
            :meth:`~zixy.qubit.pauli.num_ops.create_num_op`
        """
        from .num_ops import create_num_op  # noqa: PLC0415

        return self.commutes_with(
            cast(Any, create_num_op(cast(Any, type(self)), self.qubits)), atol
        )

    def conserves_odd_bit_hamming_weight(self, atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether :param:`self` conserves odd-bit Hamming weight.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether :param:`self` conserves odd-bit Hamming weight.

        See Also:
            :meth:`~zixy.qubit.pauli.num_ops.create_num_op_odd_bits`
        """
        from .num_ops import create_num_op_odd_bits  # noqa: PLC0415

        return self.commutes_with(
            cast(Any, create_num_op_odd_bits(cast(Any, type(self)), self.qubits)), atol
        )


class SignTerm(Term[Sign]):
    """A term consisting of a Pauli string and a sign coefficient.

    A single qubit-based term consisting of a Pauli string and a
    :class:`~zixy.container.coeffs.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.
    """

    coeff_type = Sign


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of Pauli strings and sign coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~zixy.container.coeffs.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms consisting of Pauli strings and sign coefficients.

    A set-like container of qubit-based terms with :class:`~zixy.container.coeffs.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a Pauli string and a complex sign coefficient.

    A single qubit-based term consisting of a Pauli string and a
    :class:`~zixy.container.coeffs.ComplexSign` that may be an owning instance referencing a
    single element in a :class:`~zixy.container.data.TermData` instance, or a view on an
    element in another collection.
    """

    coeff_type = ComplexSign


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms consisting of Pauli strings and complex sign coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~zixy.container.coeffs.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class ComplexSignTermSet(TermSet[ComplexSign]):
    """A collection of unique terms consisting of Pauli strings and complex sign coefficients.

    A set-like container of qubit-based terms with
    :class:`~zixy.container.coeffs.ComplexSign` coefficients that may be used to store unique
    terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexSignTerms


class RealTerm(Term[float]):
    """A term consisting of a Pauli string and a real coefficient.

    A single qubit-based term consisting of a Pauli string and a ``float`` coefficient that may be
    an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = float


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
        array = self.coeffs.np_array
        if by_magnitude:
            array = np.abs(array)
        inds = tuple(int(i) for i in np.argsort(array))
        if not ascending:
            inds = inds[::-1]
        tmp = self.reordered(inds)
        self._impl = tmp._impl


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of Pauli strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class RealTermSum(NumericTermSum[QubitPauliArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of Pauli strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Convert :param:`self` to a sparse matrix.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The resulting sparse matrix.
        """
        assert isinstance(self._impl._coeffs, RealCoeffs)
        return QubitPauliArray.lincomb_to_sparse_real(
            self._impl._cmpnts._impl,
            self._impl._coeffs._impl,
            big_endian,
        )

    @classmethod
    def from_fermionic(
        cls,
        qubits: int | Qubits,
        mapper: mappings.Mapper,
        fermion_ops: Sequence[tuple[Sequence[tuple[int, bool]], float]],
    ) -> Self:
        """Create an instance of :param:`cls` from a fermionic operator given a mapping.

        Args:
            qubits: The qubit register or qubit count.
            mapper: The mapping.
            fermion_ops: A sequence of tuples, where each tuple consists of an integer index
                indicating the mode and a boolean indicating whether it's a creation (``True``) or
                annihilation (``False``) operator.

        Returns:
            The constructed instance.
        """
        out = cls(qubits)
        out_impl = out._impl._cmpnts._impl
        out_map = out._cmpnt_set._map
        out_real_coeffs = out._impl._coeffs
        assert isinstance(out_real_coeffs, RealCoeffs)
        out_coeffs = out_real_coeffs._impl
        mapper._impl.op_encode_real(fermion_ops, out_impl, out_map, out_coeffs)
        return out

    @overload  # type: ignore[override]
    def __mul__(self, other: Coeff) -> Self: ...

    @overload
    def __mul__(self, other: Self) -> ComplexTermSum: ...

    def __mul__(self, other: Self | Coeff) -> Self | ComplexTermSum:
        """Multiplication of :param:`self` by :param:`other`."""
        if isinstance(other, Coeff):
            return super().__mul__(other)
        elif not isinstance(other, RealTermSum):
            return NotImplemented
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(other._impl._coeffs, RealCoeffs)
        lhs_impl = self._impl._cmpnts._impl
        lhs_coeffs = self._impl._coeffs._impl
        rhs_impl = other._impl._cmpnts._impl
        rhs_coeffs = other._impl._coeffs._impl
        out = ComplexTermSum._create(TermData(other._impl._cmpnts._empty_clone(), ComplexCoeffs()))
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        out_impl = out._impl._cmpnts._impl
        out_map = out._cmpnt_set._map
        out_coeffs = out._impl._coeffs._impl
        QubitPauliArray.lincomb_mul_real(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs, out_impl, out_map, out_coeffs
        )
        return out

    def project_into_ortho_subspace(self, subspace: StateStrings) -> NDArray[np.float64]:
        """Project the term into an orthogonal subspace defined by an ordered sequence of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            The resulting vector of coefficients in the subspace basis.
        """
        assert isinstance(self._impl._coeffs, RealCoeffs)
        return QubitPauliArray.lincomb_project_into_ortho_subspace_real(
            self._impl._cmpnts._impl,
            self._impl._coeffs._impl,
            subspace._impl,
        )

    def project_into_nonortho_subspace(
        self, subspace: Sequence[RealState]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project the term into a non-orthogonal subspace defined by an ordered sequence of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            A tuple containing the resulting vector of coefficients in the subspace basis and the
            overlap matrix of the subspace states.
        """
        assert isinstance(self._impl._coeffs, RealCoeffs)
        cmpnts = tuple(item._cmpnt_set._impl for item in subspace)
        maps = tuple(item._cmpnt_set._map for item in subspace)
        coeffs = []
        for item in subspace:
            assert isinstance(item._impl._coeffs, RealCoeffs)
            coeffs.append(item._impl._coeffs._impl)
        return QubitPauliArray.lincomb_project_into_nonortho_subspace_real(
            self._impl._cmpnts._impl,
            self._impl._coeffs._impl,
            cmpnts,
            maps,
            tuple(coeffs),
        )

    def __iadd__(self, rhs: RealTerm | Self | mappings.Contribution[float]) -> Self:  # type: ignore[override,misc]
        """In-place addition of :param:`self` by :param:`rhs`."""
        if isinstance(rhs, mappings.Contribution):
            assert isinstance(self._impl._coeffs, RealCoeffs)  # TODO: resolve
            rhs._mapper._impl.op_contribute_real(
                self._impl._cmpnts._impl,
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._c,
            )
        else:
            super().__iadd__(rhs)
        return self

    def __isub__(self, rhs: RealTerm | Self | mappings.Contribution[float]) -> Self:  # type: ignore[override,misc]
        """In-place subtraction of :param:`self` by :param:`rhs`."""
        if isinstance(rhs, mappings.Contribution):
            self.__iadd__(-rhs)
        else:
            super().__isub__(rhs)
        return self

    def apply(self, state: RealState) -> ComplexState:
        """Apply :param:`self` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        out = ComplexState(self.qubits)
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(state._impl._coeffs, RealCoeffs)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        self._impl._cmpnts._impl.apply_to_state_real(
            self._impl._coeffs._impl,
            state._impl._cmpnts._impl,
            state._impl._coeffs._impl,
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
        )
        return out

    def mat_elem(self, bra: RealState, ket: RealState) -> float:
        """Evaluate the matrix element of :param:`self` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(bra._impl._coeffs, RealCoeffs)
        assert isinstance(ket._impl._coeffs, RealCoeffs)
        return float(
            self._impl._cmpnts._impl.mat_elem_real(
                self._impl._coeffs._impl,
                bra._impl._cmpnts._impl,
                bra._impl._coeffs._impl,
                ket._impl._cmpnts._impl,
                ket._impl._coeffs._impl,
            )
        )

    def exp_val(self, state: RealState) -> float:
        """Evaluate the expectation value of :param:`self` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)


class ComplexTerm(Term[complex]):
    """A term consisting of a Pauli string and a complex coefficient.

    A single qubit-based term consisting of a Pauli string and a ``complex`` coefficient that may
    be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = complex


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
        array = np.abs(self.coeffs.np_array)
        inds = tuple(int(i) for i in np.argsort(array))
        if not ascending:
            inds = inds[::-1]
        tmp = self.reordered(inds)
        self._impl = tmp._impl


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms consisting of Pauli strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class ComplexTermSum(NumericTermSum[QubitPauliArray, StringSpec, complex], TermSum[complex]):
    """A sum of terms consisting of Pauli strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    @property
    def real_part(self) -> RealTermSum:
        """Return the real part of :param:`self`."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.cmpnts, self._data.coeffs.real_part)
        return RealTermSum._create(data)

    @property
    def imag_part(self) -> RealTermSum:
        """Return the imaginary part of :param:`self`."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.cmpnts, self._data.coeffs.imag_part)
        return RealTermSum._create(data)

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Convert :param:`self` to a sparse matrix.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The resulting sparse matrix.
        """
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        return QubitPauliArray.lincomb_to_sparse_complex(
            self._impl._cmpnts._impl,
            self._impl._coeffs._impl,
            big_endian,
        )

    def __mul__(self, other: Self | Coeff) -> Self | ComplexTermSum:
        """Multiplication of :param:`self` by :param:`other`."""
        if isinstance(other, Coeff):
            return super().__mul__(other)
        elif not isinstance(other, ComplexTermSum):
            return NotImplemented
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(other._impl._coeffs, ComplexCoeffs)
        lhs_impl = self._impl._cmpnts._impl
        lhs_coeffs = self._impl._coeffs._impl
        rhs_impl = other._impl._cmpnts._impl
        rhs_coeffs = other._impl._coeffs._impl
        out = ComplexTermSum._create(TermData(other._impl._cmpnts._empty_clone(), ComplexCoeffs()))
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        out_impl = out._impl._cmpnts._impl
        out_map = out._cmpnt_set._map
        out_coeffs = out._impl._coeffs._impl
        QubitPauliArray.lincomb_mul_complex(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs, out_impl, out_map, out_coeffs
        )
        return out

    def project_into_ortho_subspace(self, subspace: StateStrings) -> NDArray[np.complex128]:
        """Project the term into an orthogonal subspace defined by a set of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            The resulting vector of coefficients in the subspace basis.
        """
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        return QubitPauliArray.lincomb_project_into_ortho_subspace_complex(
            self._impl._cmpnts._impl,
            self._impl._coeffs._impl,
            subspace._impl,
        )

    def __iadd__(self, rhs: ComplexTerm | Self | mappings.Contribution[complex]) -> Self:  # type: ignore[override,misc]
        """In-place addition of :param:`self` by :param:`rhs`."""
        if isinstance(rhs, mappings.Contribution):
            assert isinstance(self._impl._coeffs, ComplexCoeffs)  # TODO: resolve
            rhs._mapper._impl.op_contribute_complex(
                self._impl._cmpnts._impl,
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._c,
            )
        else:
            super().__iadd__(rhs)
        return self

    def __isub__(self, rhs: ComplexTerm | Self | mappings.Contribution[complex]) -> Self:  # type: ignore[override,misc]
        """In-place subtraction of :param:`self` by :param:`rhs`."""
        if isinstance(rhs, mappings.Contribution):
            self.__iadd__(-rhs)
        else:
            super().__isub__(rhs)
        return self

    def apply(self, state: ComplexState) -> ComplexState:
        """Apply :param:`self` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        out = ComplexState(self.qubits)
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(state._impl._coeffs, ComplexCoeffs)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        self._impl._cmpnts._impl.apply_to_state_complex(
            self._impl._coeffs._impl,
            state._impl._cmpnts._impl,
            state._impl._coeffs._impl,
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
        )
        return out

    def mat_elem(self, bra: ComplexState, ket: ComplexState) -> complex:
        """Evaluate the matrix element of :param:`self` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(bra._impl._coeffs, ComplexCoeffs)
        assert isinstance(ket._impl._coeffs, ComplexCoeffs)
        return complex(
            self._impl._cmpnts._impl.mat_elem_complex(
                self._impl._coeffs._impl,
                bra._impl._cmpnts._impl,
                bra._impl._coeffs._impl,
                ket._impl._cmpnts._impl,
                ket._impl._coeffs._impl,
            )
        )

    def exp_val(self, state: ComplexState) -> complex:
        """Evaluate the expectation value of :param:`self` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)


class SymbolicTerm(Term[Expr]):
    """A term consisting of a Pauli string and a symbolic coefficient.

    A single qubit-based term consisting of a Pauli string and a :class:`~sympy.Expr`
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeff_type = Expr

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
        """Try to evaluate :param:`self` as a term containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.pauli._terms.RealTerm` with the evaluated
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
        """Try to evaluate :param:`self` as a term containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.pauli._terms.ComplexTerm` with the evaluated
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
    """A collection of terms consisting of Pauli strings and symbolic coefficients.

    An array-like container of qubit-based terms consisting of Pauli strings and
    :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of :param:`self`."""
        return cast(SymbolicCoeffs, self._data.coeffs[self.slice])

    @property
    def free_symbols(self) -> set[Symbol]:
        """Get the set of free (unsubstituted) symbols in :param:`self`.

        Returns:
            Union of the sets of free symbols across all coefficients in :param:`self`.
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
        """Differentiate partially with respect to :param:`variable` in-place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Note:
            This method operates in-place.
        """
        self.coeffs.idiff(variable)

    def diff(self, variable: Symbol | str) -> SymbolicTerms:
        """Differentiate partially with respect to :param:`variable` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.diff(variable)))

    def try_to_real(self) -> RealTerms:
        """Try to evaluate :param:`self` as terms containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.pauli._terms.RealTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.
        """
        return RealTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_real()))

    def try_to_complex(self) -> ComplexTerms:
        """Try to evaluate :param:`self` as terms containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.pauli._terms.ComplexTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.
        """
        return ComplexTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_complex()))


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms consisting of Pauli strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of Pauli strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    @overload  # type: ignore[override]
    def __mul__(self, other: Coeff) -> Self: ...

    @overload
    def __mul__(self, other: Self) -> Self: ...

    def __mul__(self, other: Self | Coeff) -> Self | SymbolicTermSum:
        """Multiplication of :param:`self` by :param:`other`."""
        if isinstance(other, Coeff):
            return super().__mul__(other)
        elif not isinstance(other, SymbolicTermSum):
            return NotImplemented
        out = SymbolicTermSum._create(
            TermData(other._impl._cmpnts._empty_clone(), SymbolicCoeffs())
        )
        for l_term in self:
            for r_term in other:
                product = l_term * r_term
                product.coeff = product.coeff.simplify()
                out += product  # type: ignore[arg-type]
        return out


class TermRegistry(terms.TermRegistry[QubitPauliArray, StringSpec]):
    """Registry of term types for each different coefficient type."""

    term_type_sign: type[SignTerm]
    term_type_complex_sign: type[ComplexSignTerm]
    term_type_real: type[RealTerm]
    term_type_complex: type[ComplexTerm]
    term_type_symbolic: type[SymbolicTerm]

    def __getitem__(self, coeff_type: type[CoeffT]) -> type[Term[CoeffT]]:
        """Get the term type corresponding to :param:`coeff_type`."""
        return cast(type[Term[CoeffT]], super().__getitem__(coeff_type))


String._term_registry = TermRegistry(
    term_type_sign=SignTerm,
    term_type_complex_sign=ComplexSignTerm,
    term_type_real=RealTerm,
    term_type_complex=ComplexTerm,
    term_type_symbolic=SymbolicTerm,
)
