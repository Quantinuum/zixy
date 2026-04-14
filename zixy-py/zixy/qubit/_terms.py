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

"""Base qubit-based terms and collections of such terms."""

from __future__ import annotations

from typing import Generic, cast

from zixy._zixy import Qubits
from zixy.container.cmpnts import SpecT
from zixy.container.coeffs import CoeffT, get_coeffs_type
from zixy.container.data import TermData
from zixy.container.terms import (
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.qubit._strings import ElemT, ImplT, String, Strings


class Term(Generic[ImplT, SpecT, CoeffT, ElemT], TermBase[ImplT, SpecT, CoeffT]):
    """A term consisting of a string and a coefficient.

    A single qubit-based term consisting of a component and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    cmpnts_type: type[Strings[ImplT, SpecT, ElemT]]

    def __init__(self, qubits: int | Qubits = 0, source: SpecT | None = None):
        """Initialize the term.

        Args:
            qubits: The qubit register or qubit count.
            source: The term specifier to use for default qubits and initial value.
        """
        cmpnts = self.cmpnts_type(qubits, 1)
        coeffs = get_coeffs_type(self.coeff_type).from_size(1)
        data = TermData(cmpnts, coeffs)
        TermBase.__init__(self, data)
        self.set(source)

    @property
    def string(self) -> String[ImplT, SpecT, ElemT]:
        """Get the string component of the term."""
        return cast(String[ImplT, SpecT, ElemT], self.cmpnt)

    @property
    def qubits(self) -> Qubits:
        """Get the qubits corresponding to :param:`self`."""
        return self.string._impl.qubits


class Terms(Generic[ImplT, SpecT, CoeffT, ElemT], TermsBase[ImplT, SpecT, CoeffT]):
    """A collection of terms consisting of strings and coefficients.

    An array-like container of qubit-based terms consisting of strings and coefficients that may
    be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a
    view on a slice of the elements in another collection.
    """

    term_type: type[Term[ImplT, SpecT, CoeffT, ElemT]]

    def __init__(self, qubits: int | Qubits = 0, n: int = 0):
        """Initialize the term array.

        Args:
            qubits: The qubit register or qubit count.
            n: The number of items to initialize the array with.
        """
        cmpnts = self.term_type.cmpnts_type(qubits, n)
        coeffs = get_coeffs_type(self.term_type.coeff_type).from_size(n)
        data = TermData(cmpnts, coeffs)
        TermsBase.__init__(self, data)

    @property
    def strings(self) -> Strings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(Strings[ImplT, SpecT, ElemT], self.cmpnts)

    @property
    def qubits(self) -> Qubits:
        """Get the qubits corresponding to :param:`self`."""
        return self.strings.qubits


class TermSet(Generic[ImplT, SpecT, CoeffT, ElemT], TermSetBase[ImplT, SpecT, CoeffT]):
    """A collection of unique terms consisting of strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[ImplT, SpecT, CoeffT, ElemT]]

    def __init__(self, qubits: int | Qubits = 0):
        """Initialize the term set.

        Args:
            qubits: The qubit register or qubit count.
        """
        TermSetBase.__init__(self, self.terms_type(qubits))

    @property
    def qubits(self) -> Qubits:
        """Get the qubits corresponding to :param:`self`."""
        return cast(Strings[ImplT, SpecT, ElemT], self._impl._cmpnts).qubits


class TermSum(TermSet[ImplT, SpecT, CoeffT, ElemT], TermSumBase[ImplT, SpecT, CoeffT]):
    """A sum of terms consisting of strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    def __init__(self, qubits: int | Qubits = 0):
        """Initialize the term set.

        Args:
            qubits: The qubit register or qubit count.
        """
        TermSumBase.__init__(self, self.terms_type(qubits))
