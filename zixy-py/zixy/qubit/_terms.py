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

"""Term sequences for qubit systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sympy import Expr

from zixy._zixy import Qubits  # noqa: TC001 (required for sphinx)
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.terms import Terms as TermsBase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zixy._zixy import QubitArray
    from zixy.container.term import TermSpecT
    from zixy.qubit._term import Term


class Terms[
    ImplT: QubitArray[Any, Any],
    SpecT,
    C: (float, complex, Sign, ComplexSign, Expr),
    ElemT,
](TermsBase[ImplT, SpecT, C]):
    """A collection of terms consisting of strings and coefficients.

    An array-like container of qubit-based terms consisting of strings and coefficients that may
    be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a
    view on a slice of the elements in another collection.
    """

    term_type: type[Term[ImplT, SpecT, C, ElemT]]

    def __init__(self, qubits: int | Qubits = 0, *, size: int = 0):
        """Initialize the term array.

        Args:
            qubits: The qubit register or qubit count.
            size: The number of items to initialize the array with.
        """
        strings = self.term_type.strings_type(qubits, size=size)
        coeffs = self.term_type.coeffs_type(size=size)
        data = TermData(strings, coeffs)
        TermsBase.__init__(self, data)

    @classmethod
    def from_iterable(
        cls, source: Iterable[TermSpecT[ImplT, SpecT, C]], qubits: int | Qubits = 0
    ) -> Self:
        """Create an owning collection from an iterable of terms.

        Args:
            source: Iterable of compatible terms or term specifiers.
            qubits: The qubit register or qubit count.

        Returns:
            An independent owning instance of ``cls``.
        """
        out = cls(qubits)
        out.extend(source)
        return out

    @property
    def qubits(self) -> Qubits:
        """The qubits corresponding to ``self``."""
        return self.strings.qubits
