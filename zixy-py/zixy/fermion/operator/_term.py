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

"""Individual terms and term specifications for fermionic operators."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy.container.coeff import ComplexSign, Sign
from zixy.fermion._term import Term as FermionTerm

if TYPE_CHECKING:
    from zixy._zixy import FermionArray
    from zixy.fermion.operator._strings import Strings as OperatorStrings


def _product_sign(n_cre: int, n_ann: int) -> Sign:
    r"""Compute the sign of a product of fermionic ladder operators.

    Args:
        n_cre: The number of creation operators in the product.
        n_ann: The number of annihilation operators in the product.

    Returns:
        The sign of the product of ladder operators.
    """
    return Sign(((n_cre * (n_cre - 1) // 2) + (n_ann * (n_ann - 1) // 2)) & 1)


class Term[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
    FermionTerm[ImplT, SpecT, C, ElemT]
):
    """A term consisting of a fermionic ladder-operator string and a coefficient."""

    strings_type: type[OperatorStrings[ImplT, SpecT, ElemT]]

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        pass
