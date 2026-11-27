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

"""Term sequences for fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy.container.coeff import ComplexSign, Sign
from zixy.fermion._terms import Terms as FermionTerms

if TYPE_CHECKING:
    from zixy._zixy import FermionArray
    from zixy.fermion.operator._term import Term


class Terms[ImplT: FermionArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr), ElemT](
    FermionTerms[ImplT, SpecT, C, ElemT]
):
    """A collection of terms consisting of fermionic ladder-operator strings and coefficients."""

    term_type: type[Term[ImplT, SpecT, C, ElemT]]
