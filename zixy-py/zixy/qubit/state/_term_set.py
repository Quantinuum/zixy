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

"""Term sets for qubit states."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import QubitStateArray
from zixy.container.coeff import ComplexSign, Sign
from zixy.qubit._term_set import TermSet as TermSetBase
from zixy.qubit.state._string import StringSpec
from zixy.qubit.state._terms import (
    ComplexSignTerms,
    ComplexTerms,
    RealTerms,
    SignTerms,
    SymbolicTerms,
)

if TYPE_CHECKING:
    from zixy.qubit.state._terms import Terms


class TermSet[C: (float, complex, Sign, ComplexSign, Expr)](
    TermSetBase[QubitStateArray, StringSpec, C, bool]
):
    """A collection of unique terms consisting of state strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[C]]


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms consisting of state strings and sign coefficients.

    A set-like container of qubit-based terms with :class:`~zixy.container.coeff.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class ComplexSignTermSet(TermSet[ComplexSign]):
    """A collection of unique terms consisting of state strings and complex sign coefficients.

    A set-like container of qubit-based terms with
    :class:`~zixy.container.coeff.ComplexSign` coefficients that may be used to store unique
    terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexSignTerms


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of state strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms consisting of state strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms consisting of state strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms
