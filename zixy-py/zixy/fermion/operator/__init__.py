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

"""Submodule for fermionic ladder-operator strings and terms."""

from zixy._zixy import (
    FermionSprings,
    GeneralFermionOperatorArray,
    Modes,
    NormalFermionOperatorArray,
)
from zixy.fermion.operator.normal._string import String, StringSpec
from zixy.fermion.operator.normal._strings import Strings
from zixy.fermion.operator.normal._string_set import StringSet
from zixy.fermion.operator.normal._term import (
    SignTerm,
    ComplexSignTerm,
    RealTerm,
    ComplexTerm,
    SymbolicTerm,
)
from zixy.fermion.operator.normal._terms import (
    SignTerms,
    ComplexSignTerms,
    RealTerms,
    ComplexTerms,
    SymbolicTerms,
)
from zixy.fermion.operator.normal._term_set import (
    SignTermSet,
    ComplexSignTermSet,
    RealTermSet,
    ComplexTermSet,
    SymbolicTermSet,
)
from zixy.fermion.operator.normal._term_sum import RealTermSum, ComplexTermSum, SymbolicTermSum

__all__ = [
    "Modes",
    "FermionSprings",
    "NormalFermionOperatorArray",
    "GeneralFermionOperatorArray",
    "StringSpec",
    "String",
    "Strings",
    "StringSet",
    "SignTerm",
    "SignTerms",
    "SignTermSet",
    "ComplexSignTerm",
    "ComplexSignTerms",
    "ComplexSignTermSet",
    "RealTerm",
    "RealTerms",
    "RealTermSet",
    "RealTermSum",
    "ComplexTerm",
    "ComplexTerms",
    "ComplexTermSet",
    "ComplexTermSum",
    "SymbolicTerm",
    "SymbolicTerms",
    "SymbolicTermSet",
    "SymbolicTermSum",
]
