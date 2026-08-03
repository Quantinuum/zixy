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

"""Submodule for fermionic operators, states, and mappings."""

from zixy.fermion.mappings import JordanWignerMapper
from zixy.fermion.operator import (
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    SignTerm,
    SignTerms,
    SignTermSet,
    SignTermSum,
    String,
    Strings,
    StringSet,
    StringSpec,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSet,
    SymbolicTermSum,
)
from zixy.fermion.state import (
    ComplexTerm as StateComplexTerm,
    ComplexTerms as StateComplexTerms,
    ComplexTermSet as StateComplexTermSet,
    ComplexTermSum as StateComplexTermSum,
    RealTerm as StateRealTerm,
    RealTerms as StateRealTerms,
    RealTermSet as StateRealTermSet,
    RealTermSum as StateRealTermSum,
    SignTerm as StateSignTerm,
    SignTerms as StateSignTerms,
    SignTermSet as StateSignTermSet,
    String as StateString,
    Strings as StateStrings,
    StringSet as StateStringSet,
    StringSpec as StateStringSpec,
    SymbolicTerm as StateSymbolicTerm,
    SymbolicTerms as StateSymbolicTerms,
    SymbolicTermSet as StateSymbolicTermSet,
    SymbolicTermSum as StateSymbolicTermSum,
)

__all__ = [
    "JordanWignerMapper",
    "StringSpec",
    "String",
    "Strings",
    "StringSet",
    "SignTerm",
    "SignTerms",
    "SignTermSet",
    "SignTermSum",
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
    "StateStringSpec",
    "StateString",
    "StateStrings",
    "StateStringSet",
    "StateSignTerm",
    "StateSignTerms",
    "StateSignTermSet",
    "StateRealTerm",
    "StateRealTerms",
    "StateRealTermSet",
    "StateRealTermSum",
    "StateComplexTerm",
    "StateComplexTerms",
    "StateComplexTermSet",
    "StateComplexTermSum",
    "StateSymbolicTerm",
    "StateSymbolicTerms",
    "StateSymbolicTermSet",
    "StateSymbolicTermSum",
]
