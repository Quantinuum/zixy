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

"""Submodule for qubit-based state strings."""

from __future__ import annotations

from zixy.qubit.state._strings import String, StringSet, Strings, StringSpec
from zixy.qubit.state._terms import (
    SignTerm,
    SignTerms,
    SignTermSet,
    ComplexSignTerm,
    ComplexSignTerms,
    ComplexSignTermSet,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSet,
    SymbolicTermSum,
)

__all__ = [
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
