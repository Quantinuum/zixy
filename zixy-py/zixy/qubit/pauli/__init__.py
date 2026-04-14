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

"""Submodule for qubit-based Pauli strings."""

from __future__ import annotations

from zixy.qubit.pauli._strings import String, StringSet, Strings, StringSpec
from zixy.qubit.pauli._terms import (
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

from zixy._zixy import PauliMatrix, PauliSprings

I = PauliMatrix.I  # noqa: E741
X = PauliMatrix.X  # noqa: E741
Y = PauliMatrix.Y  # noqa: E741
Z = PauliMatrix.Z  # noqa: E741

__all__ = [
    "PauliMatrix",
    "I",
    "X",
    "Y",
    "Z",
    "PauliSprings",
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
