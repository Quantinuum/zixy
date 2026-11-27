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

"""String sequences for fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zixy.fermion._strings import Strings as FermionStrings

if TYPE_CHECKING:
    from zixy._zixy import FermionArray
    from zixy.fermion.operator._string import String


class Strings[ImplT: FermionArray, SpecT, ElemT](FermionStrings[ImplT, SpecT, ElemT]):
    """A collection of fermionic ladder-operator strings."""

    string_type: type[String[ImplT, SpecT, ElemT]]
