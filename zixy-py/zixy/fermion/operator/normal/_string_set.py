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

"""String sets for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zixy.fermion._string import _serialize_modes
from zixy.fermion.operator._string_set import StringSet as OperatorStringSet
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT
from zixy.fermion.operator.normal._strings import Strings

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer


class StringSet(OperatorStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique normal-ordered fermionic ladder-operator strings.

    A set-like container of mode-based normal-ordered fermionic strings that may be used to store
    unique strings and perform set-like operations on them.
    """

    strings_type = Strings

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the unique strings, and the full mode space.
        """
        data = super().to_dict()
        data["family"] = "fermion.operator.normal"
        data["space"] = _serialize_modes(self._impl.modes)
        data["strings"] = [[list(indices) for indices in string.get_sets()] for string in self]
        return data
