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

"""String sets for qubit operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zixy.qubit._string_set import StringSet as StringSetBase
from zixy.qubit.operator._string import ElemT, ImplT, SpecT
from zixy.qubit.operator._strings import Strings

if TYPE_CHECKING:
    from zixy.container.serialization import SerializedContainer


class StringSet(StringSetBase[ImplT, SpecT, ElemT]):
    """A collection of unique Pauli strings.

    A set-like container of qubit-based Pauli strings that may be used to store unique strings
    and perform set-like operations on them.
    """

    strings_type = Strings

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the unique strings, and the full qubit
            register.
        """
        data = super().to_dict()
        data["family"] = "qubit.operator"
        data["space"] = {"qubits": self._impl.qubits.get_inds()}
        data["strings"] = [[str(p) for p in string.get_list()] for string in self]
        return data
