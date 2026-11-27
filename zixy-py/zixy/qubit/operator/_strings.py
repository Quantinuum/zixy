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

"""String sequences for qubit operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from zixy._zixy import PauliMatrix, Qubits
from zixy.container.serialization import check_schema_version
from zixy.qubit import operator as zqo
from zixy.qubit._strings import Strings as StringsBase
from zixy.qubit.operator._string import ElemT, ImplT, SpecT, String
from zixy.utils import slice_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray

    from zixy.container.base import _FilterableStrings
    from zixy.container.serialization import SerializedContainer


class Strings(StringsBase[ImplT, SpecT, ElemT]):
    """A collection of Pauli strings.

    An array-like container of qubit-based Pauli strings that may be an owning instance referencing
    a contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type = String

    @classmethod
    def _set_type(cls) -> type[zqo.StringSet]:
        """Return the string set class paired with this collection."""
        return zqo.StringSet

    def get_dicts(self) -> tuple[dict[int, PauliMatrix], ...]:
        """Get the strings as a list of dictionaries of their elements."""
        return tuple(self[i].get_dict() for i in range(len(self)))

    def iter_filter_non_identity[StringT: String](
        self: _FilterableStrings[Any, StringT],
    ) -> Iterator[StringT]:
        """Lazily filter strings of ``self``, retaining only those different to identity."""
        return self.iter_filter_populated()

    def filter_non_identity(self) -> Self:
        """Eagerly filter strings of ``self``, retaining only those different to identity."""
        return self.filter_populated()

    def compatibility_matrix(self) -> NDArray[np.uint8]:
        """Form the compatibility matrix for the strings in ``self``.

        Returns:
            Matrix with 1 where the strings represented by the row and column commute, 0 elsewhere.
        """
        if slice_equal(self.slice, slice(None, len(self._impl)), len(self._impl)):
            return self._impl.compatibility_matrix()
        else:
            # todo: have rust binding take inds to preclude this copy in this non-contiguous case.
            return self.clone()._impl.compatibility_matrix()

    def centralizer_and_remainder(self) -> tuple[Strings, Strings]:
        """Get the centralizer and remainder of ``self``.

        The centralizer of a set :math:`S` of Pauli strings is the set of strings :math:`C` that
        commute with all others in :math:`S`. The remainder is the set of Pauli strings that are in
        :math:`S` but not in :math:`C`.

        Returns:
            The centralizer and remainder as a pair of
            :class:`~zixy.qubit.operator._strings.Strings` instances.
        """
        c, r = self._impl.centralizer_and_remainder()
        return Strings._create(c), Strings._create(r)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A dictionary containing container metadata, the visible strings, and the full qubit
            register.
        """
        data = super().to_dict()
        data["family"] = "qubit.operator"
        data["space"] = {"qubits": self.qubits.get_inds()}
        data["strings"] = [[str(p) for p in string.get_list()] for string in self]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            A new owning instance of ``cls`` parsed from ``source``, with storage independent of
            the source dictionary and any existing containers.
        """
        check_schema_version(source)
        qubits = Qubits.from_inds(source["space"]["qubits"])
        rows = source["strings"]
        out = cls(qubits, size=len(rows))
        paulis = {"I": PauliMatrix.I, "X": PauliMatrix.X, "Y": PauliMatrix.Y, "Z": PauliMatrix.Z}
        for i, row in enumerate(rows):
            if len(row) != len(qubits):
                raise ValueError("Serialized string length must match the qubit register.")
            try:
                values = [paulis[p] for p in row]
            except KeyError as exc:
                raise ValueError(f"Unknown serialized Pauli matrix: {exc.args[0]!r}.") from exc
            if values:
                out[i].set(values)
        return out
