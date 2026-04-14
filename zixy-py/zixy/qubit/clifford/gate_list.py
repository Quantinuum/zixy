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

"""Clifford gate list."""

from __future__ import annotations

from zixy._zixy import CliffordGateList


class GateList:
    """List of Clifford gates."""

    _impl: CliffordGateList

    def __init__(self) -> None:
        """Initialize the list."""
        self._impl = CliffordGateList()

    def h(self, i_qubit: int) -> GateList:
        """Push a Hadamard gate into the list.

        Args:
            i_qubit: The index of the qubit to which the gate is applied.

        Returns:
            self: The list itself, for method chaining.

        Note:
            This method operates in-place.
        """
        self._impl.push_h(i_qubit)
        return self

    def s(self, i_qubit: int) -> GateList:
        """Push an S gate into the list.

        Args:
            i_qubit: The index of the qubit to which the gate is applied.

        Returns:
            self: The list itself, for method chaining.

        Note:
            This method operates in-place.
        """
        self._impl.push_s(i_qubit)
        return self

    def cx(self, i_control: int, i_target: int) -> GateList:
        """Push a CNOT gate into the list.

        Args:
            i_control: The index of the control qubit.
            i_target: The index of the target qubit.

        Returns:
            self: The list itself, for method chaining.

        Note:
            This method operates in-place.
        """
        self._impl.push_cx(i_control, i_target)
        return self

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return str(self._impl)
