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

"""JSON serialization and automatic deserialization of Zixy containers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from importlib import import_module
from inspect import isabstract
from typing import TYPE_CHECKING, TypedDict, overload

import numpy as np

from zixy import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, NotRequired, Self

    from numpy import complex128, float64, int64
    from numpy.typing import NDArray

__schema_version__ = "1.0.0"

_FAMILY_MODULES = {
    "container": "zixy.container.coeffs",
    "qubit.operator": "zixy.qubit.operator",
    "qubit.state": "zixy.qubit.state",
    "fermion.operator.general": "zixy.fermion.operator.general",
    "fermion.operator.normal": "zixy.fermion.operator.normal",
    "fermion.state": "zixy.fermion.state",
}


def check_schema_version(source: SerializedContainer) -> None:
    """Check that a serialized container uses the supported schema version."""
    if source["schema_version"] != __schema_version__:
        raise ValueError(
            f"schema version mismatch: expected {__schema_version__}, got "
            f"{source['schema_version']}"
        )


@overload
def serialize_numpy_array(array: NDArray[int64]) -> list[int]: ...
@overload
def serialize_numpy_array(array: NDArray[float64]) -> list[float]: ...
@overload
def serialize_numpy_array(array: NDArray[complex128]) -> list[list[float]]: ...


def serialize_numpy_array[T: (int64, float64, complex128)](
    array: NDArray[T],
) -> list[int] | list[float] | list[list[float]]:
    """Serialize a NumPy array to a JSON-serializable list."""
    if array.dtype == np.int64:
        return [int(x) for x in array.tolist()]
    elif array.dtype == np.float64:
        return [float(x) for x in array.tolist()]
    return [[float(x.real), float(x.imag)] for x in array.tolist()]


class SerializedContainer(TypedDict):
    """Serialized representation of a container."""

    schema_version: str
    """Version of the schema."""

    package_version: str
    """Version of the package."""

    family: str
    """Family of the container."""

    container: str
    """Name of the container."""

    strings: NotRequired[list[Any]]
    """List of strings in the container."""

    coeffs: NotRequired[list[Any]]
    """List of coefficients in the container."""

    space: NotRequired[dict[str, Any]]
    """Space for the domain of the container."""


class Serializable[SerializedT: Mapping[str, object]](ABC):
    """Abstract base class for classes that can be serialized to and deserialized from JSON."""

    @abstractmethod
    def to_dict(self) -> SerializedT:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A JSON-serializable dictionary representation of ``self``.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, source: SerializedT) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        pass

    def to_json(self, **kwargs: Any) -> str:
        """Return a JSON string representation of ``self``.

        Args:
            **kwargs: Keyword arguments to forward to ``json.dumps``.

        Returns:
            A JSON string representation of ``self``.
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, source: str, **kwargs: Any) -> Self:
        """Create an instance of ``cls`` from a JSON string representation.

        Args:
            source: JSON string representation to parse.
            **kwargs: Keyword arguments to forward to ``json.loads``.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        return cls.from_dict(json.loads(source, **kwargs))


class SerializableContainer(Serializable[SerializedContainer], ABC):
    """Abstract base class for containers that can be serialized to and deserialized from JSON."""

    def to_dict(self) -> SerializedContainer:
        """Return the common metadata for a serialized container.

        Returns:
            A new dictionary containing the schema version, package version, concrete class name,
            and the default family ``"container"``.

        Note:
            Subclasses update this dictionary with their family, contents, and space as needed.
            This base implementation contains no container contents or view metadata.
        """
        return dict(
            schema_version=__schema_version__,
            package_version=__version__,
            family="container",
            container=type(self).__name__,
        )


def serialize(obj: SerializableContainer, **kwargs: Any) -> str:
    """Return a JSON string representation of a Zixy container.

    Args:
        obj: Coefficient, string, or term container to serialize.
        **kwargs: Keyword arguments to forward to ``json.dumps``.

    Returns:
        A JSON string containing the container metadata and an independent snapshot of its
        visible contents and space, using the container's serialization format.
    """
    if not isinstance(obj, SerializableContainer):
        raise TypeError("Object must be a serializable Zixy container.")
    return obj.to_json(**kwargs)


def deserialize(source: str) -> SerializableContainer:
    """Create a Zixy container from a JSON string representation.

    Args:
        source: JSON string containing ``schema_version``, ``package_version``,
            ``family``, and ``container``, together with the encoded contents and space.

    Returns:
        A new container of the class identified by the family and container tags, owning
        independent storage. The container class determines the string and coefficient types.
    """
    if not isinstance(source, str):
        raise TypeError("Serialized source must be a JSON string.")
    data = json.loads(source)
    if not isinstance(data, dict):
        raise TypeError("Serialized JSON must contain an object.")
    metadata: dict[str, str] = {}
    for field in ("schema_version", "package_version", "family", "container"):
        if field not in data:
            raise ValueError(f"Missing serialization metadata: {field!r}.")
        value = data[field]
        if not isinstance(value, str):
            raise TypeError(f"Serialization metadata {field!r} must be a string.")
        metadata[field] = value

    document: SerializedContainer = dict(
        schema_version=metadata["schema_version"],
        package_version=metadata["package_version"],
        family=metadata["family"],
        container=metadata["container"],
    )
    check_schema_version(document)
    if "strings" in data:
        if not isinstance(data["strings"], list):
            raise TypeError("Serialized strings must be a list.")
        document["strings"] = data["strings"]
    if "coeffs" in data:
        if not isinstance(data["coeffs"], list):
            raise TypeError("Serialized coefficients must be a list.")
        document["coeffs"] = data["coeffs"]
    if "space" in data:
        if not isinstance(data["space"], dict):
            raise TypeError("Serialized space must be a dictionary.")
        document["space"] = data["space"]

    module_name = _FAMILY_MODULES.get(document["family"])
    if module_name is None:
        raise ValueError(f"Unknown serialization family: {document['family']!r}.")
    cls = getattr(import_module(module_name), document["container"], None)
    if not isinstance(cls, type) or not issubclass(cls, SerializableContainer) or isabstract(cls):
        raise ValueError(
            f"Unknown serializable container {document['container']!r} "
            f"in family {document['family']!r}."
        )
    return cls.from_dict(document)
