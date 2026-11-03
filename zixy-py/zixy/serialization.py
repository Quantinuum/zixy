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
from importlib import import_module
from inspect import isabstract
from typing import TYPE_CHECKING

from zixy.container.base import SerializableContainer, check_schema_version

if TYPE_CHECKING:
    from typing import Any

    from zixy.container.base import SerializedContainer

__all__ = ["serialize", "deserialize"]

_FAMILY_MODULES = {
    "container": "zixy.container.coeffs",
    "qubit.operator": "zixy.qubit.operator",
    "qubit.state": "zixy.qubit.state",
    "fermion.operator.general": "zixy.fermion.operator.general",
    "fermion.operator.normal": "zixy.fermion.operator.normal",
    "fermion.state": "zixy.fermion.state",
}


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
