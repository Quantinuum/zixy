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

"""Individual terms and term specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

from sympy import Expr

from zixy.container.base import Representable, ViewableItem
from zixy.container.coeff import ComplexSign, Sign, convert, typesafe_mul, unit
from zixy.container.data import TermData, _TermData
from zixy.container.serialization import SerializableContainer, check_schema_version
from zixy.container.string import String
from zixy.container.utils import _coeffs_from_str, _parse_term_str
from zixy.utils import classproperty, slice_index

if TYPE_CHECKING:
    from zixy._zixy import ImplArray
    from zixy.container.base import _HasStringsType, _HasStringType
    from zixy.container.coeff import Coeff
    from zixy.container.coeffs import Coeffs
    from zixy.container.serialization import SerializedContainer
    from zixy.container.strings import Strings

type _AnyCoeffTerm[ImplT: ImplArray, SpecT] = (
    Term[ImplT, SpecT, float]
    | Term[ImplT, SpecT, complex]
    | Term[ImplT, SpecT, Sign]
    | Term[ImplT, SpecT, ComplexSign]
    | Term[ImplT, SpecT, Expr]
)

type TermSpecT[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)] = (
    String[ImplT, SpecT]
    | SpecT
    | tuple[SpecT | String[ImplT, SpecT], C | None]
    | _AnyCoeffTerm[ImplT, SpecT]
)


class Term[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    ViewableItem[_TermData[ImplT, SpecT, C]],
    Representable,
    SerializableContainer,
):
    """A term consisting of a string and a coefficient.

    A single term consisting of a string and a coefficient that may be an owning instance
    referencing the sole element in an owned :class:`~zixy.container.data.TermData` instance, or
    a view on an element in another collection.
    """

    coeffs_type: type[Coeffs[C]]
    strings_type: type[Strings[ImplT, SpecT]]

    _impl: _TermData[ImplT, SpecT, C]
    _index: int | None

    def __init__(self, data: _TermData[ImplT, SpecT, C], indexer: int | None = None):
        """Initialize the term.

        Args:
            data: Raw term data object, of which ``self`` views one.
            indexer: Index of the term within ``data`` that ``self`` views.
        """
        if type(data.strings) is not self.strings_type:
            raise TypeError(
                f"Term data strings type {type(data.strings)} does not match "
                f"the strings type {self.strings_type} of this {type(self)}."
            )
        if data.coeffs.coeff_type is not self.coeff_type:
            raise TypeError(
                f"Term data coeff type {data.coeffs.coeff_type} does not match "
                f"the coeff type {self.coeff_type} of this {type(self)}."
            )
        if not isinstance(data.coeffs, self.coeffs_type):
            raise TypeError("Coefficient storage does not match its factory")
        if indexer is None and len(data) != 1:
            raise ValueError("Owning Term must have exactly one element in its storage.")
        if indexer is not None:
            indexer = slice_index(slice(None), indexer, len(data))
        self._impl = data
        self._index = indexer

    @classmethod
    def _create(cls, data: _TermData[ImplT, SpecT, C]) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            data: Term data object containing the data for this instance.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.

        Raises:
            TypeError: The storage does not match the concrete container factories.
            ValueError: The storage does not contain exactly one term.
        """
        out = cls.__new__(cls)
        Term.__init__(out, data)
        return out

    @classmethod
    def _view(cls, data: _TermData[ImplT, SpecT, C], indexer: int) -> Self:
        """Create a mutable view on an existing term.

        Args:
            data: Term data object containing the data for this instance.
            indexer: Index selecting the viewed term.

        Returns:
            A non-owning instance of ``cls`` sharing the selected term without copying it.

        Raises:
            IndexError: The index is outside the storage.
        """
        out = cls.__new__(cls)
        Term.__init__(out, data, indexer)
        return out

    @classmethod
    def from_string(cls, string: String[ImplT, SpecT], coeff: C | None = None) -> Self:
        """Create an owning term from a string and an optional coefficient.

        Args:
            string: String to copy from.
            coeff: Coefficient scaling the string. If omitted or ``None``, use the unit value
                of the destination coefficient type.

        Returns:
            An independent owning instance of ``cls`` with the copied string, preserving its
            space and configuration, and the given or unit coefficient.
        """
        strings = cls.strings_type.from_string(string)
        coeffs = cls.coeffs_type()
        coeffs.append(unit(cls.coeff_type) if coeff is None else coeff)
        term_data = TermData(strings, coeffs)
        return cls._create(term_data)

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        return type(self)._create(self._impl.clone(self.index))

    @classproperty
    def coeff_type(cls: type[Term[ImplT, SpecT, C]]) -> type[C]:
        """The scalar type supplied by the coefficient container."""
        return cls.coeffs_type.coeff_type

    @property
    def string_type[StringT: String[Any, Any]](
        self: _HasStringsType[_HasStringType[StringT]],
    ) -> type[StringT]:
        """The concrete string type of ``self``."""
        term = cast("Term[Any, Any, Any]", self)
        return cast("type[StringT]", term.strings_type.string_type)

    @property
    def _data(self) -> _TermData[ImplT, SpecT, C]:
        """The raw term data object underlying ``self``."""
        return self._impl

    @property
    def string[StringT: String[Any, Any]](
        self: _HasStringsType[_HasStringType[StringT]],
    ) -> StringT:
        """A view of the string of ``self``, preserving its concrete type."""
        term = cast("Term[Any, Any, Any]", self)
        return cast("StringT", term._data.strings[term.index])

    @property
    def coeff(self) -> C:
        """A copy of the coefficient of ``self``."""
        return self._data.coeffs[self.index]

    @coeff.setter
    def coeff(self, value: C) -> None:
        """Set the coefficient of ``self``.

        Args:
            value: Value to assign.

        Note:
            This method operates in-place.
        """
        self._data.coeffs[self.index] = convert(value, self._data.coeffs.coeff_type)

    def set(self, source: TermSpecT[ImplT, SpecT, C]) -> None:
        """Assign a specification, string/coefficient pair, or compatible term.

        Args:
            source: Value to copy or parse. Compatible parent and subclass terms are supported.
                A string-only input sets the coefficient to unity; a ``None`` coefficient in a
                pair also means unity. Coefficients are converted to this term's coefficient type.

        Note:
            This method operates in-place. A two-element tuple is first interpreted as a string
            specification, so tuples of Pauli matrices and normal-ordered operator sets retain
            their domain-specific meaning.
        """
        if isinstance(source, Term):
            self._set_string_coeff(source.string, source.coeff)
            return
        try:
            # A pair can also be a valid domain specification; try that interpretation first.
            self.string.set(cast("SpecT | String[ImplT, SpecT]", source))
        except (TypeError, ValueError):
            if not isinstance(source, tuple) or len(source) != 2:
                raise
        else:
            self.coeff = unit(self.coeff_type)
            return
        string, coeff = source
        self._set_string_coeff(string, coeff)

    def _set_string_coeff(self, string: SpecT | String[ImplT, SpecT], coeff: Coeff | None) -> None:
        """Validate coefficient conversion before copying or parsing the string."""
        value = unit(self.coeff_type) if coeff is None else convert(coeff, self.coeff_type)
        self.string.set(string)
        self.coeff = value

    def clear(self) -> None:
        """Set the value of the string to zero and the coefficient to unity.

        Note:
            This method operates in-place.
        """
        self.set("")  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, Term):
            return NotImplemented
        result: bool = (
            self._impl.strings._impl.strings_eq(
                [self.index], other._impl.strings._impl, [other.index]
            )
            and self._impl.coeffs[self.index] == other._impl.coeffs[other.index]
        )
        return result

    def aliases(self, other: Term[Any, Any, Any]) -> bool:
        """Determine whether ``self`` is a view of the same string as ``other``."""
        if type(self) is not type(other):
            return False
        return self.string.aliases(other.string)

    def _identity(self) -> Self:
        """Create a multiplicative identity of the same space and configuration as ``self``."""
        data = self._impl._new()
        data.resize(1)
        data.coeffs[0] = unit(self.coeff_type)
        return self._create(data)

    def __pos__(self) -> Self:
        """Return ``self``."""
        return self.clone()

    def __neg__(self) -> Self:
        """Return the negation of ``self``."""
        out = self.clone()
        out.coeff = typesafe_mul(out.coeff, -1)
        return out

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        coeff = repr(self.coeff)
        coeff = coeff.replace("-0j", "+0j")
        return f"({coeff}, {self.string})"

    @classmethod
    def from_str(cls, source: str, *args: Any, **kwargs: Any) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            *args: Positional arguments to forward to the string constructor.
            **kwargs: Keyword arguments to forward to the string constructor.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        source_coeff, source_string = _parse_term_str(source)
        strings = cls.strings_type.from_str(source_string, *args, **kwargs)
        coeffs = _coeffs_from_str(source_coeff, cls.coeffs_type, len(strings))
        data = TermData(strings, coeffs)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            An independent dictionary containing the container metadata, full space, and
            one string with its coefficient, using their respective serialization formats.
        """
        data = self.string.to_dict()
        data["container"] = type(self).__name__
        data["coeffs"] = self._data.coeffs[self.index : self.index + 1].to_dict()["coeffs"]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: Dictionary containing the schema version, space, strings, and coefficients.
                String and coefficient values must be compatible with the factories of ``cls``.

        Returns:
            A new owning instance of ``cls`` with independent string and coefficient storage.
        """
        check_schema_version(source)
        strings = cls.strings_type.from_dict(source)
        coeffs = cls.coeffs_type.from_dict(source)
        data = TermData(strings, coeffs)
        if len(data) != 1:
            raise ValueError("A serialized Term must contain exactly one term.")
        return cls._create(data)
