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

"""Coefficient vectors and conversions."""

from __future__ import annotations

import builtins
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np
from sympy import Expr, Float, Integer, Symbol, diff, sympify

from zixy import __version__, _zixy
from zixy.container.base import Representable, ViewableSequence, requires_ownership
from zixy.container.coeff import (
    Coeff,
    ComplexSign,
    Number,  # noqa: TC001 (runtime type hints)
    Sign,
    _integer_exponent,
    convert,
    normalize_scalar,
    typesafe_mul,
    unit,
)
from zixy.container.serialization import (
    SerializableContainer,
    __schema_version__,
    check_schema_version,
    serialize_numpy_array,
)
from zixy.utils import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    slice_index,
    slice_index_gen,
    slice_len,
    slice_of_slice,
    split_top_level,
)

__all__ = [
    "Coeffs",
    "SignCoeffs",
    "ComplexSignCoeffs",
    "RealCoeffs",
    "ComplexCoeffs",
    "SymbolicCoeffs",
]


if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike, NDArray

    from zixy.container.serialization import SerializedContainer

    BaseVec = _zixy.BaseVec
else:
    BaseVec = object


class Coeffs[C: (float, complex, Sign, ComplexSign, Expr)](
    ViewableSequence[C, BaseVec],
    Representable,
    SerializableContainer,
):
    """A collection of coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type: type[C]
    impl_type: type[_zixy.BaseVec]

    _impl: _zixy.BaseVec

    def __init__(self, *, size: int = 0):
        """Allocate an owning vector of unit coefficients.

        Args:
            size: Number of unit coefficients with which to initialize the vector.

        Raises:
            ValueError: The size is negative.
        """
        if size < 0:
            raise ValueError("size must be non-negative.")
        self._bind(self.impl_type(size), slice(None))

    def _bind(self, data: _zixy.BaseVec, indexer: slice) -> None:
        """Install coefficient storage and its selection without allocating or copying.

        Args:
            data: Coefficient storage matching the concrete vector factory.
            indexer: Slice selecting the coefficients; ``slice(None)`` denotes ownership.
        """
        assert type(data) is self.impl_type, (type(data), self.impl_type)
        self._impl = data
        self._slice = indexer

    @classmethod
    def _create(cls, data: _zixy.BaseVec) -> Self:
        """Create an owning instance of ``cls``.

        Args:
            data: Rust-bound object containing the data for this instance.

        Returns:
            A new owning instance of ``cls`` sharing the supplied storage.
        """
        out = cls.__new__(cls)
        out._bind(data, slice(None))
        return out

    @classmethod
    def _view(cls, data: _zixy.BaseVec, indexer: slice) -> Self:
        """Create a mutable view on existing storage.

        Args:
            data: Rust-bound object containing the data for this instance.
            indexer: Slice selecting the viewed coefficients.

        Returns:
            A non-owning instance of ``cls`` sharing the selected storage without copying it.
        """
        out = cls.__new__(cls)
        if indexer == slice(None):
            indexer = slice(0, len(data), 1)
        out._bind(data, indexer)
        return out

    def _new(self) -> Self:
        """Create an empty instance with the same type and configuration."""
        return self._create(self.impl_type(0))

    @classmethod
    def from_scalar(cls, coeff: C, n: int = 1) -> Self:
        """Get a new instance of ``cls`` with the given scalar repeated ``n`` times.

        Args:
            coeff: Coefficient value.
            n: Number of times to repeat ``coeff``.

        Returns:
            New instance with ``coeff`` repeated ``n`` times.
        """
        out = cls()
        out.append_n(n, coeff)
        return out

    @classmethod
    def from_iterable(cls, source: Iterable[C]) -> Self:
        """Create an owning vector from an iterable of coefficients.

        Args:
            source: Iterable of coefficients.

        Returns:
            Owning instance containing the coefficients in ``source``.
        """
        out = cls()
        for coeff in source:
            out.append(coeff)
        return out

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        out = self._new()
        for c in self:
            out.append(c)
        return out

    def map_index(self, index: int) -> int:
        """Map an index in ``self`` to an index in the underlying data.

        Args:
            index: Index in ``self``.

        Returns:
            Corresponding index in the underlying data.
        """
        return slice_index(self.slice, index, len(self._impl))

    @overload
    def __getitem__(self, indexer: int) -> C: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> C | Self:
        """Get the element or elements selected by ``indexer``.

        Args:
            indexer: Index or slice selecting the coefficient(s) to return.

        Returns:
            Coefficient or slice selected by ``indexer``.
        """
        if isinstance(indexer, builtins.slice):
            return type(self)._view(
                self._impl, slice_of_slice(self.slice, indexer, len(self._impl))
            )
        if issubclass(self.coeff_type, Sign) or issubclass(self.coeff_type, ComplexSign):
            coeff = self.coeff_type(self._impl[self.map_index(indexer)].to_phase())
        else:
            coeff = self._impl[self.map_index(indexer)]
        if not isinstance(coeff, self.coeff_type):
            raise TypeError(f"Expected coefficient of type {self.coeff_type}, got {type(coeff)}")
        return coeff

    def _set_scalar(self, index: int, value: C | None = None) -> None:
        """Set the indexed element to the given value.

        Args:
            index: Element within the view to set.
            value: Coefficient to set the element to.
        """
        if value is None:
            # None indicates the default value (i.e. unity)
            self[index] = unit(self.coeff_type)
            return
        if issubclass(self.coeff_type, Sign) or issubclass(self.coeff_type, ComplexSign):
            data = value._impl
        else:
            data = value
        self._impl[self.map_index(index)] = data

    def __setitem__(
        self, indexer: int | builtins.slice, source: C | Coeffs[C] | None = None
    ) -> None:
        """Set the indexed element(s) to the given value(s).

        Args:
            indexer: Index or slice of coefficient(s) within ``self`` to assign.
            source: Value(s) specifying the coefficient(s) to assign.
        """
        if isinstance(indexer, builtins.slice):
            if isinstance(source, Coeffs):
                if source._impl is self._impl:
                    # internal copy, need to ensure source is not overwritten in write.
                    source = source.clone()
                n = slice_len(indexer, len(self))
                # write a slice to a slice
                if n != len(source):
                    raise ValueError(
                        f"Length of source ({len(source)}) does not match "
                        f"length of destination ({n})"
                    )
                for i_src, i_dst in enumerate(slice_index_gen(indexer, len(self))):
                    self._set_scalar(i_dst, source[i_src])
            else:
                # write the same term value to each element of the slice.
                for i in slice_index_gen(indexer, len(self)):
                    self._set_scalar(i, source)
        else:
            if isinstance(source, Coeffs):
                raise ValueError(
                    f"Cannot assign a {self.__class__.__name__} instance to an integer indexer."
                )
            self._set_scalar(indexer, source)

    def set(self, source: Iterable[C]) -> None:
        """Assign the viewed coefficients according to the values yielded by the given iterable.

        Args:
            source: Iterable containing the values to be assigned to the viewed coefficients of
                ``self``.
        """
        for i, c in enumerate(source):
            self[i] = convert(c, self.coeff_type)

    def fill(self, coeff: C) -> None:
        """Set all the coefficients in the vector to the given value.

        Args:
            coeff: Coefficient value to which to set all coefficients in the vector.
        """
        for i in range(len(self)):
            self[i] = coeff

    @requires_ownership
    def append(self, value: C) -> None:
        """Append ``value`` to the end of ``self``.

        Args:
            value: Value to append.

        Note:
            This method operates in-place.
        """
        self.append_n(1, value)

    @requires_ownership
    def append_n(self, n: int, value: C) -> None:
        """Append ``value`` to the end of ``self`` ``n`` times.

        Args:
            n: Number of times to repeatedly append ``value``.
            value: Value to append.

        Note:
            This method operates in-place.
        """
        for _ in range(n):
            if issubclass(self.coeff_type, Sign) or issubclass(self.coeff_type, ComplexSign):
                self._impl.append(value._impl)
            else:
                self._impl.append(value)

    @requires_ownership
    def extend(self, source: Iterable[C]) -> None:
        """Append the elements of ``source`` to the end of ``self``.

        Args:
            source: Iterable of compatible coefficients.

        Raises:
            ValueError: If ``self`` is a non-owning view, including for an empty source.

        Note:
            This method operates in-place.
        """
        if isinstance(source, Coeffs) and source.impl_type is self.impl_type and source.is_owning():
            if self._impl.same_as(source._impl):
                self._impl.extend_internal()
            else:
                self._impl.extend_external(source._impl)
            return
        for value in source:
            self.append(value)

    @requires_ownership
    def swap_remove(self, index: int) -> None:
        """Set the element at ``index`` to the final element, then delete the final element.

        Args:
            index: Element index to remove.

        Note:
            This method operates in-place.
        """
        self[index] = self[-1]
        self.resize(len(self) - 1)

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, Coeffs):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=False))

    def allclose(
        self, other: Coeffs[Any], rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        """Check whether ``self`` and ``other`` are within a certain tolerance.

        Args:
            other: Other instance to compare to.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Whether ``self`` and ``other`` are within the given tolerances of each other.

        Note:
            This method operates by comparing the NumPy arrays returned by
            :meth:`~zixy.container.coeffs.Coeffs.to_numpy`.
        """
        if len(self) != len(other):
            return False
        if self is other:
            return True
        return np.allclose(self.to_numpy(), other.to_numpy(), rtol=rtol, atol=atol)

    def __pos__(self) -> Self:
        """Return ``self``."""
        return self.clone()

    def __neg__(self) -> Self:
        """Return the negation of ``self``."""
        out: Self = self._new()
        for item in self:
            out.append(-item)
        return out

    def __imul__(self, rhs: Coeff | Coeffs[C] | Iterable[Coeff]) -> Self:
        """Multiply ``self`` by ``rhs`` in-place.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                ``self``.
        """
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Coeff):
            self.scale(rhs)
        elif isinstance(rhs, Coeffs):
            self.scale_by_coeffs(rhs)
        else:
            for i, c in enumerate(rhs):
                if i >= len(self):
                    break
                self._set_scalar(i, typesafe_mul(self[i], c))
        return self

    def scale(self, scalar: Coeff) -> None:
        """Scale all elements by a given factor.

        Args:
            scalar: Scalar factor by which to scale all elements.

        Note:
            This method operates in-place.
        """
        scalar = normalize_scalar(scalar)
        for i in range(len(self)):
            self[i] = typesafe_mul(self[i], scalar)

    def scale_by_coeffs(self, coeffs: Coeffs[C]) -> None:
        """Scale all elements by the corresponding elements of ``coeffs``.

        Args:
            coeffs: Coefficients by which to scale the elements of ``self``.

        Note:
            This method operates in-place.
        """
        if len(coeffs) != len(self):
            raise ValueError(
                f"Length of coeffs ({len(coeffs)}) does not match length of self ({len(self)})"
            )

        if type(coeffs) is type(self) and self._impl.same_as(coeffs._impl):
            coeffs = coeffs.clone()

        for i in range(len(self)):
            self[i] = typesafe_mul(self[i], coeffs[i])

    def __pow__(self, exponent: int | np.integer[Any]) -> Self:
        """Return a new collection with each coefficient raised to an integer power."""
        exponent = _integer_exponent(exponent)
        out = self._new()
        for coeff in self:
            out.append(coeff**exponent)
        return out

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of ``self`` in a new order.

        Args:
            inds: Sequence of indices defining the new order. Should be a permutation of
                ``range(len(self))``.

        Returns:
            A new instance with the reordered elements.
        """
        if not all(i < len(self) for i in inds):
            raise IndexError("Index out of bounds in reorder.")
        out = self._new()
        for i in inds:
            out.append(self[i])
        return out

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return "[" + (", ".join(str(c) for c in self)) + "]"

    @classmethod
    def from_str(cls, source: str) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.

        Note:
            The string is expected to be a comma-separated list of coefficients, optionally enclosed
            in brackets. Coefficients may be enclosed in parentheses.
        """
        # Remove any brackets and whitespace
        string = source.strip()
        if string.startswith("[") != string.endswith("]"):
            raise ValueError(f"Mismatched brackets in string '{source}'.")
        string = string.removeprefix("[").removesuffix("]").strip()

        # Remove any parentheses and whitespace from each element
        items = [
            item.strip().removeprefix("(").removesuffix(")").strip()
            for item in split_top_level(string, ",", keep_empty=True)
        ]
        string = ", ".join(item for item in items if item)
        if any(not item for item in items) and string:
            raise ValueError(f"Invalid comma separator in string '{source}'.")

        # Parse the string
        out = cls()
        if not string:
            out._impl = cls.impl_type(0)
        else:
            out._impl = cls.impl_type.parse(string)

        return out

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A JSON-serializable dictionary representation of ``self``.
        """
        return dict(
            schema_version=__schema_version__,
            package_version=__version__,
            family="container",
            container=self.__class__.__name__,
            coeffs=serialize_numpy_array(self.to_numpy()),
        )

    def to_tuple(self) -> tuple[C, ...]:
        """Get a tuple of clones of the elements of ``self``."""
        return tuple(c for c in self)

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> NDArray[Any]:
        """Convert the coefficients through NumPy's array protocol.

        Args:
            dtype: Requested NumPy dtype, or ``None`` to use the native numeric dtype.
            copy: Whether copying is allowed. ``None`` and ``True`` return independent
                arrays; ``False`` is unsupported because conversion requires a copy.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order,
            converted to the requested dtype if supplied.
        """
        if copy is False:
            raise ValueError(
                "Coefficient array conversion requires a copy; copy=False is unsupported."
            )
        array = self.to_numpy()
        if dtype is None:
            return array
        requested_dtype = np.dtype(dtype)
        if np.iscomplexobj(array) and requested_dtype.kind in "buif":
            if np.any(array.imag != 0):
                raise ValueError(
                    "Cannot convert coefficients with nonzero imaginary parts to real."
                )
            array = array.real
        return array.astype(requested_dtype, copy=False)

    @abstractmethod
    def to_numpy(self) -> NDArray[np.int64] | NDArray[np.float64] | NDArray[np.complex128]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.
        """
        pass


class RootOfUnityCoeffs[C: (float, complex, Sign, ComplexSign, Expr)](Coeffs[C]):
    """A collection of :class:`RootOfUnity` coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    impl_type: type[_zixy.SignVec | _zixy.ComplexSignVec]


class SignCoeffs(RootOfUnityCoeffs[Sign]):
    """A collection of :class:`Sign`.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = Sign
    impl_type = _zixy.SignVec

    _impl: _zixy.SignVec

    @classmethod
    def from_phases(cls, source: NDArray[np.bool_] | Sequence[bool]) -> Self:
        """Get a new instance from a sequence of phases.

        Args:
            source: Sequence of phases.

        Returns:
            New instance with elements set to phases according to ``source``.
        """
        out = cls()
        if isinstance(source, np.ndarray) and source.dtype == np.bool_:
            out._impl = _zixy.SignVec.from_phases(source)
        else:
            out._impl = _zixy.SignVec.from_phases(
                np.array([i % 2 == 1 for i in source], dtype=bool)
            )
        assert len(out) == len(source)
        return out

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        check_schema_version(source)
        return cls.from_iterable([cls.coeff_type.from_scalar(value) for value in source["coeffs"]])

    def to_numpy(self) -> NDArray[np.int64]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.
        """
        return np.array([int(s) for s in self], dtype=np.int64).ravel()


class ComplexSignCoeffs(RootOfUnityCoeffs[ComplexSign]):
    """A collection of :class:`ComplexSign`.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = ComplexSign
    impl_type = _zixy.ComplexSignVec

    _impl: _zixy.ComplexSignVec

    @classmethod
    def from_phases(cls, source: NDArray[np.uint8] | Sequence[int]) -> Self:
        """Get a new instance from a sequence of phases.

        Args:
            source: Sequence of phases.

        Returns:
            New instance with elements set to phases according to ``source``.
        """
        out = cls()
        if isinstance(source, np.ndarray) and source.dtype == np.uint8:
            out._impl = _zixy.ComplexSignVec.from_phases(source)
        else:
            out._impl = _zixy.ComplexSignVec.from_phases(
                np.array([i for i in source], dtype=np.uint8)
            )
        assert len(out) == len(source)
        return out

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        check_schema_version(source)
        return cls.from_iterable(
            [cls.coeff_type.from_scalar(complex(real, imag)) for real, imag in source["coeffs"]]
        )

    def to_numpy(self) -> NDArray[np.complex128]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.
        """
        return np.array([complex(s) for s in self], dtype=np.complex128).ravel()


class NumericalCoeffs[C: (float, complex)](Coeffs[C]):
    """A collection of numerical coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    impl_type: type[_zixy.RealVec | _zixy.ComplexVec]
    _impl: _zixy.RealVec | _zixy.ComplexVec

    @classmethod
    def from_iterable(cls, source: Iterable[C] | NDArray[np.inexact]) -> Self:
        """Create an owning numeric vector from an iterable of coefficients.

        Args:
            source: Iterable of coefficients.

        Returns:
            Owning instance containing the coefficients in ``source``.
        """
        if isinstance(source, Sequence | np.ndarray):
            values = np.asarray(source, dtype=cls.coeff_type)
        else:
            values = np.fromiter(source, dtype=cls.coeff_type)
        return cls._create(cls.impl_type.from_array(values))

    def clone(self) -> Self:
        """Copy the viewed coefficients into an independent contiguous vector."""
        return self._create(self._impl.copy_slice(self.slice))

    def __neg__(self) -> Self:
        """Return a negated copy of the viewed coefficients."""
        out = self.clone()
        out.scale(-1)
        return out

    def any_significant(self, atol: float = DEFAULT_ATOL) -> bool:
        """Check whether any element of ``self`` is significantly different from zero.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether any element of ``self`` is significantly different from zero.
        """
        return not np.allclose(self.to_numpy(), 0, rtol=0, atol=atol)


def _real_from_scalar(value: float | complex) -> float:
    """Convert a numeric scalar to real without discarding an imaginary part."""
    if isinstance(value, complex | np.complexfloating):
        if value.imag != 0:
            raise ValueError("Cannot convert coefficients with nonzero imaginary parts to real.")
        return float(value.real)
    return float(value)


class RealCoeffs(NumericalCoeffs[float]):
    """A collection of ``float``.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = float
    impl_type = _zixy.RealVec

    _impl: _zixy.RealVec

    @classmethod
    def from_iterable(cls, source: Iterable[float | complex] | NDArray[np.inexact]) -> Self:
        """Create an owning real vector from an iterable of numeric coefficients.

        Args:
            source: Iterable of real or complex coefficients. Complex values are
                accepted only when their imaginary parts are exactly zero.

        Returns:
            An independent owning instance containing the coefficients in source order.

        Raises:
            ValueError: A coefficient has a nonzero imaginary part or the source array
                is not one-dimensional.
            TypeError: A coefficient cannot be converted to a real number.
        """
        if isinstance(source, Sequence | np.ndarray):
            values = np.asarray(source)
            if values.ndim != 1:
                raise ValueError("Coefficient arrays must be one-dimensional.")
            if np.iscomplexobj(values):
                if np.any(values.imag != 0):
                    raise ValueError(
                        "Cannot convert coefficients with nonzero imaginary parts to real."
                    )
                values = values.real
            if values.dtype.kind != "O":
                return super().from_iterable(values)
            source = values
        return super().from_iterable(_real_from_scalar(value) for value in source)

    def fill(self, coeff: float) -> None:
        """Fill the viewed real coefficients in Rust."""
        if len(self):
            self._impl.fill_slice(self.slice, convert(coeff, float))

    def scale(self, scalar: Coeff) -> None:
        """Scale the viewed real coefficients in Rust."""
        scalar = normalize_scalar(scalar)
        if len(self):
            self._impl.scale_slice(self.slice, convert(scalar, float))

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        check_schema_version(source)
        return cls.from_iterable([cls.coeff_type(value) for value in source["coeffs"]])

    def to_numpy(self) -> NDArray[np.float64]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.
        """
        impl = self._impl if self.is_owning() else self._impl.copy_slice(self.slice)
        return np.asarray(impl.to_array(), dtype=np.float64)


class ComplexCoeffs(NumericalCoeffs[complex]):
    """A collection of ``complex``.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = complex
    impl_type = _zixy.ComplexVec

    _impl: _zixy.ComplexVec

    @property
    def real_part(self) -> RealCoeffs:
        """The real part of ``self`` as a new instance of :class:`RealCoeffs`."""
        return RealCoeffs.from_iterable(self.to_numpy().real)

    @property
    def imag_part(self) -> RealCoeffs:
        """The imaginary part of ``self`` as a new instance of :class:`RealCoeffs`."""
        return RealCoeffs.from_iterable(self.to_numpy().imag)

    def fill(self, coeff: complex) -> None:
        """Fill the viewed complex coefficients in Rust."""
        if len(self):
            self._impl.fill_slice(self.slice, convert(coeff, complex))

    def scale(self, scalar: Coeff) -> None:
        """Scale the viewed complex coefficients in Rust."""
        scalar = normalize_scalar(scalar)
        if len(self):
            self._impl.scale_slice(self.slice, convert(scalar, complex))

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        check_schema_version(source)
        return cls.from_iterable(
            [
                cls.coeff_type(*value) if isinstance(value, Sequence) else cls.coeff_type(value)
                for value in source["coeffs"]
            ]
        )

    def to_numpy(self) -> NDArray[np.complex128]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.
        """
        impl = self._impl if self.is_owning() else self._impl.copy_slice(self.slice)
        return np.asarray(impl.to_array(), dtype=np.complex128)


class ExprListWrapper(BaseVec):
    """A collection of :class:`~sympy.Expr` coefficients.

    This class is a drop-in replacement for the Rust-bound vector types used for other coefficient
    types for the case of symbolic coefficients.
    """

    def __init__(self, n: int = 0):
        """Initialize the expression vector.

        Args:
            n: The number of items.
        """
        self._list = [sympify(1) for _ in range(n)]

    @classmethod
    def from_list(cls, source: list[Expr]) -> ExprListWrapper:
        """Create an instance of ``cls`` from a list of expressions.

        Args:
            source: The list of expressions to read from.

        Returns:
            An instance of ``cls`` containing the expressions in ``source``.
        """
        out = cls()
        assert all(isinstance(c, Expr) for c in source)
        out._list = source
        return out

    @classmethod
    def from_coeffs(cls, source: Coeff | Sequence[Coeff]) -> ExprListWrapper:
        """Create an instance of ``cls`` from a coefficient or sequence of coefficients.

        Args:
            source: The coefficient or sequence of coefficients to read from.

        Returns:
            An instance of ``cls`` containing the expressions corresponding to the coefficients
            in ``source``.
        """
        out = cls()
        out._list = ExprListWrapper._sympify_coeffs(source)
        return out

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        return len(self._list)

    def __getitem__(self, index: int) -> Expr:
        """Get the element or elements selected by ``index``.

        Args:
            index: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by ``index``.
        """
        return self._list[index]

    def __setitem__(self, index: int, value: Coeff) -> None:
        """Set the indexed element(s) to the given value(s).

        Args:
            index: Index or slice of coefficient(s) within ``self`` to assign.
            value: Value(s) specifying the coefficient(s) to assign.
        """
        self._list[index] = ExprListWrapper.simplify_integer_floats(sympify(value))

    @staticmethod
    def _sympify_coeff(coeff: Coeff) -> Expr:
        """Convert a coefficient to a SymPy expression.

        Args:
            coeff: Value to sympify.

        Returns:
            A sympy expression
        """
        if isinstance(coeff, Expr):
            return coeff
        elif isinstance(coeff, int | float | complex):
            return sympify(coeff)
        elif isinstance(coeff, Sign):
            return sympify(int(coeff))
        elif isinstance(coeff, ComplexSign):
            return sympify(complex(coeff))
        raise TypeError("Not a scalar convertible to Expr.")

    @staticmethod
    def _sympify_coeffs(coeffs: Coeff | Sequence[Coeff]) -> list[Expr]:
        """Convert a coefficient or sequence of coefficients to a list of SymPy expressions.

        Args:
            coeffs: Coefficient or sequence of coefficients to sympify.

        Returns:
            A list of sympy expressions.
        """
        if isinstance(coeffs, Sequence):
            return [ExprListWrapper._sympify_coeff(coeff) for coeff in coeffs]
        return [ExprListWrapper._sympify_coeff(coeffs)]

    def append(self, value: Expr) -> None:
        """Append ``value`` to the end of ``self``.

        Args:
            value: Value to append.

        Note:
            This method operates in-place.
        """
        self._list.append(sympify(value))

    def resize(self, n: int) -> None:
        """Resize the underlying container.

        Args:
            n: The new size of the container.

        Raises:
            ValueError: If the container is a view.

        Note:
            This method operates in-place.
        """
        self._list.extend([sympify(1) for _ in range(n - len(self))])
        self._list = self._list[:n]
        assert len(self) == n

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, ExprListWrapper):
            return NotImplemented
        return self.same_as(other) or self._list == other._list

    def same_as(self, other: ExprListWrapper) -> bool:
        return self._list is other._list

    @staticmethod
    def simplify_integer_floats(expr: Expr) -> Expr:
        """Simplify any floating point numbers that are exactly representable as integers.

        Args:
            expr: The expression to simplify.

        Returns:
            The simplified expression.
        """
        return expr.xreplace({x: Integer(int(x)) for x in expr.atoms(Float) if x.num == int(x.num)})

    @classmethod
    def from_str(cls, source: str) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        out = cls()
        out._list = [
            sympify(s)
            for s in split_top_level(source.strip().removeprefix("[").removesuffix("]"), ",")
        ]
        return out


class SymbolicCoeffs(Coeffs[Expr]):
    """A collection of :class:`~sympy.Expr`.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = Expr
    impl_type = ExprListWrapper

    _impl: ExprListWrapper

    @property
    def free_symbols(self) -> set[Symbol]:
        """The set of free (unsubstituted) symbols in ``self``.

        Returns:
            Union of the sets of free symbols across all coefficients in ``self``.
        """
        out = set()
        for coeff in self._impl._list:
            out.update(coeff.free_symbols)
        return out

    @requires_ownership
    def extend(self, source: Iterable[Expr]) -> None:
        """Append the elements of ``source`` to the end of ``self``.

        Args:
            source: Iterable of compatible coefficients.

        Raises:
            ValueError: If ``self`` is a non-owning view, including for an empty source.

        Note:
            This method operates in-place.
        """
        for value in source:
            self.append(value)

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, SymbolicCoeffs):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(
            left.simplify() == right.simplify() for left, right in zip(self, other, strict=False)
        )

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicCoeffs:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        out = self.clone()
        out.isubs(values)
        return out

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        coeffs = [coeff.subs(values) for coeff in self]
        self._impl._list[self.slice] = coeffs

    def diff(self, variable: Symbol | str) -> SymbolicCoeffs:
        """Differentiate partially with respect to ``variable`` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        out = self.clone()
        out.idiff(variable)
        return out

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to ``variable`` in-place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Note:
            This method operates in-place.
        """
        if isinstance(variable, str):
            variable = Symbol(variable)
        coeffs = [diff(coeff, variable) for coeff in self]
        self._impl._list[self.slice] = coeffs

    @classmethod
    def from_str(cls, source: str) -> Self:  # noqa: D102
        raise NotImplementedError("Cannot parse SymbolicCoeffs from string.")

    from_str.__doc__ = Coeffs.from_str.__doc__

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A JSON-serializable dictionary representation of ``self``.
        """
        raise TypeError(
            f"JSON serialization of {type(self).__name__} with symbolic coefficients in not "
            "supported."
        )

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        raise TypeError(
            f"JSON deserialization of {cls.__name__} with symbolic coefficients in not supported."
        )

    def try_to_real(self) -> RealCoeffs:
        """Try to evaluate ``self`` as a vector of real coefficients.

        Returns:
            An instance of :class:`RealCoeffs` with the evaluated coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.
        """
        out = []
        for coeff in self:
            v = coeff.evalf()
            if not np.can_cast(float, type(v), casting="safe"):
                raise TypeError(f"Cannot cast from {type(v)} to float")
            out.append(float(v))
        return RealCoeffs.from_iterable(out)

    def try_to_complex(self) -> ComplexCoeffs:
        """Try to evaluate ``self`` as a vector of complex coefficients.

        Returns:
            An instance of :class:`ComplexCoeffs` with the evaluated coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.
        """
        out = []
        for coeff in self:
            v = coeff.evalf()
            if not np.can_cast(complex, type(v), casting="safe"):
                raise TypeError(f"Cannot cast from {type(v)} to complex")
            out.append(complex(v))
        return ComplexCoeffs.from_iterable(out)

    def to_numpy(self) -> NDArray[np.float64] | NDArray[np.complex128]:
        """Return a flat NumPy array copy of the contents of ``self``.

        Returns:
            An independent one-dimensional NumPy array containing the elements in iteration order.

        Raises:
            TypeError: A coefficient contains free symbols or cannot be evaluated as a real or
                complex number.
        """
        try:
            return self.try_to_real().to_numpy()
        except TypeError:
            return self.try_to_complex().to_numpy()


def convert_vec[Cs: Coeffs[Any]](source: Coeffs[Any], t: type[Cs]) -> Cs:
    """Convert a vector of coefficients to a vector of another coefficient type.

    Args:
        source: The vector of coefficients to convert.
        t: The type of vector to convert to.

    Returns:
        The converted vector of coefficients.
    """
    out = t()
    for c in source:
        out.append(convert(c, t.coeff_type))
    return out


@overload
def get_coeffs_type(coeff_type: type[Sign]) -> type[SignCoeffs]: ...


@overload
def get_coeffs_type(coeff_type: type[ComplexSign]) -> type[ComplexSignCoeffs]: ...


@overload
def get_coeffs_type(coeff_type: type[float]) -> type[RealCoeffs]: ...


@overload
def get_coeffs_type(coeff_type: type[complex]) -> type[ComplexCoeffs]: ...


@overload
def get_coeffs_type(coeff_type: type[Expr]) -> type[SymbolicCoeffs]: ...


@overload
def get_coeffs_type[C: (float, complex, Sign, ComplexSign, Expr)](
    coeff_type: type[C],
) -> type[Coeffs[C]]: ...


def get_coeffs_type[C: (float, complex, Sign, ComplexSign, Expr)](
    coeff_type: type[C],
) -> type[Coeffs[C]]:
    """Get the coefficient vector type corresponding to a given coefficient type.

    Args:
        coeff_type: The coefficient type for which to get the corresponding vector type.

    Returns:
        The coefficient vector type corresponding to ``coeff_type``.
    """
    if issubclass(coeff_type, Sign):
        return SignCoeffs
    elif issubclass(coeff_type, ComplexSign):
        return ComplexSignCoeffs
    elif not issubclass(coeff_type, complex) and issubclass(coeff_type, float | int):
        return RealCoeffs
    elif not issubclass(coeff_type, float | int) and issubclass(coeff_type, complex):
        return ComplexCoeffs
    elif issubclass(coeff_type, Expr):
        return SymbolicCoeffs
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")
