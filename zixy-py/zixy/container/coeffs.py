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

"""Coefficients and coefficient vectors.

Coefficients are scalar values that can be square roots of unity (implemented by :class:`Sign`),
fourth roots of unity (implemented by :class:`ComplexSign`), or a built-in numeric type (e.g.
``int``, ``float``, or ``complex``). Coefficients can also be symbolic expressions (of type
:class:`~sympy.Expr`).

Coefficient vectors (:class:`Coeffs` and subclasses) store contiguous collections of the
coefficients in underlying Rust-bound data objects. They can either own the data they reference,
or reference a slice of the data in another vector.
"""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Float, I, Integer, Symbol, diff, sympify
from typing_extensions import Self, TypeIs

from zixy import _zixy
from zixy.container.base import ViewableSequence, requires_ownership
from zixy.utils import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    slice_index,
    slice_index_gen,
    slice_len,
    slice_of_slice,
)

Number: TypeAlias = int | float | complex
NumberT = TypeVar("NumberT", float, complex)


def _imul_factor_error(lhs: Any, rhs: Any, factor: Any) -> ValueError:
    """Get an exception for an invalid in-place multiplication due to the resulting factor."""
    return ValueError(
        f"Cannot multiply-assign left-hand operand {lhs} with type {type(lhs)} by value {rhs}. "
        f"Result with factor {factor} is not representable by the left-hand operand type."
    )


def _imul_error(lhs: Any, rhs: Any) -> ValueError:
    """Get an exception for an invalid in-place multiplication due to incompatible types."""
    return ValueError(
        f"Cannot multiply-assign coefficient of type {type(lhs)} by value {rhs} of type {type(rhs)}"
    )


def _convert_error(source: Any, t: type) -> ValueError:
    """Get an exception for an invalid conversion of a coefficient value to a given type."""
    return ValueError(
        f"Cannot represent coefficient value {source} of type {type(source)} as type {t}"
    )


class RootOfUnity(ABC):
    """Abstract base class for types representing roots of unity."""

    _impl: _zixy.RootOfUnity

    @abstractmethod
    def to_symbolic(self) -> Expr:
        """Convert :param:`self` to a SymPy symbolic type."""
        pass

    @abstractmethod
    def to_numeric(self) -> Number:
        """Convert :param:`self` to a built-in numeric type."""
        pass

    @abstractmethod
    def from_numeric(cls, value: Number) -> Self:
        """Construct an instance of :param:`cls` from a built-in numeric type."""
        pass

    def _sympy_(self) -> Expr:  # noqa: PLW3201
        """Hook for SymPy to convert this object to a SymPy expression."""
        return self.to_symbolic()

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return str(self._impl)


class Sign(RootOfUnity):
    """A real sign, i.e. a square root of unity (either +1 or -1)."""

    _impl: _zixy.Sign

    def __init__(self, source: bool | int | _zixy.Sign | Sign = False):
        """Initialize the sign.

        Args:
            source: Either a boolean or integer interpreted as the exponent :math:`k` such that the
                value of the sign is :math:`(-1)^k`, a Rust-bound
                :class:`~zixy._zixy.Sign` instance,
                or an existing :class:`Sign` instance.
        """
        if isinstance(source, _zixy.Sign):
            self._impl = source
        elif isinstance(source, Sign):
            self._impl = _zixy.Sign(source.phase)
        else:
            self._impl = _zixy.Sign(bool(source))

    @property
    def phase(self) -> bool:
        """Get the phase :math:`k` satisfying the value of the sign as :math:`(-1)^k`."""
        return self._impl.to_phase()

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, Sign):
            return NotImplemented
        return self._impl == other._impl

    def __mul__(self, other: OtherCoeffT) -> OtherCoeffT:
        """Out-of-place multiplication of :param:`self` by :param:`other`.

        Note:
            The return type is determined by the type of the right-hand operand :param:`other`,
            with appropriate type promotion rules.
        """
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, Sign):
            return Sign(self._impl.to_phase() ^ other._impl.to_phase())
        elif isinstance(other, ComplexSign):
            return ComplexSign(other._impl.to_phase() + (2 if self._impl.to_phase() else 0))
        return self.to_numeric() * other

    def __rmul__(self, other: OtherCoeffT) -> OtherCoeffT:
        """Out-of-place multiplication of :param:`other` by :param:`self`.

        See Also:
            :meth:`__mul__`
        """
        return self * other

    def __imul__(self, rhs: Coeff) -> Self:
        """Multiply :param:`self` by :param:`rhs`.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                :param:`self`.
        """
        if isinstance(rhs, Sign):
            Sign.__init__(self, self._impl.to_phase() ^ rhs._impl.to_phase())
            return self
        elif isinstance(rhs, ComplexSign):
            return self.__imul__(int(rhs))
        elif isinstance(rhs, Number):
            if int(rhs.real) == rhs:
                if int(rhs.real) == -1:
                    Sign.__init__(self, not self._impl.to_phase())
                    return self
                elif int(rhs.real) == 1:
                    return self
        elif isinstance(rhs, Expr):
            try:
                return self.__imul__(int(rhs))
            except TypeError:
                pass
        raise _imul_error(self, rhs)

    @overload
    def __truediv__(self, other: float) -> float: ...
    @overload
    def __truediv__(self, other: complex) -> complex: ...
    @overload
    def __truediv__(self, other: Sign) -> Sign: ...
    @overload
    def __truediv__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __truediv__(self, other: Expr) -> Expr: ...

    def __truediv__(self, other: OtherCoeffT) -> OtherCoeffT:
        """Out-of-place division of :param:`self` by :param:`other`.

        See Also:
            :meth:`__mul__`
        """
        if isinstance(other, Sign):
            return self * other
        elif isinstance(other, ComplexSign):
            self_phase = 2 if self._impl.to_phase() else 0
            return ComplexSign(self_phase - other._impl.to_phase())
        return self.to_numeric() / other

    @overload
    def __rtruediv__(self, other: float) -> float: ...
    @overload
    def __rtruediv__(self, other: complex) -> complex: ...
    @overload
    def __rtruediv__(self, other: Sign) -> Sign: ...
    @overload
    def __rtruediv__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __rtruediv__(self, other: Expr) -> Expr: ...

    def __rtruediv__(self, other: OtherCoeffT) -> OtherCoeffT:
        """Out-of-place division of :param:`other` by :param:`self`.

        See Also:
            :meth:`__mul__`, :meth:`__truediv__`
        """
        if isinstance(other, Sign):
            return self * other
        elif isinstance(other, ComplexSign):
            return ComplexSign(other._impl.to_phase() + (2 if self._impl.to_phase() else 0))
        return other / self.to_numeric()

    @classmethod
    def from_int(cls, value: int | float | complex) -> Sign:
        """Construct an instance of :param:`cls` from an integer.

        Raises:
            ValueError: If :param:`value` is not exactly representable as :param:`cls`.
        """
        if value == 1:
            return cls(0)
        elif value == -1:
            return cls(1)
        raise ValueError(f"value {value} is not an exact square root of unity.")

    from_numeric = from_int

    def __int__(self) -> int:
        """Convert :param:`self` to an integer value."""
        return -1 if self._impl.to_phase() else 1

    def __float__(self) -> float:
        """Convert :param:`self` to a floating point value."""
        return float(int(self))

    def __complex__(self) -> complex:
        """Convert :param:`self` to a complex floating point value."""
        return complex(int(self))

    def __pos__(self) -> Self:
        """Return :param:`self`."""
        return self

    def __neg__(self) -> Sign:
        """Return the negation of :param:`self`."""
        return Sign(not self._impl.to_phase())

    def __abs__(self) -> Sign:
        """Return the absolute value of :param:`self`."""
        return Sign(False)

    def to_symbolic(self) -> Expr:
        """Convert :param:`self` to a SymPy symbolic constant."""
        return sympify(int(self))

    def to_numeric(self) -> int:
        """Convert :param:`self` to a built-in numeric type."""
        return int(self)


class ComplexSign(RootOfUnity):
    """A complex sign, i.e. a fourth root of unity (1, i, -1, or -i)."""

    _impl: _zixy.ComplexSign

    def __init__(self, source: int | _zixy.ComplexSign | ComplexSign = 0):
        """Initialize the complex sign.

        Args:
            source: Either an integer interpreted as the exponent :math:`k` such that the value of
                the complex sign is :math:`i^k`, a Rust-bound
                :class:`~zixy._zixy.ComplexSign`
                instance, or an existing :class:`ComplexSign` instance.
        """
        if isinstance(source, _zixy.ComplexSign):
            self._impl = source
        elif isinstance(source, ComplexSign):
            self._impl = _zixy.ComplexSign(source.phase)
        else:
            self._impl = _zixy.ComplexSign(source % 4)

    @property
    def phase(self) -> int:
        """Get the phase :math:`k` satisfying the value of the sign as :math:`(-1)^k`."""
        return self._impl.to_phase()

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, ComplexSign):
            return NotImplemented
        return self._impl == other._impl

    @overload
    def __mul__(self, other: float) -> complex: ...
    @overload
    def __mul__(self, other: complex) -> complex: ...
    @overload
    def __mul__(self, other: Sign) -> ComplexSign: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __mul__(self, other: Expr) -> Expr: ...

    def __mul__(self, other: Coeff) -> Coeff:
        """Out-of-place multiplication of :param:`self` by :param:`other`.

        Note:
            The return type is determined by the type of the right-hand operand :param:`other`,
            with appropriate type promotion rules.
        """
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(self._impl.to_phase() + other._impl.to_phase())
        elif isinstance(other, Sign):
            return NotImplemented  # handled in Sign.__mul__
        return self.to_numeric() * other

    @overload
    def __rmul__(self, other: float) -> complex: ...
    @overload
    def __rmul__(self, other: complex) -> complex: ...
    @overload
    def __rmul__(self, other: Sign) -> ComplexSign: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __rmul__(self, other: Expr) -> Expr: ...

    def __rmul__(self, other: Coeff) -> Coeff:
        """Out-of-place multiplication of :param:`other` by :param:`self`.

        See Also:
            :meth:`__mul__`
        """
        return self * other

    def __imul__(self, rhs: Coeff) -> Self:  # type: ignore[misc]
        """Multiply :param:`self` by :param:`rhs`.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                :param:`self`.
        """
        if isinstance(rhs, Sign):
            return self.__imul__(complex(rhs))
        elif isinstance(rhs, ComplexSign):
            ComplexSign.__init__(self, self._impl.to_phase() + rhs._impl.to_phase())
            return self
        elif isinstance(rhs, Number):
            if int(rhs.real) == rhs or complex(0, int(rhs.imag)) == rhs:
                return self.__imul__(ComplexSign.from_complex(rhs))
        else:
            try:
                return self.__imul__(ComplexSign.from_complex(complex(rhs)))
            except TypeError:
                pass
        raise _imul_error(self, rhs)

    @overload
    def __truediv__(self, other: float) -> complex: ...
    @overload
    def __truediv__(self, other: complex) -> complex: ...
    @overload
    def __truediv__(self, other: Sign) -> ComplexSign: ...
    @overload
    def __truediv__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __truediv__(self, other: Expr) -> Expr: ...

    def __truediv__(self, other: Coeff) -> Coeff:
        """Out-of-place division of :param:`self` by :param:`other`.

        See Also:
            :meth:`__mul__`
        """
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(self._impl.to_phase() - other._impl.to_phase())
        elif isinstance(other, Sign):
            return ComplexSign(self._impl.to_phase() + (2 if other._impl.to_phase() else 0))
        return self.to_numeric() / other

    @overload
    def __rtruediv__(self, other: float) -> complex: ...
    @overload
    def __rtruediv__(self, other: complex) -> complex: ...
    @overload
    def __rtruediv__(self, other: Sign) -> ComplexSign: ...
    @overload
    def __rtruediv__(self, other: ComplexSign) -> ComplexSign: ...
    @overload
    def __rtruediv__(self, other: Expr) -> Expr: ...

    def __rtruediv__(self, other: Coeff) -> Coeff:
        """Out-of-place division of :param:`other` by :param:`self`.

        See Also:
            :meth:`__mul__`, :meth:`__truediv__`
        """
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(other._impl.to_phase() - self._impl.to_phase())
        elif isinstance(other, Sign):
            return ComplexSign((2 if other._impl.to_phase() else 0) - self._impl.to_phase())
        return other / self.to_numeric()

    @classmethod
    def from_complex(cls, value: complex | float) -> ComplexSign:
        """Construct an instance of :param:`cls` from a complex value.

        Raises:
            ValueError: If :param:`value` is not exactly representable as :param:`cls`.
        """
        if value == complex(1, 0):
            return cls(0)
        elif value == complex(0, 1):
            return cls(1)
        elif value == complex(-1, 0):
            return cls(2)
        elif value == complex(0, -1):
            return cls(3)
        raise ValueError(f"value {value} is not an exact fourth root of unity.")

    from_numeric = from_complex

    def __int__(self) -> int:
        """Convert :param:`self` to an integer value.

        Raises:
            ValueError: If the value of :param:`self` is not representable as an integer.
        """
        phase = self._impl.to_phase()
        assert phase >= 0 and phase < 4
        if phase == 0:
            return 1
        elif phase == 2:
            return -1
        raise ValueError(f"Cannot convert {type(self)} value {self} to int.")

    def __float__(self) -> float:
        """Convert :param:`self` to a floating point value.

        Raises:
            ValueError: If the value of :param:`self` is not representable as a float.

        See Also:
            :meth:`__int__`
        """
        return float(int(self))

    def __complex__(self) -> complex:
        """Convert :param:`self` to a complex floating point value."""
        phase = self._impl.to_phase()
        assert phase >= 0 and phase < 4
        if phase == 0:
            return complex(1, 0)
        elif phase == 1:
            return complex(0, 1)
        elif phase == 2:
            return complex(-1, 0)
        else:
            return complex(0, -1)

    def __pos__(self) -> Self:
        """Return :param:`self`."""
        return self

    def __neg__(self) -> ComplexSign:
        """Return the negation of :param:`self`."""
        return ComplexSign(self._impl.to_phase() + 2)

    def __abs__(self) -> ComplexSign:
        """Return the absolute value of :param:`self`."""
        return ComplexSign(0)

    def to_symbolic(self) -> Expr:
        """Convert :param:`self` to a SymPy symbolic constant."""
        phase = self._impl.to_phase()
        assert phase >= 0 and phase < 4
        if phase == 0:
            return sympify(1)
        elif phase == 1:
            return I
        elif phase == 2:
            return sympify(-1)
        else:
            return -I

    def to_numeric(self) -> complex:
        """Convert :param:`self` to a built-in numeric type."""
        return complex(self)


Scalar: TypeAlias = Number | Sign | ComplexSign
Coeff: TypeAlias = Scalar | Expr
CoeffT = TypeVar("CoeffT", float, complex, Sign, ComplexSign, Expr)
OtherCoeffT = TypeVar("OtherCoeffT", float, complex, Sign, ComplexSign, Expr)


def zero(coeff_type: type[CoeffT]) -> CoeffT:
    """Zero as the given coeff type.

    Args:
        coeff_type: Type in which to return the value.

    Returns:
        An instance of :param:`coeff_type` that is equal to zero.
    """
    if issubclass(coeff_type, RootOfUnity):
        return coeff_type(0)
    elif issubclass(coeff_type, Expr):
        return sympify(0)
    elif issubclass(coeff_type, Number):
        return coeff_type(0)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")


def unit(coeff_type: type[CoeffT]) -> CoeffT:
    """Unity as the given coeff type.

    Args:
        coeff_type: Type in which to return the value.

    Returns:
        An instance of :param:`coeff_type` that is equal to unity.
    """
    if issubclass(coeff_type, RootOfUnity):
        return coeff_type()
    elif issubclass(coeff_type, Expr):
        return sympify(1)
    elif issubclass(coeff_type, Number):
        return coeff_type(1)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")


def _convert_symbolic(source: Expr, t: type[CoeffT]) -> CoeffT:
    """Convert the symbolic expression :param:`source` to a coefficient of type :param:`t`.

    Args:
        source: Symbolic expression to convert.
        t: Type of coefficient with which to represent :param:`source`.

    Returns:
        An instance of :param:`t` equal to :param:`source`.

    Raises:
        ValueError: If :param:`source` cannot be represented as type :param:`t`.
    """
    if t is int:
        try:
            if float(int(source)) == float(source):
                return convert(int(source), t)
        except (TypeError, ValueError):
            pass
    try:
        return convert(float(source), t)
    except (TypeError, ValueError):
        pass
    try:
        return convert(complex(source), t)
    except (TypeError, ValueError):
        pass
    raise _convert_error(source, t)


def _is_int(cls: type[Coeff]) -> TypeIs[type[int]]:
    """Whether :param:`cls` is of type ``int``."""
    return cls is int


def _is_float(cls: type[Coeff]) -> TypeIs[type[float]]:
    """Whether :param:`cls` is of type ``float``."""
    return cls is float


def _is_complex(cls: type[Coeff]) -> TypeIs[type[complex]]:
    """Whether :param:`cls` is of type ``complex``."""
    return cls is complex


def _is_expr(cls: type[Coeff]) -> TypeIs[type[Expr]]:
    """Whether :param:`cls` is of type :class:`~sympy.Expr`."""
    return issubclass(cls, Expr)


def _is_sign(cls: type[Coeff]) -> TypeIs[type[Sign]]:
    """Whether :param:`cls` is of type :class:`Sign`."""
    return issubclass(cls, Sign)


def _is_complex_sign(cls: type[Coeff]) -> TypeIs[type[ComplexSign]]:
    """Whether :param:`cls` is of type :class:`ComplexSign`."""
    return issubclass(cls, ComplexSign)


def convert(source: Coeff, cls: type[CoeffT]) -> CoeffT:
    """Convert the coefficient :param:`source` to a coefficient of type :param:`cls`.

    Args:
        source: Coefficient to convert.
        cls: Type of coefficient with which to represent :param:`source`.

    Returns:
        An instance of :param:`cls` equal to :param:`source`.

    Raises:
        ValueError: If :param:`source` cannot be represented as type :param:`cls`.
    """
    if not isinstance(source, Coeff):
        raise TypeError
    elif isinstance(source, cls):
        return source
    elif isinstance(source, Expr):
        return _convert_symbolic(source, cls)
    elif _is_int(cls):
        if isinstance(source, float | complex):
            if source.imag == 0 and int(source.real) == source.real:
                return cls(source.real)
        elif isinstance(source, float):
            if int(source) == source:
                return cls(source)
        else:
            return cls(source)
    elif _is_float(cls):
        if isinstance(source, complex):
            if source.imag == 0:
                return cls(source.real)
        else:
            return cls(source)
    elif _is_complex(cls):
        return cls(source)
    elif _is_expr(cls):
        if isinstance(source, RootOfUnity):
            return source.to_symbolic()
        else:
            return sympify(source)
    elif _is_sign(cls):
        return Sign.from_int(convert(source, int))
    elif _is_complex_sign(cls):
        return ComplexSign.from_complex(complex(source))
    raise _convert_error(source, cls)


def common_type(lhs: Coeff, rhs: Coeff) -> type[Coeff]:
    """Find the most narrow coefficient type that can represent both :param:`lhs` and :param:`rhs`.

    Args:
        lhs: First coefficient value.
        rhs: Second coefficient value.

    Returns:
        The most specific coefficient type to which both values can be converted.
    """
    types = (Sign, ComplexSign, int, float, complex, Expr)
    for t in types:
        try:
            convert(lhs, t)  # type: ignore[type-var]
            convert(rhs, t)  # type: ignore[type-var]
            return t
        except (ValueError, TypeError):
            pass
    raise ValueError(
        f"Could not find common coefficient type for {lhs} of type "
        f"{type(lhs)} and {rhs} of type {type(rhs)}."
    )


def typesafe_mul(lhs: CoeffT, rhs: Coeff) -> CoeffT:
    """Multiply :param:`lhs` by :param:`rhs` with the result conserving the type of :param:`lhs`.

    Args:
        lhs: Left hand operand, also defining the required result type.
        rhs: Right hand operand.

    Returns:
        The product of :param:`lhs` and :param:`rhs`, of the same type as :param:`lhs`.

    Raises:
        ValueError: Product of :param:`lhs` and :param:`rhs` is not representable as the type of
            :param:`lhs`.
    """
    cls = type(lhs) if not isinstance(lhs, Expr) else Expr
    result = lhs * convert(rhs, cls)
    ok = isinstance(result, type(lhs)) or (isinstance(result, Expr) and isinstance(lhs, Expr))
    if not ok:
        raise ValueError(
            f"Could not safely multiply {lhs} of type {type(lhs)} by {rhs} of type {type(rhs)}. "
            f"Result of multiplication is {result} of type {type(result)}."
        )
    return result


if TYPE_CHECKING:
    BaseVec = _zixy.BaseVec
else:
    BaseVec = object


class Coeffs(Generic[CoeffT], ViewableSequence[CoeffT, BaseVec]):
    """A collection of coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type: type[CoeffT]
    coeffs_type: type[_zixy.BaseVec]

    _impl: _zixy.BaseVec

    def __init__(self, data: _zixy.BaseVec | None = None, s: slice = slice(None)):
        """Initialize the coefficient vector.

        Args:
            data: Rust-bound object storing a vector of coefficients. If ``None``, an empty vector
                of the appropriate type is initialized.
            s: Slice over which the data is to be viewed.
        """
        self._impl = data if data is not None else self.coeffs_type(0)
        self._slice = s

    def fill(self, coeff: CoeffT) -> None:
        """Set all the coefficients in the vector to the given value.

        Args:
            coeff: Coefficient value to which to set all coefficients in the vector.
        """
        for i in range(len(self)):
            self[i] = coeff

    def set(self, source: Iterable[CoeffT]) -> None:
        """Assign the viewed coefficients according to the values yielded by the given iterable.

        Args:
            source: Iterable containing the values to be assigned to the viewed coefficients of
                :param:`self`.
        """
        for i, c in enumerate(source):
            self[i] = convert(c, self.coeff_type)

    @classmethod
    def _create(cls, data: _zixy.BaseVec, s: slice = slice(None)) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            data: Rust-bound object containing the data for this sequence.
            s: Slice of the data in :param:`data` that this instance should view. If ``None``, this
                instance is considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        assert type(data) is cls.coeffs_type, (type(data), cls.coeffs_type)
        Coeffs.__init__(out, data, s)
        return out

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        out = self._empty_clone()
        for c in self:
            out.append(c)
        return out

    def __pos__(self) -> Self:
        """Return :param:`self`."""
        return self

    def __neg__(self) -> Self:
        """Return the negation of :param:`self`."""
        out: Self = self._empty_clone()
        for item in self:
            out.append(-item)
        return out

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return "[" + (", ".join(str(c) for c in self)) + "]"

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, Coeffs):
            return NotImplemented
        return all(left == right for left, right in zip(self, other, strict=False))

    def map_index(self, index: int) -> int:
        """Map an index in :param:`self` to an index in the underlying data.

        Args:
            index: Index in :param:`self`.

        Returns:
            Corresponding index in the underlying data.
        """
        return slice_index(self.slice, index, len(self._impl))

    @overload
    def __getitem__(self, indexer: int) -> CoeffT: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> CoeffT | Self:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the coefficient(s) to return.

        Returns:
            Coefficient or slice selected by :param:`indexer`.
        """
        if isinstance(indexer, builtins.slice):
            return type(self)._create(
                self._impl, slice_of_slice(self.slice, indexer, len(self._impl))
            )
        if _is_sign(self.coeff_type) or _is_complex_sign(self.coeff_type):
            coeff = self.coeff_type(self._impl[self.map_index(indexer)].to_phase())
        else:
            coeff = self._impl[self.map_index(indexer)]
        if not isinstance(coeff, self.coeff_type):
            raise TypeError(f"Expected coefficient of type {self.coeff_type}, got {type(coeff)}")
        return coeff

    def _set_scalar(self, index: int, value: CoeffT | None = None) -> None:
        """Set the indexed element to the given value.

        Args:
            index: Element within the view to set.
            value: Coefficient to set the element to.
        """
        if value is None:
            # None indicates the default value (i.e. unity)
            self[index] = unit(self.coeff_type)
            return
        if _is_sign(self.coeff_type) or _is_complex_sign(self.coeff_type):
            data = value._impl
        else:
            data = value
        self._impl[self.map_index(index)] = data

    def __setitem__(
        self, indexer: int | builtins.slice, source: CoeffT | Coeffs[CoeffT] | None = None
    ) -> None:
        """Set the indexed element(s) to the given value(s).

        Args:
            indexer: Index or slice of coefficient(s) within :param:`self` to assign.
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

    def scale(self, scalar: Coeff) -> None:
        """Scale all elements by a given factor.

        Args:
            scalar: Scalar factor by which to scale all elements.

        Note:
            This method operates in-place.
        """
        for i in range(len(self)):
            self[i] = typesafe_mul(self[i], scalar)

    def __imul__(self, rhs: Coeff | Iterable[Coeff]) -> Self:
        """Multiply :param:`self` by :param:`rhs` in-place.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                :param:`self`.
        """
        if isinstance(rhs, Coeff):
            self.scale(rhs)
        else:
            for i, c in enumerate(rhs):
                if i >= len(self):
                    break
                self._set_scalar(i, typesafe_mul(self[i], c))
        return self

    def to_tuple(self) -> tuple[CoeffT, ...]:
        """Get a tuple of clones of the elements of :param:`self`."""
        return tuple(c for c in self)

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return self._create(self.coeffs_type(0))

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of :param:`self` in a new order.

        Args:
            inds: Sequence of indices defining the new order. Should be a permutation of
                ``range(len(self))``.

        Returns:
            A new instance with the reordered elements.
        """
        if not all(i < len(self) for i in inds):
            raise IndexError("Index out of bounds in reorder.")
        out = self._empty_clone()
        for i in inds:
            out.append(self[i])
        return out

    @requires_ownership
    def append_n(self, n: int, value: CoeffT) -> None:
        """Append :param:`value` to the end of :param:`self` :param:`n` times.

        Args:
            n: Number of times to repeatedly append :param:`value`.
            value: Value to append.

        Note:
            This method operates in-place.
        """
        for _ in range(n):
            if _is_sign(self.coeff_type) or _is_complex_sign(self.coeff_type):
                self._impl.append(value._impl)
            else:
                self._impl.append(value)

    @requires_ownership
    def append(self, value: CoeffT) -> None:
        """Append :param:`value` to the end of :param:`self`.

        Args:
            value: Value to append.

        Note:
            This method operates in-place.
        """
        self.append_n(1, value)

    @requires_ownership
    def extend(self, other: Self) -> None:
        """Append the elements of :param:`other` to the end of :param:`self`.

        Args:
            other: Other instance whose elements are appended to :param:`self`.

        Note:
            This method operates in-place.
        """
        if self._impl.same_as(other._impl):
            self._impl.extend_internal()
        else:
            self._impl.extend_external(other._impl)

    @requires_ownership
    def swap_remove(self, index: int) -> None:
        """Set the element at :param:`index` to the final element, then delete the final element.

        Args:
            index: Element index to remove.

        Note:
            This method operates in-place.
        """
        self[index] = self[-1]
        self.resize(len(self) - 1)

    @classmethod
    def parse(cls, source: str) -> Self:
        """Construct an instance of :param:`cls` from a string representation.

        Args:
            source: The string to read from.

        Returns:
            An instance of :param:`cls` represented by :param:`source`.
        """
        out = cls()
        out._impl = cls.coeffs_type.parse(source)
        return out

    @property
    @abstractmethod
    def np_array(self) -> NDArray[np.generic]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        pass

    def allclose(
        self, other: Coeffs[Any], rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        """Check whether :param:`self` and :param:`other` are within a certain tolerance.

        Args:
            other: Other instance to compare to.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Whether :param:`self` and :param:`other` are within the given tolerances of each other.

        Note:
            This method operates by comparing the NumPy arrays returned by
            :attr:`~zixy.container.coeffs.Coeffs.np_array`.
        """
        if len(self) != len(other):
            return False
        if self is other:
            return True
        return np.allclose(self.np_array, other.np_array, rtol=rtol, atol=atol)

    @classmethod
    def from_scalar(cls, coeff: CoeffT, n: int = 1) -> Self:
        """Get a new instance of :param:`cls` with the given scalar repeated :param:`n` times.

        Args:
            coeff: Coefficient value.
            n: Number of times to repeat :param:`coeff`.

        Returns:
            New instance with :param:`coeff` repeated :param:`n` times.
        """
        out = cls()
        out.append_n(n, coeff)
        return out

    @classmethod
    def from_sequence(cls, source: Sequence[CoeffT]) -> Self:
        """Get a new instance from a sequence of coefficients.

        Args:
            source: Sequence of coefficients.

        Returns:
            New instance with elements set according to :param:`source`.
        """
        out = cls()
        for coeff in source:
            out.append(coeff)
        assert len(out) == len(source)
        return out


class RootOfUnityCoeffs(Coeffs[CoeffT]):
    """A collection of :class:`RootOfUnity` coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeffs_type: type[_zixy.SignVec | _zixy.ComplexSignVec]


class SignCoeffs(RootOfUnityCoeffs[Sign]):
    """A collection of :class:`Sign`s.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = Sign
    coeffs_type = _zixy.SignVec

    _impl: _zixy.SignVec

    @classmethod
    def from_phases(cls, source: NDArray[np.bool_] | Sequence[bool]) -> Self:
        """Get a new instance from a sequence of phases.

        Args:
            source: Sequence of phases.

        Returns:
            New instance with elements set to phases according to :param:`source`.
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

    @property
    def np_array(self) -> NDArray[np.int64]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        return np.array([int(s) for s in self], dtype=np.int64)[self.slice].ravel()


class ComplexSignCoeffs(RootOfUnityCoeffs[ComplexSign]):
    """A collection of :class:`ComplexSign`s.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = ComplexSign
    coeffs_type = _zixy.ComplexSignVec

    _impl: _zixy.ComplexSignVec

    @classmethod
    def from_phases(cls, source: NDArray[np.uint8] | Sequence[int]) -> Self:
        """Get a new instance from a sequence of phases.

        Args:
            source: Sequence of phases.

        Returns:
            New instance with elements set to phases according to :param:`source`.
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

    @property
    def np_array(self) -> NDArray[np.complex128]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        return np.array([complex(s) for s in self], dtype=np.complex128)[self.slice].ravel()


class NumericalCoeffs(Coeffs[NumberT]):
    """A collection of numerical coefficients.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeffs_type: type[_zixy.RealVec | _zixy.ComplexVec]

    def any_significant(self, atol: float = DEFAULT_ATOL) -> bool:
        """Check whether any element of :param:`self` is significantly different from zero.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether any element of :param:`self` is significantly different from zero.
        """
        return not np.allclose(self.np_array, 0, rtol=0, atol=atol)

    @classmethod
    def from_sequence(cls, source: Sequence[NumberT] | NDArray[np.inexact]) -> Self:
        """Get a new instance from a sequence of coefficients.

        Args:
            source: Sequence of coefficients.

        Returns:
            New instance with elements set according to :param:`source`.
        """
        out = cls()
        out._impl = cls.coeffs_type.from_array(np.asarray(source, dtype=cls.coeff_type))
        assert len(out) == len(source)
        return out


class RealCoeffs(NumericalCoeffs[float]):
    """A collection of ``float``s.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = float
    coeffs_type = _zixy.RealVec

    _impl: _zixy.RealVec

    @property
    def np_array(self) -> NDArray[np.float64]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        return np.asarray(self._impl.to_array(), dtype=np.float64)[self.slice].ravel()


class ComplexCoeffs(NumericalCoeffs[complex]):
    """A collection of ``complex``s.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = complex
    coeffs_type = _zixy.ComplexVec

    _impl: _zixy.ComplexVec

    @property
    def np_array(self) -> NDArray[np.complex128]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        return np.asarray(self._impl.to_array(), dtype=np.complex128)[self.slice].ravel()

    @property
    def real_part(self) -> RealCoeffs:
        """Get the real part of :param:`self` as a new instance of :class:`RealCoeffs`."""
        return RealCoeffs.from_sequence(self.np_array.real)

    @property
    def imag_part(self) -> RealCoeffs:
        """Get the imaginary part of :param:`self` as a new instance of :class:`RealCoeffs`."""
        return RealCoeffs.from_sequence(self.np_array.imag)


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

    def __len__(self) -> int:
        """Get the number of elements in :param:`self`."""
        return len(self._list)

    @classmethod
    def parse(self, string: str) -> Self:
        """Construct an instance of :param:`cls` from a string representation.

        Args:
            string: The string to read from.

        Returns:
            An instance of :param:`cls` represented by :param:`string`.
        """
        out = self()
        out._list = [sympify(s) for s in string.split(",")]
        return out

    def __getitem__(self, index: int) -> Expr:
        """Get the element or elements selected by :param:`index`.

        Args:
            index: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`index`.
        """
        return self._list[index]

    def __setitem__(self, index: int, value: Coeff) -> None:
        """Set the indexed element(s) to the given value(s).

        Args:
            index: Index or slice of coefficient(s) within :param:`self` to assign.
            value: Value(s) specifying the coefficient(s) to assign.
        """
        self._list[index] = ExprListWrapper.simplify_integer_floats(sympify(value))

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
    def from_list(cls, source: list[Expr]) -> ExprListWrapper:
        """Create an instance of :param:`cls` from a list of expressions.

        Args:
            source: The list of expressions to read from.

        Returns:
            An instance of :param:`cls` containing the expressions in :param:`source`.
        """
        out = cls()
        assert all(isinstance(c, Expr) for c in source)
        out._list = source
        return out

    @classmethod
    def from_coeffs(cls, source: Coeff | Sequence[Coeff]) -> ExprListWrapper:
        """Create an instance of :param:`cls` from a coefficient or sequence of coefficients.

        Args:
            source: The coefficient or sequence of coefficients to read from.

        Returns:
            An instance of :param:`cls` containing the expressions corresponding to the coefficients
            in :param:`source`.
        """
        out = cls()
        out._list = ExprListWrapper._sympify_coeffs(source)
        return out

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
        """Append :param:`value` to the end of :param:`self`.

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


class SymbolicCoeffs(Coeffs[Expr]):
    """A collection of :class:`~sympy.Expr`s.

    A resizable vector-like container of coefficients that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    coeff_type = Expr
    coeffs_type = ExprListWrapper

    _impl: ExprListWrapper

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, SymbolicCoeffs):
            return NotImplemented
        return all(
            left.simplify() == right.simplify() for left, right in zip(self, other, strict=False)
        )

    @property
    def free_symbols(self) -> set[Symbol]:
        """Get the set of free (unsubstituted) symbols in :param:`self`.

        Returns:
            Union of the sets of free symbols across all coefficients in :param:`self`.
        """
        out = set()
        for coeff in self._impl._list:
            out.update(coeff.free_symbols)
        return out

    def extend(self, other: Self) -> None:
        """Append the elements of :param:`other` to the end of :param:`self`.

        Args:
            other: Other instance whose elements are appended to :param:`self`.

        Note:
            This method operates in-place.
        """
        for c in other:
            self.append(c)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        coeffs = [coeff.subs(values) for coeff in self]
        self._impl._list[self.slice] = coeffs

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

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to :param:`variable` in-place.

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

    def diff(self, variable: Symbol | str) -> SymbolicCoeffs:
        """Differentiate partially with respect to :param:`variable` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        out = self.clone()
        out.idiff(variable)
        return out

    def try_to_real(self) -> RealCoeffs:
        """Try to evaluate :param:`self` as a vector of real coefficients.

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
        return RealCoeffs.from_sequence(out)

    def try_to_complex(self) -> ComplexCoeffs:
        """Try to evaluate :param:`self` as a vector of complex coefficients.

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
        return ComplexCoeffs.from_sequence(out)

    @classmethod
    def parse(cls, source: str) -> Self:  # noqa: D102
        raise NotImplementedError("Cannot parse SymbolicCoeffs from string.")

    parse.__doc__ = Coeffs.parse.__doc__

    @property
    def np_array(self) -> NDArray[np.float64 | np.complex128]:
        """Get the contents of :param:`self` copied as a flat NumPy array.

        Returns:
            NumPy array containing all elements of :param:`self`.
        """
        try:
            return self.try_to_real().np_array
        except TypeError:
            return self.try_to_complex().np_array


def convert_vec(source: Coeffs[Any], t: type[Coeffs[CoeffT]]) -> Coeffs[CoeffT]:
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


def get_coeffs_type(t: type[CoeffT]) -> type[Coeffs[CoeffT]]:
    """Get the coefficient vector type corresponding to a given coefficient type.

    Args:
        t: The coefficient type for which to get the corresponding vector type.

    Returns:
        The coefficient vector type corresponding to :param:`t`.
    """
    if _is_sign(t):
        return SignCoeffs
    elif _is_complex_sign(t):
        return ComplexSignCoeffs
    elif _is_int(t) or _is_float(t):
        return RealCoeffs
    elif _is_complex(t):
        return ComplexCoeffs
    elif _is_expr(t):
        return SymbolicCoeffs
    else:
        raise TypeError(f"Unsupported coefficient type {t}.")


__all__ = [
    "Coeff",
    "Coeffs",
    "Sign",
    "SignCoeffs",
    "ComplexSign",
    "ComplexSignCoeffs",
    "SymbolicCoeffs",
    "RealCoeffs",
    "ComplexCoeffs",
]
