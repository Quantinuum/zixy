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

"""Scalar coefficients and arithmetic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, TypeAlias, overload

import numpy as np
from sympy import Expr, I, sympify

from zixy import _zixy
from zixy.container.base import Representable

__all__ = ["Coeff", "Sign", "ComplexSign"]


Number: TypeAlias = int | float | complex  # noqa: UP040


@overload
def normalize_scalar(value: np.integer[Any]) -> int: ...


@overload
def normalize_scalar(value: np.floating[Any]) -> float: ...


@overload
def normalize_scalar(value: np.complexfloating[Any, Any]) -> complex: ...


@overload
def normalize_scalar[T](value: T) -> T: ...


def normalize_scalar[T](value: T) -> T | int | float | complex:
    """Convert NumPy integer, floating and complex scalars to their Python equivalents."""
    if isinstance(value, np.integer) and value.dtype.kind in "iu":
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.complexfloating):
        return complex(value)
    return value


def _integer_exponent(exponent: int | np.integer[Any]) -> int:
    """Return the integer value of a power exponent."""
    if isinstance(exponent, int):
        return exponent
    if isinstance(exponent, np.integer) and exponent.dtype.kind in {"i", "u"}:
        return int(exponent)
    raise TypeError(f"Powers require an integer exponent, got {exponent!r}.")


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


class RootOfUnity(Representable, ABC):
    """Abstract base class for types representing roots of unity."""

    _impl: _zixy.RootOfUnity

    __array_priority__ = 1000

    @classmethod
    @abstractmethod
    def from_scalar(cls, value: Number) -> Self:
        """Construct an instance from an exactly representable scalar value.

        Args:
            value: Numeric value to represent, rather than a phase exponent.

        Returns:
            An instance of ``cls`` representing ``value``.

        Raises:
            ValueError: The value is not exactly representable as ``cls``.
        """
        pass

    @abstractmethod
    def conjugate(self) -> Self:
        """Return the complex conjugate of ``self``."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return str(self._impl)

    def _sympy_(self) -> Expr:  # noqa: PLW3201
        """Hook for SymPy to convert this object to a SymPy expression."""
        return self.to_symbolic()

    @abstractmethod
    def to_numeric(self) -> Number:
        """Convert ``self`` to a built-in numeric type."""
        pass

    @abstractmethod
    def to_symbolic(self) -> Expr:
        """Convert ``self`` to a SymPy symbolic type."""
        pass


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

    @classmethod
    def from_scalar(cls, value: Number) -> Self:
        """Construct a sign from its scalar value.

        Args:
            value: Numeric value, which must equal +1 or -1. This is not a phase exponent.

        Returns:
            An instance of ``cls`` representing ``value``.

        Raises:
            ValueError: If ``value`` is not exactly representable as ``cls``.
        """
        if value == 1:
            return cls(0)
        elif value == -1:
            return cls(1)
        raise ValueError(f"value {value} is not an exact square root of unity.")

    @property
    def phase(self) -> bool:
        """The phase :math:`k` such that the sign equals :math:`(-1)^k`."""
        return self._impl.to_phase()

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, Sign):
            return NotImplemented
        return self._impl == other._impl

    def __pos__(self) -> Self:
        """Return ``self``."""
        return type(self)(self.phase)

    def __neg__(self) -> Sign:
        """Return the negation of ``self``."""
        return Sign(not self.phase)

    def __abs__(self) -> Sign:
        """Return the absolute value of ``self``."""
        return Sign(False)

    def __mul__[C: (float, complex, Expr, Sign, ComplexSign)](self, other: C) -> C:
        """Out-of-place multiplication of ``self`` by ``other``.

        Note:
            The return type is determined by the type of the right-hand operand ``other``,
            with appropriate type promotion rules.
        """
        other = normalize_scalar(other)
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, Sign):
            return Sign(self.phase ^ other.phase)
        elif isinstance(other, ComplexSign):
            return ComplexSign(other.phase + (2 if self.phase else 0))
        return self.to_numeric() * other

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, other: C) -> C:
        """Out-of-place multiplication of ``other`` by ``self``.

        See Also:
            :meth:`__mul__`
        """
        return self * other

    def __imul__(self, rhs: Coeff) -> Self:
        """Multiply ``self`` by ``rhs``.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                ``self``.
        """
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Sign):
            Sign.__init__(self, self.phase ^ rhs.phase)
            return self
        elif isinstance(rhs, ComplexSign):
            return self.__imul__(int(rhs))
        elif isinstance(rhs, Number):
            if int(rhs.real) == rhs:
                if int(rhs.real) == -1:
                    Sign.__init__(self, not self.phase)
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

    def __truediv__[C: (float, complex, Sign, ComplexSign, Expr)](self, other: C) -> C:
        """Out-of-place division of ``self`` by ``other``.

        See Also:
            :meth:`__mul__`
        """
        other = normalize_scalar(other)
        if isinstance(other, Sign):
            return self * other
        elif isinstance(other, ComplexSign):
            self_phase = 2 if self.phase else 0
            return ComplexSign(self_phase - other.phase)
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

    def __rtruediv__[C: (float, complex, Sign, ComplexSign, Expr)](self, other: C) -> C:
        """Out-of-place division of ``other`` by ``self``.

        See Also:
            :meth:`__mul__`, :meth:`__truediv__`
        """
        other = normalize_scalar(other)
        if isinstance(other, Sign):
            return self * other
        elif isinstance(other, ComplexSign):
            return ComplexSign(other.phase + (2 if self.phase else 0))
        return other / self.to_numeric()

    def __pow__(self, exponent: int | np.integer[Any]) -> Self:
        """Return this sign raised to an integer power."""
        exponent = _integer_exponent(exponent)
        return type(self)(self.phase and exponent % 2)

    def conjugate(self) -> Self:
        """Return the complex conjugate of ``self``."""
        return type(self)(self.phase)

    @classmethod
    def from_str(cls, source: str) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.

        Raises:
            ValueError: If ``source`` is not exactly representable as ``cls``.
        """
        try:
            value = int(source)
            return cls.from_scalar(value)
        except ValueError as e:
            raise ValueError(f"string {source} is not an exact square root of unity.") from e

    def __int__(self) -> int:
        """Convert ``self`` to an integer value."""
        return -1 if self.phase else 1

    def __float__(self) -> float:
        """Convert ``self`` to a floating point value."""
        return float(int(self))

    def __complex__(self) -> complex:
        """Convert ``self`` to a complex floating point value."""
        return complex(int(self))

    def to_numeric(self) -> int:
        """Convert ``self`` to a built-in numeric type."""
        return int(self)

    def to_symbolic(self) -> Expr:
        """Convert ``self`` to a SymPy symbolic constant."""
        return sympify(int(self))


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

    @classmethod
    def from_scalar(cls, value: Number) -> Self:
        """Construct a complex sign from its scalar value.

        Args:
            value: Numeric value, which must equal +1, -1, +i, or -i. This is not a phase exponent.

        Returns:
            An instance of ``cls`` representing ``value``.

        Raises:
            ValueError: If ``value`` is not exactly representable as ``cls``.
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

    @property
    def phase(self) -> int:
        """The phase :math:`k` such that the sign equals :math:`i^k`."""
        return self._impl.to_phase()

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, ComplexSign):
            return NotImplemented
        return self._impl == other._impl

    def __pos__(self) -> Self:
        """Return ``self``."""
        return type(self)(self.phase)

    def __neg__(self) -> ComplexSign:
        """Return the negation of ``self``."""
        return ComplexSign(self.phase + 2)

    def __abs__(self) -> ComplexSign:
        """Return the absolute value of ``self``."""
        return ComplexSign(0)

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
        """Out-of-place multiplication of ``self`` by ``other``.

        Note:
            The return type is determined by the type of the right-hand operand ``other``,
            with appropriate type promotion rules.
        """
        other = normalize_scalar(other)
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(self.phase + other.phase)
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
        """Out-of-place multiplication of ``other`` by ``self``.

        See Also:
            :meth:`__mul__`
        """
        return self * other

    def __imul__(self, rhs: Coeff) -> Self:  # type: ignore[misc]
        """Multiply ``self`` by ``rhs``.

        Raises:
            ValueError: If the result of the multiplication cannot be represented by the type of
                ``self``.
        """
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Sign):
            return self.__imul__(complex(rhs))
        elif isinstance(rhs, ComplexSign):
            ComplexSign.__init__(self, self.phase + rhs.phase)
            return self
        elif isinstance(rhs, Number):
            if int(rhs.real) == rhs or complex(0, int(rhs.imag)) == rhs:
                return self.__imul__(ComplexSign.from_scalar(rhs))
        else:
            try:
                return self.__imul__(ComplexSign.from_scalar(complex(rhs)))
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
        """Out-of-place division of ``self`` by ``other``.

        See Also:
            :meth:`__mul__`
        """
        other = normalize_scalar(other)
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(self.phase - other.phase)
        elif isinstance(other, Sign):
            return ComplexSign(self.phase + (2 if other.phase else 0))
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
        """Out-of-place division of ``other`` by ``self``.

        See Also:
            :meth:`__mul__`, :meth:`__truediv__`
        """
        other = normalize_scalar(other)
        if not isinstance(other, Coeff):
            return NotImplemented
        if isinstance(other, ComplexSign):
            return ComplexSign(other.phase - self.phase)
        elif isinstance(other, Sign):
            return ComplexSign((2 if other.phase else 0) - self.phase)
        return other / self.to_numeric()

    def __pow__(self, exponent: int | np.integer[Any]) -> Self:
        """Return this sign raised to an integer power."""
        exponent = _integer_exponent(exponent)
        return type(self)(self.phase * exponent)

    def conjugate(self) -> Self:
        """Return the complex conjugate of ``self``."""
        return type(self)(-self.phase)

    @classmethod
    def from_str(cls, source: str) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.

        Raises:
            ValueError: If ``source`` is not exactly representable as ``cls``.
        """
        try:
            if source == "+i":
                source = "1j"
            elif source == "-i":
                source = "-1j"
            value = complex(source)
            return cls.from_scalar(value)
        except ValueError as e:
            raise ValueError(f"string {source} is not an exact fourth root of unity.") from e

    def __int__(self) -> int:
        """Convert ``self`` to an integer value.

        Raises:
            ValueError: If the value of ``self`` is not representable as an integer.
        """
        phase = self.phase
        assert phase >= 0 and phase < 4
        if phase == 0:
            return 1
        elif phase == 2:
            return -1
        raise ValueError(f"Cannot convert {type(self)} value {self} to int.")

    def __float__(self) -> float:
        """Convert ``self`` to a floating point value.

        Raises:
            ValueError: If the value of ``self`` is not representable as a float.

        See Also:
            :meth:`__int__`
        """
        return float(int(self))

    def __complex__(self) -> complex:
        """Convert ``self`` to a complex floating point value."""
        phase = self.phase
        assert phase >= 0 and phase < 4
        if phase == 0:
            return complex(1, 0)
        elif phase == 1:
            return complex(0, 1)
        elif phase == 2:
            return complex(-1, 0)
        else:
            return complex(0, -1)

    def to_numeric(self) -> complex:
        """Convert ``self`` to a built-in numeric type."""
        return complex(self)

    def to_symbolic(self) -> Expr:
        """Convert ``self`` to a SymPy symbolic constant."""
        phase = self.phase
        assert phase >= 0 and phase < 4
        if phase == 0:
            return sympify(1)
        elif phase == 1:
            return I
        elif phase == 2:
            return sympify(-1)
        else:
            return -I


Scalar: TypeAlias = Number | Sign | ComplexSign  # noqa: UP040


Coeff: TypeAlias = Scalar | Expr  # noqa: UP040


def zero[C: (float, complex, Sign, ComplexSign, Expr)](coeff_type: type[C]) -> C:
    """Zero as the given coeff type.

    Args:
        coeff_type: Type in which to return the value.

    Returns:
        An instance of ``coeff_type`` that is equal to zero.
    """
    if issubclass(coeff_type, RootOfUnity):
        return coeff_type(0)
    elif issubclass(coeff_type, Expr):
        return sympify(0)
    elif issubclass(coeff_type, Number):
        return coeff_type(0)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")


def unit[C: (float, complex, Sign, ComplexSign, Expr)](coeff_type: type[C]) -> C:
    """Unity as the given coeff type.

    Args:
        coeff_type: Type in which to return the value.

    Returns:
        An instance of ``coeff_type`` that is equal to unity.
    """
    if issubclass(coeff_type, RootOfUnity):
        return coeff_type()
    elif issubclass(coeff_type, Expr):
        return sympify(1)
    elif issubclass(coeff_type, Number):
        return coeff_type(1)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")


def _convert_symbolic[C: (float, complex, Sign, ComplexSign, Expr)](source: Expr, t: type[C]) -> C:
    """Convert the symbolic expression ``source`` to a coefficient of type ``t``.

    Args:
        source: Symbolic expression to convert.
        t: Type of coefficient with which to represent ``source``.

    Returns:
        An instance of ``t`` equal to ``source``.

    Raises:
        ValueError: If ``source`` cannot be represented as type ``t``.
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


def convert[C: (float, complex, Sign, ComplexSign, Expr)](source: Coeff, cls: type[C]) -> C:
    """Convert the coefficient ``source`` to a coefficient of type ``cls``.

    Args:
        source: Coefficient to convert.
        cls: Type of coefficient with which to represent ``source``.

    Returns:
        An instance of ``cls`` equal to ``source``.

    Raises:
        ValueError: If ``source`` cannot be represented as type ``cls``.
    """
    source = normalize_scalar(source)
    if not isinstance(source, Coeff):
        raise TypeError
    elif isinstance(source, cls):
        return source
    elif isinstance(source, Expr):
        return _convert_symbolic(source, cls)
    elif issubclass(cls, int):
        if isinstance(source, float | complex):
            if source.imag == 0 and int(source.real) == source.real:
                return cls(source.real)
        else:
            return cls(source)
    elif issubclass(cls, float):
        if isinstance(source, complex):
            if source.imag == 0:
                return cls(source.real)
        else:
            return cls(source)
    elif issubclass(cls, complex):
        return cls(source)
    elif issubclass(cls, Expr):
        if isinstance(source, RootOfUnity):
            return source.to_symbolic()
        else:
            return sympify(source)
    elif issubclass(cls, Sign):
        return Sign.from_scalar(convert(source, int))
    elif issubclass(cls, ComplexSign):
        return ComplexSign.from_scalar(complex(source))
    raise _convert_error(source, cls)


def common_type(lhs: Coeff, rhs: Coeff) -> type[Coeff]:
    """Find the most narrow coefficient type that can represent both ``lhs`` and ``rhs``.

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


def typesafe_mul[C: (float, complex, Sign, ComplexSign, Expr)](lhs: C, rhs: Coeff) -> C:
    """Multiply ``lhs`` by ``rhs`` with the result conserving the type of ``lhs``.

    Args:
        lhs: Left hand operand, also defining the required result type.
        rhs: Right hand operand.

    Returns:
        The product of ``lhs`` and ``rhs``, of the same type as ``lhs``.

    Raises:
        ValueError: Product of ``lhs`` and ``rhs`` is not representable as the type of
            ``lhs``.
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
