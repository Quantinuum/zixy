//! Introduces numeric multipliers to MappedIterableElements to provide linear combinations.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg};
use std::str::FromStr;

use num_complex::Complex64;
use num_traits::{FromPrimitive, Zero};
use serde::{Deserialize, Serialize};

use crate::container::coeffs::complex_sign::{ComplexSign, ComplexSignVec};
use crate::container::coeffs::sign::{Sign, SignVec};
use crate::container::coeffs::unity::{Unity, UnityVec};
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::{Elements, NewWithLen};

/// Error returned when a value cannot be represented exactly in a target coefficient type.
#[derive(Debug)]
pub struct Unrepresentable(String, String, String);

impl std::fmt::Display for Unrepresentable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value {} of type \"{}\" is not representable as type \"{}\"",
            self.0, self.1, self.2
        )
    }
}
impl std::error::Error for Unrepresentable {}

impl Unrepresentable {
    /// Return the final path segment of `T`'s type name for use in error messages.
    fn short_type_name<T>() -> &'static str {
        let full = std::any::type_name::<T>();
        full.rsplit_once("::").map(|(_, last)| last).unwrap_or(full)
    }

    /// Build an `Unrepresentable` error describing conversion of `inp` into `Output` type.
    pub fn new<Output, Input: Display>(inp: &Input) -> Unrepresentable {
        Unrepresentable(
            inp.to_string(),
            Self::short_type_name::<Input>().to_owned(),
            Self::short_type_name::<Output>().to_owned(),
        )
    }
}

/// In-place multiplication x *= y can either be successful, or unsuccessful because the type of x cannot represent the
/// product x * y. In case of Failure, y is returned as an `AnyNumRepr`.
#[must_use]
pub enum IMulResult {
    Success,
    Failure(AnyNumRepr),
}

/// Representation of a number
pub trait NumRepr:
    Copy
    + Clone
    + Default
    + Display
    + Debug
    + Into<AnyNumRepr>
    + Mul<Output = Self>
    + MulAssign
    + PartialEq
    + Serialize
    + for<'a> Deserialize<'a>
{
    /// Container type used to store sequences of values represented by this coefficient type.
    type Vector: NumReprVec<Element = Self> + Serialize + for<'a> Deserialize<'a>;
    const ONE: Self;

    /// Get the complex conjugate (default is for real types)
    fn conj(&self) -> Self {
        *self
    }

    /// Try to represent the given value as an instance of `Self`.
    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable>;

    /// Try to represent the given value as an instance of `Self`.
    fn try_represent<T: NumRepr>(value: T) -> Result<Self, Unrepresentable> {
        Self::try_represent_any(value.into())
    }

    /// Represent self as a `Complex64`.
    fn to_complex(&self) -> Complex64 {
        Complex64::try_represent(*self).unwrap()
    }

    /// Try to multiply self in-place by any number.
    fn try_imul_any(&mut self, value: AnyNumRepr) -> IMulResult {
        match Self::try_represent_any(value) {
            Ok(x) => {
                *self *= x;
                IMulResult::Success
            }
            Err(_) => IMulResult::Failure(value),
        }
    }

    /// Try to multiply self in-place by any number.
    fn try_imul<T: NumRepr>(&mut self, value: T) -> IMulResult {
        self.try_imul_any(value.into())
    }

    /// Exponentiate self in-place by any integer.
    fn ipow(&mut self, exp: i32);

    /// Exponentiate self out-of-place by any integer.
    fn pow(&self, exp: i32) -> Self {
        let mut this = self.to_owned();
        this.ipow(exp);
        this
    }

    /// Return a squared value of self.
    fn squared(&self) -> Self {
        self.pow(2)
    }

    /// Get coefficient from a string representation
    fn parse(s: &str) -> Result<Self, Unrepresentable>;
}

/// A type-erased coefficient representation covering all supported scalar coefficient types.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AnyNumRepr {
    Unity(Unity),
    Sign(Sign),
    ComplexSign(ComplexSign),
    Whole(usize),
    Real(f64),
    Complex(Complex64),
}

/// Implement for all `implementing` types `ImplType` and all `SourceType`s for which the `ImplType` can
/// exactly represent all values of type `SourceType`.
pub trait Represent<SourceType>: Sized {
    /// Converts value into `Self` without loss of information.
    fn represent(value: SourceType) -> Self;
}

/// Indicates whether `Self` can exactly represent every value of another [`NumRepr`] type.
pub trait IsSuperTypeOf<T: NumRepr>: NumRepr {
    const IS_SUPERTYPE_OF_T: bool;
}

// Each type is trivially supertyped by itself.
impl<T: NumRepr> IsSuperTypeOf<T> for T {
    const IS_SUPERTYPE_OF_T: bool = true;
}

macro_rules! impl_is_super_type_of {
    ($super: ty, $sub: ty, $bool: expr) => {
        impl IsSuperTypeOf<$sub> for $super {
            const IS_SUPERTYPE_OF_T: bool = $bool;
        }
    };
}

impl_is_super_type_of!(Unity, Sign, false);
impl_is_super_type_of!(Unity, ComplexSign, false);
impl_is_super_type_of!(Unity, f64, false);
impl_is_super_type_of!(Unity, Complex64, false);

impl_is_super_type_of!(Sign, Unity, true);
impl_is_super_type_of!(Sign, ComplexSign, false);
impl_is_super_type_of!(Sign, f64, false);
impl_is_super_type_of!(Sign, Complex64, false);

impl_is_super_type_of!(ComplexSign, Unity, true);
impl_is_super_type_of!(ComplexSign, Sign, true);
impl_is_super_type_of!(ComplexSign, f64, false);
impl_is_super_type_of!(ComplexSign, Complex64, false);

impl_is_super_type_of!(f64, Unity, true);
impl_is_super_type_of!(f64, Sign, true);
impl_is_super_type_of!(f64, ComplexSign, false);
impl_is_super_type_of!(f64, Complex64, false);

impl_is_super_type_of!(Complex64, Unity, true);
impl_is_super_type_of!(Complex64, Sign, true);
impl_is_super_type_of!(Complex64, ComplexSign, true);
impl_is_super_type_of!(Complex64, f64, true);

/// For numeric vectors that can be initialized to store a number of unit values.
pub trait NewUnitsWithLen: NewWithLen + Sized {
    /// Create a vector with `n_element` unit values. `Default` is for the case that the implementor's default is unity.
    fn new_units_with_len(n_element: usize) -> Self {
        Self::new_with_len(n_element)
    }
}

/// A vector-like container for elements `implementing` [`NumRepr`].
pub trait NumReprVec:
    Clone + Debug + Default + NewUnitsWithLen + PartialEq + Serialize + for<'a> Deserialize<'a>
{
    /// Scalar coefficient type stored by this vector implementation.
    type Element: NumRepr<Vector = Self>;

    /// Get an iterator over the elements of self.
    fn copied_iter(&self) -> impl Iterator<Item = Self::Element> {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }

    /// Get the element at position `index`.
    fn get_unchecked(&self, index: usize) -> Self::Element;

    /// Get the element at position `index`, or return None if the index is out of bounds.
    fn get(&self, index: usize) -> Result<Self::Element, OutOfBounds> {
        OutOfBounds::check(index, self.len(), Dimension::Element)?;
        Ok(self.get_unchecked(index))
    }

    /// `Set` the element at position `index` with value value.
    fn set_unchecked(&mut self, index: usize, value: Self::Element);

    /// `Set` the element at position `index` with value value, or return None if the index is out of bounds.
    fn set(&mut self, index: usize, value: Self::Element) -> Result<(), OutOfBounds> {
        OutOfBounds::check(index, self.len(), Dimension::Element)?;
        self.set_unchecked(index, value);
        Ok(())
    }

    /// Multiply the element at position `index` in-place by value.
    fn imul_elem_unchecked(&mut self, index: usize, value: Self::Element) {
        self.set_unchecked(index, self.get_unchecked(index) * value);
    }

    /// Multiply the element at position `index` in-place by value, or return None if the index is out of bounds.
    fn imul_elem(&mut self, index: usize, value: Self::Element) -> Result<(), OutOfBounds> {
        OutOfBounds::check(index, self.len(), Dimension::Element)?;
        self.imul_elem_unchecked(index, value);
        Ok(())
    }

    /// Multiply all elements in-place by value
    fn imul(&mut self, value: Self::Element) {
        for i in 0..self.len() {
            self.imul_elem_unchecked(i, value);
        }
    }

    /// Multiply the element at position `index` in-place by itself.
    fn square_unchecked(&mut self, index: usize) {
        self.set_unchecked(index, self.get_unchecked(index).squared())
    }

    /// Multiply the element at position `index` in-place by itself, or return None if the index is out of bounds.
    fn square(&mut self, index: usize) -> Result<(), OutOfBounds> {
        OutOfBounds::check(index, self.len(), Dimension::Element)?;
        self.square_unchecked(index);
        Ok(())
    }

    /// Swap the values of the elements at positions `i` and `j`.
    fn swap_unchecked(&mut self, i: usize, j: usize) {
        let tmp = self.get_unchecked(i);
        self.set_unchecked(i, self.get_unchecked(j));
        self.set_unchecked(j, tmp);
    }

    /// `Set` the value of the elements at position `i_dst` to that of `i_src`.
    fn copy_unchecked(&mut self, i_dst: usize, i_src: usize) {
        self.set_unchecked(i_dst, self.get_unchecked(i_src));
    }

    /// `Set` the size of the vector to `n` elements assigning the default value to newly created elements.
    fn resize(&mut self, n: usize);

    /// `Set` the size of the vector to `n` elements assigning the unit value to newly created elements.
    fn resize_with_units(&mut self, n: usize) {
        let init_len = self.len();
        self.resize(n);
        if let Ok(c) = self.get(0) {
            if c != Self::Element::ONE && n > init_len {
                for i in init_len..n {
                    self.set_unchecked(i, Self::Element::ONE);
                }
            }
        }
    }

    /// `Set` the size of `self` to zero.
    fn clear(&mut self) {
        self.resize(0);
    }

    /// Push a default valued element to the end of `self`.
    fn push_default(&mut self);

    /// Push value to the end of `self`.
    fn push(&mut self, value: Self::Element) {
        let i = self.len();
        self.push_default();
        self.set_unchecked(i, value);
    }

    /// Append value to the end of `self`.
    fn append(&mut self, value: &Self) {
        for value in value.copied_iter() {
            self.push(value);
        }
    }

    /// Push all elements of `self` to the end of the `self`.
    fn append_self(&mut self) {
        let n = self.len();
        for i in 0..n {
            self.push(self.get_unchecked(i));
        }
    }

    /// Copy the value of the last element into position `index`, then drop the last element.
    fn pop_and_swap(&mut self, index: usize) {
        let i_end = self.len().saturating_sub(1);
        self.copy_unchecked(index, i_end);
        self.resize(i_end);
    }

    /// Multiply all elements in-place by those of `other`. This operation is only applied to the first n elements
    /// where n is the minimum of the two vector lengths.
    fn imul_elemwise(&mut self, other: &Self) {
        let n = self.len().min(other.len());
        for i in 0..n {
            self.set_unchecked(i, self.get_unchecked(i) * other.get_unchecked(i));
        }
    }

    /// Multiply all elements out-of-place by those of `other`.
    fn mul_elemwise(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.imul_elemwise(other);
        out
    }

    /// Multiply all elements out-of-place by those of `other`, which can be of any type
    fn try_imul_elemwise<T: NumReprVec>(&mut self, other: &T) -> Result<(), Unrepresentable> {
        let n = self.len().min(other.len());
        for i in 0..n {
            let c: T::Element = other.get_unchecked(i);
            let c = Self::Element::try_represent(c)?;
            self.imul_elem_unchecked(i, c);
        }
        Ok(())
    }

    /// Try to exactly represent `other` with an instance of `Self`.
    fn try_represent<T: NumReprVec>(other: &T) -> Result<Self, Unrepresentable> {
        let mut this = Self::default();
        for elem in other.copied_iter() {
            this.push(Self::Element::try_represent(elem)?);
        }
        Ok(this)
    }

    /// Try to create an instance of `Self` from an iterator over parsed elements.
    fn try_from_parsed(
        iter: impl Iterator<Item = Result<Self::Element, Unrepresentable>>,
    ) -> Result<Self, Unrepresentable> {
        let mut this = Self::default();
        for item in iter {
            let item = item?;
            this.push(item);
        }
        Ok(this)
    }

    /// Try to create an instance of `Self` from a string, which can be in one of two formats:
    /// - coefficient, component pairs "(coeff0, cmpnt0), (coeff1, cmpnt1), ..."
    /// - comma-separated list of coeffs "coeff0, coeff1, ..."
    fn try_parse(s: &str) -> Result<Self, Unrepresentable> {
        if s.trim().starts_with("(") {
            // extract the coeffs from (coeff, cmpnt) pairs
            Self::try_from_parsed(
                s.trim()
                    .trim_start_matches("(")
                    .split("(")
                    .map(|x| Self::Element::parse(x.split(",").next().unwrap())),
            )
        } else {
            // assume a comma-separated list of coeffs
            Self::try_from_parsed(s.split(",").map(Self::Element::parse))
        }
    }

    /// Get a new instance in which the elements with indices given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn select(&self, inds: impl Iterator<Item = usize>) -> Self {
        let mut out = Self::default();
        for i in inds {
            if i < self.len() {
                out.push(self.get_unchecked(i));
            }
        }
        out
    }

    /// Get a new instance in which the elements with indices not given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn deselect(&self, mut inds: impl Iterator<Item = usize>) -> Self {
        let mut out = Self::default();
        let mut next = inds.next();
        for i in 0..self.len() {
            if let Some(j) = next {
                if i == j {
                    next = inds.next();
                    continue;
                }
            }
            out.push(self.get_unchecked(i));
        }
        out
    }

    /// Get two new instances: the first with the elements selected in `inds` and the second with the remainder.
    fn bipartition(&self, mut inds: impl Iterator<Item = usize>) -> (Self, Self) {
        let mut out = (Self::default(), Self::default());
        let mut next = inds.next();
        for i in 0..self.len() {
            if let Some(j) = next {
                if i == j {
                    next = inds.next();
                    out.0.push(self.get_unchecked(i));
                    continue;
                }
            }
            out.1.push(self.get_unchecked(i));
        }
        out
    }
}

/// A type-erased vector of coefficient values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnyNumReprVec {
    Unity(UnityVec),
    Sign(SignVec),
    ComplexSign(ComplexSignVec),
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

impl Elements for AnyNumReprVec {
    fn len(&self) -> usize {
        match self {
            AnyNumReprVec::Unity(x) => x.len(),
            AnyNumReprVec::Sign(x) => x.len(),
            AnyNumReprVec::ComplexSign(x) => x.len(),
            AnyNumReprVec::Real(x) => x.len(),
            AnyNumReprVec::Complex(x) => x.len(),
        }
    }
}

/// For types that contain a vector of coefficients.
pub trait HasCoeffs<C: NumRepr>: Elements {
    /// Returns the coefficient vector.
    fn get_coeffs(&self) -> &C::Vector;
}

/// For types that contain a mutable vector of coefficients.
pub trait HasCoeffsMut<C: NumRepr>: HasCoeffs<C> {
    /// Returns a mutable reference to the coefficient vector.
    fn get_coeffs_mut(&mut self) -> &mut C::Vector;
}

/// Representation of an element of a field e.g. real or complex number
/// future: symbolic
pub trait FieldElem:
    NumRepr<Vector = Vec<Self>>
    + Neg<Output = Self>
    + PartialEq
    + Clone
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Display
    + FromStr
    + Debug
    + Default
    + FromPrimitive
    + Sum
    + Zero
{
    // use the same default values as numpy
    const RTOL_DEFAULT: f64 = 1e-5;
    const ATOL_DEFAULT: f64 = 1e-8;
    const COMMUTES_ATOL_DEFAULT: f64 = 1e-12;
    const ZERO: Self;
    const TWO: Self;
    const HALF: Self;
    const SQRT2: Self;
    const ISQRT2: Self;

    /// Multiply the given complex value out of place by this coefficient
    fn scaled_complex(&self, complex: Complex64) -> Complex64;

    /// Return whether this coefficient's real part is greater in magnitude than the given tolerance.
    fn real_part_is_significant(&self, atol: f64) -> bool;

    /// Return whether this coefficient's imaginary part is greater in magnitude than the given tolerance.
    fn imag_part_is_significant(&self, atol: f64) -> bool;

    /// Return whether this coefficient is greater in magnitude than the given tolerance.
    fn is_significant(&self, atol: f64) -> bool {
        self.magnitude() > atol
    }

    /// Represent the f64 as `Self` type
    fn from_real(v: f64) -> Self;

    /// Get coefficient from a complex number if representable, else return error.
    fn try_from_complex(v: Complex64) -> Result<Self, Unrepresentable>;

    /// Return whether this coefficient is close in value to another within given tolerances.
    fn is_close(&self, other: Self, rtol: f64, atol: f64) -> bool;

    /// Return whether this coefficient is close in value to another within default tolerances.
    fn is_close_default(&self, other: Self) -> bool {
        self.is_close(other, Self::RTOL_DEFAULT, Self::ATOL_DEFAULT)
    }

    /// Absolute value of the coefficient.
    fn magnitude(&self) -> f64;

    /// Square of absolute value of the coefficient.
    fn magnitude_sq(&self) -> f64 {
        (self.complex_conj() * *self).magnitude()
    }

    /// Ge the complex conjugate of the coefficient
    fn complex_conj(&self) -> Self;

    /// Return real part of v if `Self` is real, return v if `Self` is complex.
    fn complex_part(v: Complex64) -> Self;
}

/// Helper trait for sequences of field element numbers (i.e. real or complex).
pub trait FieldElemVec: Deref<Target = [Self::ElemType]> {
    type ElemType: FieldElem;

    /// Return whether every element has magnitude at most `atol`.
    fn all_insignificant(&self, atol: f64) -> bool {
        !self.iter().any(|elem| elem.is_significant(atol))
    }

    /// Return whether the real part of a vector of `Self` is significant with the given tolerance.
    fn real_part_is_significant(&self, atol: f64) -> bool {
        self.iter().all(|elem| elem.real_part_is_significant(atol))
    }

    /// Return whether the imaginary part of a vector of `Self` is significant with the given tolerance.
    fn imag_part_is_significant(&self, atol: f64) -> bool {
        self.iter().all(|elem| elem.imag_part_is_significant(atol))
    }

    /// Get an iterator over the indices of significant indices in value.
    fn significant_inds(&self, atol: f64) -> impl Iterator<Item = usize> {
        self.iter().enumerate().filter_map(move |(i, c)| {
            if c.is_significant(atol) {
                Some(i)
            } else {
                None
            }
        })
    }
}

impl<C: FieldElem> FieldElemVec for Vec<C> {
    type ElemType = C;
}

/// Trait for splitting complex-valued collections into separate real and imaginary parts.
pub trait ComplexParts {
    /// Return the real parts of the values as a `Vec<f64>`.
    fn real_part(&self) -> Vec<f64>;
    /// Return the imaginary parts of the values as a `Vec<f64>`.
    fn imag_part(&self) -> Vec<f64>;
}

impl ComplexParts for Vec<Complex64> {
    fn real_part(&self) -> Vec<f64> {
        self.iter().map(|x| x.re).collect()
    }

    fn imag_part(&self) -> Vec<f64> {
        self.iter().map(|x| x.im).collect()
    }
}

/// Marks types that are elements of a finite cyclic group e.g. `Unity`, `Sign`, `ComplexSign`.
pub trait DiscretePhase: NumRepr + Mul + MulAssign {}

/// For NumReprs that support negation. Excludes `Unity`, but includes all other NumReprs.
pub trait Signed: NumRepr + Neg<Output = Self> {}

/// For types that can absorb a factor of the imaginary unit.
pub trait ComplexSigned: NumRepr + Signed {
    /// Return `self * i`.
    fn mul_by_i(&self) -> Self;
    /// Return `self / i`.
    fn div_by_i(&self) -> Self {
        -self.mul_by_i()
    }
    /// Multiply `self` in place by the given complex-sign phase factor.
    fn imul_complex_sign(&mut self, factor: ComplexSign) {
        match factor.0 {
            1 => *self = self.mul_by_i(),
            2 => *self = -*self,
            3 => *self = self.div_by_i(),
            _ => {}
        }
    }
}

/// Representation of the nth roots of unity
pub trait RootUnity: Neg {}

/// Marker trait for field elements that can be represented exactly as `Complex64`.
pub trait IsComplex: FieldElem {
    /// Return this value as a `Complex64`.
    fn get(&self) -> Complex64;
}

impl IsComplex for Complex64 {
    fn get(&self) -> Complex64 {
        *self
    }
}

/// Marker trait for field elements that can be represented exactly as `f64`.
pub trait IsFloat: FieldElem {
    /// Return this value as an `f64`.
    fn get(&self) -> f64;
}

impl IsFloat for f64 {
    fn get(&self) -> f64 {
        *self
    }
}

/// Classification of an input string as coefficients only, components only, or `(coeff, cmpnt)` pairs.
pub enum StringKind {
    CoeffsOnly,
    CmpntsOnly,
    Both,
}

impl From<&str> for StringKind {
    fn from(value: &str) -> Self {
        if value.starts_with("(") {
            Self::Both
        } else if Vec::<Complex64>::try_parse(value).is_ok() {
            // Every numrepr string is parsable as complex
            Self::CoeffsOnly
        } else {
            Self::CmpntsOnly
        }
    }
}

impl From<&String> for StringKind {
    fn from(value: &String) -> Self {
        <Self as From<&str>>::from(value.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("0.123", Some(0.123))]
    #[case("0.123 ", Some(0.123))]
    #[case(" 0.123", Some(0.123))]
    #[case("-0.231123", Some(-0.231123))]
    #[case("-e30.231123", None)]
    #[case("", None)]
    #[case(" ", None)]
    #[case("  ", None)]
    #[case("-", None)]
    #[case("0000", Some(0.0))]
    #[case("1e-3", Some(0.001))]
    #[case("-0.673E-3", Some(-0.000673))]
    fn test_parse_real(#[case] s: &str, #[case] r: Option<f64>) {
        let res = f64::parse(s);
        assert_eq!(r.is_some(), res.is_ok());
        if let Some(r) = r {
            assert_eq!(r, res.unwrap());
        }
    }

    #[rstest]
    #[case("0.1+1j", Some(Complex64::new(0.1, 1.0)))]
    #[case("0.1", Some(Complex64::new(0.1, 0.0)))]
    #[case("0.1-1i", Some(Complex64::new(0.1, -1.0)))]
    #[case("0.1 -1i", Some(Complex64::new(0.1, -1.0)))]
    #[case("0.1  -1i", Some(Complex64::new(0.1, -1.0)))]
    #[case(" 0.1  -1i", Some(Complex64::new(0.1, -1.0)))]
    #[case("123e-3+456e-3i", Some(Complex64::new(0.123, 0.456)))]
    fn test_parse_complex(#[case] s: &str, #[case] r: Option<Complex64>) {
        let res = Complex64::parse(s);
        assert_eq!(r.is_some(), res.is_ok());
        if let Some(r) = r {
            assert_eq!(r, res.unwrap());
        }
    }

    #[rstest]
    #[case("0.123, 0.456", Some(vec![0.123, 0.456]))]
    #[case(" 0.123,   0.456", Some(vec![0.123, 0.456]))]
    #[case(" 0.123,   0.456, -0.789", Some(vec![0.123, 0.456, -0.789]))]
    #[case(" 0.123, 0.456, -0h.789", None)]
    #[case(" 0.123, 0.456, ,-0.789", None)]
    #[case("(0.123, hello), (0.456, world)", Some(vec![0.123, 0.456]))]
    #[case("  (0.123, hello), (0.456, world)", Some(vec![0.123, 0.456]))]
    #[case("(0.123  , hello), ( 0.456, world)", Some(vec![0.123, 0.456]))]
    #[case("(0.123  , hello), ( 0.456, world) ", Some(vec![0.123, 0.456]))]
    #[case("(, hello), ( 0.456, world) ", None)]
    #[case("(), ( 0.456, world) ", None)]
    #[case("()", None)]
    #[case("", None)]
    fn test_parse_real_vec(#[case] s: &str, #[case] r: Option<Vec<f64>>) {
        let res = Vec::<f64>::try_parse(s);
        assert_eq!(r.is_some(), res.is_ok());
        if let Some(r) = r {
            assert_eq!(r, res.unwrap());
        }
    }

    #[rstest]
    #[case("0.1+1j, 0.2+2j", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("0.1+1j, 0.2", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 0.0)]))]
    #[case("0.1+1j,, 0.2", None)]
    #[case("0.1 +1j,  0.2+2j", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("0.1 +1j,  0.2+ 2j", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case(" 0.1 +1j,0.2+ 2j ", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("(0.1+1j, hello), (0.2+2j, world)", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("( 0.1+1j, hello), ( 0.2+ 2j, world)", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("(0.1+1j, hello), (0.2+2j,   world)", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case(" (0.1 +1j, hello),  (0.2 +2j, world)", Some(vec![Complex64::new(0.1, 1.0), Complex64::new(0.2, 2.0)]))]
    #[case("()", None)]
    #[case("", None)]
    fn test_parse_complex_vec(#[case] s: &str, #[case] r: Option<Vec<Complex64>>) {
        let res = Vec::<Complex64>::try_parse(s);
        assert_eq!(r.is_some(), res.is_ok());
        if let Some(r) = r {
            assert_eq!(r, res.unwrap());
        }
    }
}
