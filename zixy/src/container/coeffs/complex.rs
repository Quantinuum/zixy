//! Complex coefficient representations.

use num_complex::Complex64;

use crate::container::coeffs::traits::{
    AnyNumRepr, ComplexSigned, FieldElem, NewUnitsWithLen, NumRepr, NumReprVec, Represent, Signed,
    Unrepresentable,
};
use crate::container::coeffs::{complex_sign::ComplexSign, sign::Sign, unity::Unity};
use crate::container::traits::{Elements, NewWithLen};

impl NumRepr for Complex64 {
    type Vector = Vec<Complex64>;
    const ONE: Self = Complex64 { re: 1.0, im: 0.0 };

    fn conj(&self) -> Self {
        Complex64::conj(self)
    }

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, super::traits::Unrepresentable> {
        match value {
            AnyNumRepr::Unity(_) => Ok(Complex64::ONE),
            AnyNumRepr::Sign(x) => Ok(Complex64::represent(x)),
            AnyNumRepr::ComplexSign(x) => Ok(Complex64::represent(x)),
            AnyNumRepr::Whole(x) => Ok(Complex64::represent(x)),
            AnyNumRepr::Real(x) => Ok(Self { re: x, im: 0.0 }),
            AnyNumRepr::Complex(x) => Ok(x),
        }
    }

    fn ipow(&mut self, exp: i32) {
        *self = Complex64::powi(self, exp);
    }

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<Complex64, _> = s.trim().parse();
        num.map_err(|_| Unrepresentable::new::<Complex64, _>(&s))
    }
}

impl Elements for Vec<Complex64> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl NewWithLen for Vec<Complex64> {
    fn new_with_len(n_element: usize) -> Self {
        vec![Complex64::ZERO; n_element]
    }
}

impl NewUnitsWithLen for Vec<Complex64> {
    fn new_units_with_len(n_element: usize) -> Self {
        vec![Complex64::ONE; n_element]
    }
}

impl NumReprVec for Vec<Complex64> {
    type Element = Complex64;

    fn get_unchecked(&self, index: usize) -> Self::Element {
        self[index]
    }

    fn set_unchecked(&mut self, index: usize, value: Self::Element) {
        self[index] = value
    }

    fn push_default(&mut self) {
        self.push(Complex64::ZERO);
    }

    fn resize(&mut self, n: usize) {
        (self as &mut Vec<Complex64>).resize(n, Complex64::ZERO);
    }
}

impl From<Complex64> for AnyNumRepr {
    fn from(value: Complex64) -> Self {
        Self::Complex(value)
    }
}

impl FieldElem for Complex64 {
    fn from_real(v: f64) -> Self {
        Self::new(v, 0.0)
    }

    fn try_from_complex(v: Complex64) -> Result<Self, Unrepresentable> {
        Ok(v)
    }

    fn is_close(&self, other: Self, rtol: f64, atol: f64) -> bool {
        (*self - other).norm() <= (atol + rtol * other.norm())
    }

    fn scaled_complex(&self, complex: Complex64) -> Complex64 {
        *self * complex
    }

    fn magnitude(&self) -> f64 {
        self.norm()
    }

    fn complex_conj(&self) -> Self {
        self.conj()
    }

    fn complex_part(v: Complex64) -> Self {
        v
    }

    const ZERO: Self = Self {
        re: f64::ZERO,
        im: 0.0,
    };
    const TWO: Self = Self {
        re: f64::TWO,
        im: 0.0,
    };
    const HALF: Self = Self {
        re: f64::HALF,
        im: 0.0,
    };
    const SQRT2: Self = Self {
        re: f64::SQRT2,
        im: 0.0,
    };
    const ISQRT2: Self = Self {
        re: f64::ISQRT2,
        im: 0.0,
    };

    fn real_part_is_significant(&self, atol: f64) -> bool {
        self.re.is_significant(atol)
    }

    fn imag_part_is_significant(&self, atol: f64) -> bool {
        self.im.is_significant(atol)
    }
}

impl Signed for Complex64 {}
impl ComplexSigned for Complex64 {
    fn mul_by_i(&self) -> Self {
        Self::new(-self.im, self.re)
    }
}

impl Represent<Unity> for Complex64 {
    fn represent(_value: Unity) -> Self {
        Complex64::ONE
    }
}
impl Represent<Sign> for Complex64 {
    fn represent(value: Sign) -> Self {
        Self::new(f64::represent(value), 0.0)
    }
}
impl Represent<ComplexSign> for Complex64 {
    fn represent(value: ComplexSign) -> Self {
        match value.0 & 3 {
            0 => Self::ONE,
            1 => Self::I,
            2 => -Self::ONE,
            3 => -Self::I,
            _ => unreachable!("phase is in the range [0, 4)"),
        }
    }
}
impl Represent<usize> for Complex64 {
    fn represent(value: usize) -> Self {
        Self::new(value as f64, 0.0)
    }
}
impl Represent<f64> for Complex64 {
    fn represent(value: f64) -> Self {
        Self::new(value, 0.0)
    }
}

#[test]
fn test_parse() {
    assert!(Complex64::parse("ad").is_err());

    assert!(Complex64::parse("1.234").is_ok_and(|x| x == Complex64::new(1.234, 0.0)));
    assert!(Complex64::parse(" 1.234").is_ok_and(|x| x == Complex64::new(1.234, 0.0)));
    assert!(Complex64::parse("+1.234").is_ok_and(|x| x == Complex64::new(1.234, 0.0)));
    assert!(Complex64::parse("-1.234").is_ok_and(|x| x == Complex64::new(-1.234, 0.0)));
    assert!(Complex64::parse("i-1.234").is_ok_and(|x| x == Complex64::new(-1.234, 1.0)));
    assert!(Complex64::parse("i - 1.234").is_ok_and(|x| x == Complex64::new(-1.234, 1.0)));
    assert!(Complex64::parse(" +i - 1.234").is_ok_and(|x| x == Complex64::new(-1.234, 1.0)));
}
