//! Real number coefficient representations.

use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

use num_complex::Complex64;

use crate::container::coeffs::sign::Sign;
use crate::container::coeffs::traits::{
    AnyNumRepr, FieldElem, NewUnitsWithLen, NumRepr, NumReprVec, Represent, Signed, Unrepresentable,
};
use crate::container::coeffs::unity::Unity;
use crate::container::traits::{Elements, NewWithLen};

impl NumRepr for f64 {
    type Vector = Vec<f64>;
    const ONE: Self = 1.0;

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable> {
        match value {
            AnyNumRepr::Unity(_) => Ok(1.0),
            AnyNumRepr::Sign(x) => Ok(if x.0 { -1.0 } else { 1.0 }),
            AnyNumRepr::ComplexSign(x) => {
                if x.0 == 0 {
                    Ok(1.0)
                } else if x.0 == 2 {
                    Ok(-1.0)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Whole(x) => Ok(x as f64),
            AnyNumRepr::Real(x) => Ok(x),
            AnyNumRepr::Complex(x) => {
                if x.im == 0.0 {
                    Ok(x.re)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
        }
    }

    fn ipow(&mut self, exp: i32) {
        *self = f64::powi(*self, exp)
    }

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<f64, _> = s.trim().parse();
        num.map_err(|_| Unrepresentable::new::<f64, _>(&s))
    }
}

impl From<f64> for AnyNumRepr {
    fn from(value: f64) -> Self {
        Self::Real(value)
    }
}

impl FieldElem for f64 {
    fn from_real(v: f64) -> Self {
        v
    }

    fn try_from_complex(v: Complex64) -> Result<Self, Unrepresentable> {
        if v.im == 0.0 {
            Ok(v.re)
        } else {
            Err(Unrepresentable::new::<f64, Complex64>(&v))
        }
    }

    fn is_close(&self, other: Self, rtol: f64, atol: f64) -> bool {
        (self - other).abs() <= (atol + rtol * other.abs())
    }

    fn scaled_complex(&self, complex: Complex64) -> Complex64 {
        Complex64::new(complex.re * self, complex.im * self)
    }

    fn magnitude(&self) -> f64 {
        self.abs()
    }

    fn complex_conj(&self) -> Self {
        *self
    }

    fn complex_part(v: Complex64) -> Self {
        v.re
    }

    const ZERO: Self = 0.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const SQRT2: Self = SQRT_2;
    const ISQRT2: Self = FRAC_1_SQRT_2;

    fn real_part_is_significant(&self, atol: f64) -> bool {
        self.is_significant(atol)
    }

    fn imag_part_is_significant(&self, _atol: f64) -> bool {
        false
    }
}

impl Signed for f64 {}

impl Represent<Unity> for f64 {
    fn represent(_value: Unity) -> Self {
        1.0
    }
}
impl Represent<Sign> for f64 {
    fn represent(value: Sign) -> Self {
        if value.0 {
            -1.0
        } else {
            1.0
        }
    }
}

impl Elements for Vec<f64> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl NewWithLen for Vec<f64> {
    fn new_with_len(n_element: usize) -> Self {
        vec![0.0; n_element]
    }
}

impl NewUnitsWithLen for Vec<f64> {
    fn new_units_with_len(n_element: usize) -> Self {
        vec![1.0; n_element]
    }
}

impl NumReprVec for Vec<f64> {
    type Element = f64;

    fn get_unchecked(&self, index: usize) -> Self::Element {
        self[index]
    }

    fn set_unchecked(&mut self, index: usize, value: Self::Element) {
        self[index] = value
    }

    fn push_default(&mut self) {
        self.push(0.0);
    }

    fn resize(&mut self, n: usize) {
        (self as &mut Vec<f64>).resize(n, 0.0);
    }
}

#[test]
fn test_parse() {
    assert!(f64::parse("ad").is_err());
    assert!(f64::parse("2i").is_err());
    assert!(f64::parse("+").is_err());
    assert!(f64::parse("-").is_err());

    assert!(f64::parse("1.234").is_ok_and(|x| x == 1.234));
    assert!(f64::parse(" 1.234").is_ok_and(|x| x == 1.234));
    assert!(f64::parse("+1.234").is_ok_and(|x| x == 1.234));
    assert!(f64::parse("-1.234").is_ok_and(|x| x == -1.234));
}
