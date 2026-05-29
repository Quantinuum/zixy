//! Whole-number coefficient representations.

use crate::container::coeffs::traits::{
    AnyNumRepr, NewUnitsWithLen, NumRepr, NumReprVec, Represent, Unrepresentable,
};
use crate::container::coeffs::unity::Unity;
use crate::container::traits::{Elements, NewWithLen};

impl NumRepr for usize {
    type Vector = Vec<usize>;
    const ONE: Self = 1;

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable> {
        match value {
            AnyNumRepr::Unity(_) => Ok(1),
            AnyNumRepr::Sign(x) => {
                if !x.0 {
                    Ok(1)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::ComplexSign(x) => {
                if x.0 == 0 {
                    Ok(1)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Whole(x) => Ok(x),
            AnyNumRepr::Real(x) => {
                if x.fract() != 0.0 {
                    return Err(Unrepresentable::new::<Self, _>(&x));
                }
                let truncated = x as usize;
                if truncated as f64 == x && x >= 0.0 {
                    Ok(truncated)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Complex(x) => {
                if x.im != 0.0 {
                    return Err(Unrepresentable::new::<Self, _>(&x));
                }
                if x.re.fract() != 0.0 {
                    return Err(Unrepresentable::new::<Self, _>(&x));
                }
                let truncated = x.re as usize;
                if truncated as f64 == x.re && x.re >= 0.0 {
                    Ok(truncated)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
        }
    }

    // This method computes the power of the whole number. If the exponent is negative, it sets the value to 0.
    fn ipow(&mut self, exp: i32) {
        *self = if exp < 0 {
            0
        } else {
            usize::pow(*self, exp as u32)
        }
    }

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<usize, _> = s.trim().parse();
        num.map_err(|_| Unrepresentable::new::<usize, _>(&s))
    }
}

impl From<usize> for AnyNumRepr {
    fn from(value: usize) -> Self {
        Self::Whole(value)
    }
}

impl Represent<Unity> for usize {
    fn represent(_value: Unity) -> Self {
        1
    }
}

impl Elements for Vec<usize> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl NewWithLen for Vec<usize> {
    fn new_with_len(n_element: usize) -> Self {
        vec![0; n_element]
    }
}

impl NewUnitsWithLen for Vec<usize> {
    fn new_units_with_len(n_element: usize) -> Self {
        vec![1; n_element]
    }
}

impl NumReprVec for Vec<usize> {
    type Element = usize;

    fn get_unchecked(&self, index: usize) -> Self::Element {
        self[index]
    }

    fn set_unchecked(&mut self, index: usize, value: Self::Element) {
        self[index] = value
    }

    fn push_default(&mut self) {
        self.push(0);
    }

    fn resize(&mut self, n: usize) {
        (self as &mut Vec<usize>).resize(n, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::coeffs::{complex_sign::ComplexSign, sign::Sign, unity::Unity};
    use num_complex::Complex64;
    use rstest::rstest;

    #[rstest]
    #[case(AnyNumRepr::Unity(Unity {}), 1)]
    #[case(AnyNumRepr::Sign(Sign(false)), 1)]
    #[case(AnyNumRepr::ComplexSign(ComplexSign(0)), 1)]
    #[case(AnyNumRepr::Whole(0), 0)]
    #[case(AnyNumRepr::Whole(1), 1)]
    #[case(AnyNumRepr::Real(0.0), 0)]
    #[case(AnyNumRepr::Real(1.0), 1)]
    #[case(AnyNumRepr::Complex(Complex64::new(0.0, 0.0)), 0)]
    #[case(AnyNumRepr::Complex(Complex64::new(1.0, 0.0)), 1)]
    fn test_try_represent_any_valid(
        #[case] input: AnyNumRepr,
        #[case] expected: usize,
    ) -> Result<(), Unrepresentable> {
        assert_eq!(usize::try_represent_any(input)?, expected);
        Ok(())
    }

    #[rstest]
    #[case(AnyNumRepr::Sign(Sign(true)))]
    #[case(AnyNumRepr::ComplexSign(ComplexSign(1)))]
    #[case(AnyNumRepr::ComplexSign(ComplexSign(2)))]
    #[case(AnyNumRepr::ComplexSign(ComplexSign(3)))]
    #[case(AnyNumRepr::Real(1.5))]
    #[case(AnyNumRepr::Real(-1.5))]
    #[case(AnyNumRepr::Real(-1.0))]
    #[case(AnyNumRepr::Real(f64::INFINITY))]
    #[case(AnyNumRepr::Real(f64::NEG_INFINITY))]
    #[case(AnyNumRepr::Real(f64::NAN))]
    #[case(AnyNumRepr::Complex(Complex64::new(1.0, 1.0)))]
    #[case(AnyNumRepr::Complex(Complex64::new(0.0, 1.0)))]
    #[case(AnyNumRepr::Complex(Complex64::new(f64::NAN, 0.0)))]
    #[case(AnyNumRepr::Complex(Complex64::new(f64::INFINITY, 0.0)))]
    fn test_try_represent_any_invalid(#[case] input: AnyNumRepr) {
        assert!(matches!(
            usize::try_represent_any(input),
            Err(Unrepresentable { .. })
        ));
    }

    #[rstest]
    #[case(2, 3, 8)]
    #[case(5, 0, 1)]
    #[case(7, -1, 0)]
    fn test_ipow(#[case] base: usize, #[case] exp: i32, #[case] expected: usize) {
        let mut value = base;
        value.ipow(exp);
        assert_eq!(value, expected);
    }

    #[rstest]
    #[case("42", 42)]
    #[case("  123  ", 123)]
    #[case("0", 0)]
    fn test_parse_valid(
        #[case] input: &str,
        #[case] expected: usize,
    ) -> Result<(), Unrepresentable> {
        assert_eq!(usize::parse(input)?, expected);
        Ok(())
    }

    #[rstest]
    #[case("abc")]
    #[case("-1")]
    #[case("42.42")]
    #[case("")]
    fn test_parse_invalid(#[case] input: &str) {
        assert!(matches!(usize::parse(input), Err(Unrepresentable { .. })));
    }
}
