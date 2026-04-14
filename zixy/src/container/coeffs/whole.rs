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
                let truncated = x as usize;
                if truncated as f64 == x && x > 0.0 {
                    Ok(truncated)
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Complex(x) => {
                if x.im == 0.0 {
                    let truncated = x.re as usize;
                    if truncated as f64 == x.re && x.re > 0.0 {
                        Ok(truncated)
                    } else {
                        Err(Unrepresentable::new::<Self, _>(&x))
                    }
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
        }
    }

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
