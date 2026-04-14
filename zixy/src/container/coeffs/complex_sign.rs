//! Compact representations of phases in `{+1, +i, -1, -i}` and their packed vectors.

use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, MulAssign, Neg};

use itertools::Itertools;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::container::coeffs::sign::Sign;
use crate::container::coeffs::traits::{
    AnyNumRepr, ComplexSigned, DiscretePhase, NewUnitsWithLen, NumRepr, NumReprVec, Represent,
    RootUnity, Signed, Unrepresentable,
};
use crate::container::coeffs::unity::Unity;
use crate::container::traits::{Elements, NewWithLen};
use crate::container::two_bit_vec::TwoBitVec;

/// The fourth roots of unity {1, i, -1, -i} stored as bytes 0, 1, 2, 3 respectively.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComplexSign(pub u8);

impl ComplexSign {
    /// The multiplicative identity, represented as `+1`.
    pub const ONE: ComplexSign = ComplexSign(0);

    /// The positive imaginary unit, represented as `+i`.
    pub const I: ComplexSign = ComplexSign(1);
}

impl<T> From<T> for ComplexSign
where
    T: num_traits::PrimInt + num_traits::Unsigned,
{
    fn from(value: T) -> Self {
        Self((value.to_usize().unwrap() & 3) as u8)
    }
}

impl From<ComplexSign> for AnyNumRepr {
    fn from(value: ComplexSign) -> Self {
        AnyNumRepr::ComplexSign(value)
    }
}

impl DiscretePhase for ComplexSign {}
impl Signed for ComplexSign {}
impl ComplexSigned for ComplexSign {
    fn mul_by_i(&self) -> Self {
        *self * ComplexSign(1)
    }

    fn div_by_i(&self) -> Self {
        *self * ComplexSign(3)
    }
}

impl Mul for ComplexSign {
    type Output = ComplexSign;

    fn mul(self, rhs: Self) -> Self::Output {
        ComplexSign(self.0.add(rhs.0))
    }
}

impl MulAssign for ComplexSign {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = (*self * rhs).0
    }
}

impl Neg for ComplexSign {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self((self.0 + 2) & 3)
    }
}

impl RootUnity for ComplexSign {}

/// A packed vector of [`ComplexSign`] values stored in a [`TwoBitVec`].
///
/// Each entry uses two bits to encode one of the four supported roots of unity.
#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComplexSignVec(pub TwoBitVec);

impl NumRepr for ComplexSign {
    type Vector = ComplexSignVec;
    const ONE: Self = ComplexSign(0);

    fn conj(&self) -> Self {
        Self::from(self.0 * 3)
    }

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable> {
        match value {
            AnyNumRepr::Unity(_) => Ok(ComplexSign(0)),
            AnyNumRepr::Sign(x) => Ok(ComplexSign(if x.0 { 2 } else { 0 })),
            AnyNumRepr::ComplexSign(x) => Ok(x),
            AnyNumRepr::Whole(x) => {
                if x == 1 {
                    Ok(Self(0))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Real(x) => {
                if x == 1.0 {
                    Ok(Self(0))
                } else if x == -1.0 {
                    Ok(Self(2))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Complex(x) => {
                if x == Complex64::ONE {
                    Ok(Self(0))
                } else if x == -Complex64::ONE {
                    Ok(Self(2))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
        }
    }

    fn ipow(&mut self, exp: i32) {
        if exp < 0 {
            *self = -*self;
            self.ipow(exp.abs());
        }
        self.0 = ((self.0 as i32 * exp) & 3) as u8
    }

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<Complex64, _> = s.trim().parse();
        if let Ok(num) = num {
            if num == Complex64::ONE {
                return Ok(Self(0));
            } else if num == Complex64::I {
                return Ok(Self(1));
            } else if num == -Complex64::ONE {
                return Ok(Self(2));
            } else if num == -Complex64::I {
                return Ok(Self(3));
            }
        }
        Err(Unrepresentable::new::<ComplexSign, _>(&s))
    }
}

impl Elements for ComplexSignVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for ComplexSignVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(TwoBitVec::new_with_len(n_element))
    }
}

impl NewUnitsWithLen for ComplexSignVec {}

impl Debug for ComplexSignVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ComplexSignVec").field(&self.0).finish()
    }
}

impl NumReprVec for ComplexSignVec {
    type Element = ComplexSign;

    fn get_unchecked(&self, index: usize) -> Self::Element {
        ComplexSign(self.0.get_unchecked(index))
    }

    fn set_unchecked(&mut self, index: usize, value: Self::Element) {
        self.0.set_unchecked(index, value.0)
    }

    fn push_default(&mut self) {
        self.0.push(0);
    }

    fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    fn imul_elemwise(&mut self, other: &Self) {
        self.0.iadd_elemwise(&other.0);
    }
}

impl Display for ComplexSign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self.0 & 3 {
                0 => "+1",
                1 => "+i",
                2 => "-1",
                3 => "-i",
                _ => unreachable!("phase is in the range [0, 4)"),
            }
        )
    }
}

impl Display for ComplexSignVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.copied_iter().join(", "))
    }
}

impl Represent<Unity> for ComplexSign {
    fn represent(_value: Unity) -> Self {
        Self(0)
    }
}
impl Represent<Sign> for ComplexSign {
    fn represent(value: Sign) -> Self {
        Self(if value.0 { 2 } else { 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        assert!(ComplexSign::parse("2").is_err());
        assert!(ComplexSign::parse("").is_err());
        assert!(ComplexSign::parse("2i").is_err());
        assert!(ComplexSign::parse("++i").is_err());

        assert!(ComplexSign::parse("1").is_ok_and(|x| x == ComplexSign(0)));
        assert!(ComplexSign::parse("1 ").is_ok_and(|x| x == ComplexSign(0)));
        assert!(ComplexSign::parse(" +1").is_ok_and(|x| x == ComplexSign(0)));
        assert!(ComplexSign::parse("+1 ").is_ok_and(|x| x == ComplexSign(0)));

        assert!(ComplexSign::parse("i").is_ok_and(|x| x == ComplexSign(1)));
        assert!(ComplexSign::parse("i ").is_ok_and(|x| x == ComplexSign(1)));
        assert!(ComplexSign::parse(" +i").is_ok_and(|x| x == ComplexSign(1)));
        assert!(ComplexSign::parse("+i ").is_ok_and(|x| x == ComplexSign(1)));

        assert!(ComplexSign::parse("-1").is_ok_and(|x| x == ComplexSign(2)));
        assert!(ComplexSign::parse("-1 ").is_ok_and(|x| x == ComplexSign(2)));
        assert!(ComplexSign::parse(" -1").is_ok_and(|x| x == ComplexSign(2)));

        assert!(ComplexSign::parse("-i").is_ok_and(|x| x == ComplexSign(3)));
        assert!(ComplexSign::parse("-i ").is_ok_and(|x| x == ComplexSign(3)));
        assert!(ComplexSign::parse(" -i").is_ok_and(|x| x == ComplexSign(3)));
    }

    #[test]
    fn test_imul() {
        let mut v = ComplexSignVec::new_units_with_len(4);
        assert_eq!(v.to_string(), "[+1, +1, +1, +1]");
        v.imul_elem_unchecked(1, -ComplexSign::I);
        assert_eq!(v.to_string(), "[+1, -i, +1, +1]");
        v.imul_elem_unchecked(3, -ComplexSign::I);
        assert_eq!(v.to_string(), "[+1, -i, +1, -i]");
        v.imul_elem_unchecked(1, -ComplexSign::I);
        assert_eq!(v.to_string(), "[+1, -1, +1, -i]");
    }
}
