//! Compact representations of phases in `{+1, -1}` and their packed vectors.

use itertools::Itertools;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::ops::{BitXor, Mul, MulAssign, Neg};

use crate::container::coeffs::traits::{
    AnyNumRepr, DiscretePhase, NewUnitsWithLen, NumRepr, NumReprVec, RootUnity, Signed,
    Unrepresentable,
};
use crate::container::traits::{Elements, NewWithLen};
use crate::container::two_bit_vec::BitVec;

/// A square root of unity represented as either `+1` or `-1`.
///
/// The inner `bool` stores `false` for `+1` and `true` for `-1`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sign(pub bool);

impl Display for Sign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}1", if self.0 { "-" } else { "+" })
    }
}

impl From<Sign> for AnyNumRepr {
    fn from(value: Sign) -> Self {
        Self::Sign(value)
    }
}

impl DiscretePhase for Sign {}
impl Signed for Sign {}

impl Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        Sign(self.0.bitxor(rhs.0))
    }
}

impl MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = (*self * rhs).0
    }
}

impl Neg for Sign {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(!self.0)
    }
}

impl RootUnity for Sign {}

/// A packed vector of [`Sign`] values stored in a [`BitVec`].
#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SignVec(pub BitVec);

impl SignVec {
    /// Returns an iterator over the signs stored in the vector.
    pub fn iter(&self) -> impl Iterator<Item = Sign> + '_ {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }

    /// Builds a packed sign vector from raw phase bits.
    pub fn from_phases(phases: &[bool]) -> Self {
        let mut out = Self::default();
        for phase in phases.iter().copied() {
            out.push(Sign(phase));
        }
        out
    }
}

impl Debug for SignVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SignVec").field(&self.0).finish()
    }
}

impl NumRepr for Sign {
    type Vector = SignVec;
    const ONE: Self = Sign(false);

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable> {
        match value {
            AnyNumRepr::Unity(_) => Ok(Self(false)),
            AnyNumRepr::Sign(x) => Ok(x),
            AnyNumRepr::ComplexSign(x) => {
                if x.0 == 0 {
                    Ok(Self(false))
                } else if x.0 == 2 {
                    Ok(Self(true))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Whole(x) => {
                if x == 1 {
                    Ok(Self(false))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Real(x) => {
                if x == 1.0 {
                    Ok(Self(false))
                } else if x == -1.0 {
                    Ok(Self(true))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
            AnyNumRepr::Complex(x) => {
                if x == Complex64::ONE {
                    Ok(Self(false))
                } else if x == -Complex64::ONE {
                    Ok(Self(true))
                } else {
                    Err(Unrepresentable::new::<Self, _>(&x))
                }
            }
        }
    }

    fn ipow(&mut self, exp: i32) {
        if exp.abs() & 1 == 1 {
            *self = -*self;
        }
    }

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<i8, _> = s.trim().parse();
        if let Ok(n) = num {
            if n.abs() == 1 {
                return Ok(Self(n == -1));
            }
        }
        Err(Unrepresentable::new::<Sign, _>(&s))
    }
}

impl Elements for SignVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for SignVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(BitVec::new_with_len(n_element))
    }
}

impl NewUnitsWithLen for SignVec {}

impl NumReprVec for SignVec {
    type Element = Sign;

    fn get_unchecked(&self, index: usize) -> Self::Element {
        Sign(self.0.get_unchecked(index))
    }

    fn set_unchecked(&mut self, index: usize, value: Self::Element) {
        self.0.set_unchecked(index, value.0)
    }

    fn push_default(&mut self) {
        self.0.push(false);
    }

    fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }
    fn imul_elemwise(&mut self, other: &Self) {
        self.0.iadd_elemwise(&other.0);
    }
}

impl Display for SignVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.iter().join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        assert!(Sign::parse("12").is_err());
        assert!(Sign::parse("asd").is_err());
        assert!(Sign::parse("1").is_ok_and(|x| x == Sign(false)));
        assert!(Sign::parse(" 1").is_ok_and(|x| x == Sign(false)));
        assert!(Sign::parse("+1").is_ok_and(|x| x == Sign(false)));
        assert!(Sign::parse("-1").is_ok_and(|x| x == Sign(true)));
        assert!(Sign::parse("-1 ").is_ok_and(|x| x == Sign(true)));
        assert!(Sign::parse(" -1 ").is_ok_and(|x| x == Sign(true)));
    }
}
