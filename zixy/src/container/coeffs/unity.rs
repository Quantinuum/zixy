//! Unit coefficient representation. Defined largely to detect when a coefficient of another type is a multiplicative identity.

use std::{
    fmt::Display,
    ops::{Mul, MulAssign},
};

use itertools::Itertools;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::container::{
    coeffs::traits::{
        AnyNumRepr, DiscretePhase, NewUnitsWithLen, NumRepr, NumReprVec, Unrepresentable,
    },
    traits::{Elements, NewWithLen},
};

/// Representation of unity. Only one allowed value, obviously!
#[derive(Copy, Clone, Default, PartialEq, Hash, Eq, Debug, Serialize, Deserialize)]
pub struct Unity {}

impl DiscretePhase for Unity {}

impl Mul for Unity {
    type Output = Unity;

    fn mul(self, _rhs: Self) -> Self::Output {
        self
    }
}

impl MulAssign for Unity {
    fn mul_assign(&mut self, _rhs: Self) {}
}

impl From<Unity> for AnyNumRepr {
    fn from(value: Unity) -> Self {
        Self::Unity(value)
    }
}

/// Vector of unit elements (no physical storage required, just an extent)
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnityVec(pub usize);

impl NumRepr for Unity {
    type Vector = UnityVec;
    const ONE: Self = Self {};

    fn try_represent_any(value: AnyNumRepr) -> Result<Self, Unrepresentable> {
        match value {
            AnyNumRepr::Unity(value) => Ok(value),
            AnyNumRepr::Sign(value) => {
                if !value.0 {
                    Ok(Self::default())
                } else {
                    Err(Unrepresentable::new::<Unity, _>(&value))
                }
            }
            AnyNumRepr::ComplexSign(value) => {
                if value.0 == 0 {
                    Ok(Self::default())
                } else {
                    Err(Unrepresentable::new::<Unity, _>(&value))
                }
            }
            AnyNumRepr::Whole(value) => {
                if value == 1 {
                    Ok(Self::default())
                } else {
                    Err(Unrepresentable::new::<Unity, _>(&value))
                }
            }
            AnyNumRepr::Real(value) => {
                if value == 1.0 {
                    Ok(Self::default())
                } else {
                    Err(Unrepresentable::new::<Unity, _>(&value))
                }
            }
            AnyNumRepr::Complex(value) => {
                if value == Complex64::ONE {
                    Ok(Self::default())
                } else {
                    Err(Unrepresentable::new::<Unity, _>(&value))
                }
            }
        }
    }

    fn ipow(&mut self, _exp: i32) {}

    fn parse(s: &str) -> Result<Self, Unrepresentable> {
        let num: Result<i8, _> = s.trim().parse();
        if let Ok(n) = num {
            if n == 1 {
                return Ok(Self {});
            }
        }
        Err(Unrepresentable::new::<Unity, _>(&s))
    }
}

impl Display for Unity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl Elements for UnityVec {
    fn len(&self) -> usize {
        self.0
    }
}

impl NewWithLen for UnityVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(n_element)
    }
}

impl NewUnitsWithLen for UnityVec {}

impl NumReprVec for UnityVec {
    type Element = Unity;

    fn get_unchecked(&self, _index: usize) -> Self::Element {
        Unity {}
    }

    fn set_unchecked(&mut self, _index: usize, _value: Self::Element) {}

    fn push_default(&mut self) {
        self.0 += 1;
    }

    fn resize(&mut self, n: usize) {
        self.0 = n;
    }

    fn imul_elemwise(&mut self, other: &Self) {
        self.0 = self.0.max(other.0)
    }
}

impl Display for UnityVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", std::iter::repeat_n("1", self.len()).join(", "))
    }
}

#[test]
fn test_parse() {
    assert!(Unity::parse("2").is_err());
    assert!(Unity::parse("-1").is_err());
    assert!(Unity::parse("").is_err());
    assert!(Unity::parse("1").is_ok_and(|x| x == Unity {}));
    assert!(Unity::parse(" 1").is_ok_and(|x| x == Unity {}));
    assert!(Unity::parse("1 ").is_ok_and(|x| x == Unity {}));
    assert!(Unity::parse("+1").is_ok_and(|x| x == Unity {}));
}
