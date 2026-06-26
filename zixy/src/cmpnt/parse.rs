//! Component parsing module.

use std::fmt::Display;

use crate::cmpnt::springs::BadParse;
use crate::container::coeffs::traits::Unrepresentable;
use crate::container::errors::OutOfBounds;

/// Any of the errors that can result from a failed attempt to parse.
#[must_use]
#[derive(Debug)]
pub enum ParseError {
    BadParse(BadParse),
    ModeBounds(OutOfBounds),
    CoeffUnrepresentable(Unrepresentable),
}

impl From<BadParse> for ParseError {
    fn from(value: BadParse) -> Self {
        Self::BadParse(value)
    }
}
impl From<OutOfBounds> for ParseError {
    fn from(value: OutOfBounds) -> Self {
        Self::ModeBounds(value)
    }
}
impl From<Unrepresentable> for ParseError {
    fn from(value: Unrepresentable) -> Self {
        Self::CoeffUnrepresentable(value)
    }
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::BadParse(x) => x.fmt(f),
            ParseError::ModeBounds(x) => x.fmt(f),
            ParseError::CoeffUnrepresentable(x) => x.fmt(f),
        }
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::BadParse(x) => Some(x),
            ParseError::ModeBounds(x) => Some(x),
            ParseError::CoeffUnrepresentable(x) => Some(x),
        }
    }
}
