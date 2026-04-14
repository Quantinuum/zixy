//! Shared container utility types.

use std::fmt::{Debug, Display};

/// A pair of distinct values.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DistinctPair<T: PartialEq + Copy + Display = usize>(T, T);

impl<T: PartialEq + Copy + Display + Debug> DistinctPair<T> {
    /// Return `Some` when `a` and `b` are distinct, otherwise return `None`.
    pub fn new(a: T, b: T) -> Option<Self> {
        if a != b {
            Some(DistinctPair::<T>(a, b))
        } else {
            None
        }
    }

    /// Construct a `DistinctPair`, returning `IndistinctError` when inputs are equal.
    pub fn try_new(a: T, b: T) -> Result<Self, crate::container::errors::IndistinctError<T>> {
        if a != b {
            Ok(DistinctPair::<T>(a, b))
        } else {
            Err(crate::container::errors::IndistinctError { ind: a })
        }
    }

    /// Return the two stored values as a tuple.
    pub fn get(&self) -> (T, T) {
        (self.0, self.1)
    }
}
