//! Shared container error types.

use std::fmt::Display;

use pluralizer::pluralize;

/// Error returned when both values of a [`crate::container::utils::DistinctPair`] would be identical.
#[derive(Debug, PartialEq)]
pub struct IndistinctError<T: PartialEq + Copy + Display = usize> {
    pub ind: T,
}

impl<T: PartialEq + Copy + Display> std::fmt::Display for IndistinctError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value {} is both elements of what is supposed to be a distinct pair.",
            self.ind
        )
    }
}

impl<T: PartialEq + Copy + Display + std::fmt::Debug> std::error::Error for IndistinctError<T> {}

/// Either the Row (major) dimension or the Bit (minor) dimension of a bit matrix,
/// or some other dimension descriptor.
#[derive(Debug, PartialEq)]
pub enum Dimension {
    Row,
    Bit,
    Element,
    Cmpnt,
    Mode,
    String,
}

/// Error type for table dimension bounds.
#[derive(Debug, PartialEq)]
pub struct OutOfBounds(usize, usize, Dimension);

impl OutOfBounds {
    /// Error if i >= n.
    pub fn check(i: usize, n: usize, kind: Dimension) -> Result<(), OutOfBounds> {
        if i < n {
            Ok(())
        } else {
            Err(OutOfBounds(i, n, kind))
        }
    }
}

impl std::fmt::Display for OutOfBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.2 {
            Dimension::Row => write!(
                f,
                "Row index {} is out-of-bounds for bit matrix with {}.",
                self.0,
                pluralize("row", self.1 as isize, true)
            ),
            Dimension::Bit => write!(
                f,
                "Bit index {} is out-of-bounds for bit matrix with {} per row.",
                self.0,
                pluralize("bit", self.1 as isize, true)
            ),
            Dimension::Element => write!(
                f,
                "Element index {} is out-of-bounds for a container with {}.",
                self.0,
                pluralize("element", self.1 as isize, true)
            ),
            Dimension::Cmpnt => write!(
                f,
                "Component index {} is out-of-bounds for component list with {}.",
                self.0,
                pluralize("components", self.1 as isize, true)
            ),
            Dimension::Mode => write!(
                f,
                "Mode index {} is out-of-bounds for component list with {} per component.",
                self.0,
                pluralize("mode", self.1 as isize, true)
            ),
            Dimension::String => write!(
                f,
                "String index {} is out-of-bounds for string list with {}.",
                self.0,
                pluralize("string", self.1 as isize, true)
            ),
        }
    }
}
