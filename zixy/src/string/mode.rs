//! Basic degrees of freedom for quantum mechanical states, operators and spaces made up of them.

pub use std::sync::Arc;

/// Error returned when two mode spaces or mode-indexed objects have different numbers of modes.
#[derive(Debug, PartialEq)]
pub struct DifferentModeCounts {
    pub m_mode: usize,
    pub n_mode: usize,
}

impl DifferentModeCounts {
    /// Check whether `m` and `n` are equal, otherwise return a `DifferentModeCounts` error.
    pub fn check(m: usize, n: usize) -> Result<(), DifferentModeCounts> {
        if m == n {
            Ok(())
        } else {
            Err(DifferentModeCounts {
                m_mode: m,
                n_mode: n,
            })
        }
    }
}

impl std::fmt::Display for DifferentModeCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mode counts {} and {} differ.", self.m_mode, self.n_mode,)
    }
}
impl std::error::Error for DifferentModeCounts {}
