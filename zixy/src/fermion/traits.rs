//! Functionality common among collections of fermion space-based types.

use crate::fermion::mode::Modes;

/// Error returned when operands are defined on different fermionic mode spaces.
#[derive(Debug, PartialEq)]
pub struct DifferentSpaces {}

impl DifferentSpaces {
    /// Check whether the `ModesBased` inputs are based on the same fermionic space.
    pub fn check<L: ModesBased, R: ModesBased>(lhs: &L, rhs: &R) -> Result<(), DifferentSpaces> {
        if lhs.same_qubits(rhs) {
            Ok(())
        } else {
            Err(DifferentSpaces {})
        }
    }

    /// Check whether the three `ModesBased` inputs are all based on the same fermionic space.
    pub fn check_transitive<L: ModesBased, M: ModesBased, R: ModesBased>(
        lhs: &L,
        mid: &M,
        rhs: &R,
    ) -> Result<(), DifferentSpaces> {
        Self::check(lhs, rhs)?;
        Self::check(mid, rhs)
    }
}

impl std::fmt::Display for DifferentSpaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qubits-based objects are based on different qubit spaces."
        )
    }
}
impl std::error::Error for DifferentSpaces {}

/// Any object that is based on a space of many fermion modes.
pub trait ModesBased {
    /// Get a reference to the `Modes` instance on which this object is defined.
    fn modes(&self) -> &Modes;

    /// Get clone of the `Modes` instance on which this object is defined.
    fn to_modes(&self) -> Modes {
        self.modes().clone()
    }

    /// Return whether `self` is based on the same fermion modes as `other`.
    fn same_qubits<T: ModesBased>(&self, other: &T) -> bool {
        self.modes() == other.modes()
    }
}
