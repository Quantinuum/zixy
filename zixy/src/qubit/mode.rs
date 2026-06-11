//! Definitions related to qubits as "modes" i.e. quantum mechanical degrees of freedom, and their collections as "spaces".

use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;

pub use crate::cmpnt::mode::*;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::{coeffs::complex_sign::ComplexSign, traits::Elements};
use crate::utils::arith::ceil_log2;
use num_enum::TryFromPrimitive;
use pluralizer::pluralize;
use serde::{Deserialize, Serialize};

/// Symbols representing the Pauli matrices including the identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
#[repr(u8)]
pub enum PauliMatrix {
    /// 2x2 identity matrix
    I = 0,
    /// Pauli X matrix
    X = 1,
    /// Pauli Y matrix
    Y = 2,
    /// Pauli Z matrix
    Z = 3,
}

impl Display for PauliMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PauliMatrix::I => "I",
                PauliMatrix::X => "X",
                PauliMatrix::Y => "Y",
                PauliMatrix::Z => "Z",
            }
        )
    }
}

/// Compute the product of two Pauli matrices as another Pauli matrix and a phase given in terms of factors of the imag unit.
pub fn pauli_matrix_product(lhs: PauliMatrix, rhs: PauliMatrix) -> (PauliMatrix, ComplexSign) {
    use PauliMatrix::*;
    let (pauli, phase): (PauliMatrix, u8) = match lhs {
        I => (rhs, 0),
        X => match rhs {
            I => (X, 0),
            X => (I, 0),
            Y => (Z, 1),
            Z => (Y, 3),
        },
        Y => match rhs {
            I => (Y, 0),
            X => (Z, 3),
            Y => (I, 0),
            Z => (X, 1),
        },
        Z => match rhs {
            I => (Z, 0),
            X => (Y, 1),
            Y => (X, 3),
            Z => (I, 0),
        },
    };
    (pauli, ComplexSign(phase))
}

/// Symbols representing the Pauli X and Z
/// Used to identify the X or Z part of the symplectic representation of Paulis
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
#[repr(u8)]
pub enum SymplecticPart {
    /// X part
    X = 0,
    /// Z part
    Z = 1,
}

impl Display for SymplecticPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SymplecticPart::X => "X",
                SymplecticPart::Z => "Z",
            }
        )
    }
}

/// The valid representations of the qubits field of objects acting on qubit spaces
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Kind {
    /// Just a number of qubits
    Count(usize),
    /// Just a number of qubits offset from the start of the standard register.
    Offset(usize, usize),
    /// Mapping from the indices of the `QubitsBased` object to the qubits in the standard register.
    Mapped(Vec<usize>),
}

/// Qubit register.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qubits(Kind);

impl Qubits {
    /// Create an instance from count.
    pub fn from_count(n: usize) -> Qubits {
        Qubits(Kind::Count(n))
    }

    /// Create an instance from offset and count.
    pub fn from_offset(i: usize, n: usize) -> Qubits {
        if i == 0 {
            Qubits(Kind::Count(n))
        } else {
            Qubits(Kind::Offset(i, n))
        }
    }

    /// Create an instance from general indices.
    pub fn from_inds(inds: Vec<usize>) -> Result<Qubits, BasisError> {
        CoincidentIndex::check(&inds)?;
        Ok(match inds.first() {
            Some(&i) => {
                if inds.iter().copied().eq(i..(i + inds.len())) {
                    Self::from_offset(i, inds.len())
                } else {
                    Self(Kind::Mapped(inds))
                }
            }
            None => Self::from_count(0),
        })
    }

    /// Create an instance from hilbert space dimension.
    pub fn from_hilbert_space_dim(n: usize) -> Qubits {
        Self::from_count(ceil_log2(n).unwrap_or_default())
    }

    /// Return the index of a mode from a given order index.
    pub fn get_unchecked(&self, i: usize) -> usize {
        match &self.0 {
            Kind::Count(_) => i,
            Kind::Offset(offset, _) => i + offset,
            Kind::Mapped(inds) => inds[i],
        }
    }

    /// Return the index of a mode from a given order index with bounds checking.
    pub fn get(&self, i: usize) -> Result<usize, OutOfBounds> {
        OutOfBounds::check(i, self.len(), Dimension::Mode).map(|()| self.get_unchecked(i))
    }

    /// Return all indices as a vector.
    pub fn inds(&self) -> Vec<usize> {
        (0..self.len()).map(|i| self.get_unchecked(i)).collect()
    }

    /// Return all indices as an iterator.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }

    /// Return the number of qubits.
    pub fn n_qubit(&self) -> usize {
        self.len()
    }
}

impl Elements for Qubits {
    fn len(&self) -> usize {
        match &self.0 {
            Kind::Count(n) => *n,
            Kind::Offset(_, n) => *n,
            Kind::Mapped(inds) => inds.len(),
        }
    }
}

impl Eq for Qubits {}

impl Hash for Qubits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(&self.0).hash(state);
    }
}

/// Error returned when two `Qubits` descriptors refer to different qubit spaces.
#[derive(Debug, PartialEq)]
pub struct IncompatibleSpaces {
    pub lhs_nqubit: usize,
    pub rhs_nqubit: usize,
}

impl IncompatibleSpaces {
    /// Check whether the two qubits spaces are the same.
    pub fn check(lhs: Qubits, rhs: Qubits) -> Result<(), IncompatibleSpaces> {
        if lhs == rhs {
            Ok(())
        } else {
            Err(IncompatibleSpaces {
                lhs_nqubit: lhs.n_qubit(),
                rhs_nqubit: rhs.n_qubit(),
            })
        }
    }
}

impl std::fmt::Display for IncompatibleSpaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qubit space of {} is incompatible with another of {}.",
            pluralize("mode", self.lhs_nqubit as isize, true),
            pluralize("mode", self.rhs_nqubit as isize, true)
        )
    }
}
impl std::error::Error for IncompatibleSpaces {}

/// Error returned when trying to remap a qubit register with a permutation vector that repeats any index.
#[derive(Debug, PartialEq)]
pub struct CoincidentIndex {
    pub i_qubit: usize,
}

impl CoincidentIndex {
    /// Check whether any of the items in the given vector are the same.
    pub fn check(old_inds: &Vec<usize>) -> Result<(), CoincidentIndex> {
        let mut set = HashSet::<usize>::default();
        for &i_qubit in old_inds {
            if set.contains(&i_qubit) {
                return Err(CoincidentIndex { i_qubit });
            } else {
                set.insert(i_qubit);
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for CoincidentIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qubit mode {} appears more than once in the new basis.",
            self.i_qubit
        )
    }
}
impl std::error::Error for CoincidentIndex {}

/// Errors that can arise while constructing or validating a qubit basis description.
#[derive(Debug)]
pub enum BasisError {
    Bounds(OutOfBounds),
    Counts(DifferentModeCounts),
    Coincident(CoincidentIndex),
}

impl From<OutOfBounds> for BasisError {
    fn from(value: OutOfBounds) -> Self {
        Self::Bounds(value)
    }
}
impl From<DifferentModeCounts> for BasisError {
    fn from(value: DifferentModeCounts) -> Self {
        Self::Counts(value)
    }
}
impl From<CoincidentIndex> for BasisError {
    fn from(value: CoincidentIndex) -> Self {
        Self::Coincident(value)
    }
}

impl Display for BasisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisError::Bounds(x) => x.fmt(f),
            BasisError::Counts(x) => x.fmt(f),
            BasisError::Coincident(x) => x.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::qubit::mode::Qubits;

    #[test]
    fn test_qubits_equality() {
        assert_eq!(Qubits::from_count(6), Qubits::from_count(6));
        assert_ne!(Qubits::from_count(6), Qubits::from_count(8));
        assert_eq!(Qubits::from_count(6), Qubits::from_offset(0, 6));
        assert_ne!(Qubits::from_count(6), Qubits::from_offset(1, 6));
        assert_ne!(Qubits::from_count(6), Qubits::from_offset(1, 7));
        assert_eq!(Qubits::from_count(0), Qubits::from_inds([].into()).unwrap());
        assert_eq!(
            Qubits::from_count(1),
            Qubits::from_inds([0].into()).unwrap()
        );
        assert_ne!(
            Qubits::from_count(1),
            Qubits::from_inds([1].into()).unwrap()
        );
        assert_eq!(
            Qubits::from_offset(1, 1),
            Qubits::from_inds([1].into()).unwrap()
        );
        assert_eq!(
            Qubits::from_count(6),
            Qubits::from_inds([0, 1, 2, 3, 4, 5].into()).unwrap()
        );
        assert_eq!(
            Qubits::from_offset(4, 4),
            Qubits::from_inds([4, 5, 6, 7].into()).unwrap()
        );
        assert_ne!(
            Qubits::from_offset(4, 4),
            Qubits::from_inds([4, 5, 6, 8].into()).unwrap()
        );
    }
}
