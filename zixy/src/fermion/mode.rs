//! Fermionic modes i.e. spin orbitals and lattice sites.

use serde::{Deserialize, Serialize};

use crate::container::traits::Elements;

/// The valid representations of the qubits field of objects acting on qubit spaces
#[derive(Debug, Hash, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Kind {
    /// A number of fermionic modes with no implicit spin labelling.
    Count(usize),
    /// A number of spatial orbitals or sites with spin orbitals in uuu...ddd... ordering.
    SpinMajorPairs(usize),
    /// A number spatial orbitals or sites with of spin orbitals in ududud... ordering.
    SpinMinorPairs(usize),
}

/// Fermionic mode-space descriptor, including whether spin pairs are stored contiguously or interleaved.
#[derive(Debug, Hash, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Modes(pub Kind);

impl Modes {
    /// Create an instance from a number of modes only.
    pub fn from_count(n: usize) -> Modes {
        Modes(Kind::Count(n))
    }

    /// Create an instance from the number of spin orbital pairs, assuming spin major ordering.
    pub fn from_pair_count_spin_major(n_pair: usize) -> Modes {
        Modes(Kind::SpinMajorPairs(n_pair))
    }

    /// Create an instance from the number of spin orbital pairs, assuming spin minor ordering.
    pub fn from_pair_count_spin_minor(n_pair: usize) -> Modes {
        Modes(Kind::SpinMinorPairs(n_pair))
    }

    /// Return the index of the `i`-th mode.
    pub fn get_unchecked(&self, i: usize) -> usize {
        i
    }

    /// Return the index of the `i`-th mode with bounds checking.
    pub fn get(&self, i: usize) -> Option<usize> {
        if i < self.len() {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Return all mode indices as a vector.
    pub fn inds(&self) -> Vec<usize> {
        (0..self.len()).map(|i| self.get_unchecked(i)).collect()
    }

    /// Iterate over all mode indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }
}

impl Elements for Modes {
    fn len(&self) -> usize {
        match self.0 {
            Kind::Count(n) => n,
            Kind::SpinMajorPairs(n) | Kind::SpinMinorPairs(n) => 2 * n,
        }
    }
}
