//! Fermionic modes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zixy::container::traits::Elements;
use zixy::fermion::mode::Modes as Modes_;

/// Python wrapper for fermionic mode spaces.
#[pyclass]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Modes(pub Modes_);

#[pymethods]
impl Modes {
    /// Create using the first n fermionic modes.
    #[staticmethod]
    pub fn from_count(n: usize) -> PyResult<Self> {
        Ok(Self(Modes_::from_count(n)))
    }

    /// Create from a number of spin-orbital pairs in spin-major ordering.
    #[staticmethod]
    pub fn from_pair_count_spin_major(n: usize) -> PyResult<Self> {
        Ok(Self(Modes_::from_pair_count_spin_major(n)))
    }

    /// Create from a number of spin-orbital pairs in spin-minor ordering.
    #[staticmethod]
    pub fn from_pair_count_spin_minor(n: usize) -> PyResult<Self> {
        Ok(Self(Modes_::from_pair_count_spin_minor(n)))
    }

    /// Create by taking the ceiling of the log2 of the size of the Fock space.
    #[staticmethod]
    pub fn from_fock_space_dim(n: usize) -> PyResult<Self> {
        if n == 0 {
            Err(PyErr::new::<PyValueError, _>(
                "Fock space dimension must be non-zero",
            ))
        } else {
            Ok(Self(Modes_::from_fock_space_dim(n)))
        }
    }

    /// Return number of modes.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Return whether self is equal to other.
    pub fn __eq__(&self, other: Modes) -> bool {
        self.0 == other.0
    }
}

impl Elements for Modes {
    fn len(&self) -> usize {
        self.0.len()
    }
}
