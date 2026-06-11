//! Qubit modes and collections made up of them.
use std::fmt::Display;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::PyAny;
use zixy::container::traits::Elements;
use zixy::qubit::mode::PauliMatrix as PauliMatrix_;
use zixy::qubit::mode::Qubits as Qubits_;
use zixy::qubit::mode::SymplecticPart as SymplecticPart_;

use crate::standard_dunders;
use crate::utils::ToPyResult;

/// Python wrapper for Qubits, which can either take the form of a vector of modes
/// or simply a number of modes
#[pyclass]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Qubits(pub Qubits_);

#[pymethods]
impl Qubits {
    /// Create using the first n modes of the default register.
    #[staticmethod]
    pub fn from_count(n: usize) -> PyResult<Self> {
        Ok(Self(Qubits_::from_count(n)))
    }

    /// Create using the modes i to i+n of the default register.
    #[staticmethod]
    pub fn from_offset(i: usize, n: usize) -> PyResult<Self> {
        Ok(Self(Qubits_::from_offset(i, n)))
    }

    /// Create using the modes indexed.
    #[staticmethod]
    pub fn from_inds(inds: Vec<usize>) -> PyResult<Self> {
        Ok(Self(Qubits_::from_inds(inds).to_py_result()?))
    }

    /// Create by taking the ceiling of the log2 of the size of the hilbert space
    #[staticmethod]
    pub fn from_hilbert_space_dim(n: usize) -> PyResult<Self> {
        if n == 0 {
            Err(PyErr::new::<PyValueError, _>(
                "Hilbert space dimension must be non-zero",
            ))
        } else {
            Ok(Self(Qubits_::from_hilbert_space_dim(n)))
        }
    }

    /// Return number of elements.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Return whether self is equal to other.
    pub fn __eq__(&self, other: Qubits) -> bool {
        self.0 == other.0
    }
}

impl Elements for Qubits {
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Pauli matrices and the identity
#[pyclass]
#[derive(Clone, Copy, Debug)]
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
        Into::<PauliMatrix_>::into(*self).fmt(f)
    }
}

impl PartialEq for PauliMatrix {
    fn eq(&self, other: &Self) -> bool {
        Into::<PauliMatrix_>::into(*self) == Into::<PauliMatrix_>::into(*other)
    }
}

#[pymethods]
impl PauliMatrix {
    #[classattr]
    const ALL: [Self; 4] = [Self::I, Self::X, Self::Y, Self::Z];

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.to_string())
    }
}
standard_dunders!(PauliMatrix);

impl From<PauliMatrix_> for PauliMatrix {
    fn from(val: PauliMatrix_) -> Self {
        match val {
            PauliMatrix_::I => PauliMatrix::I,
            PauliMatrix_::X => PauliMatrix::X,
            PauliMatrix_::Y => PauliMatrix::Y,
            PauliMatrix_::Z => PauliMatrix::Z,
        }
    }
}

impl From<PauliMatrix> for PauliMatrix_ {
    fn from(val: PauliMatrix) -> Self {
        match val {
            PauliMatrix::I => PauliMatrix_::I,
            PauliMatrix::X => PauliMatrix_::X,
            PauliMatrix::Y => PauliMatrix_::Y,
            PauliMatrix::Z => PauliMatrix_::Z,
        }
    }
}

/// Symbols representing the Pauli X and Z
/// Used to identify the X or Z part of the symplectic representation of Paulis
#[pyclass]
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum SymplecticPart {
    /// X part
    X = 0,
    /// Z part
    Z = 1,
}

impl Display for SymplecticPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Into::<SymplecticPart_>::into(*self).fmt(f)
    }
}

impl PartialEq for SymplecticPart {
    fn eq(&self, other: &Self) -> bool {
        Into::<SymplecticPart_>::into(*self) == Into::<SymplecticPart_>::into(*other)
    }
}

#[pymethods]
impl SymplecticPart {
    #[classattr]
    const ALL: [Self; 2] = [Self::X, Self::Z];

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.to_string())
    }
}
standard_dunders!(SymplecticPart);

impl From<SymplecticPart_> for SymplecticPart {
    fn from(val: SymplecticPart_) -> Self {
        match val {
            SymplecticPart_::X => SymplecticPart::X,
            SymplecticPart_::Z => SymplecticPart::Z,
        }
    }
}

impl From<SymplecticPart> for SymplecticPart_ {
    fn from(val: SymplecticPart) -> Self {
        match val {
            SymplecticPart::X => SymplecticPart_::X,
            SymplecticPart::Z => SymplecticPart_::Z,
        }
    }
}
