//! Sparse strings.
use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, Bound, PyAny, PyErr, PyResult};
use zixy::cmpnt::springs::{ModeInd, ModeSettings};
use zixy::container::traits::Elements;

use crate::utils::ToPyResult;
use crate::{standard_dunders, wrapped_str};

/// Python wrapper for a vector of qubit operator sparse strings, i.e. there can be many parts to each
/// component.
#[pyclass(subclass)]
#[pyo3(name = "ManyPartPauliSprings")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct ManyPartPauliSprings(pub Vec<zixy::qubit::pauli::springs::Springs>);

#[pymethods]
impl ManyPartPauliSprings {
    /// Constructor.
    #[new]
    pub fn __init__(s: Option<String>) -> PyResult<Self> {
        let s: String = s.unwrap_or_default();
        Ok(Self(
            zixy::qubit::pauli::springs::Springs::all_parts_from_str(&s).to_py_result()?,
        ))
    }

    /// Get string representation.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(if self.0.is_empty() {
            String::default()
        } else {
            self.0.iter().map(|x| format!("({x})")).join(" ")
        })
    }

    /// Return number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.iter().map(Elements::len).min().unwrap_or(0))
    }

    /// Return shallow copy.
    pub fn __copy__(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Return deep copy.
    pub fn __deepcopy__(&self, _memo: Bound<PyAny>) -> PyResult<Self> {
        Ok(self.clone())
    }
}

/// Python wrapper for qubit Pauli sparse strings
#[pyclass(subclass)]
#[pyo3(name = "PauliSprings")]
#[repr(transparent)]
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct PauliSprings(pub zixy::qubit::pauli::springs::Springs);

#[pymethods]
impl PauliSprings {
    /// Constructor.
    #[new]
    pub fn __init__(s: Option<String>) -> PyResult<Self> {
        let many_part = ManyPartPauliSprings::__init__(s)?;
        let n_part = many_part.0.len();
        if let Some(part) = many_part.0.into_iter().next() {
            if n_part == 1 {
                return Ok(Self(part));
            }
        }
        Err(PyErr::new::<PyValueError, _>(
            "Springs object does not have exactly one part",
        ))
    }

    /// Number of modes required in the smallest space of modes that can contain modes indexed in [0, self.max].
    pub fn default_n_qubit(&self) -> ModeInd {
        self.0.get_mode_inds().default_n_mode()
    }

    /// Get number of springs.
    pub fn __len__(&self) -> usize {
        self.0.len()
    }
}
wrapped_str!(PauliSprings);
standard_dunders!(PauliSprings);
