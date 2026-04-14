//! Wrapper around the zixy Map type.
use pyo3::{pyclass, pymethods, PyResult};
use zixy::container::map::Map as Map_;
use zixy::container::traits::Elements;

/// A Map wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "Map")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Clone)]
pub struct Map(pub Map_);

#[pymethods]
impl Map {
    /// Constructor.
    #[new]
    pub fn __init__() -> PyResult<Self> {
        Ok(Self(Map_::default()))
    }

    /// Get number of entries.
    pub fn __len__(&self) -> usize {
        self.0.len()
    }
}
