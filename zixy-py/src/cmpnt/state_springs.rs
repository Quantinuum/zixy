//! Storage format for lists of state mode settings.
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, Bound, PyAny, PyErr, PyResult};
use zixy::cmpnt::springs::ModeSettings;
use zixy::cmpnt::state_springs::BinarySprings as BinarySprings_;

use crate::utils::ToPyResult;
use crate::{standard_dunders, wrapped_str};

/// Python wrapper for binary sparse strings
#[pyclass(subclass)]
#[pyo3(name = "BinarySprings")]
#[repr(transparent)]
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct BinarySprings(pub BinarySprings_);

#[pymethods]
impl BinarySprings {
    /// Constructor.
    #[new]
    pub fn __init__(s: Option<String>) -> PyResult<Self> {
        let s = s.unwrap_or_default();
        let all_parts = BinarySprings_::all_parts_from_str(&s).to_py_result()?;
        let n_part = all_parts.len();
        if let Some(part) = all_parts.into_iter().next() {
            if n_part == 1 {
                return Ok(Self(part));
            }
        }
        Err(PyErr::new::<PyValueError, _>(
            "Binary springs object does not have exactly one part",
        ))
    }
}
wrapped_str!(BinarySprings);
standard_dunders!(BinarySprings);
