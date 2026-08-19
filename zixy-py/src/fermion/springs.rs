//! Sparse-string parser for fermionic ladder operators.

use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, Bound, PyAny, PyErr, PyResult};
use zixy::cmpnt::springs::{ModeInd, ModeSettings};
use zixy::container::traits::Elements;
use zixy::fermion::operator::normal::springs::Springs as Springs_;

use crate::utils::{try_py_index, ToPyResult};
use crate::{standard_dunders, wrapped_str};

/// Python wrapper for fermionic ladder-operator sparse strings.
#[pyclass(subclass)]
#[pyo3(name = "FermionSprings")]
#[repr(transparent)]
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct FermionSprings(pub Springs_);

#[pymethods]
impl FermionSprings {
    /// Constructor.
    #[new]
    pub fn __init__(s: Option<String>) -> PyResult<Self> {
        let s = s.unwrap_or_default();
        let all_parts = Springs_::all_parts_from_str(&s).to_py_result()?;
        let n_part = all_parts.len();
        if let Some(part) = all_parts.into_iter().next() {
            if n_part == 1 {
                return Ok(Self(part));
            }
        }
        Err(PyErr::new::<PyValueError, _>(
            "Fermion springs object does not have exactly one part",
        ))
    }

    /// Number of modes required in the smallest space of modes that can contain modes indexed in [0, self.max].
    pub fn default_n_mode(&self) -> ModeInd {
        self.0.get_mode_inds().default_n_mode()
    }

    /// Get number of springs.
    pub fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Get one sparse string as `(mode, is_creation)` pairs.
    pub fn get_ops(&self, i: isize) -> PyResult<Vec<(usize, bool)>> {
        let i = try_py_index(i, self.0.len())?;
        Ok(self
            .0
            .get_ladder_op_iter(i)
            .map(|(is_cre, mode)| (usize::from(mode), is_cre))
            .collect())
    }
}
wrapped_str!(FermionSprings);
standard_dunders!(FermionSprings);
