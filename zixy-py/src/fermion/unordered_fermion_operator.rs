//! Fermion operators which are not stored in a general, variable width format.
use pyo3::{pyclass, pymethods, PyResult};
use zixy::fermion::mappings::operators::Op;

/// Sum of fermionic operators that are not necessarily in normal order
#[pyclass(subclass)]
#[pyo3(name = "UnorderedFermionOpReal")]
#[repr(transparent)]
#[derive(Clone)]
pub struct UnorderedFermionOpReal(pub Vec<(Vec<Op>, f64)>);

#[pymethods]
impl UnorderedFermionOpReal {
    /// Constructor.
    #[new]
    pub fn __init__(ops: Vec<(Vec<Op>, f64)>) -> PyResult<Self> {
        Ok(Self(ops))
    }
}
