//! Clifford gates and compositions of Clifford gates.
use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyIterator};
use pyo3::{pyclass, pymethods, Bound, PyAny, PyErr, PyResult};
use zixy::container::utils::DistinctPair;
use zixy::qubit::clifford::Gate as Gate_;

use crate::standard_dunders;
use crate::utils::ErrorToException;

/// A python wrapper for a list of Clifford gates
#[pyclass(subclass)]
#[pyo3(name = "CliffordGateList")]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct CliffordGateList(pub Vec<Gate_>);

#[pymethods]
impl CliffordGateList {
    /// Constructor.
    #[new]
    pub fn __init__() -> PyResult<Self> {
        Ok(Self(Vec::default()))
    }

    /// Push a Hadamard gate to the end of this vector of Clifford gates.
    fn push_h(&mut self, i_qubit: usize) {
        self.0.push(Gate_::H(i_qubit));
    }

    /// Push a phase gate to the end of this vector of Clifford gates.
    fn push_s(&mut self, i_qubit: usize) {
        self.0.push(Gate_::S(i_qubit));
    }

    /// Push a CNOT gate to the end of this vector of Clifford gates.
    fn push_cx(&mut self, i_control: usize, i_target: usize) -> PyResult<()> {
        match DistinctPair::try_new(i_control, i_target) {
            Ok(pair) => {
                self.0.push(Gate_::CX(pair));
                Ok(())
            }
            Err(e) => Err(e.get_exception()),
        }
    }

    /// Remove the last-pushed gate if this vector is non-empty.
    fn pop(&mut self) {
        self.0.pop();
    }
}

/// Kinds of Clifford gate
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub enum CliffordGate {
    /// Hadamard
    H,
    /// Square root of Z
    S,
    /// Controlled not gate
    CX,
}

#[pymethods]
impl CliffordGateList {
    /// Create a new list of Clifford gates from an iterator over tuples e.g.
    /// `(S, 0), (H, 1), (CX, 3, 2), (S, 1)`
    #[staticmethod]
    #[pyo3(signature = (iterable))]
    pub fn from_iterable(iterable: &Bound<PyAny>) -> PyResult<Self> {
        let mut out = Self(vec![]);
        for item in PyIterator::from_object(iterable)? {
            let item = item?;
            if let Ok((gate, ind)) = item.extract::<(CliffordGate, usize)>() {
                match gate {
                    CliffordGate::H => out.push_h(ind),
                    CliffordGate::S => out.push_s(ind),
                    CliffordGate::CX => {
                        return Err(PyErr::new::<PyValueError, _>(
                            "CX gate requires two qubit indices (control, target), only one given.",
                        ))
                    }
                }
            } else if let Ok((gate, i_control, i_target)) =
                item.extract::<(CliffordGate, usize, usize)>()
            {
                match gate {
                    CliffordGate::H => {
                        return Err(PyErr::new::<PyValueError, _>(
                            "H gate requires only one qubit index, two given.",
                        ))
                    }
                    CliffordGate::S => {
                        return Err(PyErr::new::<PyValueError, _>(
                            "S gate requires only one qubit index, two given.",
                        ))
                    }
                    CliffordGate::CX => {
                        out.push_cx(i_control, i_target)?;
                    }
                }
            } else {
                return Err(PyErr::new::<PyValueError, _>(
                    "Iterator item cannot be interpreted as a Clifford gate.",
                ));
            }
        }
        Ok(out)
    }

    /// Get a string representation.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self.0.iter().join(", "))
    }

    /// Return the number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }
}
standard_dunders!(CliffordGateList);
