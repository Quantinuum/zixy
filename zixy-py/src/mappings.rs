//! Python extension implementations for fermionic-to-Pauli mappers.

use num_complex::Complex64;
use pyo3::{pyclass, pymethods};
use zixy::container::traits::RefElements;
use zixy::container::word_iters::Elem;
use zixy::fermion::state::cmpnt_list::CmpntRef as FermionStateRef;
use zixy::mappings::bk::BravyiKitaevMapper as BravyiKitaevMapper_;
use zixy::mappings::jw::JordanWignerMapper as JordanWignerMapper_;
use zixy::mappings::paraparticular::ParaparticularMapper as ParaparticularMapper_;
use zixy::mappings::parity::ParityMapper as ParityMapper_;
use zixy::mappings::Mapper;
use zixy::qubit::pauli::cmpnt_major::term_set;
use zixy::qubit::state::cmpnt_list::CmpntList as QubitStateCmpntList;

use crate::container::coeffs::ComplexVec;
use crate::container::map::Map;
use crate::fermion::state::Array as FermionStateArray;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::Array;
use crate::qubit::state::Array as QubitStateArray;

fn apply<M>(mapper: &mut M, ladder_operators: Vec<(usize, bool)>) -> (Array, ComplexVec, Map)
where
    for<'a> M: Mapper<&'a [(usize, bool)], term_set::TermSet<Complex64>>,
{
    let output = mapper.apply(ladder_operators.as_slice());
    (
        Array(output.terms.word_iters),
        ComplexVec(output.terms.coeffs),
        Map(output.map),
    )
}

fn apply_state<M>(mapper: &mut M, state: &FermionStateArray, index: usize) -> QubitStateArray
where
    for<'a> M: Mapper<FermionStateRef<'a>, Elem<QubitStateCmpntList>>,
{
    let output = mapper.apply(state.0.get_elem_ref(index));
    QubitStateArray(output.0)
}

/// A Jordan--Wigner mapper.
#[pyclass]
#[derive(Clone)]
pub struct JordanWignerMapper(JordanWignerMapper_);

#[pymethods]
impl JordanWignerMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(JordanWignerMapper_::new(qubits.0, mode_ordering))
    }

    /// Apply the mapper to a ladder-operator product.
    pub fn apply(&mut self, ladder_operators: Vec<(usize, bool)>) -> (Array, ComplexVec, Map) {
        apply(&mut self.0, ladder_operators)
    }

    /// Apply the mapper to a fermionic occupation-number state.
    pub fn apply_state(&mut self, state: &FermionStateArray, index: usize) -> QubitStateArray {
        apply_state(&mut self.0, state, index)
    }
}

/// A Bravyi--Kitaev mapper.
#[pyclass]
#[derive(Clone)]
pub struct BravyiKitaevMapper(BravyiKitaevMapper_);

#[pymethods]
impl BravyiKitaevMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(BravyiKitaevMapper_::new(qubits.0, mode_ordering))
    }

    /// Apply the mapper to a ladder-operator product.
    pub fn apply(&mut self, ladder_operators: Vec<(usize, bool)>) -> (Array, ComplexVec, Map) {
        apply(&mut self.0, ladder_operators)
    }

    /// Apply the mapper to a fermionic occupation-number state.
    pub fn apply_state(&mut self, state: &FermionStateArray, index: usize) -> QubitStateArray {
        apply_state(&mut self.0, state, index)
    }
}

/// A parity mapper.
#[pyclass]
#[derive(Clone)]
pub struct ParityMapper(ParityMapper_);

#[pymethods]
impl ParityMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(ParityMapper_::new(qubits.0, mode_ordering))
    }

    /// Apply the mapper to a ladder-operator product.
    pub fn apply(&mut self, ladder_operators: Vec<(usize, bool)>) -> (Array, ComplexVec, Map) {
        apply(&mut self.0, ladder_operators)
    }

    /// Apply the mapper to a fermionic occupation-number state.
    pub fn apply_state(&mut self, state: &FermionStateArray, index: usize) -> QubitStateArray {
        apply_state(&mut self.0, state, index)
    }
}

/// A paraparticular mapper.
#[pyclass]
#[derive(Clone)]
pub struct ParaparticularMapper(ParaparticularMapper_);

#[pymethods]
impl ParaparticularMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(ParaparticularMapper_::new(qubits.0, mode_ordering))
    }

    /// Apply the mapper to a ladder-operator product.
    pub fn apply(&mut self, ladder_operators: Vec<(usize, bool)>) -> (Array, ComplexVec, Map) {
        apply(&mut self.0, ladder_operators)
    }

    /// Apply the mapper to a fermionic occupation-number state.
    pub fn apply_state(&mut self, state: &FermionStateArray, index: usize) -> QubitStateArray {
        apply_state(&mut self.0, state, index)
    }
}
