//! Mappings from fermionic ladder operators to pauli strings.

use num_complex::Complex64;
use pyo3::{pyclass, pymethods, PyClassInitializer, PyResult};
use zixy::fermion::mappings::bk::BravyiKitaevMapper as BravyiKitaevMapper_;
use zixy::fermion::mappings::jw::JordanWignerMapper as JordanWignerMapper_;
use zixy::fermion::mappings::operators::Operators;
use zixy::fermion::mappings::paraparticular::ParapartiularMapper as ParapartiularMapper_;
use zixy::fermion::mappings::parity::ParityMapper as ParityMapper_;
use zixy::qubit::pauli::cmpnt_major as pauli;
use zixy::qubit::traits::DifferentQubits;

use crate::container::coeffs::{ComplexVec, RealVec};
use crate::container::map::Map;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::Array;
use crate::utils::ToPyResult;

/// Base class for fermion to qubit transformation mappers.
#[pyclass(subclass)]
#[repr(transparent)]
#[derive(Clone)]
pub struct Mapper(pub Operators);

#[pymethods]
impl Mapper {
    /// Update the state of self to a mapping of the given ladder operators.
    pub fn op_load_product(&mut self, ladder_operators: Vec<(usize, bool)>) -> PyResult<()> {
        self.0.load_product(ladder_operators.as_slice());
        Ok(())
    }

    /// Using the mapping defined by `self`, contribute a fermionic ladder operator product to the operator (cmpnt, map, coeffs)
    pub fn op_contribute_real(
        &mut self,
        cmpnts: &mut Array,
        map: &mut Map,
        coeffs: &mut RealVec,
        scalar: f64,
    ) -> PyResult<()> {
        DifferentQubits::check(&cmpnts.0, &self.0).to_py_result()?;
        let mut op = pauli::term_set::ViewMut {
            word_iters: &mut cmpnts.0,
            map: &mut map.0,
            coeffs: &mut coeffs.0,
        };
        self.0.contribute_real(&mut op, scalar);
        Ok(())
    }

    /// Using the mapping defined by `self`, contribute a fermionic ladder operator product to the operator (cmpnt, map, coeffs)
    pub fn op_contribute_complex(
        &mut self,
        cmpnts: &mut Array,
        map: &mut Map,
        coeffs: &mut ComplexVec,
        scalar: Complex64,
    ) -> PyResult<()> {
        DifferentQubits::check(&cmpnts.0, &self.0).to_py_result()?;
        let mut op = pauli::term_set::ViewMut {
            word_iters: &mut cmpnts.0,
            map: &mut map.0,
            coeffs: &mut coeffs.0,
        };
        self.0.contribute_complex(&mut op, scalar);
        Ok(())
    }
}

/// A Jordan-Wigner transformation mapper.
#[pyclass(extends=Mapper, subclass)]
#[pyo3(name = "JordanWignerMapper")]
#[derive(Clone, Copy)]
pub struct JordanWignerMapper;

#[pymethods]
impl JordanWignerMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(
        qubits: Qubits,
        mode_ordering: Option<Vec<usize>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let ops = Operators::new::<JordanWignerMapper_>(qubits.0, mode_ordering);
        Ok(PyClassInitializer::from(Mapper(ops)).add_subclass(Self))
    }
}

/// A Bravyi-Kitaev transformation mapper.
#[pyclass(extends=Mapper, subclass)]
#[pyo3(name = "BravyiKitaevMapper")]
#[derive(Clone, Copy)]
pub struct BravyiKitaevMapper;

#[pymethods]
impl BravyiKitaevMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(
        qubits: Qubits,
        mode_ordering: Option<Vec<usize>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let ops = Operators::new::<BravyiKitaevMapper_>(qubits.0, mode_ordering);
        Ok(PyClassInitializer::from(Mapper(ops)).add_subclass(Self))
    }
}

/// A parity transformation mapper.
#[pyclass(extends=Mapper, subclass)]
#[pyo3(name = "ParityMapper")]
#[derive(Clone, Copy)]
pub struct ParityMapper;

#[pymethods]
impl ParityMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(
        qubits: Qubits,
        mode_ordering: Option<Vec<usize>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let ops = Operators::new::<ParityMapper_>(qubits.0, mode_ordering);
        Ok(PyClassInitializer::from(Mapper(ops)).add_subclass(Self))
    }
}

/// A parapartiular transformation mapper.
#[pyclass(extends=Mapper, subclass)]
#[pyo3(name = "ParapartiularMapper")]
#[derive(Clone, Copy)]
pub struct ParapartiularMapper;

#[pymethods]
impl ParapartiularMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(
        qubits: Qubits,
        mode_ordering: Option<Vec<usize>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let ops = Operators::new::<ParapartiularMapper_>(qubits.0, mode_ordering);
        Ok(PyClassInitializer::from(Mapper(ops)).add_subclass(Self))
    }
}
