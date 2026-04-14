//! Mappings from fermionic ladder operators to pauli strings.

use num_complex::Complex64;
use pyo3::{pyclass, pymethods, PyResult};
use zixy::fermion::mappings::jw::JordanWignerMapper as JordanWignerMapper_;
use zixy::fermion::mappings::operators::Operators;
use zixy::qubit::pauli::cmpnt_major as pauli;
use zixy::qubit::traits::DifferentQubits;

use crate::container::coeffs::{ComplexVec, RealVec};
use crate::container::map::Map;
use crate::fermion::unordered_fermion_operator::UnorderedFermionOpReal;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::Array;
use crate::utils::ToPyResult;

/// A Jordan-Wigner transformation mapper.
#[pyclass(subclass)]
#[pyo3(name = "JordanWignerMapper")]
#[repr(transparent)]
#[derive(Clone)]
pub struct JordanWignerMapper(pub Operators);

#[pymethods]
impl JordanWignerMapper {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits, mode_ordering=None))]
    pub fn __init__(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> PyResult<Self> {
        let ops = Operators::new::<JordanWignerMapper_>(qubits.0, mode_ordering);
        Ok(Self(ops))
    }

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

    /// Using the mapping defined by `self`, contribute a list of scaled fermionic ladder operator products to the operator (cmpnt, map, coeffs)
    pub fn op_encode_real(
        &mut self,
        fermion_ops: UnorderedFermionOpReal,
        cmpnts: &mut Array,
        map: &mut Map,
        coeffs: &mut RealVec,
    ) -> PyResult<()> {
        DifferentQubits::check(&cmpnts.0, &self.0).to_py_result()?;
        let mut mut_refs = pauli::term_set::ViewMut {
            word_iters: &mut cmpnts.0,
            map: &mut map.0,
            coeffs: &mut coeffs.0,
        };
        for (ops, coeff) in fermion_ops.0.iter() {
            self.0.load_product(ops.as_slice());
            self.0.contribute_real(&mut mut_refs, *coeff);
        }
        Ok(())
    }
}
