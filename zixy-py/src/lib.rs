//! Python bindings for Zixy.
pub mod cmpnt;
pub mod container;
pub mod fermion;
pub mod qubit;
pub mod utils;

use pyo3::prelude::*;

/// The Python bindings to Zixy.
#[pymodule]
fn _zixy(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Coefficient and coefficient vector types.
    m.add_class::<crate::container::coeffs::Sign>()?;
    m.add_class::<crate::container::coeffs::SignVec>()?;
    m.add_class::<crate::container::coeffs::ComplexSign>()?;
    m.add_class::<crate::container::coeffs::ComplexSignVec>()?;
    m.add_class::<crate::container::coeffs::RealVec>()?;
    m.add_class::<crate::container::coeffs::ComplexVec>()?;
    m.add_class::<crate::container::map::Map>()?;

    // // Lists of qubit pauli strings and terms thereof with coefficient vectors of each type.
    m.add_class::<crate::qubit::pauli::Array>()?;
    m.add_class::<crate::qubit::state::Array>()?;

    m.add_class::<crate::fermion::unordered_fermion_operator::UnorderedFermionOpReal>()?;
    m.add_class::<crate::fermion::mappings::JordanWignerMapper>()?;

    m.add_class::<crate::cmpnt::state_springs::BinarySprings>()?;
    m.add_class::<crate::qubit::clifford::CliffordGateList>()?;
    m.add_class::<crate::qubit::clifford::CliffordGate>()?;
    m.add_class::<crate::qubit::springs::PauliSprings>()?;
    m.add_class::<crate::qubit::mode::Qubits>()?;
    m.add_class::<crate::qubit::mode::PauliMatrix>()?;
    m.add_class::<crate::qubit::mode::SymplecticPart>()?;
    Ok(())
}
