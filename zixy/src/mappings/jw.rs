//! Jordan--Wigner mapper implementation.

use num_complex::Complex64;

use crate::fermion::state::cmpnt_list::CmpntRef as FermionStateRef;
use crate::mappings::operators::OperatorMapper;
use crate::mappings::traits::UpdateParityRho;
use crate::mappings::Mapper;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major::term_set;
use crate::qubit::state::cmpnt::BasisState;

/// Jordan--Wigner mapper from fermionic ladder-operator products to Pauli term sums.
#[derive(Clone)]
pub struct JordanWignerMapper(OperatorMapper);

impl JordanWignerMapper {
    /// Create a mapper for `qubits` and an optional mode ordering.
    pub fn new(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(OperatorMapper::new::<JordanWignerRule>(
            qubits,
            mode_ordering,
        ))
    }
}

impl Mapper<&[(usize, bool)], term_set::TermSet<Complex64>> for JordanWignerMapper {
    fn apply(&mut self, input: &[(usize, bool)]) -> term_set::TermSet<Complex64> {
        self.0.apply(input)
    }
}

impl Mapper<&[(usize, bool)], term_set::TermSet<f64>> for JordanWignerMapper {
    fn apply(&mut self, input: &[(usize, bool)]) -> term_set::TermSet<f64> {
        self.0.apply(input)
    }
}

impl Mapper<FermionStateRef<'_>, BasisState> for JordanWignerMapper {
    fn apply(&mut self, input: FermionStateRef<'_>) -> BasisState {
        self.0.apply_state(input)
    }
}

#[derive(Clone, Copy)]
pub(super) struct JordanWignerRule();

impl UpdateParityRho for JordanWignerRule {
    fn update_set(_: usize, _: usize) -> Vec<usize> {
        Vec::default()
    }

    fn parity_set(i: usize, _: usize) -> Vec<usize> {
        (0..i).collect()
    }

    fn rho_set(i: usize, n_mode: usize) -> Vec<usize> {
        Self::parity_set(i, n_mode)
    }
}
