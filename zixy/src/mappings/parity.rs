//! Parity mapper.

use num_complex::Complex64;

use crate::mappings::operators::OperatorMapper;
use crate::mappings::traits::UpdateParityRho;
use crate::mappings::Mapper;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major::term_set;

/// Parity mapper from fermionic ladder-operator products to Pauli term sums.
#[derive(Clone)]
pub struct ParityMapper(OperatorMapper);

impl ParityMapper {
    /// Create a mapper for `qubits` and an optional mode ordering.
    pub fn new(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        Self(OperatorMapper::new::<ParityRule>(qubits, mode_ordering))
    }
}

impl Mapper<&[(usize, bool)], term_set::TermSet<Complex64>> for ParityMapper {
    fn apply(&mut self, input: &[(usize, bool)]) -> term_set::TermSet<Complex64> {
        self.0.apply(input)
    }
}

/// Parity mapper rule.
#[derive(Clone, Copy)]
pub(super) struct ParityRule();

impl UpdateParityRho for ParityRule {
    fn update_set(i: usize, n_mode: usize) -> Vec<usize> {
        ((i + 1)..n_mode).collect()
    }

    fn parity_set(i: usize, _: usize) -> Vec<usize> {
        if i == 0 {
            Vec::default()
        } else {
            vec![i - 1]
        }
    }

    fn rho_set(_: usize, _: usize) -> Vec<usize> {
        Vec::default()
    }
}

#[cfg(test)]
mod tests {
    use crate::mappings::traits::UpdateParityRho;

    use super::ParityRule;

    #[test]
    fn test_sets_for_four_modes() {
        assert_eq!(ParityRule::update_set(0, 4), vec![1, 2, 3]);
        assert_eq!(ParityRule::update_set(1, 4), vec![2, 3]);
        assert_eq!(ParityRule::update_set(2, 4), vec![3]);
        assert_eq!(ParityRule::update_set(3, 4), Vec::<usize>::new());

        assert_eq!(ParityRule::parity_set(0, 4), Vec::<usize>::new());
        assert_eq!(ParityRule::parity_set(1, 4), vec![0]);
        assert_eq!(ParityRule::parity_set(2, 4), vec![1]);
        assert_eq!(ParityRule::parity_set(3, 4), vec![2]);

        assert_eq!(ParityRule::rho_set(0, 4), Vec::<usize>::new());
        assert_eq!(ParityRule::rho_set(1, 4), Vec::<usize>::new());
        assert_eq!(ParityRule::rho_set(2, 4), Vec::<usize>::new());
        assert_eq!(ParityRule::rho_set(3, 4), Vec::<usize>::new());
    }
}
