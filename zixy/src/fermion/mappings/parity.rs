//! Parity mapper.

use crate::fermion::mappings::traits::UpdateParityRho;

/// Struct for implementing Parity Mapper.
#[derive(Clone, Copy)]
pub struct ParityMapper();

impl UpdateParityRho for ParityMapper {
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
    use crate::fermion::mappings::traits::UpdateParityRho;

    use super::ParityMapper;

    #[test]
    fn test_sets_for_four_modes() {
        assert_eq!(ParityMapper::update_set(0, 4), vec![1, 2, 3]);
        assert_eq!(ParityMapper::update_set(1, 4), vec![2, 3]);
        assert_eq!(ParityMapper::update_set(2, 4), vec![3]);
        assert_eq!(ParityMapper::update_set(3, 4), Vec::<usize>::new());

        assert_eq!(ParityMapper::parity_set(0, 4), Vec::<usize>::new());
        assert_eq!(ParityMapper::parity_set(1, 4), vec![0]);
        assert_eq!(ParityMapper::parity_set(2, 4), vec![1]);
        assert_eq!(ParityMapper::parity_set(3, 4), vec![2]);

        assert_eq!(ParityMapper::rho_set(0, 4), Vec::<usize>::new());
        assert_eq!(ParityMapper::rho_set(1, 4), Vec::<usize>::new());
        assert_eq!(ParityMapper::rho_set(2, 4), Vec::<usize>::new());
        assert_eq!(ParityMapper::rho_set(3, 4), Vec::<usize>::new());
    }
}
