//! Jordan-Wigner transformation.

use crate::fermion::mappings::traits::UpdateParityRho;

/// Struct for implementing Jordan Wigner Mapper.
#[derive(Clone, Copy)]
pub struct JordanWignerMapper();

impl UpdateParityRho for JordanWignerMapper {
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
