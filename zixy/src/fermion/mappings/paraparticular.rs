//! Parapartiular mapper.

use crate::fermion::mappings::traits::UpdateParityRho;

/// Struct for implementing Parapartiular Mapper.
#[derive(Clone, Copy)]
pub struct ParapartiularMapper();

impl UpdateParityRho for ParapartiularMapper {
    fn update_set(_: usize, _: usize) -> Vec<usize> {
        Vec::default()
    }

    fn parity_set(_: usize, _: usize) -> Vec<usize> {
        Vec::default()
    }

    fn rho_set(_: usize, _: usize) -> Vec<usize> {
        Vec::default()
    }
}
