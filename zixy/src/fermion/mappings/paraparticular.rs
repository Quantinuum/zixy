//! Paraparticular mapper.

use crate::fermion::mappings::traits::UpdateParityRho;

/// Struct for implementing Paraparticular Mapper.
#[derive(Clone, Copy)]
pub struct ParaparticularMapper();

impl UpdateParityRho for ParaparticularMapper {
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
