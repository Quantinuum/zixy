//! Bravyi-Kitaev transformation.

use crate::fermion::mappings::traits::UpdateParityRho;

/// Mapper implementing the Bravyi-Kitaev transform through update, parity, and rho index sets.
/// Uses Fenwick tree bit tricks to build up the BK linear transform.
#[derive(Clone, Copy)]
pub struct BravyiKitaevMapper();

impl BravyiKitaevMapper {
    fn prefix_set(mut i_one_based: usize) -> Vec<usize> {
        let mut out = Vec::default();
        while i_one_based != 0 {
            out.push(i_one_based - 1);
            i_one_based &= i_one_based - 1;
        }
        out
    }

    fn occupation_set(i: usize) -> Vec<usize> {
        let i_one_based = i + 1;
        let parent = i_one_based & (i_one_based - 1);
        let mut out = vec![i];
        let mut cursor = i_one_based - 1;
        while cursor != parent {
            out.push(cursor - 1);
            cursor &= cursor - 1;
        }
        out
    }

    fn flip_set(i: usize) -> Vec<usize> {
        Self::occupation_set(i)
            .into_iter()
            .filter(|&j| j != i)
            .collect()
    }

    fn remainder_set(i: usize, n_mode: usize) -> Vec<usize> {
        let flip = Self::flip_set(i);
        Self::parity_set(i, n_mode)
            .into_iter()
            .filter(|j| !flip.contains(j))
            .collect()
    }
}

impl UpdateParityRho for BravyiKitaevMapper {
    fn update_set(i: usize, n_mode: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cursor = i + 1;
        loop {
            // add only the least significant set bit.
            cursor += cursor & cursor.wrapping_neg();
            if cursor > n_mode || cursor == 0 {
                break;
            }
            out.push(cursor - 1);
        }
        out
    }

    fn parity_set(i: usize, _: usize) -> Vec<usize> {
        if i == 0 {
            Vec::default()
        } else {
            Self::prefix_set(i)
        }
    }

    fn rho_set(i: usize, n_mode: usize) -> Vec<usize> {
        if i.is_multiple_of(2) {
            Self::parity_set(i, n_mode)
        } else {
            Self::remainder_set(i, n_mode)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fermion::mappings::traits::UpdateParityRho;

    use super::BravyiKitaevMapper;

    #[test]
    fn test_sets_for_four_modes() {
        assert_eq!(BravyiKitaevMapper::update_set(0, 4), vec![1, 3]);
        assert_eq!(BravyiKitaevMapper::update_set(1, 4), vec![3]);
        assert_eq!(BravyiKitaevMapper::update_set(2, 4), vec![3]);
        assert_eq!(BravyiKitaevMapper::update_set(3, 4), Vec::<usize>::new());

        assert_eq!(BravyiKitaevMapper::parity_set(0, 4), Vec::<usize>::new());
        assert_eq!(BravyiKitaevMapper::parity_set(1, 4), vec![0]);
        assert_eq!(BravyiKitaevMapper::parity_set(2, 4), vec![1]);
        assert_eq!(BravyiKitaevMapper::parity_set(3, 4), vec![2, 1]);

        assert_eq!(BravyiKitaevMapper::rho_set(0, 4), Vec::<usize>::new());
        assert_eq!(BravyiKitaevMapper::rho_set(1, 4), Vec::<usize>::new());
        assert_eq!(BravyiKitaevMapper::rho_set(2, 4), vec![1]);
        assert_eq!(BravyiKitaevMapper::rho_set(3, 4), Vec::<usize>::new());
    }
}
