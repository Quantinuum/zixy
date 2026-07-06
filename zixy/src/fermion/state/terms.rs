//! Extends `CmpntList` with associated coefficients.

use std::collections::HashSet;

use crate::container::bit_matrix::AsRowMutRef;
use crate::container::coeffs::traits::NumRepr;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::{Elements, MutRefElements};
use crate::container::word_iters::terms::AsViewMut as _;
use crate::container::word_iters::{terms, HasWordIters, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::state::cmpnt_list::CmpntList;
use crate::fermion::traits::ModesBased;

/// Stores one coeff for each component of a `CmpntList`.
pub type Terms<C /*: NumRepr*/> = terms::Terms<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = terms::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = terms::ViewMut<'a, CmpntList, C>;

pub type TermRef<'a, C /*: NumRepr*/> = terms::TermRef<'a, CmpntList, C>;
pub type TermMutRef<'a, C /*: NumRepr*/> = terms::TermMutRef<'a, CmpntList, C>;

pub trait AsView<C: NumRepr>: terms::AsView<CmpntList, C> {
    /// Return the particle number if all Slater determinants have the same particle number, otherwise return None.
    fn particle_number(&self) -> Option<usize> {
        self.view().word_iters.hamming_weight()
    }
}

/// Trait for structs that mutably view a [`Terms`].
pub trait AsViewMut<C: NumRepr>: terms::AsViewMut<CmpntList, C> {
    /// Append a Slater determinant from an occupation vector.
    fn push_vec(&mut self, value: Vec<bool>) -> Result<(), OutOfBounds> {
        let mut self_mut_ref = self.view_mut();
        let n_mode = self_mut_ref.get_word_iters().modes().len();
        OutOfBounds::check(value.len().saturating_sub(1), n_mode, Dimension::Mode)?;
        let i_cmpnt = self_mut_ref.len();
        self_mut_ref.push_clear();
        self_mut_ref
            .get_elem_mut_ref(i_cmpnt)
            .get_word_iter_mut_ref()
            .assign_vec_unchecked(value);
        Ok(())
    }

    /// Append a Slater determinant from the set of set bit positions.
    fn push_set(&mut self, value: HashSet<usize>) -> Result<(), OutOfBounds> {
        let mut self_mut_ref = self.view_mut();
        let n_mode = self_mut_ref.get_word_iters().modes().len();
        if let Some(max_ind) = value.iter().max() {
            OutOfBounds::check(*max_ind, n_mode, Dimension::Mode)?;
        }
        let i_cmpnt = self_mut_ref.len();
        self_mut_ref.push_clear();
        self_mut_ref
            .get_elem_mut_ref(i_cmpnt)
            .get_word_iter_mut_ref()
            .assign_set_unchecked(value);
        Ok(())
    }

    /// Append a Slater determinant with the given coefficient.
    fn push_set_with_coeff(&mut self, value: HashSet<usize>, coeff: C) -> Result<(), OutOfBounds> {
        self.push_set(value)?;
        let idx = self.view_mut().len() - 1;
        self.view_mut().get_elem_mut_ref(idx).set_coeff(coeff);
        Ok(())
    }
}

impl<C: NumRepr> AsView<C> for Terms<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for Terms<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> Terms<C> {
    /// Create a new list of state strings on the given space of modes.
    pub fn new(modes: Modes) -> Self {
        Self {
            word_iters: CmpntList::new(modes),
            coeffs: C::Vector::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::coeffs::unity::Unity;
    use crate::container::traits::{Elements, RefElements};
    use num_complex::Complex64;

    #[test]
    fn test_empty() {
        let list_unity = Terms::<Unity>::new(Modes::from_count(3));
        let list_real = Terms::<f64>::new(Modes::from_count(3));
        let list_complex = Terms::<Complex64>::new(Modes::from_count(3));
        assert!(list_unity.is_empty());
        assert!(list_real.is_empty());
        assert!(list_complex.is_empty());
    }

    #[test]
    fn test_particle_number_uniform() {
        let mut terms = Terms::<f64>::new(Modes::from_count(4));
        terms.push_set(HashSet::from([0, 1])).unwrap();
        terms.push_set(HashSet::from([1, 3])).unwrap();
        assert_eq!(terms.particle_number(), Some(2));
    }

    #[test]
    fn test_particle_number_mixed() {
        let mut terms = Terms::<f64>::new(Modes::from_count(4));
        terms.push_set(HashSet::from([0, 1])).unwrap();
        terms.push_set(HashSet::from([0, 1, 2])).unwrap();
        assert_eq!(terms.particle_number(), None);
    }

    #[test]
    fn test_push_vec_and_set() {
        let modes = Modes::from_count(4);
        let mut terms = Terms::<Unity>::new(modes.clone());
        terms.push_vec(vec![false, true, false, true]).unwrap();
        terms.push_set(HashSet::from([0, 2])).unwrap();
        assert_eq!(terms.len(), 2);
        assert_eq!(
            terms.get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![false, true, false, true]
        );
        assert_eq!(
            terms.get_elem_ref(1).get_word_iter_ref().to_vec(),
            vec![true, false, true, false]
        );
    }

    #[test]
    fn test_push_set_out_of_bounds() {
        let modes = Modes::from_count(4);
        let mut terms = Terms::<Unity>::new(modes.clone());
        let result = terms.push_set(HashSet::from([0, 4])); // 4 is out of bounds
        assert!(result.is_err());
    }
}
