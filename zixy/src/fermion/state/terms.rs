//! Extends `CmpntList` with associated coefficients.

use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::coeffs::traits::NewUnitsWithLen;
use crate::container::coeffs::traits::NumReprVec;
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

    /// Create state terms from parsed binary springs on the given mode space, using unit coefficients.
    pub fn from_springs(modes: Modes, springs: &BinarySprings) -> Result<Self, ParseError> {
        let cmpnts = CmpntList::from_springs(modes, springs)?;
        let coeffs = C::Vector::new_units_with_len(cmpnts.len());
        Ok(Self::from((cmpnts, coeffs)))
    }

    /// Create state terms from parsed binary springs, inferring a count-based mode space and using unit coefficients.
    pub fn from_springs_default(springs: &BinarySprings) -> Result<Self, ParseError> {
        let n_mode = springs.get_mode_inds().default_n_mode() as usize;
        Self::from_springs(Modes::from_count(n_mode), springs)
    }

    /// Create from the given sparse strings and coeff vector.
    /// If `coeffs` is shorter than `springs`, it is padded to the length of `springs`` before attempting to absorb phases.
    /// Else if `springs` is shorter than `coeffs`, it is padded to the length of `coeffs` with empty strings.
    pub fn from_springs_coeffs(
        modes: Modes,
        mut springs: BinarySprings,
        mut coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        if coeffs.len() < springs.len() {
            coeffs.resize_with_units(springs.len());
        }
        springs.append_empty(springs.len().saturating_sub(coeffs.len()));
        let list = CmpntList::from_springs(modes, &springs)?;
        Ok(Self::from((list, coeffs)))
    }

    /// Create from the given sparse strings and coeff vector, absorbing any phases into the coefficient if representable, else return error.
    /// Infer a Count-type mode space from the springs object.
    pub fn from_springs_coeffs_default(
        springs: BinarySprings,
        coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        let n_mode = springs.get_mode_inds().default_n_mode() as usize;
        Self::from_springs_coeffs(Modes::from_count(n_mode), springs, coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::coeffs::unity::Unity;
    use crate::container::traits::{Elements, RefElements};
    use crate::fermion::state::terms::AsView;
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

    #[test]
    fn test_terms_from_springs_out_of_bounds() -> Result<(), ParseError> {
        let modes = Modes::from_count(2);
        let springs = BinarySprings::from_str("[1, 0, 1, 0]")?;
        assert!(Terms::<f64>::from_springs(modes, &springs).is_err());
        Ok(())
    }

    #[test]
    fn test_terms_from_springs() -> Result<(), ParseError> {
        let modes = Modes::from_count(3);
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let terms = Terms::<f64>::from_springs(modes, &springs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        Ok(())
    }

    #[test]
    fn test_terms_from_springs_default() -> Result<(), ParseError> {
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let terms = Terms::<f64>::from_springs_default(&springs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        assert_eq!(terms.coeffs[0], 1.0);
        Ok(())
    }

    #[test]
    fn test_terms_from_springs_coeffs() -> Result<(), ParseError> {
        let modes = Modes::from_count(3);
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let coeffs = vec![0.5];
        let terms = Terms::<f64>::from_springs_coeffs(modes, springs, coeffs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        assert_eq!(terms.coeffs[0], 0.5);
        Ok(())
    }

    #[test]
    fn test_terms_from_springs_coeffs_default() -> Result<(), ParseError> {
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let coeffs = vec![0.5];
        let terms = Terms::<f64>::from_springs_coeffs_default(springs, coeffs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        assert_eq!(terms.coeffs[0], 0.5);
        Ok(())
    }
}
