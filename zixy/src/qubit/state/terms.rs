//! Extends `CmpntList` with a vector of associated coefficients.

use std::collections::HashSet;

use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::bit_matrix::AsRowMutRef;
use crate::container::coeffs::traits::{NewUnitsWithLen, NumRepr, NumReprVec};
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::{Elements, MutRefElements};
use crate::container::word_iters::terms::AsViewMut as _;
use crate::container::word_iters::{terms, HasWordIters, WordIters};
use crate::qubit::mode::Qubits;
use crate::qubit::state::cmpnt_list::CmpntList;
use crate::qubit::traits::QubitsBased;

/// Stores one coeff for each component of a `CmpntList`.
pub type Terms<C /*: NumRepr*/> = terms::Terms<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = terms::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = terms::ViewMut<'a, CmpntList, C>;

/// Trait for structs that immutably view a [`Terms`].
pub trait AsView<C: NumRepr>: terms::AsView<CmpntList, C> {
    /// Return Some with the Hamming weight if all terms have the same Hamming weight, else return None
    fn hamming_weight(&self) -> Option<usize> {
        self.view().word_iters.hamming_weight()
    }
}

/// Trait for structs that mutably view a [`Terms`].
pub trait AsViewMut<C: NumRepr>: terms::AsViewMut<CmpntList, C> {
    /// Append a basis-state component from a dense vector of bit settings.
    fn push_vec(&mut self, value: Vec<bool>) -> Result<(), OutOfBounds> {
        let mut self_mut_ref = self.view_mut();
        let n_qubit = self_mut_ref.get_word_iters().qubits().len();
        OutOfBounds::check(value.len().saturating_sub(1), n_qubit, Dimension::Mode)?;
        let i_cmpnt = self_mut_ref.len();
        self_mut_ref.push_clear();
        self_mut_ref
            .get_elem_mut_ref(i_cmpnt)
            .get_word_iter_mut_ref()
            .assign_vec_unchecked(value);
        Ok(())
    }

    /// Append a basis-state component from the set of set bit positions.
    fn push_set(&mut self, value: HashSet<usize>) -> Result<(), OutOfBounds> {
        let mut self_mut_ref = self.view_mut();
        let n_qubit = self_mut_ref.get_word_iters().qubits().len();
        OutOfBounds::check(value.len().saturating_sub(1), n_qubit, Dimension::Mode)?;
        let i_cmpnt = self_mut_ref.len();
        self_mut_ref.push_clear();
        self_mut_ref
            .get_elem_mut_ref(i_cmpnt)
            .get_word_iter_mut_ref()
            .assign_set_unchecked(value);
        Ok(())
    }
}

impl<C: NumRepr> AsView<C> for Terms<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for Terms<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> Terms<C> {
    /// Create a new list of state strings on the given space of qubits.
    pub fn new(qubits: Qubits) -> Self {
        Self {
            word_iters: CmpntList::new(qubits),
            coeffs: C::Vector::default(),
        }
    }

    /// Create state terms from parsed binary springs on the given qubit space, using unit coefficients.
    pub fn from_springs(qubits: Qubits, springs: &BinarySprings) -> Result<Self, ParseError> {
        let cmpnts = CmpntList::from_springs(qubits, springs)?;
        let coeffs = C::Vector::new_units_with_len(cmpnts.len());
        Ok(Self::from((cmpnts, coeffs)))
    }

    /// Create state terms from parsed binary springs, inferring a count-based qubit space and using unit coefficients.
    pub fn from_springs_default(springs: &BinarySprings) -> Result<Self, ParseError> {
        let n_qubit = springs.get_mode_inds().default_n_mode() as usize;
        Self::from_springs(Qubits::from_count(n_qubit), springs)
    }

    /// Create from the given sparse strings and coeff vector.
    /// If `coeffs` is shorter than `springs`, it is padded to the length of `springs`` before attempting to absorb phases.
    /// Else if `springs` is shorter than `coeffs`, it is padded to the length of `coeffs` with empty strings.
    pub fn from_springs_coeffs(
        qubits: Qubits,
        mut springs: BinarySprings,
        mut coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        if coeffs.len() < springs.len() {
            coeffs.resize_with_units(springs.len());
        }
        springs.append_empty(springs.len().saturating_sub(coeffs.len()));
        let list = CmpntList::from_springs(qubits, &springs)?;
        Ok(Self::from((list, coeffs)))
    }

    /// Create from the given sparse strings and coeff vector, absorbing any phases into the coefficient if representable, else return error.
    /// Infer a Count-type qubit space from the springs object.
    pub fn from_springs_coeffs_default(
        springs: BinarySprings,
        coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        let n_qubit = springs.get_mode_inds().default_n_mode() as usize;
        Self::from_springs_coeffs(Qubits::from_count(n_qubit), springs, coeffs)
    }
}

pub type TermRef<'a, C /*: NumRepr*/> = terms::TermRef<'a, CmpntList, C>;
pub type TermMutRef<'a, C /*: NumRepr*/> = terms::TermMutRef<'a, CmpntList, C>;

#[cfg(test)]
mod tests {
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::traits::RefElements;

    use super::*;

    #[test]
    fn test_hamming_weight_sets() -> Result<(), OutOfBounds> {
        let mut terms = Terms::<f64>::new(Qubits::from_count(4));
        terms.push_set(HashSet::from([2, 3]))?; // 0011 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_set(HashSet::from([0, 3]))?; // 1001 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_set(HashSet::from([0, 1]))?; // 1100 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_set(HashSet::from([0, 1, 2]))?; // 1110 has a hamming weight of 3
        assert_eq!(terms.hamming_weight(), None);
        Ok(())
    }

    #[test]
    fn test_hamming_weight_vector() -> Result<(), OutOfBounds> {
        let mut terms = Terms::<f64>::new(Qubits::from_count(4));
        terms.push_vec(vec![false, false, true, true])?; // 0011 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_vec(vec![true, false, false, true])?; // 1001 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_vec(vec![true, true, false, false])?; // 1100 has a hamming weight of 2
        assert_eq!(terms.hamming_weight(), Some(2));
        terms.push_vec(vec![true, true, true, false])?; // 1110 has a hamming weight of 3
        assert_eq!(terms.hamming_weight(), None);
        Ok(())
    }

    #[test]
    fn test_push_vec_out_of_bounds() {
        let mut terms = Terms::<f64>::new(Qubits::from_count(2));
        assert!(terms.push_vec(vec![true, true, true, true]).is_err());
    }

    #[test]
    fn test_push_set_out_of_bounds() {
        let mut terms = Terms::<f64>::new(Qubits::from_count(2));
        assert!(terms.push_set(HashSet::from([0, 1, 2, 3])).is_err());
    }

    #[test]
    fn test_terms_from_springs_out_of_bounds() {
        let qubits = Qubits::from_count(2);
        let springs = BinarySprings::from_str("[1, 0, 1, 0]").unwrap();
        assert!(Terms::<f64>::from_springs(qubits, &springs).is_err());
    }

    #[test]
    fn test_terms_from_springs() -> Result<(), ParseError> {
        use crate::container::word_iters::terms::AsView;
        let qubits = Qubits::from_count(3);
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let terms = Terms::<f64>::from_springs(qubits, &springs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        Ok(())
    }

    #[test]
    fn test_terms_from_springs_default() -> Result<(), ParseError> {
        use crate::container::word_iters::terms::AsView;
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let terms = Terms::<f64>::from_springs_default(&springs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        assert_eq!(terms.coeffs[0], 1.0);
        Ok(())
    }

    #[test]
    fn test_terms_from_springs_coeffs() -> Result<(), ParseError> {
        use crate::container::word_iters::terms::AsView;
        let qubits = Qubits::from_count(3);
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let coeffs = vec![0.5];
        let terms = Terms::<f64>::from_springs_coeffs(qubits, springs, coeffs)?;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, true]
        );
        assert_eq!(terms.coeffs[0], 0.5);
        Ok(())
    }
}
