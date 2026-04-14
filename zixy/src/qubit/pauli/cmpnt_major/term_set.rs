//! Stores Pauli terms with the help of a `Map` to ensure each string appears at most once.

use crate::container::coeffs::traits::NumRepr;
use crate::container::errors::OutOfBounds;
use crate::container::map::Map;
use crate::container::traits::proj::Borrow;
use crate::container::word_iters;
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::cmpnt::PauliWord;
use crate::qubit::pauli::cmpnt_major::cmpnt_list::CmpntList;
use crate::qubit::pauli::cmpnt_major::terms::Terms;
use crate::qubit::traits::QubitsBased;

pub type TermSet<C /*: NumRepr*/> = word_iters::term_set::TermSet<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = word_iters::term_set::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = word_iters::term_set::ViewMut<'a, CmpntList, C>;

/// Trait for structs that immutably view a [`TermSet`].
pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<CmpntList, C> {
    /// Lookup the given vector of Paulis.
    fn lookup_coeff_pauli_vec(&self, paulis: Vec<PauliMatrix>) -> Result<Option<C>, OutOfBounds> {
        let self_ref = self.view();
        Ok(self.lookup_coeff_elem_ref(
            PauliWord::from_vec(self_ref.qubits().clone(), paulis)?.borrow(),
        ))
    }
}

/// Trait for structs that mutably view a [`TermSet`].
pub trait AsViewMut<C: NumRepr>: word_iters::term_set::AsViewMut<CmpntList, C> {}

impl<C: NumRepr> AsView<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}
impl<'a, C: NumRepr> AsView<C> for ViewMut<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> TermSet<C> {
    /// Create a new instance.
    pub fn new(qubits: Qubits) -> Self {
        Self {
            terms: Terms::new(qubits),
            map: Map::default(),
        }
    }
}

impl<C: NumRepr> QubitsBased for TermSet<C> {
    fn qubits(&self) -> &Qubits {
        self.terms.word_iters.qubits()
    }
}

impl<'a, C: NumRepr> QubitsBased for View<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<'a, C: NumRepr> QubitsBased for ViewMut<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmpnt::springs::ModeSettings;
    use crate::container::coeffs::unity::Unity;
    use crate::container::quicksort::{LexicographicSort, QuickSortNoCoeffs};
    use crate::container::traits::proj::BorrowMut;
    use crate::container::traits::Elements;
    use crate::container::word_iters::term_set::{AsView, AsViewMut};
    use crate::qubit::mode::PauliMatrix;
    use crate::qubit::pauli::springs::Springs;
    use crate::qubit::test::HEHP_STO3G_HAM_JW_INPUT;

    use bincode::config;
    use PauliMatrix::*;

    #[test]
    fn test_contains() {
        let qubits = Qubits::from_count(4);
        let make_op = |v: Vec<PauliMatrix>| PauliWord::from_vec(qubits.clone(), v).unwrap();
        let mut set = TermSet::<Unity>::new(qubits.clone());
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert!(
            set.borrow_mut()
                .insert_elem_ref_default(make_op(vec![I, X, Y, Z]).borrow())
                .1
        );
        assert!(set
            .borrow()
            .contains_elem_ref(make_op(vec![I, X, Y, Z]).borrow()));
        assert_eq!(set.len(), 1);
        assert!(
            set.borrow_mut()
                .insert_elem_ref_default(make_op(vec![I, X, Y, Z]).borrow())
                == (0, false)
        );
        assert!(set
            .borrow()
            .contains_elem_ref(make_op(vec![I, X, Y, Z]).borrow()));
        assert_eq!(set.len(), 1);
        assert!(
            set.borrow_mut()
                .insert_elem_ref_default(make_op(vec![Z, Y, X, I]).borrow())
                .1
        );
        assert!(set
            .borrow()
            .contains_elem_ref(make_op(vec![Z, Y, X, I]).borrow()));
        assert_eq!(set.len(), 2);
        assert!(
            set.borrow_mut()
                .insert_elem_ref_default(make_op(vec![Z, I, X, Y]).borrow())
                .1
        );
        assert!(set
            .borrow()
            .contains_elem_ref(make_op(vec![Z, I, X, Y]).borrow()));
        assert_eq!(set.len(), 3);
        assert!(set
            .borrow_mut()
            .drop_elem_ref(make_op(vec![I, X, Y, Z]).borrow()));
        assert!(!set
            .borrow()
            .contains_elem_ref(make_op(vec![I, X, Y, Z]).borrow()));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_to_from_binary() {
        let springs = Springs::from_str(HEHP_STO3G_HAM_JW_INPUT);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let list = CmpntList::from_springs_default(&springs).0;
        let set = TermSet::<Unity>::from(list.clone());
        let encoded = bincode::serde::encode_to_vec(&set, config::standard()).unwrap();
        let (decoded, len): (TermSet<Unity>, usize) =
            bincode::serde::decode_from_slice(&encoded[..], config::standard()).unwrap();
        assert_eq!(len, encoded.len());
        assert!(set.borrow() == decoded.borrow());

        let mut sorted_list = list.clone();
        LexicographicSort { ascending: true }.sort(&mut sorted_list);
        // the sorted op vec is not the same as the original...
        assert_ne!(list, sorted_list);
        // but the set created from it is still equal to the decoded set created from the original
        assert!(decoded.borrow() == TermSet::<Unity>::from(sorted_list).borrow());
    }
}
