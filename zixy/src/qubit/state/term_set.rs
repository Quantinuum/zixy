//! Stores qubit state terms with the help of a `Map` to ensure each string appears at most once.

use crate::container::coeffs::traits::NumRepr;
use crate::container::map::Map;
use crate::container::word_iters;
use crate::qubit::mode::Qubits;
use crate::qubit::state::cmpnt_list::CmpntList;
use crate::qubit::state::terms::Terms;
use crate::qubit::traits::QubitsBased;

pub type TermSet<C /*: NumRepr*/> = word_iters::term_set::TermSet<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = word_iters::term_set::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = word_iters::term_set::ViewMut<'a, CmpntList, C>;

/// Trait for structs that immutably view a [`TermSet`].
pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<CmpntList, C> {}

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
mod tests {}
