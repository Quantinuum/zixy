//! Extend the `WordIters`-implementing types by associating a numeric coefficient with each element.
//! The iterable elements are represented by three kinds of type:
//! - "u64it": the iterator over u64 words themselves,
//! - "elem_ref": a reference to an element within an iterable elements object
//! - "term_ref": a reference to an element within an iterable elements with coeffs object
//!
//! So "term" means an iterable element - coeff pair.

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::container::coeffs::traits::{
    ComplexParts, FieldElem, HasCoeffs, HasCoeffsMut, IsComplex, NumRepr, NumReprVec,
};
use crate::container::map::Map;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::traits::{proj, Elements, EmptyClone, EmptyFrom, RefElements};
use crate::container::word_iters::set::{AsView as SetView, AsViewMut as SetViewMut};
use crate::container::word_iters::terms::{
    self, AsView as TermsView, AsViewMut as TermsViewMut, TermRef,
};
use crate::container::word_iters::traits::{InsertU64It, TransformCoeffs};
use crate::container::word_iters::{self, WordIters};
use crate::container::word_iters::{HasWordIters, HasWordItersMut};

/// `Terms`, but with a map to enforce uniqueness of iterable element entries, and
/// to provide constant-time lookup.
#[derive(Debug, Clone)]
pub struct TermSet<T: WordIters, C: NumRepr> {
    pub terms: terms::Terms<T, C>,
    pub map: Map,
}

/// Borrowed immutable view of the fields of a [`TermSet`] collection.
pub struct View<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a T,
    pub coeffs: &'a C::Vector,
    pub map: &'a Map,
}

/// Borrowed mutable view of the fields of a [`TermSet`] collection.
pub struct ViewMut<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a mut T,
    pub coeffs: &'a mut C::Vector,
    pub map: &'a mut Map,
}

/// Trait for structs that immutably view a [`TermSet`].
pub trait AsView<T: WordIters, C: NumRepr> {
    /// Return an immutable view over the unique terms, their coefficients, and the lookup map.
    fn view<'a>(&'a self) -> View<'a, T, C>;

    /// Reborrow this mapped term set as a plain terms view without the lookup map.
    fn as_terms<'a>(&'a self) -> terms::View<'a, T, C>
    where
        T: 'a,
        C: 'a,
    {
        let self_ref = self.view();
        terms::View {
            word_iters: self_ref.word_iters,
            coeffs: self_ref.coeffs,
        }
    }

    /// Reborrow this mapped term set as a uniqueness-enforcing set view over the word iterators alone.
    fn as_mapped_word_iters<'a>(&'a self) -> super::set::View<'a, T>
    where
        T: 'a,
        C: 'a,
    {
        let self_ref = self.view();
        super::set::View {
            word_iters: self_ref.word_iters,
            map: self_ref.map,
        }
    }

    /// Look up the coefficient associated with the iterable element identified by `u64it`.
    fn lookup_coeff_u64it(&self, u64it: impl Iterator<Item = u64> + Clone) -> Option<C> {
        let self_ref = self.view();
        self_ref
            .as_mapped_word_iters()
            .lookup(u64it)
            .map(|i| self_ref.get_coeffs().get_unchecked(i))
    }

    /// Look up the coefficient associated with `elem_ref`.
    fn lookup_coeff_elem_ref(&self, elem_ref: word_iters::ElemRef<T>) -> Option<C> {
        self.lookup_coeff_u64it(elem_ref.get_u64it())
    }

    /// Return whether the referenced iterable element is in the set.
    fn contains_u64it(&self, u64it: impl Iterator<Item = u64> + Clone) -> bool {
        self.lookup_coeff_u64it(u64it).is_some()
    }

    /// Return whether `elem_ref` is present in the mapped term set.
    fn contains_elem_ref(&self, elem_ref: word_iters::ElemRef<T>) -> bool {
        self.contains_u64it(elem_ref.get_u64it())
    }

    /// Return whether `self` and `other` contain the same elements with identical coefficients in any order.
    fn equal(&self, other: &Self) -> bool {
        let one_way = |l: &Self, r: &Self| -> bool {
            let l_ref = l.view();
            for (i, elem_ref) in l_ref.word_iters.iter().enumerate() {
                match r.lookup_coeff_elem_ref(elem_ref) {
                    Some(x) => {
                        if x != l_ref.get_coeffs().get_unchecked(i) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        };
        one_way(self, other) && one_way(other, self)
    }

    /// Return whether all terms in the linear combination are insignificant with respect to the given atol.
    fn all_insignificant(&self, atol: f64) -> bool
    where
        C: FieldElem,
    {
        self.view()
            .get_coeffs()
            .iter()
            .all(|x| x.is_close(C::ZERO, 0.0, atol))
    }

    /// Return whether all elements are close with those of another instance within given tolerances.
    fn all_close(&self, other: &Self, rtol: f64, atol: f64) -> bool
    where
        C: FieldElem,
    {
        let one_way = |lhs: &Self, rhs: &Self| -> bool {
            let lhs_ref = lhs.view();
            for (i, elem_ref) in lhs_ref.get_word_iters().iter().enumerate() {
                match rhs.lookup_coeff_elem_ref(elem_ref) {
                    Some(x) => {
                        if !lhs_ref.get_coeffs()[i].is_close(x, rtol, atol) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        };
        one_way(self, other) && one_way(other, self)
    }

    /// Return whether all elements are close with those of another instance within default tolerances.
    fn all_close_default(&self, other: &Self) -> bool
    where
        C: FieldElem,
    {
        self.all_close(other, C::RTOL_DEFAULT, C::ATOL_DEFAULT)
    }

    /// Return whether every coefficient has an insignificant imaginary part with tolerance `atol`.
    fn all_imag_insignificant(&self, atol: f64) -> bool
    where
        C: IsComplex,
    {
        self.as_terms().all_imag_insignificant(atol)
    }

    /// Return whether every coefficient has an insignificant real part with tolerance `atol`.
    fn all_real_insignificant(&self, atol: f64) -> bool
    where
        C: IsComplex,
    {
        self.as_terms().all_real_insignificant(atol)
    }

    /// Return the real parts of the coefficients as a new mapped term set with `f64` coefficients.
    fn real_part(&self) -> TermSet<T, f64>
    where
        C::Vector: ComplexParts,
    {
        self.as_terms().real_part().into()
    }

    /// Return the imaginary parts of the coefficients as a new mapped term set with `f64` coefficients.
    fn imag_part(&self) -> TermSet<T, f64>
    where
        C::Vector: ComplexParts,
    {
        self.as_terms().imag_part().into()
    }
}

impl<T: WordIters, C: NumRepr> AsView<T, C> for TermSet<T, C> {
    fn view<'a>(&'a self) -> View<'a, T, C> {
        self.borrow()
    }
}

impl<'a, T: WordIters, C: NumRepr> AsView<T, C> for View<'a, T, C> {
    fn view<'b>(&'b self) -> View<'b, T, C> {
        View {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            map: self.map,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> AsView<T, C> for ViewMut<'a, T, C> {
    fn view<'b>(&'b self) -> View<'b, T, C> {
        View {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            map: self.map,
        }
    }
}

/// Trait for structs that mutably view a [`TermSet`].
pub trait AsViewMut<T: WordIters, C: NumRepr>: AsView<T, C> {
    /// Return a mutable view over the unique terms, their coefficients, and the lookup map.
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T, C>;

    /// Reborrow this mapped term set as a mutable plain terms view without the lookup map.
    fn as_terms_mut<'a>(&'a mut self) -> terms::ViewMut<'a, T, C>
    where
        T: 'a,
        C: 'a,
    {
        let self_mut = self.view_mut();
        terms::ViewMut {
            word_iters: self_mut.word_iters,
            coeffs: self_mut.coeffs,
        }
    }

    /// Reborrow this mapped term set as a mutable set view over the word iterators alone.
    fn as_mapped_word_iters_mut<'a>(&'a mut self) -> super::set::ViewMut<'a, T>
    where
        T: 'a,
        C: 'a,
    {
        let self_mut = self.view_mut();
        super::set::ViewMut {
            word_iters: self_mut.word_iters,
            map: self_mut.map,
        }
    }

    /// Insert `u64it` with coefficient `c`, or return the existing index if that element is already present.
    fn insert_u64it_or_get_index(
        &mut self,
        u64it: impl Iterator<Item = u64> + Clone,
        c: C,
    ) -> (usize, bool) {
        let index = {
            let (index, inserted) = self
                .as_mapped_word_iters_mut()
                .insert_u64it_or_get_index(u64it);
            if !inserted {
                return (index, false);
            }
            index
        };
        self.as_terms_mut().sync_sizes();
        self.view_mut().coeffs.set_unchecked(index, c);
        (index, true)
    }

    /// Insert `elem_ref` with coefficient `c`, or return the index of the existing element if it is already present.
    fn insert_elem_ref_or_get_index(
        &mut self,
        elem_ref: word_iters::ElemRef<T>,
        c: C,
    ) -> (usize, bool) {
        self.insert_u64it_or_get_index(elem_ref.get_u64it(), c)
    }

    /// Insert `u64it` with coefficient `c`, or overwrite the coefficient if the element is already present.
    fn insert_u64it_or_update(
        &mut self,
        u64it: impl Iterator<Item = u64> + Clone,
        c: C,
    ) -> (usize, bool) {
        let (i, inserted) = self.insert_u64it_or_get_index(u64it, c);
        if !inserted {
            self.view_mut().coeffs.set_unchecked(i, c);
        }
        (i, inserted)
    }

    /// Insert `elem_ref` with coefficient `c`, or overwrite the coefficient if it is already present.
    fn insert_elem_ref_or_update(
        &mut self,
        elem_ref: word_iters::ElemRef<T>,
        c: C,
    ) -> (usize, bool) {
        self.insert_u64it_or_update(elem_ref.get_u64it(), c)
    }

    /// Insert `term_ref`, or overwrite the coefficient of the existing matching element.
    fn insert_term_ref_or_update(&mut self, term_ref: TermRef<T, C>) -> (usize, bool) {
        self.insert_elem_ref_or_update(term_ref.get_word_iter_ref(), term_ref.get_coeff())
    }

    /// Insert `u64it` using `C::default()` as the coefficient, or update the existing entry to that value.
    fn insert_u64it_default(&mut self, u64it: impl Iterator<Item = u64> + Clone) -> (usize, bool) {
        self.insert_u64it_or_update(u64it, C::default())
    }

    /// Insert `elem_ref` using `C::default()` as the coefficient, or update the existing entry to that value.
    fn insert_elem_ref_default(&mut self, elem_ref: word_iters::ElemRef<T>) -> (usize, bool) {
        self.insert_elem_ref_or_update(elem_ref, C::default())
    }

    /// Remove the term at `index` using swap-remove (or pop-and-swap), keeping the map and coefficient vector in sync.
    fn drop_index(&mut self, index: usize) -> bool {
        if self.as_mapped_word_iters_mut().drop(index) {
            self.view_mut().coeffs.pop_and_swap(index);
            true
        } else {
            false
        }
    }

    /// Remove the term identified by `u64it` if it is present.
    fn drop_u64it(&mut self, u64it: impl Iterator<Item = u64> + Clone) -> bool {
        match self.as_mapped_word_iters().lookup(u64it) {
            Some(i) if self.as_mapped_word_iters_mut().drop(i) => {
                self.as_terms_mut().sync_sizes();
                true
            }
            None => false,
            Some(_) => false,
        }
    }

    /// Remove the term referenced by `elem_ref` if it is present.
    fn drop_elem_ref(&mut self, elem_ref: word_iters::ElemRef<T>) -> bool {
        self.drop_u64it(elem_ref.get_u64it())
    }

    /// Remove every stored term, coefficient, and map entry.
    fn clear(&mut self) {
        self.as_terms_mut().clear();
        self.as_mapped_word_iters_mut().clear();
    }

    /// Remove all terms with a coefficient of absolute value less than or equal to atol.
    fn drop_all_insignificant(&mut self, atol: f64)
    where
        C: FieldElem,
    {
        let mut self_mut = self.view_mut();
        for i in (0..self_mut.word_iters.len()).rev() {
            if !self_mut.coeffs[i].is_significant(atol) {
                self_mut.drop_index(i);
            }
        }
    }
}

impl<T: WordIters, C: NumRepr> AsViewMut<T, C> for TermSet<T, C> {
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T, C> {
        ViewMut {
            word_iters: &mut self.terms.word_iters,
            coeffs: &mut self.terms.coeffs,
            map: &mut self.map,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> AsViewMut<T, C> for ViewMut<'a, T, C> {
    fn view_mut<'b, 'c: 'b>(&'c mut self) -> ViewMut<'b, T, C> {
        ViewMut {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            map: self.map,
        }
    }
}

impl<T: WordIters, C: NumRepr> Elements for TermSet<T, C> {
    fn len(&self) -> usize {
        self.terms.len()
    }
}

impl<T: WordIters, C: NumRepr> HasWordIters<T> for TermSet<T, C> {
    fn get_word_iters(&self) -> &T {
        self.terms.get_word_iters()
    }
}

impl<T: WordIters, C: NumRepr> HasWordItersMut<T> for TermSet<T, C> {
    fn get_word_iters_mut(&mut self) -> &mut T {
        self.terms.get_word_iters_mut()
    }
}

impl<T: WordIters, C: NumRepr> HasCoeffs<C> for TermSet<T, C> {
    fn get_coeffs(&self) -> &<C as NumRepr>::Vector {
        self.terms.get_coeffs()
    }
}

impl<T: WordIters, C: NumRepr> HasCoeffsMut<C> for TermSet<T, C> {
    fn get_coeffs_mut(&mut self) -> &mut <C as NumRepr>::Vector {
        self.terms.get_coeffs_mut()
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::Borrow<'a> for TermSet<T, C> {
    type RefType = View<'a, T, C>;

    fn borrow(&'a self) -> Self::RefType {
        View {
            word_iters: &self.terms.word_iters,
            coeffs: &self.terms.coeffs,
            map: &self.map,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::BorrowMut<'a> for TermSet<T, C> {
    type MutRefType = ViewMut<'a, T, C>;

    fn borrow_mut(&'a mut self) -> Self::MutRefType {
        ViewMut {
            word_iters: &mut self.terms.word_iters,
            coeffs: &mut self.terms.coeffs,
            map: &mut self.map,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> Elements for View<'a, T, C> {
    fn len(&self) -> usize {
        self.word_iters.len()
    }
}

impl<'a, T: WordIters, C: NumRepr> HasWordIters<T> for View<'a, T, C> {
    fn get_word_iters(&self) -> &T {
        self.word_iters
    }
}

impl<'a, T: WordIters, C: NumRepr> HasCoeffs<C> for View<'a, T, C> {
    fn get_coeffs(&self) -> &<C as NumRepr>::Vector {
        self.coeffs
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::ToOwned for View<'a, T, C> {
    type OwnedType = TermSet<T, C>;

    fn to_owned(&self) -> Self::OwnedType {
        TermSet::from(self.as_terms().to_owned())
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::EmptyOwned for View<'a, T, C> {
    fn empty_owned(&self) -> Self::OwnedType {
        TermSet::from(self.as_terms().empty_owned())
    }
}

impl<'a, T: WordIters, C: NumRepr> Elements for ViewMut<'a, T, C> {
    fn len(&self) -> usize {
        self.word_iters.len()
    }
}

impl<'a, T: WordIters, C: NumRepr> HasWordIters<T> for ViewMut<'a, T, C> {
    fn get_word_iters(&self) -> &T {
        self.word_iters
    }
}

impl<'a, T: WordIters, C: NumRepr> HasWordItersMut<T> for ViewMut<'a, T, C> {
    fn get_word_iters_mut(&mut self) -> &mut T {
        self.word_iters
    }
}

impl<'a, T: WordIters, C: NumRepr> HasCoeffs<C> for ViewMut<'a, T, C> {
    fn get_coeffs(&self) -> &<C as NumRepr>::Vector {
        self.coeffs
    }
}

impl<'a, T: WordIters, C: NumRepr> HasCoeffsMut<C> for ViewMut<'a, T, C> {
    fn get_coeffs_mut(&mut self) -> &mut <C as NumRepr>::Vector {
        self.coeffs
    }
}

impl<T: WordIters, C: NumRepr> From<terms::Terms<T, C>> for TermSet<T, C> {
    fn from(value: terms::Terms<T, C>) -> Self {
        // load in the elements with an empty Map.
        let mut out = Self {
            terms: value.empty_clone(),
            map: Map::default(),
        };
        for term_ref in value.iter() {
            out.insert_term_ref_or_update(term_ref);
        }
        out
    }
}

impl<T: WordIters, C: NumRepr> From<TermSet<T, C>> for terms::Terms<T, C> {
    fn from(val: TermSet<T, C>) -> Self {
        val.terms
    }
}

impl<T: WordIters, C: NumRepr> From<T> for TermSet<T, C> {
    fn from(value: T) -> Self {
        let mut coeffs = C::Vector::default();
        coeffs.resize(value.len());
        Self::from(terms::Terms {
            word_iters: value,
            coeffs,
        })
    }
}

impl<'a, T: WordIters + Serialize, C: NumRepr> Serialize for View<'a, T, C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Terms", 2)?;
        state.serialize_field("word_iters", self.word_iters)?;
        state.serialize_field("coeffs", self.coeffs)?;
        state.end()
    }
}

impl<T: WordIters + Serialize, C: NumRepr> Serialize for TermSet<T, C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.borrow().serialize(serializer)
    }
}

impl<'de, T: WordIters + for<'a> Deserialize<'a>, C: NumRepr> Deserialize<'de> for TermSet<T, C> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::from(terms::Terms::<T, C>::deserialize(deserializer)?))
    }
}

impl<'a, 'b, T: WordIters + 'a, C: NumRepr + 'a> RefElements<'a> for View<'b, T, C> {
    type Output = TermRef<'a, T, C>;

    fn get_elem_ref(&'a self, index: usize) -> Self::Output {
        TermRef {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            index,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> PartialEq for View<'a, T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}

impl<T: WordIters, C: NumRepr> EmptyClone for TermSet<T, C> {
    fn empty_clone(&self) -> Self {
        Self {
            terms: self.terms.empty_clone(),
            map: Map::default(),
        }
    }
}

impl<T: WordIters, C: NumRepr> EmptyFrom<T> for TermSet<T, C> {
    fn empty_from(value: &T) -> Self {
        Self {
            terms: terms::Terms::empty_from(value),
            map: Map::default(),
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> InsertU64It<C> for ViewMut<'a, T, C> {
    fn insert_u64it(&mut self, it: impl Iterator<Item = u64> + Clone, c: C) -> Option<usize> {
        Some(self.insert_u64it_or_update(it, c).0)
    }
}

impl<T: WordIters, C: NumRepr> InsertU64It<C> for TermSet<T, C> {
    fn insert_u64it(&mut self, it: impl Iterator<Item = u64> + Clone, c: C) -> Option<usize> {
        self.borrow_mut().insert_u64it(it, c)
    }
}

impl<'a, T: WordIters, InpC: NumRepr, OutC: NumRepr> TransformCoeffs<T, InpC, OutC>
    for View<'a, T, InpC>
{
    /// Resulting mapped term-set type after transforming the coefficient representation to `OutC`.
    type Output = TermSet<T, OutC>;
}

impl<'a, T: WordIters, C: NumRepr> Display for View<'a, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_terms().fmt(f)
    }
}

impl<T: WordIters, C: NumRepr> Display for TermSet<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.borrow().fmt(f)
    }
}

pub mod test_defs {
    use crate::container::word_iters::set::AsViewMut as SetViewMut;
    use serde::{Deserialize, Serialize};

    use crate::container::table::Table;
    use crate::container::traits::{Compatible, Elements};
    use crate::container::word_iters::{set, Elem, ElemMutRef, WordIters};
    use crate::utils::arith::divceil;

    use super::*;

    /// Test helper `WordIters` implementation that stores fixed-width strings in a `Table`.
    #[derive(Default, Serialize, Deserialize)]
    pub struct StringCmpnts {
        pub n_char_max: usize,
        pub table: Table,
    }

    fn str_to_vec_u64(s: &str, n_char_max: usize) -> Vec<u64> {
        let mut padded = s[0..s.len().min(n_char_max)].to_string();
        let padding = n_char_max.saturating_sub(padded.len());
        padded.extend(std::iter::repeat_n('\0', padding));
        padded
            .as_bytes()
            .chunks(size_of::<u64>())
            .map(|chunk| {
                let mut padded = [0u8; 8];
                padded[..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(padded)
            })
            .collect()
    }

    fn vec_u64_to_string(v: Vec<u64>) -> String {
        let mut bytes = Vec::with_capacity(v.len() * size_of::<u64>());
        for word in v {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let str_bytes = bytes
            .iter()
            .copied()
            .take_while(|&b| b != 0)
            .collect::<Vec<_>>();
        String::from_utf8(str_bytes).unwrap()
    }

    impl Elements for StringCmpnts {
        fn len(&self) -> usize {
            self.table.len()
        }
    }

    impl Clone for StringCmpnts {
        fn clone(&self) -> Self {
            Self {
                n_char_max: self.n_char_max,
                table: self.table.clone(),
            }
        }
    }

    impl EmptyClone for StringCmpnts {
        fn empty_clone(&self) -> Self {
            Self {
                n_char_max: self.n_char_max,
                table: self.table.empty_clone(),
            }
        }
    }

    impl WordIters for StringCmpnts {
        fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
            self.table[i].iter().copied()
        }

        fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
            self.table[i].iter_mut()
        }

        fn u64it_size(&self) -> usize {
            self.table.get_row_size()
        }

        fn pop_and_swap(&mut self, i_row: usize) {
            self.table.pop_and_swap(i_row);
        }

        fn fmt_elem(&self, i: usize) -> String {
            vec_u64_to_string(self.elem_u64it(i).collect::<Vec<u64>>())
        }

        fn resize(&mut self, n: usize) {
            self.table.resize(n);
        }
    }

    impl Compatible for StringCmpnts {
        fn compatible_with(&self, other: &Self) -> bool {
            self.n_char_max == other.n_char_max
        }
    }

    impl StringCmpnts {
        /// Create a string-backed `WordIters` helper with capacity for strings up to `n_char_max` bytes.
        pub fn new(n_char_max: usize) -> Self {
            Self {
                n_char_max,
                table: Table::new(divceil(n_char_max, size_of::<u64>())),
            }
        }
    }

    impl<'a> ElemMutRef<'a, StringCmpnts> {
        /// Overwrite the referenced string element with the contents of `s`, truncated or padded as needed.
        pub fn assign_from_str(&mut self, s: &str) {
            let n_char_max = self.as_ref().word_iters.n_char_max;
            let dst = self.get_u64it_mut();
            let src = str_to_vec_u64(s, n_char_max).into_iter();
            dst.zip(src).for_each(|(dst, src)| *dst = src);
        }
    }

    impl<'a> set::ViewMut<'a, StringCmpnts> {
        /// Insert the string `s`, or return the existing index if an equal string is already present.
        pub fn insert_or_get_from_str(&mut self, s: &str) -> (usize, bool) {
            let mut tmp = Elem::<StringCmpnts>::from(&*self.word_iters);
            tmp.borrow_mut().assign_from_str(s);
            self.insert_or_get_index(tmp.borrow())
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::test_defs::StringCmpnts;
    use crate::container::{
        traits::EmptyFrom,
        word_iters::{
            set::{self, AsView},
            WordIters,
        },
    };

    #[test]
    fn test_keys() {
        const N_CHAR_MAX: usize = 20;
        let mut cmpnts = set::Set::empty_from(&StringCmpnts::new(N_CHAR_MAX));
        assert_eq!(cmpnts.borrow().get_word_iters().u64it_size(), 3);
        assert_eq!(
            cmpnts.borrow_mut().insert_or_get_from_str("Hello, "),
            (0, true)
        );
        assert_eq!(
            cmpnts.borrow_mut().insert_or_get_from_str("World!"),
            (1, true)
        );
        assert_eq!(
            cmpnts.borrow_mut().insert_or_get_from_str("World!"),
            (1, false)
        );
        assert_eq!(
            cmpnts.borrow_mut().insert_or_get_from_str("Hello, "),
            (0, false)
        );
        assert_eq!(cmpnts.borrow().get_word_iters().fmt_elem(0), "Hello, ");
        assert_eq!(cmpnts.borrow().get_word_iters().fmt_elem(1), "World!");
        assert_eq!(
            cmpnts
                .borrow_mut()
                .insert_or_get_from_str("This is a long string that will be truncated!"),
            (2, true)
        );
        assert_eq!(
            cmpnts.borrow().get_word_iters().fmt_elem(2),
            "This is a long strin"
        );
        assert_eq!(
            cmpnts
                .borrow_mut()
                .insert_or_get_from_str("This is a long string that will be truncated!"),
            (2, false)
        );
        assert_eq!(
            cmpnts
                .borrow_mut()
                .insert_or_get_from_str("This is a long strin"),
            (2, false)
        );
    }
}
