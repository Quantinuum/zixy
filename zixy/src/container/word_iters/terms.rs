//! Extend the `word_iters` types by associating a numeric coefficient with each iterable element.
//! The iterable elements are represented by three kinds of type:
//! - "u64it": the iterator over u64 words themselves,
//! - "elem_ref": a reference to an element within an iterable elements object
//! - "term_ref": a reference to an element within an iterable elements with coeffs object
//!
//! So "term" means an iterable element - coeff pair.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::container::coeffs::complex_sign::ComplexSign;
use crate::container::coeffs::traits::{
    AnyNumRepr, ComplexParts, ComplexSigned, FieldElem, HasCoeffs, HasCoeffsMut, IMulResult,
    IsComplex, NumRepr, NumReprVec,
};
use crate::container::traits::proj::{AsRef, Borrow, BorrowMut};
use crate::container::traits::{
    proj, Elements, EmptyClone, EmptyFrom, HasIndex, MutRefElements, RefElements,
};
use crate::container::word_iters::traits::{InsertU64It, TransformCoeffs};
use crate::container::word_iters::{self, ElemRef};
use crate::container::word_iters::{HasWordIters, HasWordItersMut, WordIters};

/// Combine a `WordIters` type with a vector of associated coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terms<T: WordIters, C: NumRepr> {
    pub word_iters: T,
    pub coeffs: C::Vector,
}

impl<T: WordIters, C: NumRepr> proj::ToOwned for Terms<T, C> {
    type OwnedType = Self;

    fn to_owned(&self) -> Self::OwnedType {
        self.clone()
    }
}

/// Trait for structs that immutably view a [`Terms`].
pub trait AsView<T: WordIters, C: NumRepr> {
    /// Return an immutable view over the word iterators and their coefficient vector.
    fn view<'a>(&'a self) -> View<'a, T, C>;

    /// Return whether the terms with indices `i` and `j` are equal.
    fn elem_equal(&self, i: usize, j: usize) -> bool {
        let self_ref = self.view();
        if i == j {
            true
        } else {
            let (ref_i, ref_j) = self_ref.get_pair_refs(i, j);
            ref_i == ref_j && self_ref.coeffs.get_unchecked(i) == self_ref.coeffs.get_unchecked(j)
        }
    }

    /// Multiply out-of-place by a scalar of the coefficient type.
    fn mul_scalar(&self, c: C) -> Terms<T, C> {
        let self_ref = self.view();
        let mut out = proj::ToOwned::to_owned(&self_ref);
        out.imul_scalar(c);
        out
    }

    /// Format the term at index `i`, including its coefficient when non-empty.
    fn fmt_elem(&self, i: usize) -> String {
        let self_ref = self.view();
        let c_str = self_ref.coeffs.get_unchecked(i).to_string();
        if c_str.is_empty() {
            self_ref.word_iters.fmt_elem(i)
        } else {
            format!("({}, {})", c_str, self_ref.word_iters.fmt_elem(i))
        }
    }

    /// Get a new instance in which the terms with indices given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn select(&self, inds: impl Iterator<Item = usize>) -> Terms<T, C> {
        let self_ref = self.view();
        let inds = inds.collect::<Vec<_>>();
        Terms {
            word_iters: self_ref.word_iters.select(inds.iter().copied()),
            coeffs: self_ref.coeffs.select(inds.iter().copied()),
        }
    }

    /// Get a new instance in which the terms with indices not given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn deselect(&self, inds: impl Iterator<Item = usize>) -> Terms<T, C> {
        let self_ref = self.view();
        let inds = inds.collect::<Vec<_>>();
        Terms {
            word_iters: self_ref.word_iters.deselect(inds.iter().copied()),
            coeffs: self_ref.coeffs.deselect(inds.iter().copied()),
        }
    }

    /// Get two new instances: the first with the terms selected in `inds` and the second with the remainder.
    fn bipartition(&self, inds: impl Iterator<Item = usize>) -> (Terms<T, C>, Terms<T, C>) {
        let self_ref = self.view();
        let inds = inds.collect::<Vec<_>>();
        let word_iters = self_ref.word_iters.bipartition(inds.iter().copied());
        let coeffs = self_ref.coeffs.bipartition(inds.iter().copied());
        (
            Terms {
                word_iters: word_iters.0,
                coeffs: coeffs.0,
            },
            Terms {
                word_iters: word_iters.1,
                coeffs: coeffs.1,
            },
        )
    }

    /// Get a new instance in which the duplicate elements have been removed.
    fn without_duplicates(&self) -> Terms<T, C> {
        self.deselect(self.view().word_iters.find_duplicates())
    }

    /// Return the term with the largest coefficient magnitude, if any term is present.
    fn dominant_term<'a>(&'a self) -> Option<(ElemRef<'a, T>, C)>
    where
        C: FieldElem + 'a,
    {
        self.view()
            .coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.magnitude().partial_cmp(&b.1.magnitude()).unwrap())
            .map(|(i, c)| (self.view().word_iters.get_elem_ref(i), *c))
    }

    /// Return whether all terms are insignificant with respect to the given atol.
    fn all_insignificant(&self, atol: f64) -> bool
    where
        C: FieldElem,
    {
        use crate::container::coeffs::traits::FieldElemVec;
        self.view().get_coeffs().all_insignificant(atol)
    }

    /// Return whether every coefficient has an insignificant imaginary part with tolerance `atol`.
    fn all_imag_insignificant(&self, atol: f64) -> bool
    where
        C: IsComplex,
    {
        self.view()
            .get_coeffs()
            .iter()
            .all(|c| c.get().im.is_close(f64::ZERO, 0.0, atol))
    }

    /// Return whether every coefficient has an insignificant real part with tolerance `atol`.
    fn all_real_insignificant(&self, atol: f64) -> bool
    where
        C: IsComplex,
    {
        self.view()
            .get_coeffs()
            .iter()
            .all(|c| c.get().re.is_close(f64::ZERO, 0.0, atol))
    }

    /// Return a new term list whose coefficients are the real parts of this list's coefficients.
    fn real_part(&self) -> Terms<T, f64>
    where
        C::Vector: ComplexParts,
    {
        let self_ref = self.view();
        Terms {
            word_iters: self_ref.word_iters.clone(),
            coeffs: self_ref.coeffs.real_part(),
        }
    }

    /// Return a new term list whose coefficients are the imaginary parts of this list's coefficients.
    fn imag_part(&self) -> Terms<T, f64>
    where
        C::Vector: ComplexParts,
    {
        let self_ref = self.view();
        Terms {
            word_iters: self_ref.word_iters.clone(),
            coeffs: self_ref.coeffs.imag_part(),
        }
    }
}

impl<T: WordIters, C: NumRepr> AsView<T, C> for Terms<T, C> {
    fn view<'a>(&'a self) -> View<'a, T, C> {
        self.borrow()
    }
}

impl<'a, T: WordIters, C: NumRepr> AsView<T, C> for View<'a, T, C> {
    fn view(&self) -> Self {
        View {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> AsView<T, C> for ViewMut<'a, T, C> {
    fn view<'b>(&'b self) -> View<'b, T, C> {
        View {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
        }
    }
}

/// Trait for structs that mutably view a [`Terms`].
pub trait AsViewMut<T: WordIters, C: NumRepr>: AsView<T, C> {
    /// Return a mutable view over the word iterators and their coefficient vector.
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T, C>;

    /// Ensure that the coefficients vector is brought to the same size as the u64 iterable elements.
    fn sync_sizes(&mut self) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.coeffs.resize(self_mut_ref.view().len());
    }

    /// Swap the indexed elements.
    fn swap(&mut self, i: usize, j: usize) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.swap(i, j);
        self_mut_ref.coeffs.swap_unchecked(i, j);
    }

    /// Copy the i_src indexed element to the i_dst indexed element
    fn copy(&mut self, i_dst: usize, i_src: usize) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.copy(i_dst, i_src);
        self_mut_ref.coeffs.copy_unchecked(i_dst, i_src);
    }

    /// `Set` the size of the cmpnts and coeffs.
    fn resize(&mut self, n: usize) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.resize(n);
        self_mut_ref.coeffs.resize(n);
    }

    /// Insert a default (zero) valued element at the end.
    fn push_clear(&mut self) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.push_clear();
        self_mut_ref.coeffs.push_default();
    }

    /// Push a new element with the given iterator value and coefficient.
    fn push_u64it(&mut self, u64it: impl Iterator<Item = u64>, c: C) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.push_u64it(u64it);
        self_mut_ref.coeffs.push(c);
    }

    /// Push a new element with the given iterator value and coefficient.
    fn push_elem_coeff(&mut self, elem_ref: word_iters::ElemRef<T>, c: C) {
        let mut self_mut_ref = self.view_mut();
        self_mut_ref.push_u64it(elem_ref.get_u64it(), c);
    }

    /// Push a new element with the given iterator value.
    fn push_term_ref(&mut self, term_ref: TermRef<T, C>) {
        let mut self_mut_ref = self.view_mut();
        self_mut_ref.push_elem_coeff(term_ref.get_word_iter_ref(), term_ref.get_coeff());
    }

    /// Push all elements of the other list onto the end of self.
    fn append(&mut self, other: &Self) {
        let other_ref = other.view();
        other_ref
            .word_iters
            .iter()
            .enumerate()
            .for_each(|(i, elem)| self.push_elem_coeff(elem, other_ref.coeffs.get_unchecked(i)))
    }

    /// Duplicate all elements of self.
    fn self_append(&mut self) {
        let n = self.view_mut().len();
        for i in 0..n {
            self.push_clear();
            self.copy(n + i, i);
        }
    }

    /// Clear the contents of the indexed element, set all u64 words to zero.
    fn clear_elem(&mut self, i: usize) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.clear_elem(i);
        self_mut_ref.coeffs.set_unchecked(i, C::default());
    }

    /// Clear all contents, i.e. set the len of the collection to zero.
    fn clear(&mut self) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.clear();
        self_mut_ref.coeffs.clear();
    }

    /// Replace the indexed element with the last one in the collection, then drop the last element.
    fn pop_and_swap(&mut self, i: usize) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.word_iters.pop_and_swap(i);
        self_mut_ref.coeffs.pop_and_swap(i);
    }

    /// Multiply in-place by a scalar of the coefficient type.
    fn imul_scalar(&mut self, c: C) {
        let self_mut_ref = self.view_mut();
        self_mut_ref.coeffs.imul(c)
    }
}

impl<T: WordIters, C: NumRepr> AsViewMut<T, C> for Terms<T, C> {
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T, C> {
        ViewMut {
            word_iters: &mut self.word_iters,
            coeffs: &mut self.coeffs,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> AsViewMut<T, C> for ViewMut<'a, T, C> {
    fn view_mut<'b, 'c: 'b>(&'c mut self) -> ViewMut<'b, T, C> {
        ViewMut {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::Borrow<'a> for Terms<T, C> {
    type RefType = View<'a, T, C>;

    fn borrow(&'a self) -> Self::RefType {
        View {
            word_iters: &self.word_iters,
            coeffs: &self.coeffs,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::BorrowMut<'a> for Terms<T, C> {
    type MutRefType = ViewMut<'a, T, C>;

    fn borrow_mut(&'a mut self) -> Self::MutRefType {
        ViewMut {
            word_iters: &mut self.word_iters,
            coeffs: &mut self.coeffs,
        }
    }
}

/// Borrowed immutable view of the fields of a [`Terms`] collection.
pub struct View<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a T,
    pub coeffs: &'a C::Vector,
}

impl<'a, T: WordIters, C: NumRepr> proj::ToOwned for View<'a, T, C> {
    type OwnedType = Terms<T, C>;

    fn to_owned(&self) -> Self::OwnedType {
        Self::OwnedType {
            word_iters: self.word_iters.clone(),
            coeffs: self.coeffs.clone(),
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::EmptyOwned for View<'a, T, C> {
    fn empty_owned(&self) -> Self::OwnedType {
        Self::OwnedType {
            word_iters: self.word_iters.empty_clone(),
            coeffs: C::Vector::default(),
        }
    }
}

impl<T: WordIters, C: NumRepr> Display for Terms<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.borrow().fmt(f)
    }
}
impl<'a, T: WordIters, C: NumRepr> Display for View<'a, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.len()).map(|i| self.fmt_elem(i)).join(", ")
        )
    }
}

impl<'a, 'b, T: WordIters + 'a + 'b, C: NumRepr + 'a + 'b> RefElements<'b> for View<'a, T, C> {
    type Output = TermRef<'b, T, C>;

    fn get_elem_ref(&'b self, index: usize) -> Self::Output {
        TermRef {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            index,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> PartialEq for View<'a, T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.word_iters.eq(other.word_iters) && self.coeffs == other.coeffs
    }
}

impl<T: WordIters, C: NumRepr> PartialEq for Terms<T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.borrow() == other.borrow()
    }
}

impl<'a, T: WordIters, C: NumRepr> Elements for View<'a, T, C> {
    fn len(&self) -> usize {
        self.word_iters.len()
    }
}

impl<T: WordIters, C: NumRepr> Elements for Terms<T, C> {
    fn len(&self) -> usize {
        self.borrow().len()
    }
}

impl<T: WordIters, C: NumRepr> EmptyClone for Terms<T, C> {
    fn empty_clone(&self) -> Self {
        Self {
            word_iters: self.word_iters.empty_clone(),
            coeffs: C::Vector::default(),
        }
    }
}

impl<T: WordIters, C: NumRepr> EmptyFrom<T> for Terms<T, C> {
    fn empty_from(value: &T) -> Self {
        Self {
            word_iters: value.empty_clone(),
            coeffs: C::Vector::default(),
        }
    }
}

impl<T: WordIters, C: NumRepr> From<(T, C::Vector)> for Terms<T, C> {
    fn from(value: (T, C::Vector)) -> Self {
        Self {
            word_iters: value.0,
            coeffs: value.1,
        }
    }
}

impl<T: WordIters, C: NumRepr> From<(&T, &C::Vector)> for Terms<T, C> {
    fn from(value: (&T, &C::Vector)) -> Self {
        Self {
            word_iters: value.0.clone(),
            coeffs: value.1.clone(),
        }
    }
}

/// Borrowed mutable view of the fields of a [`Terms`] collection.
pub struct ViewMut<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a mut T,
    pub coeffs: &'a mut C::Vector,
}

impl<'a, T: WordIters, C: NumRepr> proj::ToOwned for ViewMut<'a, T, C> {
    type OwnedType = Terms<T, C>;

    fn to_owned(&self) -> Self::OwnedType {
        Terms {
            word_iters: self.word_iters.clone(),
            coeffs: self.coeffs.clone(),
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> Elements for ViewMut<'a, T, C> {
    fn len(&self) -> usize {
        self.word_iters.len()
    }
}

impl<'a, 'b, T: WordIters + 'a + 'b, C: NumRepr + 'a + 'b> RefElements<'b> for ViewMut<'a, T, C> {
    type Output = TermRef<'b, T, C>;

    fn get_elem_ref(&'b self, index: usize) -> Self::Output {
        TermRef {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            index,
        }
    }
}

impl<'a, 'b, T: WordIters + 'a + 'b, C: NumRepr + 'a + 'b> MutRefElements<'b>
    for ViewMut<'a, T, C>
{
    /// Resulting `Terms` type after converting the coefficient representation to `C`.
    type Output = TermMutRef<'b, T, C>;

    /// Return a mutable reference-like view of the term at `index`.
    fn get_elem_mut_ref(&'b mut self, index: usize) -> TermMutRef<'b, T, C> {
        TermMutRef {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            index,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> InsertU64It<C> for ViewMut<'a, T, C> {
    fn insert_u64it(&mut self, iter: impl Iterator<Item = u64>, c: C) -> Option<usize> {
        self.push_u64it(iter, c);
        Some(self.len().saturating_sub(1))
    }
}

impl<T: WordIters, C: NumRepr> InsertU64It<C> for Terms<T, C> {
    fn insert_u64it(&mut self, iter: impl Iterator<Item = u64> + Clone, c: C) -> Option<usize> {
        self.borrow_mut().insert_u64it(iter, c)
    }
}

impl<'a, T: WordIters, C: NumRepr> HasWordIters<T> for View<'a, T, C> {
    fn get_word_iters(&self) -> &T {
        self.word_iters
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

impl<T: WordIters, C: NumRepr> HasWordIters<T> for Terms<T, C> {
    fn get_word_iters(&self) -> &T {
        &self.word_iters
    }
}

impl<T: WordIters, C: NumRepr> HasWordItersMut<T> for Terms<T, C> {
    fn get_word_iters_mut(&mut self) -> &mut T {
        &mut self.word_iters
    }
}

impl<'a, T: WordIters, C: NumRepr> HasCoeffs<C> for View<'a, T, C> {
    fn get_coeffs(&self) -> &<C as NumRepr>::Vector {
        self.coeffs
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

impl<T: WordIters, C: NumRepr> HasCoeffs<C> for Terms<T, C> {
    fn get_coeffs(&self) -> &<C as NumRepr>::Vector {
        &self.coeffs
    }
}

impl<T: WordIters, C: NumRepr> HasCoeffsMut<C> for Terms<T, C> {
    fn get_coeffs_mut(&mut self) -> &mut <C as NumRepr>::Vector {
        &mut self.coeffs
    }
}

impl<T: WordIters, InpC: NumRepr, OutC: NumRepr> TransformCoeffs<T, InpC, OutC> for Terms<T, InpC> {
    type Output = Terms<T, OutC>;
}

/// Reference type for one element of a [`WordIters`] container with coefficients.
pub struct TermRef<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a T,
    pub coeffs: &'a C::Vector,
    pub index: usize,
}

/// Owned single-term container holding one iterable element together with its coefficient as a single vector entry.
#[derive(Clone)]
pub struct Term<T: WordIters, C: NumRepr> {
    pub word_iters: T,
    pub coeffs: C::Vector,
}

impl<T: WordIters, C: NumRepr> proj::ToOwned for Term<T, C> {
    type OwnedType = Self;

    fn to_owned(&self) -> Term<T, C> {
        self.clone()
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::ToOwned for TermRef<'a, T, C> {
    type OwnedType = Term<T, C>;

    fn to_owned(&self) -> Self::OwnedType {
        Self::OwnedType {
            word_iters: self.word_iters.clone(),
            coeffs: self.coeffs.clone(),
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::EmptyOwned for TermRef<'a, T, C> {
    fn empty_owned(&self) -> Self::OwnedType {
        Self::OwnedType {
            word_iters: self.word_iters.empty_clone(),
            coeffs: C::Vector::default(),
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::Borrow<'a> for Term<T, C> {
    type RefType = TermRef<'a, T, C>;

    fn borrow(&'a self) -> Self::RefType {
        Self::RefType {
            word_iters: &self.word_iters,
            coeffs: &self.coeffs,
            index: 0,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> proj::BorrowMut<'a> for Term<T, C> {
    type MutRefType = TermMutRef<'a, T, C>;

    fn borrow_mut(&'a mut self) -> Self::MutRefType {
        Self::MutRefType {
            word_iters: &mut self.word_iters,
            coeffs: &mut self.coeffs,
            index: 0,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> TermRef<'a, T, C> {
    /// Return the coefficient of the referenced term.
    pub fn get_coeff(&self) -> C {
        self.coeffs.get_unchecked(self.index)
    }

    /// Return an immutable reference-like view of the referenced word-iterator element.
    pub fn get_word_iter_ref(&self) -> word_iters::ElemRef<'_, T> {
        self.word_iters.get_elem_ref(self.index)
    }

    /// Return the referenced element view together with its coefficient.
    pub fn unpack(&self) -> (word_iters::ElemRef<'_, T>, C) {
        (self.get_word_iter_ref(), self.get_coeff())
    }

    /// Return the `u64` words of the referenced iterable element as an iterator.
    pub fn get_u64it(&'a self) -> impl Iterator<Item = u64> + Clone + 'a {
        self.word_iters.elem_u64it(self.index)
    }
}

impl<'a, T: WordIters, C: NumRepr> HasIndex for TermRef<'a, T, C> {
    fn get_index(&self) -> usize {
        self.index
    }
}

impl<'a, T: WordIters, C: NumRepr> PartialEq for TermRef<'a, T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.word_iters.compatible_with(other.word_iters)
            && self
                .get_word_iter_ref()
                .get_u64it()
                .eq(other.get_word_iter_ref().get_u64it())
            && self.get_coeff() == other.get_coeff()
    }
}

impl<'a, T: WordIters, C: NumRepr> Display for TermRef<'a, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_word_iter_ref())
    }
}

/// Mutable handle to one term inside a [`Terms`] collection, giving access to the indexed iterable element and coefficient.
pub struct TermMutRef<'a, T: WordIters, C: NumRepr> {
    pub word_iters: &'a mut T,
    pub coeffs: &'a mut C::Vector,
    pub index: usize,
}

impl<'a, T: WordIters, C: NumRepr> TermMutRef<'a, T, C> {
    /// `Set` the contents of the referenced term to equal that of the other term.
    pub fn assign(&mut self, other: TermRef<T, C>) {
        self.get_word_iter_mut_ref()
            .assign(other.get_word_iter_ref());
        self.set_coeff(other.get_coeff());
    }

    /// Clear the contents of the referenced term, including setting the coefficient to its default value.
    pub fn clear(&mut self) {
        self.get_word_iter_mut_ref().clear();
        self.set_coeff(C::default());
    }

    /// Reassign the coefficient of this term with the given value.
    pub fn set_coeff(&mut self, value: C) {
        self.coeffs.set_unchecked(self.index, value);
    }

    /// Multiply the coefficient of this term in-place by a given value of the same type.
    pub fn imul_coeff(&mut self, value: C) {
        let c = self.coeffs.get_unchecked(self.index) * value;
        self.coeffs.set_unchecked(self.index, c);
    }

    /// Try to multiply the coefficient of this term in-place by a given value of any type.
    pub fn try_imul_coeff_any(&mut self, factor: AnyNumRepr) -> IMulResult {
        let mut c = self.as_ref().get_coeff();
        let res = c.try_imul_any(factor);
        if let IMulResult::Success = res {
            self.set_coeff(c);
        }
        res
    }

    /// Try to multiply the coefficient of this term in-place by a given value of a generic `NumRepr` type.
    pub fn try_imul_coeff<U: NumRepr>(&mut self, factor: U) -> IMulResult {
        self.try_imul_coeff_any(factor.into())
    }

    /// No "try" here: complex sign is exactly representable as `Self`.
    pub fn imul_complex_sign(&mut self, factor: ComplexSign)
    where
        C: ComplexSigned,
    {
        let mut c = self.as_ref().get_coeff();
        c.imul_complex_sign(factor);
        self.set_coeff(c);
    }

    /// Return a mutable reference-like view of the referenced word-iterator element.
    pub fn get_word_iter_mut_ref(&mut self) -> word_iters::ElemMutRef<'_, T> {
        self.word_iters.get_elem_mut_ref(self.index)
    }

    /// Return an immutable reference-like view of the referenced word-iterator element.
    pub fn get_word_iter_ref(&mut self) -> word_iters::ElemRef<'_, T> {
        self.word_iters.get_elem_ref(self.index)
    }
}

impl<'a, T: WordIters, C: NumRepr> proj::AsRef<'a> for TermMutRef<'a, T, C> {
    type RefType = TermRef<'a, T, C>;

    fn as_ref(&'a self) -> Self::RefType {
        TermRef {
            word_iters: self.word_iters,
            coeffs: self.coeffs,
            index: self.index,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> RefElements<'a> for Terms<T, C> {
    type Output = TermRef<'a, T, C>;

    fn get_elem_ref(&'a self, index: usize) -> Self::Output {
        TermRef {
            word_iters: &self.word_iters,
            coeffs: &self.coeffs,
            index,
        }
    }
}

impl<'a, T: WordIters + 'a, C: NumRepr + 'a> MutRefElements<'a> for Terms<T, C> {
    type Output = TermMutRef<'a, T, C>;

    fn get_elem_mut_ref(&'a mut self, index: usize) -> <Self as MutRefElements<'a>>::Output {
        TermMutRef {
            word_iters: &mut self.word_iters,
            coeffs: &mut self.coeffs,
            index,
        }
    }
}

impl<'a, T: WordIters, C: NumRepr> HasIndex for TermMutRef<'a, T, C> {
    fn get_index(&self) -> usize {
        self.index
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
}
