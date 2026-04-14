//! Defines functionality for structs that have elements definable as iterators over u64 words
//! "u64it" is the shorthand used to refer to iterators like impl `Iterator`<`Item` = u64>.

use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::container::traits::{
    proj, Compatible, Elements, EmptyClone, EmptyFrom, HasIndex, MutRefElements, RefElements,
};
use crate::container::word_iters::set::AsViewMut as SetViewMut;

/// Compute the hash of an iterator over n u64 words
fn hash_u64it(it: impl Iterator<Item = u64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    it.for_each(|x| x.hash(&mut hasher));
    hasher.finish()
}

/// Functionality for structs that expose elements of raw data as iterators over u64
pub trait WordIters: Elements + EmptyClone + Compatible {
    /// Get an iterator over the u64 words corresponding to an indexed element
    fn elem_u64it(&self, index: usize) -> impl Iterator<Item = u64> + Clone;

    /// Get an iterator over mutable refs to the u64 words corresponding to an indexed element
    fn elem_u64it_mut(&mut self, index: usize) -> impl Iterator<Item = &mut u64>;

    /// Swap the indexed elements.
    fn swap(&mut self, i: usize, j: usize) {
        if i == j || i >= self.len() || j >= self.len() {
            return;
        }
        let self_ptr = std::ptr::from_mut(self);
        // need unsafe to have two living mut refs simultaneously.
        // above checks guarantee no aliasing since i != j
        unsafe {
            (*self_ptr)
                .elem_u64it_mut(i)
                .zip((*self_ptr).elem_u64it_mut(j))
                .for_each(|(i, j)| std::mem::swap(i, j));
        }
    }

    /// Copy the i_src indexed element to the i_dst indexed element
    fn copy(&mut self, i_dst: usize, i_src: usize) {
        if i_dst == i_src || i_dst > self.len() || i_src > self.len() {
            return;
        }
        let self_ptr = std::ptr::from_mut(self);
        // need unsafe to have a living ref and mut ref simultaneously.
        // above checks guarantee no aliasing since i != j
        unsafe {
            (*self_ptr)
                .elem_u64it_mut(i_dst)
                .zip((*self_ptr).elem_u64it(i_src))
                .for_each(|(i, j)| *i = j);
        }
    }

    /// Compare the indexed elements.
    fn cmp(&self, i: usize, j: usize) -> Ordering {
        self.elem_u64it(i).cmp(self.elem_u64it(j))
    }

    /// Count the set bits in an entire iterable element.
    fn elem_hamming_weight(&self, i: usize) -> usize {
        self.elem_u64it(i)
            .map(|word| word.count_ones())
            .sum::<u32>() as usize
    }

    /// Return Some with the Hamming weight if all elements have the same Hamming weight, else return None
    fn hamming_weight(&self) -> Option<usize> {
        let mut out: Option<usize> = None;
        for i_cmpnt in 0..self.len() {
            let n = self.elem_hamming_weight(i_cmpnt);
            match out {
                Some(m) => {
                    if m != n {
                        return None;
                    }
                }
                None => {
                    out = Some(n);
                }
            }
        }
        out
    }

    // Modular divisor of hash function, useful in tests and debugging, but should be None
    // otherwise to use the full width of the key type for the minimum occurrence of collisions
    const HASH_MOD: Option<u64> = None;

    /// Compute the hash of the iterator corresponding to element with index `index`
    fn hash_u64it(&self, it: impl Iterator<Item = u64>) -> u64 {
        let hash = hash_u64it(it);
        match Self::HASH_MOD {
            Some(div) => hash % div,
            None => hash,
        }
    }

    /// Compute the hash of the iterator corresponding to element with index `index`.
    fn hash_at_index(&self, index: usize) -> u64 {
        self.hash_u64it(self.elem_u64it(index))
    }

    /// Number of u64 words in the iterator of each element.
    fn u64it_size(&self) -> usize;

    /// Resize the collection to hold `n` elements, zero-initializing any new element storage.
    fn resize(&mut self, n: usize);

    /// Insert a default (zero) valued element at the end.
    fn push_clear(&mut self) {
        self.resize(self.len() + 1);
    }

    /// Remove the last element from the collection if one exists.
    fn pop(&mut self) {
        self.resize(self.len().saturating_sub(1));
    }

    /// Push a new element with the given iterator value.
    fn push_u64it(&mut self, it: impl Iterator<Item = u64>) {
        let i = self.len();
        self.push_clear();
        for (dst, src) in self.elem_u64it_mut(i).zip(it) {
            *dst = src;
        }
    }

    /// Push a copy of the element referenced by `elem_ref`.
    fn push_elem_ref(&mut self, elem_ref: ElemRef<Self>) {
        self.push_u64it(elem_ref.get_u64it())
    }

    /// Clear the contents of the indexed element, set all u64 words to zero.
    fn clear_elem(&mut self, index: usize) {
        for word in self.elem_u64it_mut(index) {
            *word = 0;
        }
    }

    /// Clear all contents, i.e. set the len of the collection to zero.
    fn clear(&mut self) {
        self.resize(0);
    }

    /// Replace the indexed element with the last one in the collection, then drop the last element.
    fn pop_and_swap(&mut self, index: usize);

    /// Format the element at `index` for display.
    fn fmt_elem(&self, index: usize) -> String {
        format!("{:?}", self.elem_u64it(index).collect::<Vec<_>>())
    }

    /// Return whether `self` and `other` have matching metadata and identical element contents.
    fn eq(&self, other: &Self) -> bool {
        if !self.compatible_with(other) {
            return false;
        }
        if self.len() != other.len() || self.u64it_size() != other.u64it_size() {
            return false;
        }
        (0..self.len()).all(|i| self.get_elem_ref(i) == other.get_elem_ref(i))
    }

    /// Push all elements of `other` onto the end of `self`
    fn append(&mut self, other: &Self) {
        let n = other.len();
        for i in 0..n {
            self.push_elem_ref(other.get_elem_ref(i));
        }
    }

    /// Push all elements of `self` onto the end of `self`
    fn self_append(&mut self) {
        let n = self.len();
        for i in 0..n {
            self.push_clear();
            self.copy(n + i, i);
        }
    }

    /// Find the indices of all elements that come after another of the same value and return them as an iterator.
    fn find_duplicates(&self) -> impl Iterator<Item = usize> {
        let mut mapped = crate::container::word_iters::set::Set::<Self>::empty_from(self);
        self.iter().filter_map(move |elem_ref| {
            let (i, inserted) = mapped
                .borrow_mut()
                .insert_u64it_or_get_index(elem_ref.get_u64it());
            if !inserted {
                Some(i)
            } else {
                None
            }
        })
    }

    /// Get a new instance in which the elements with indices given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn select(&self, inds: impl Iterator<Item = usize>) -> Self {
        let mut out = self.empty_clone();
        for i in inds {
            if i < self.len() {
                out.push_elem_ref(self.get_elem_ref(i));
            }
        }
        out
    }

    /// Get a new instance in which the elements with indices not given in the `inds` iterator are stored
    /// contiguously. Out-of-bounds indices are ignored.
    fn deselect(&self, mut inds: impl Iterator<Item = usize>) -> Self {
        let mut out = self.empty_clone();
        let mut next = inds.next();
        for i in 0..self.len() {
            if let Some(j) = next {
                if i == j {
                    next = inds.next();
                    continue;
                }
            }
            out.push_elem_ref(self.get_elem_ref(i));
        }
        out
    }

    /// Get two new instances: the first with the components selected in `inds` and the second with the remainder.
    fn bipartition(&self, mut inds: impl Iterator<Item = usize>) -> (Self, Self) {
        let mut out = (self.empty_clone(), self.empty_clone());
        let mut next = inds.next();
        for i in 0..self.len() {
            if let Some(j) = next {
                if i == j {
                    next = inds.next();
                    out.0.push_elem_ref(self.get_elem_ref(i));
                    continue;
                }
            }
            out.1.push_elem_ref(self.get_elem_ref(i));
        }
        out
    }
}

impl<T: WordIters> proj::ToOwned for T {
    type OwnedType = T;

    fn to_owned(&self) -> Self::OwnedType {
        self.clone()
    }
}

impl<T: WordIters> proj::EmptyOwned for T {
    fn empty_owned(&self) -> Self::OwnedType {
        self.empty_clone()
    }
}

/// For any object that contains a `WordIters` instance.
pub trait HasWordIters<T: WordIters>: Elements {
    /// Return the embedded `WordIters` value.
    fn get_word_iters(&self) -> &T;
    /// Return the `u64` iterator for the element at `index` within the embedded word-iterator container.
    fn get_u64it<'a>(&'a self, index: usize) -> impl Iterator<Item = u64> + Clone + 'a
    where
        T: 'a,
    {
        self.get_word_iters().elem_u64it(index)
    }
}

/// For any object that contains a mutable `WordIters` instance.
pub trait HasWordItersMut<T: WordIters>: HasWordIters<T> {
    /// Return mutable access to the embedded `WordIters` value.
    fn get_word_iters_mut(&mut self) -> &mut T;
    /// Return mutable `u64` words for the element at `index` within the embedded word-iterator container.
    fn get_u64it_mut<'a>(&'a mut self, index: usize) -> impl Iterator<Item = &'a mut u64>
    where
        T: 'a,
    {
        self.get_word_iters_mut().elem_u64it_mut(index)
    }
}

impl<'a, T: WordIters + 'a> RefElements<'a> for T {
    type Output = ElemRef<'a, T>;

    fn get_elem_ref(&'a self, index: usize) -> ElemRef<'a, T> {
        ElemRef {
            word_iters: self,
            index,
        }
    }
}

impl<'a, T: WordIters + 'a> MutRefElements<'a> for T {
    type Output = ElemMutRef<'a, T>;

    fn get_elem_mut_ref(&'a mut self, index: usize) -> ElemMutRef<'a, T> {
        ElemMutRef {
            word_iters: self,
            index,
        }
    }
}

/// Reference type for one element of a [`WordIters`] container.
#[derive(Clone)]
pub struct ElemRef<'a, T: WordIters> {
    pub word_iters: &'a T,
    pub index: usize,
}

impl<'a, T: WordIters> ElemRef<'a, T> {
    /// Return the `u64` words of the referenced element as an iterator.
    pub fn get_u64it(&self) -> impl Iterator<Item = u64> + Clone + '_ {
        self.word_iters.elem_u64it(self.index)
    }

    /// Return whether every word in the referenced element is zero.
    pub fn is_clear(&self) -> bool {
        self.get_u64it().all(|x| x == 0)
    }

    /// Return the total number of set bits in the referenced element.
    pub fn hamming_weight(&self) -> usize {
        self.word_iters.elem_hamming_weight(self.index)
    }

    /// Return the hash of the referenced word iterator.
    pub fn hash(&self) -> u64 {
        self.word_iters.hash_at_index(self.index)
    }
}

impl<'a, T: WordIters> std::cmp::PartialOrd for ElemRef<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: WordIters> Eq for ElemRef<'a, T> {}

impl<'a, T: WordIters> std::cmp::Ord for ElemRef<'a, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.get_u64it().cmp(other.get_u64it())
    }
}

impl<'a, T: WordIters> HasIndex for ElemRef<'a, T> {
    fn get_index(&self) -> usize {
        self.index
    }
}

impl<'a, T: WordIters> Display for ElemRef<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word_iters.fmt_elem(self.index))
    }
}

impl<'a, T: WordIters> PartialEq for ElemRef<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_u64it().eq(other.get_u64it())
    }
}

/// Mutable reference type for one element of a [`WordIters`] container.
pub struct ElemMutRef<'a, T: WordIters> {
    pub word_iters: &'a mut T,
    pub index: usize,
}

impl<'a, T: WordIters> proj::AsRef<'a> for ElemMutRef<'a, T> {
    type RefType = ElemRef<'a, T>;

    fn as_ref(&'a self) -> Self::RefType {
        ElemRef {
            word_iters: self.word_iters,
            index: self.index,
        }
    }
}

impl<'a, T: WordIters> ElemMutRef<'a, T> {
    /// Overwrite the referenced element with the contents of `other`.
    pub fn assign(&mut self, other: ElemRef<T>) {
        self.get_u64it_mut()
            .zip(other.get_u64it())
            .for_each(|(dst, src)| *dst = src);
    }

    /// Reborrow this mutable element view as an immutable element view.
    pub fn as_ref(&self) -> ElemRef<'_, T> {
        ElemRef {
            word_iters: self.word_iters,
            index: self.index,
        }
    }

    /// Return mutable access to the `u64` words of the referenced element.
    pub fn get_u64it_mut(&mut self) -> impl Iterator<Item = &mut u64> {
        self.word_iters.elem_u64it_mut(self.index)
    }

    /// Zero every word in the referenced element.
    pub fn clear(&mut self) {
        let index = self.index;
        self.word_iters.clear_elem(index);
    }
}

impl<'a, T: WordIters> HasIndex for ElemMutRef<'a, T> {
    fn get_index(&self) -> usize {
        self.index
    }
}

/// Owned wrapper around a single element [`WordIters`] container.
#[derive(Clone, PartialEq, Eq)]
pub struct Elem<T: WordIters>(pub T);

impl<T: WordIters> proj::ToOwned for Elem<T> {
    type OwnedType = Self;

    fn to_owned(&self) -> Self::OwnedType {
        self.clone()
    }
}

impl<'a, T: WordIters + 'a> proj::Borrow<'a> for Elem<T> {
    type RefType = ElemRef<'a, T>;

    fn borrow(&'a self) -> Self::RefType {
        ElemRef {
            word_iters: &self.0,
            index: 0,
        }
    }
}

impl<'a, T: WordIters + 'a> proj::BorrowMut<'a> for Elem<T> {
    type MutRefType = ElemMutRef<'a, T>;

    fn borrow_mut(&'a mut self) -> Self::MutRefType {
        ElemMutRef {
            word_iters: &mut self.0,
            index: 0,
        }
    }
}

impl<T: WordIters> From<&T> for Elem<T> {
    fn from(value: &T) -> Self {
        let mut tmp = value.empty_clone();
        tmp.resize(1);
        Elem(tmp)
    }
}

pub mod test_defs {
    use serde::{Deserialize, Serialize};

    use crate::container::table::Table;
    use crate::container::traits::{Compatible, Elements, EmptyClone};
    use crate::container::word_iters::WordIters;

    /// Test helper `WordIters` implementation formed of two internal `Table`s.
    #[derive(Serialize, Deserialize)]
    pub struct StructWithInternalTables {
        pub table_first: Table,
        pub table_second: Table,
    }

    impl Clone for StructWithInternalTables {
        fn clone(&self) -> Self {
            Self {
                table_first: self.table_first.clone(),
                table_second: self.table_second.clone(),
            }
        }
    }

    impl Elements for StructWithInternalTables {
        fn len(&self) -> usize {
            assert_eq!(self.table_first.len(), self.table_second.len());
            self.table_first.len()
        }
    }

    impl EmptyClone for StructWithInternalTables {
        fn empty_clone(&self) -> Self {
            Self::new(
                self.table_first.get_row_size(),
                self.table_second.get_row_size(),
                0,
            )
        }
    }

    impl WordIters for StructWithInternalTables {
        // set a small hash mod to increase likelihood of collision
        const HASH_MOD: Option<u64> = Option::Some(6);

        fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
            assert!(i < self.len());
            self.table_first[i]
                .iter()
                .copied()
                .chain(self.table_second[i].iter().copied())
        }

        fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
            assert!(i < self.len());
            self.table_first[i]
                .iter_mut()
                .chain(self.table_second[i].iter_mut())
        }

        fn u64it_size(&self) -> usize {
            self.table_first.get_row_size() + self.table_second.get_row_size()
        }

        fn pop_and_swap(&mut self, i_row: usize) {
            self.table_first.pop_and_swap(i_row);
            self.table_second.pop_and_swap(i_row);
        }

        fn resize(&mut self, n: usize) {
            self.table_first.resize(n);
            self.table_second.resize(n);
        }
    }

    impl Compatible for StructWithInternalTables {
        fn compatible_with(&self, other: &Self) -> bool {
            self.table_first.compatible_with(&other.table_first)
                && self.table_second.compatible_with(&other.table_second)
        }
    }

    impl StructWithInternalTables {
        /// Create a test helper with two internal tables and `n_row` zero rows.
        pub fn new(row_size_first: usize, row_size_second: usize, n_row: usize) -> Self {
            let mut this = Self {
                table_first: Table::new(row_size_first),
                table_second: Table::new(row_size_second),
            };
            this.table_first.resize(n_row);
            this.table_second.resize(n_row);
            this
        }

        /// Append row `i_row` from `source` to both internal tables.
        fn push_from_other(&mut self, source: &StructWithInternalTables, i_row: usize) {
            let i_back = self.len();
            self.table_first.push_clear();
            self.table_first[i_back].copy_from_slice(&source.table_first[i_row]);
            self.table_second.push_clear();
            self.table_second[i_back].copy_from_slice(&source.table_second[i_row]);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::container::quicksort::QuickSortNoCoeffs;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::test_defs::*;
    use super::*;

    #[test]
    fn test_quicksort() {
        use crate::container::quicksort::LexicographicSort;
        // instantiate an object for holding and mapping many multi-u64 word elements
        let mut tables = StructWithInternalTables::new(1, 2, 0);

        let seed = [45u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let modulo: u64 = 3;

        // each element requires 3 integers
        type Key = [u64; 3];

        let mut v: Vec<Key> = vec![];

        // setup all elements of the tables and the check vector in a random order
        let n_elem = 300_usize;
        for i in 0..n_elem {
            assert_eq!(tables.len(), i);
            assert_eq!(v.len(), i);
            let key: Key = [
                rng.random::<u64>() % modulo,
                rng.random::<u64>() % modulo,
                rng.random::<u64>() % modulo,
            ];
            v.push(key);
            tables.push_u64it(key.into_iter());
        }

        v.sort();
        assert_eq!(v.len(), n_elem);
        LexicographicSort { ascending: true }.sort(&mut tables);
        assert_eq!(tables.len(), n_elem);

        // check that elements in vector and table are in the same order
        for (i, item) in v.iter().enumerate() {
            assert!(tables.table_first[i].iter().eq(item.split_at(1).0.iter()));
            assert!(tables.table_second[i].iter().eq(item.split_at(1).1.iter()));
        }
    }
}
