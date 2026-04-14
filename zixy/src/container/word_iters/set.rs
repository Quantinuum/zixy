//! Combines `WordIters` with `Map` to ensure uniqueness among elements.

use serde::Serialize;

use crate::container::map::{LookupResult, Map};
use crate::container::traits::proj;
use crate::container::traits::{Elements, EmptyFrom};
use crate::container::word_iters::{ElemRef, WordIters};

/// Provides a non-owning constant-time accessor for data composed of iterable elements of u64 words
/// This is done indirectly by using the hash of the iterables as the key of the `Map`
///
/// Pop-and-swap is used to remove elements, meaning removal by overwriting an arbitrary element with
/// the last element in the sequence, and then shortening the sequence by one element, maintaining gaplessness.
/// This does not preserve insertion order like a shift-remove would, but it is constant complexity.

#[derive(Serialize)]
pub struct Set<T: WordIters> {
    pub word_iters: T,
    #[serde(skip_serializing)]
    pub map: Map,
}

/// Borrowed immutable view of the fields of a [`Set`] collection.
pub struct View<'a, T: WordIters> {
    pub word_iters: &'a T,
    pub map: &'a Map,
}

/// Borrowed mutable view of the fields of a [`Set`] collection.
pub struct ViewMut<'a, T: WordIters> {
    pub word_iters: &'a mut T,
    pub map: &'a mut Map,
}

/// Trait for structs that immutably view a [`Set`].
pub trait AsView<T: WordIters> {
    /// Return an immutable view over the set and its lookup map.
    fn view<'a>(&'a self) -> View<'a, T>;

    /// Return the underlying `WordIters` storage referenced by this view.
    fn get_word_iters(&self) -> &T {
        self.view().word_iters
    }

    /// Try to find and return an element's index by value in the structure.
    fn lookup(&self, u64it: impl Iterator<Item = u64> + Clone) -> Option<usize> {
        let self_ref = self.view();
        let elements = self_ref.word_iters;
        let map = self_ref.map;
        match map.lookup(elements.hash_u64it(u64it.clone())) {
            LookupResult::One(x) => {
                if elements.elem_u64it(x).eq(u64it.clone()) {
                    Some(x)
                } else {
                    None
                }
            }
            LookupResult::Many(items) => items
                .iter()
                .find(|i| elements.elem_u64it(**i).eq(u64it.clone()))
                .copied(),
            LookupResult::None => None,
        }
    }

    /// Return the `u64` iterator for the element at index `i`.
    fn elem_u64it<'a>(&'a self, i: usize) -> impl Iterator<Item = u64> + Clone + 'a
    where
        T: 'a,
    {
        self.view().word_iters.elem_u64it(i)
    }

    /// For testing: assert that the hash map and element storage agree on every stored element and index.
    fn consistency_check(&self) {
        let self_ref = self.view();
        assert_eq!(self_ref.word_iters.len(), self_ref.map.len());
        for i in 0..self_ref.word_iters.len() {
            let lookup_result = self.lookup(self_ref.word_iters.elem_u64it(i));
            assert!(lookup_result.is_some());
            assert_eq!(lookup_result.unwrap(), i);
        }
    }
}

/// Trait for structs that mutably view a [`Set`].
pub trait AsViewMut<T: WordIters>: AsView<T> {
    /// Return a mutable view over the set and its lookup map.
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T>;

    /// Insert a new element from the given iterator at the end and insert the (k, v) pair into the map
    /// or, if the iterator is already contained in the map by value, return the existing position
    /// also return whether the element was inserted as a boolean
    fn insert_u64it_or_get_index(
        &mut self,
        u64it: impl Iterator<Item = u64> + Clone,
    ) -> (usize, bool) {
        if let Some(x) = self.lookup(u64it.clone()) {
            return (x, false);
        }
        let self_mut = self.view_mut();
        self_mut.word_iters.push_u64it(u64it);
        let v = self_mut.map.len();
        let k = self_mut.word_iters.hash_at_index(v);
        assert!(self_mut.map.insert(k, v));
        (v, true)
    }

    /// Insert a new element from the given element reference.
    fn insert_or_get_index<'b>(&mut self, elem: ElemRef<'b, T>) -> (usize, bool) {
        self.insert_u64it_or_get_index(elem.get_u64it())
    }

    /// Try to remove the (k, v) pair corresponding to the iterable element indexed by `index`. Return true if it was
    /// successfully removed, and false otherwise
    fn drop(&mut self, v: usize) -> bool {
        let self_mut = self.view_mut();
        let k = self_mut.word_iters.hash_at_index(v);
        if !self_mut.map.remove(k, v) {
            return false;
        }
        if v == self_mut.map.len() {
            self_mut.word_iters.pop_and_swap(v);
            return true;
        }
        let v_last = self_mut.map.len();
        let k_last = self_mut.word_iters.hash_at_index(v_last);
        self_mut.map.remove(k_last, v_last);
        self_mut.map.insert(k_last, v);
        self_mut.word_iters.pop_and_swap(v);
        true
    }

    /// Remove all elements and reset the map.
    fn clear(&mut self) {
        let self_mut = self.view_mut();
        self_mut.word_iters.clear();
        self_mut.map.clear();
    }
}

impl<T: WordIters> AsView<T> for Set<T> {
    fn view<'a>(&'a self) -> View<'a, T> {
        self.borrow()
    }
}

impl<'a, T: WordIters> AsView<T> for View<'a, T> {
    fn view<'b>(&'b self) -> View<'b, T> {
        View {
            word_iters: self.word_iters,
            map: self.map,
        }
    }
}

impl<'a, T: WordIters> AsView<T> for ViewMut<'a, T> {
    fn view<'b>(&'b self) -> View<'b, T> {
        View {
            word_iters: self.word_iters,
            map: self.map,
        }
    }
}

impl<T: WordIters> AsViewMut<T> for Set<T> {
    fn view_mut<'a, 'b: 'a>(&'b mut self) -> ViewMut<'a, T> {
        self.borrow_mut()
    }
}

impl<'a, T: WordIters> AsViewMut<T> for ViewMut<'a, T> {
    fn view_mut<'b, 'c: 'b>(&'c mut self) -> ViewMut<'b, T> {
        ViewMut {
            word_iters: self.word_iters,
            map: self.map,
        }
    }
}

impl<T: WordIters> Elements for Set<T> {
    fn len(&self) -> usize {
        self.word_iters.len()
    }
}

impl<'a, T: WordIters + 'a> proj::Borrow<'a> for Set<T> {
    type RefType = View<'a, T>;

    fn borrow(&'a self) -> Self::RefType {
        View {
            word_iters: &self.word_iters,
            map: &self.map,
        }
    }
}

impl<'a, T: WordIters + 'a> proj::BorrowMut<'a> for Set<T> {
    type MutRefType = ViewMut<'a, T>;

    fn borrow_mut(&'a mut self) -> Self::MutRefType {
        ViewMut {
            word_iters: &mut self.word_iters,
            map: &mut self.map,
        }
    }
}

impl<T: WordIters> Set<T> {
    /// Borrow an immutable set view over the stored elements and lookup map.
    pub fn borrow(&self) -> View<'_, T> {
        View {
            word_iters: &self.word_iters,
            map: &self.map,
        }
    }

    /// Borrow a mutable set view over the stored elements and lookup map.
    pub fn borrow_mut(&mut self) -> ViewMut<'_, T> {
        ViewMut {
            word_iters: &mut self.word_iters,
            map: &mut self.map,
        }
    }
}

impl<T: WordIters> EmptyFrom<T> for Set<T> {
    fn empty_from(elements: &T) -> Self {
        Self {
            word_iters: elements.empty_clone(),
            map: Map::default(),
        }
    }
}

impl<T: WordIters> From<&T> for Set<T> {
    fn from(value: &T) -> Self {
        let mut this = Self::empty_from(value);
        for i in 0..value.len() {
            this.borrow_mut()
                .insert_u64it_or_get_index(value.elem_u64it(i));
        }
        this
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::container::traits::Elements;
    use crate::container::word_iters::set::{AsView, AsViewMut, Set as MappedWordIters};
    use crate::container::word_iters::test_defs::StructWithInternalTables;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_randomized() {
        // instantiate an object for holding and mapping many multi-u64 word elements
        let mut tables = MappedWordIters::from(&StructWithInternalTables::new(1, 2, 0));

        let seed = [45u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let modulo: u64 = 3;

        // each element requires 3 integers
        type Key = [u64; 3];

        // use a set of these keys to keep track of what should be in the mapped `WordIters` object
        let mut set_chk: HashSet<Key> = HashSet::new();

        // initially false, but set to true whenever there's a non-zero number of collisions in tables.map
        let mut many_bins_tested = false;

        // generate random keys, add key to tables and set_chk if it is not found in these objects, else remove it
        for _ in 0..(1 << 12) {
            // should have the same number of elements in tables and the checking set
            assert_eq!(tables.len(), set_chk.len());
            // same with the map within tables
            assert_eq!(tables.map.len(), set_chk.len());
            let key: Key = [
                rng.random::<u64>() % modulo,
                rng.random::<u64>() % modulo,
                rng.random::<u64>() % modulo,
            ];

            // keep track of whether the many-bins functionality of Map is tested at least once.
            many_bins_tested |= tables.map.n_entry_many_bins() != 0;

            match tables.borrow().lookup(key.iter().copied()) {
                Some(i) => {
                    // the found element should match the key iterator used in the lookup
                    assert!(tables.borrow().elem_u64it(i).eq(key.iter().copied()));
                    // the checking set must contain the looked-up key
                    assert!(set_chk.contains(&key));
                    // trying to insert an existing key into the mapped `WordIters` object should return the index of the existing element
                    // and false to indicate that the insertion was not successful
                    assert_eq!(
                        tables
                            .borrow_mut()
                            .insert_u64it_or_get_index(key.iter().copied()),
                        (i, false)
                    );
                    // should be the same number of elements in the mapped `WordIters` object as in the checking set
                    assert_eq!(tables.len(), set_chk.len());
                    // same with the map within tables
                    assert_eq!(tables.map.len(), set_chk.len());
                    // take the key out of the checking set, asserting that it was already present
                    assert!(set_chk.remove(&key));
                    // take the key out of tables, asserting that it was already present
                    assert!(tables.borrow_mut().drop(i));
                    // check that the number of elements is still correct after the drop
                    assert_eq!(tables.len(), set_chk.len());
                    assert_eq!(tables.map.len(), set_chk.len());
                }
                None => {
                    // element wasn't found, so it shouldn't be in the checking set either
                    assert!(!set_chk.contains(&key));
                    // should be the same number of elements in the mapped `WordIters` object as in the checking set
                    assert_eq!(tables.len(), set_chk.len());
                    // same with the map within tables
                    assert_eq!(tables.map.len(), set_chk.len());
                    // insert the key into the checking set, asserting that it was not already present
                    assert!(set_chk.insert(key));
                    // inserting should succeed, i.e. the index inserted at should be the last one in tables
                    assert_eq!(
                        tables
                            .borrow_mut()
                            .insert_u64it_or_get_index(key.iter().copied()),
                        (tables.len().saturating_sub(1), true)
                    );
                    // check that the number of elements is still correct after the insertion
                    assert_eq!(tables.len(), set_chk.len());
                    assert_eq!(tables.map.len(), set_chk.len());
                    // the previously not found element should now be found
                    assert!(tables.borrow().lookup(key.iter().copied()).is_some());
                    // and found at the last index
                    assert_eq!(
                        tables.borrow().lookup(key.iter().copied()).unwrap() + 1,
                        tables.len()
                    );
                }
            }
            tables.borrow().consistency_check();
        }
        // for a thorough test, the collision resolution of Map should have been used at some point
        assert!(many_bins_tested);
    }
}
