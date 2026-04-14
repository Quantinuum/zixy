//! Hash-map helpers for containers based on iterable word elements.

use std::collections::HashMap;

use crate::container::traits::{Elements, RefElements};
use crate::container::word_iters::WordIters;

/// Although rare, different elements can hash to the same key, so resolve the collision using a vec of
/// indices whenever necessary
pub enum LookupResult<'a> {
    One(usize),
    Many(&'a Vec<usize>),
    None,
}

/// `Map` from the hash of an iterator over an element's words to its position in a `WordIters` structure.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct Map {
    /// Storage for hash -> index when hashes are unique.
    one_bins: HashMap<u64, usize>,
    /// Storage for hash -> indices when many indices have the same hash.
    many_bins: HashMap<u64, Vec<usize>>,
    /// Total number of entries stored in this `Map`.
    n_entry: usize,
}

impl Map {
    /// Clear the stored contents.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Put the (k, v) pair into the map.
    /// If the key is already in the many_bins container, then append to its vector value,
    /// If the key is already in the one_bins container, then "upgrade" the key to the many_bins
    /// Otherwise, insert the (k, v) pair in the one_bins container
    /// Returns true if the (k, v) pair was not already in the map, else return false
    pub fn insert(&mut self, k: u64, v: usize) -> bool {
        self.n_entry += 1;
        if !self.many_bins.is_empty() {
            if let Some(x) = self.many_bins.get_mut(&k) {
                assert_ne!(x.len(), 0);
                if x.contains(&v) {
                    return false;
                }
                x.push(v);
                return true;
            }
        }
        match self.one_bins.get(&k) {
            Some(&x) => {
                if x == v {
                    return false;
                };
                self.many_bins.insert(k, vec![x, v]);
                self.one_bins.remove(&k);
            }
            None => {
                self.one_bins.insert(k, v);
            }
        }
        true
    }

    /// If the (k, v) pair is in the map, remove it.
    /// If the key is in the many_bins container, there are two cases:
    ///     If there are only two inds in the value, "downgrade" the key to one_bins
    ///     Otherwise, just remove the appropriate value element
    /// If the (k, v) pair is in the one_bins, simply remove it
    /// Return true if the pair was found and removed, false if the pair was not found
    pub fn remove(&mut self, k: u64, v: usize) -> bool {
        assert_ne!(self.n_entry, 0);
        self.n_entry -= 1;
        if !self.many_bins.is_empty() {
            if let Some(x) = self.many_bins.get_mut(&k) {
                let n_v = x.len();
                if n_v <= 2 {
                    if n_v == 2 {
                        self.one_bins.insert(k, x[if x[0] == v { 1 } else { 0 }]);
                    }
                    self.many_bins.remove(&k);
                    return true;
                } else {
                    x.retain(|&i| i != v);
                    assert_eq!(x.len() + 1, n_v);
                    return true;
                }
            }
        }
        if let Some(x) = self.one_bins.remove(&k) {
            assert_eq!(x, v);
            return true;
        }
        false
    }

    /// Try to find the (One or Many) values corresponding to the key k in the map
    pub fn lookup(&self, k: u64) -> LookupResult<'_> {
        if !self.many_bins.is_empty() {
            if let Some(x) = self.many_bins.get(&k) {
                return LookupResult::Many(x);
            }
        }
        if let Some(&x) = self.one_bins.get(&k) {
            return LookupResult::One(x);
        }
        LookupResult::None
    }

    /// Get the number of entries stored in the `many_bins` structure.
    pub fn n_entry_many_bins(&self) -> usize {
        self.len().saturating_sub(self.one_bins.len())
    }

    /// Clear `self` and repopulate from `elements`.
    pub fn populate_from<T: WordIters>(&mut self, elements: &T) {
        self.clear();
        for (i, elem_ref) in elements.iter().enumerate() {
            self.insert(elem_ref.hash(), i);
        }
    }
}

impl Elements for Map {
    fn len(&self) -> usize {
        self.n_entry
    }
}
