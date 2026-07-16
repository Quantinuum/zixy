//! Fermionic modes i.e. spin orbitals and lattice sites.

use crate::container::table::Table;
use crate::container::traits::Elements;
use crate::utils::arith::divceil;
use serde::{Deserialize, Serialize};

/// The valid representations of the qubits field of objects acting on qubit spaces
#[derive(Debug, Hash, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Kind {
    /// A number of fermionic modes with no implicit spin labelling.
    Count(usize),
    /// A number of spatial orbitals or sites with spin orbitals in uuu...ddd... ordering.
    SpinMajorPairs(usize),
    /// A number spatial orbitals or sites with of spin orbitals in ududud... ordering.
    SpinMinorPairs(usize),
}

/// Fermionic mode-space descriptor, including whether spin pairs are stored contiguously or interleaved.
#[derive(Debug, Hash, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Modes(pub Kind);

impl Modes {
    /// Create an instance from a number of modes only.
    pub fn from_count(n: usize) -> Modes {
        Modes(Kind::Count(n))
    }

    /// Create an instance from the number of spin orbital pairs, assuming spin major ordering.
    pub fn from_pair_count_spin_major(n_pair: usize) -> Modes {
        Modes(Kind::SpinMajorPairs(n_pair))
    }

    /// Create an instance from the number of spin orbital pairs, assuming spin minor ordering.
    pub fn from_pair_count_spin_minor(n_pair: usize) -> Modes {
        Modes(Kind::SpinMinorPairs(n_pair))
    }

    /// Return the index of the `i`-th mode.
    pub fn get_unchecked(&self, i: usize) -> usize {
        i
    }

    /// Return the index of the `i`-th mode with bounds checking.
    pub fn get(&self, i: usize) -> Option<usize> {
        if i < self.len() {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Return all mode indices as a vector.
    pub fn inds(&self) -> Vec<usize> {
        (0..self.len()).map(|i| self.get_unchecked(i)).collect()
    }

    /// Iterate over all mode indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }
}

impl Elements for Modes {
    fn len(&self) -> usize {
        match self.0 {
            Kind::Count(n) => n,
            Kind::SpinMajorPairs(n) | Kind::SpinMinorPairs(n) => 2 * n,
        }
    }
}

// A matrix of packed integers stored in a 'Table' buffer.
#[derive(Clone)]
pub struct ModeInds {
    table: Table,
    n_bits: usize,
}

impl ModeInds {
    /// Create a new empty row
    pub fn new(n_bits: usize, max_len: usize) -> Self {
        assert!((1..=64).contains(&n_bits), "n_bits must be in 1..=64");
        let slots_per_word = 64 / n_bits;
        let row_size = divceil(max_len as isize, slots_per_word as isize) as usize;
        Self {
            table: Table::new(row_size),
            n_bits,
        }
    }

    /// Calculate which u64 and bit offset within it corresponds to slot `i_slot`.
    pub fn get_offset(&self, i_slot: usize) -> (usize, usize) {
        let slots_per_word = 64 / self.n_bits;
        let i_u64 = i_slot / slots_per_word;
        let bit_offset = (i_slot % slots_per_word) * self.n_bits;
        (i_u64, bit_offset)
    }

    /// Push a new row of packed integers into the matrix.
    pub fn push_vec(&mut self, values: &[usize]) {
        self.push_iter(values.iter().copied());
    }

    pub fn push_iter<I: IntoIterator<Item = usize>>(&mut self, values: I) {
        self.table.push_clear();
        let last_row = self.table.len() - 1;
        for (i, value) in values.into_iter().enumerate() {
            let (i_u64, bit_offset) = self.get_offset(i);
            self.table[last_row][i_u64] |= (value as u64) << bit_offset;
        }
    }

    /// Read back integer stored at slot `i_slot` in row `i_row` by reversing the packing from `push_vec`.
    pub fn get_value(&self, i_row: usize, i_slot: usize) -> usize {
        let (i_u64, bit_offset) = self.get_offset(i_slot);
        // Extra safety measure to ensure only the n_bit bits for this slot
        let mask = if self.n_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.n_bits) - 1
        };
        let shifted = self.table[i_row][i_u64] >> bit_offset;
        (shifted & mask) as usize
    }

    /// Read back all mode indices in row `i_row` up to `length` slots, ignoring padding.
    pub fn read_row(&self, i_row: usize, length: usize) -> Vec<usize> {
        let mut vec: Vec<usize> = Vec::new();
        for i in 0..length {
            let value = self.get_value(i_row, i);
            vec.push(value);
        }
        vec
    }

    /// Return an iterator over the raw u64 words in row `i`.
    pub fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone + use<'_> {
        self.table[i].iter().copied()
    }

    /// Return a mutable iterator over the raw u64 words in row `i`.
    pub fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.table[i].iter_mut()
    }

    /// Return the number of u64 words per row.
    pub fn u64it_size(&self) -> usize {
        self.table.get_row_size()
    }

    /// Remove element at `index` by replacing it with the last element.
    pub fn pop_and_swap(&mut self, index: usize) {
        self.table.pop_and_swap(index);
    }

    /// Resize to `n` rows.
    pub fn resize(&mut self, n: usize) {
        self.table.resize(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_slots_spanning_multiple_words() {
        let mut inds = ModeInds::new(3, 32);
        let values: Vec<usize> = (0..32).map(|i| i % 8).collect();
        inds.push_vec(&values);
        assert_eq!(inds.read_row(0, values.len()), values);
    }

    #[test]
    fn test_roundtrip_n_bits_64() {
        let mut inds = ModeInds::new(64, 3);
        let values = vec![u64::MAX as usize, 0, 42];
        inds.push_vec(&values);
        assert_eq!(inds.read_row(0, values.len()), values);
    }

    #[test]
    fn test_slots_never_straddle_word_boundary() {
        let inds = ModeInds::new(5, 16);
        let (i_u64, bit_offset) = inds.get_offset(12);
        assert_eq!((i_u64, bit_offset), (1, 0));
        let (i_u64, bit_offset) = inds.get_offset(11);
        assert_eq!((i_u64, bit_offset), (0, 55));
    }

    #[test]
    fn test_multiple_rows_roundtrip() {
        let mut inds = ModeInds::new(3, 32);
        let row0: Vec<usize> = (0..32).map(|i| i % 8).collect();
        let row1: Vec<usize> = (0..32).map(|i| (31 - i) % 8).collect();
        inds.push_vec(&row0);
        inds.push_vec(&row1);
        assert_eq!(inds.read_row(0, row0.len()), row0);
        assert_eq!(inds.read_row(1, row1.len()), row1);
    }
}
