//! A matrix of bits stored in a `Table` buffer.

use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use itertools::Itertools;
use ndarray::Array2 as Matrix;
use serde::{Deserialize, Serialize};

use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::table::Table;
use crate::container::traits::{Compatible, Elements, EmptyClone, HasIndex, RefElements};
use crate::container::word_iters::{self, WordIters};
use crate::utils::arith::{divceil, divrem};

/// A [`WordIters`] implementation viewable as a bit-matrix.
pub trait AsBitMatrix: WordIters {
    /// Return the underlying row-major `Table` storage.
    fn get_table(&self) -> &Table;
    /// Return mutable access to the underlying row-major `Table` storage.
    fn get_table_mut(&mut self) -> &mut Table;
    /// Return the number of bits stored in each row.
    fn n_bit(&self) -> usize;

    /// Return an error if `index` is not a valid row index.
    fn check_row_bounds(&self, index: usize) -> Result<(), OutOfBounds> {
        OutOfBounds::check(index, self.len(), Dimension::Row)
    }

    /// Return an error if `index` is not a valid bit index within a row.
    fn check_bit_bounds(&self, index: usize) -> Result<(), OutOfBounds> {
        OutOfBounds::check(index, self.n_bit(), Dimension::Bit)
    }

    /// Return an error unless both the row and bit indices are in bounds.
    fn check_bounds(&self, i_row: usize, i_bit: usize) -> Result<(), OutOfBounds> {
        self.check_row_bounds(i_row)?;
        self.check_bit_bounds(i_bit)
    }

    /// Clear every stored bit in row `i_row`.
    fn clear_row(&mut self, i_row: usize) {
        self.get_table_mut().clear_row(i_row);
    }

    /// Read the indexed bit value.
    fn get_bit_unchecked(&self, i_row: usize, i_bit: usize) -> bool {
        let (i_u64, i_bit) = divrem(i_bit, 64);
        debug_assert!(i_bit < 64);
        let mask: u64 = 1u64 << i_bit;
        self.get_table()[i_row][i_u64] & mask != 0
    }

    /// Read the indexed bit value with bounds checking.
    fn get_bit(&self, i_row: usize, i_bit: usize) -> Result<bool, OutOfBounds> {
        self.check_bounds(i_row, i_bit)?;
        Ok(self.get_bit_unchecked(i_row, i_bit))
    }

    /// Assign the row at index `i_row` a value from a vector of bit values.
    fn assign_vec_unchecked(&mut self, i_row: usize, value: Vec<bool>) {
        self.clear_row(i_row);
        for (i, excited) in value.into_iter().enumerate() {
            if excited {
                self.set_bit_unchecked(i_row, i, true);
            }
        }
    }

    /// Assign the row at index `i_row` a value from a vector of bit values with bounds checking.
    fn assign_vec(&mut self, i_row: usize, value: Vec<bool>) -> Result<(), OutOfBounds> {
        self.check_bounds(i_row, value.len().saturating_sub(1))?;
        self.assign_vec_unchecked(i_row, value);
        Ok(())
    }

    /// Assign the row at index `i_row` a value from a vector of set bit indices.
    fn assign_set_unchecked(&mut self, i_row: usize, value: HashSet<usize>) {
        self.clear_row(i_row);
        for i in value.into_iter() {
            self.set_bit_unchecked(i_row, i, true);
        }
    }

    /// Assign the row at index `i_row` a value from a vector of set bit indices with bounds checking.
    fn assign_set(&mut self, i_row: usize, value: HashSet<usize>) -> Result<(), OutOfBounds> {
        self.check_bounds(i_row, value.iter().max().copied().unwrap_or_default())?;
        self.assign_set_unchecked(i_row, value);
        Ok(())
    }

    /// Assign the given value to the indexed bit.
    fn set_bit_unchecked(&mut self, i_row: usize, i_bit: usize, value: bool) {
        let (i_u64, i_bit) = divrem(i_bit, 64);
        let word = &mut self.get_table_mut()[i_row][i_u64];
        debug_assert!(i_bit < 64);
        let mask: u64 = 1u64 << i_bit;
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    /// Assign a value to the indexed bit.
    fn set_bit(&mut self, i_row: usize, i_bit: usize, value: bool) -> Result<(), OutOfBounds> {
        self.check_bounds(i_row, i_bit)?;
        self.set_bit_unchecked(i_row, i_bit, value);
        Ok(())
    }

    /// Push a new row initialized from `values`.
    fn push_vec(&mut self, values: Vec<bool>) -> Result<(), OutOfBounds> {
        let i_row = self.len();
        self.push_clear();
        self.assign_vec(i_row, values)
    }

    /// Push a new row whose set bits are given by `i_bits`.
    fn push_set(&mut self, i_bits: HashSet<usize>) -> Result<(), OutOfBounds> {
        let i_row = self.len();
        self.push_clear();
        self.assign_set(i_row, i_bits)
    }

    /// Push the mode settings and index i at the back of this vector.
    fn push_spring(&mut self, springs: &BinarySprings, index: usize) -> Result<(), OutOfBounds> {
        let i_row = self.len();
        self.push_clear();
        for (_, i_bit) in springs.get_iter(index) {
            self.set_bit(i_row, i_bit as usize, true)?;
        }
        Ok(())
    }

    /// Create a new instance from given `BinarySprings`, returning the associated phases as a `TwoBitVec`.
    fn set_from_springs(&mut self, springs: &BinarySprings) -> Result<(), OutOfBounds> {
        self.get_table_mut().clear();
        for i in 0..springs.len() {
            self.push_spring(springs, i)?;
        }
        Ok(())
    }

    /// A bit is redundant with respect to a bit matrix list if it is clear in all rows.
    fn bit_redundant(&self, i_bit: usize) -> Result<bool, OutOfBounds> {
        self.check_bit_bounds(i_bit)?;
        Ok((0..self.len()).all(|i_row| self.get_bit_unchecked(i_row, i_bit)))
    }
}

/// Immutable view over a single row of an [`AsBitMatrix`] value, exposing row-level bit operations.
pub trait AsRowRef: HasIndex {
    /// Return the bit matrix this row view refers to.
    fn bit_mat(&self) -> &impl AsBitMatrix;

    /// Return the logical number of bits in the referenced row.
    fn n_bit(&self) -> usize {
        self.bit_mat().n_bit()
    }

    /// Return the raw `u64` words that store the referenced row.
    fn get_slice(&self) -> &[u64] {
        let index = self.get_index();
        &self.bit_mat().get_table()[index]
    }

    /// Extract the bit value stored at the given bit index.
    fn get_bit_unchecked(&self, i_bit: usize) -> bool {
        self.bit_mat().get_bit_unchecked(self.get_index(), i_bit)
    }

    /// Extract the bit value stored at the given index.
    fn get(&self, i_bit: usize) -> Result<bool, OutOfBounds> {
        self.bit_mat().check_bit_bounds(i_bit)?;
        Ok(self.bit_mat().get_bit_unchecked(self.get_index(), i_bit))
    }

    /// Represent the referenced contents as an iterator of bit values.
    fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.n_bit()).map(|i_bit| self.get_bit_unchecked(i_bit))
    }

    /// Represent the referenced contents as a vector of bit values.
    fn to_vec(&self) -> Vec<bool> {
        self.iter().collect::<Vec<_>>()
    }

    /// Represent the referenced contents as a set of indices of excited local states.
    fn to_set(&self) -> HashSet<usize> {
        self.iter()
            .enumerate()
            .filter_map(|(i, b)| if b { Some(i) } else { None })
            .collect()
    }

    /// Return the number of occurrences of the given bit value in the referenced component.
    fn count(&self, value: bool) -> usize {
        if !value {
            self.n_bit().saturating_sub(self.count(true))
        } else {
            self.bit_mat().elem_hamming_weight(self.get_index())
        }
    }

    /// Return the next set bit at or above `(i_word, i_bit)`, as `(word_index, bit_index)`.
    /// Returns `None` when no further set bit exists.
    fn lowest_set_word_bit_not_before(
        &self,
        i_word: usize,
        i_bit: usize,
    ) -> Option<(usize, usize)> {
        let i_bit_usize = i_bit;
        let mut i_word = i_word + (i_bit_usize / 64);
        let start_bit = (i_bit_usize % 64) as u32;

        let iter = self.get_slice().split_at(i_word).1.iter();
        let mut mask: u64 = if start_bit == 0 {
            0
        } else {
            (1u64 << start_bit) - 1
        };
        for word in iter {
            let masked = *word & !mask;
            if masked != 0 {
                let i_bit_word = masked.trailing_zeros() as usize;
                return Some((i_word, i_bit_word));
            }
            i_word += 1;
            mask = 0;
        }
        None
    }

    /// Return the next set bit strictly above `(i_word, i_bit)`, as `(word_index, bit_index)`.
    /// Returns `None` when no further set bit exists.
    fn lowest_set_word_bit_after(&self, i_word: usize, i_bit: usize) -> Option<(usize, usize)> {
        self.lowest_set_word_bit_not_before(i_word, i_bit + 1)
    }

    /// Return the next set bit at or above `i_bit`.
    /// Returns `None` when no further set bit exists.
    fn lowest_set_bit_not_before(&self, i_bit: usize) -> Option<usize> {
        let (i_word, i_bit) = divrem(i_bit, 64);
        self.lowest_set_word_bit_not_before(i_word, i_bit)
            .map(|(i_word, i_bit)| i_word * 64 + i_bit)
    }

    /// Return the next set bit strictly above `(i_word, i_bit)`, as `(word_index, bit_index)`.
    /// Returns `None` when no further set bit exists.
    fn lowest_set_bit_after(&self, i_bit: usize) -> Option<usize> {
        self.lowest_set_bit_not_before(i_bit + 1)
    }

    /// Return the lowest set bit overall.
    /// Returns `None` when the cmpnt is clear.
    fn lowest_set_bit(&self) -> Option<usize> {
        self.lowest_set_bit_not_before(0)
    }

    /// Return the next set bit at or below `(i_word, i_bit)`, as `(word_index, bit_index)`.
    /// Returns `None` when no further set bit exists.
    fn next_highest_set_word_bit(&self, i_word: usize, i_bit: usize) -> Option<(usize, usize)> {
        let words = self.get_slice();
        if words.is_empty() {
            return None;
        }

        let mut i_word = (i_word + (i_bit / 64)).min(words.len().saturating_sub(1));
        let start_bit = (i_bit % 64) as u32;

        let mut mask: u64 = if start_bit == 63 {
            u64::MAX
        } else {
            (1u64 << (start_bit + 1)) - 1
        };

        loop {
            let masked = words[i_word] & mask;
            if masked != 0 {
                return Some((
                    i_word,
                    63usize.saturating_sub(masked.leading_zeros() as usize),
                ));
            }
            if i_word == 0 {
                break;
            }
            i_word -= 1;
            mask = u64::MAX;
        }
        None
    }

    /// Return the next set bit at or below `i_bit`.
    /// Returns `None` when no further set bit exists.
    fn next_highest_set_bit(&self, i_bit: usize) -> Option<usize> {
        let (i_word, i_bit) = divrem(i_bit, 64);
        self.next_highest_set_word_bit(i_word, i_bit)
            .map(|(i_word, i_bit)| i_word * 64 + i_bit)
    }

    /// Return the highest set bit overall.
    /// Returns `None` when the cmpnt is clear.
    fn highest_set_bit(&self) -> Option<usize> {
        self.next_highest_set_bit(0)
    }

    /// Iterate over all set bits from low to high, yielding `(word_index, bit_index)`.
    fn iter_set_word_bits(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut i_word = 0usize;
        let mut i_bit = 0usize;
        std::iter::from_fn(move || {
            let next = self.lowest_set_word_bit_not_before(i_word, i_bit)?;
            i_word = next.0;
            i_bit = next.1.saturating_add(1);
            Some(next)
        })
    }

    /// Iterate over all set bits from high to low, yielding `(word_index, bit_index)`.
    fn iter_set_word_bits_rev(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut i_word = self.get_slice().len().saturating_sub(1);
        let mut i_bit = 63usize;
        let mut done = self.get_slice().is_empty();

        std::iter::from_fn(move || {
            if done {
                return None;
            }
            let next = self.next_highest_set_word_bit(i_word, i_bit)?;
            i_word = next.0;
            if next.1 == 0 {
                if i_word == 0 {
                    done = true;
                } else {
                    i_word -= 1;
                    i_bit = 63;
                }
            } else {
                i_bit = next.1 - 1;
            }
            Some(next)
        })
    }

    /// Iterate over all set bits from low to high as flat indices `word * 64 + bit`.
    fn iter_set_bits_flat(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter_set_word_bits()
            .map(|(i, j)| i.saturating_mul(64).saturating_add(j))
    }

    /// Iterate over all set bits from high to low as flat indices `word * 64 + bit`.
    fn iter_set_bits_flat_rev(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter_set_word_bits_rev()
            .map(|(i, j)| i.saturating_mul(64).saturating_add(j))
    }

    /// Return the number of set bit not before the given overall bit position in the slice.
    fn count_set_bits_not_before(&self, i_bit_start: usize) -> usize {
        let words = self.get_slice();
        let i_word_start = i_bit_start / 64;
        let i_bit_word_start = (i_bit_start % 64) as u32;

        if i_word_start >= words.len() {
            return 0;
        }

        let mut n_set: usize =
            (words[i_word_start] & (!0u64 << i_bit_word_start)).count_ones() as usize;
        for word in &words[(i_word_start + 1)..] {
            n_set = n_set.saturating_add(word.count_ones() as usize);
        }
        n_set
    }

    /// Return the number of set bits strictly after the given overall bit position in the slice.
    fn count_set_bits_after(&self, i_bit: usize) -> usize {
        self.count_set_bits_not_before(i_bit.saturating_add(1))
    }

    /// Return the number of set bits at or before the given overall bit position in the slice.
    fn count_set_bits_not_after(&self, i_bit: usize) -> usize {
        self.count(true)
            .saturating_sub(self.count_set_bits_after(i_bit))
    }

    /// Return the number of set bits strictly before the given overall bit position in the slice.
    fn count_set_bits_before(&self, i_bit: usize) -> usize {
        self.count_set_bits_not_after(i_bit.saturating_sub(1))
    }
}

/// Mutable view over a single row of an [`AsBitMatrix`] value, exposing row-level bit operations.
pub trait AsRowMutRef: HasIndex {
    /// Return the bit matrix this row view refers to.
    fn bit_mat(&self) -> &impl AsBitMatrix;
    /// Return mutable access to the bit matrix this row view refers to.
    fn bit_mat_mut(&mut self) -> &mut impl AsBitMatrix;

    /// Return the logical number of bits in the referenced row.
    fn n_bit(&self) -> usize {
        self.bit_mat().n_bit()
    }

    /// Return the raw `u64` words that store the referenced row, mutably.
    fn get_slice_mut(&mut self) -> &mut [u64] {
        let index = self.get_index();
        &mut self.bit_mat_mut().get_table_mut()[index]
    }

    /// Assign this mutable ref a value from a vector of bit values, either excited (true) or ground (false).
    fn assign_vec_unchecked(&mut self, value: Vec<bool>) {
        let i_row = self.get_index();
        self.bit_mat_mut().assign_vec_unchecked(i_row, value);
    }

    /// Assign this mutable ref a value from a vector of bit values, either excited (true) or ground (false).
    fn assign_vec(&mut self, value: Vec<bool>) -> Result<(), OutOfBounds> {
        let i_row = self.get_index();
        self.bit_mat_mut().assign_vec(i_row, value)?;
        Ok(())
    }

    /// Assign this mutable ref a value from a set of set bit indices.
    fn assign_set_unchecked(&mut self, value: HashSet<usize>) {
        let i_row = self.get_index();
        self.bit_mat_mut().assign_set_unchecked(i_row, value);
    }

    /// Assign this mutable ref a value from a set of set bit indices.
    fn assign_set(&mut self, value: HashSet<usize>) -> Result<(), OutOfBounds> {
        let i_row = self.get_index();
        self.bit_mat_mut().assign_set(i_row, value)
    }

    /// Assign the given value to the indexed bit.
    fn set_bit_unchecked(&mut self, i_bit: usize, value: bool) {
        let i_row = self.get_index();
        self.bit_mat_mut().set_bit_unchecked(i_row, i_bit, value);
    }

    /// Assign the given value to the indexed bit.
    fn set_bit(&mut self, i_bit: usize, value: bool) -> Result<(), OutOfBounds> {
        let i_row = self.get_index();
        self.bit_mat_mut().set_bit(i_row, i_bit, value)
    }

    /// Assign the given springs element to the viewed row.
    fn set_spring(&mut self, springs: &BinarySprings, i: usize) -> Result<(), OutOfBounds> {
        for (b, i_bit) in springs.get_iter(i) {
            self.set_bit(i_bit as usize, b != 0)?;
        }
        Ok(())
    }

    /// Get the value stored at the given bit index.
    fn get_bit_unchecked(&self, i_bit: usize) -> bool {
        self.bit_mat().get_bit_unchecked(self.get_index(), i_bit)
    }

    /// Get the value stored at the given bit index.
    fn get_bit(&self, i_bit: usize) -> Result<bool, OutOfBounds> {
        self.bit_mat().get_bit(self.get_index(), i_bit)
    }
}

/// Contiguous and compact storage for computational basis states defined on a given basis of `Qubits`.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct BitMatrix {
    /// Raw storage table for the bitsets.
    table: Table,
    /// Number of bits per cmpnt
    n_bit: usize,
}

impl BitMatrix {
    /// Create an empty `CmpntList` on the given number of bits.
    pub fn new(n_bit: usize) -> Self {
        let row_size = divceil(n_bit, 64) as usize;
        Self {
            table: Table::new(row_size),
            n_bit,
        }
    }

    /// Create an empty square bit matrix
    pub fn new_square(n: usize) -> Self {
        let mut out = Self::new(n);
        out.resize(n);
        out
    }

    /// Create a new instance from given `BinarySprings`.
    pub fn from_springs(n_bit: usize, springs: &BinarySprings) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(n_bit);
        this.set_from_springs(springs)?;
        Ok(this)
    }

    /// Infer the row width from `springs` and build a matrix containing those rows.
    pub fn from_springs_default(springs: &BinarySprings) -> Self {
        Self::from_springs(springs.get_mode_inds().default_n_mode() as usize, springs).unwrap()
    }

    /// Convert the bit matrix into a dense `ndarray` matrix of `0` and `1` byte values.
    pub fn to_matrix(&self) -> Matrix<u8> {
        let shape = (self.len(), self.n_bit);
        let mut out = Matrix::<u8>::zeros(shape);
        for i_row in 0..self.len() {
            for i_col in 0..self.n_bit {
                if self.get_bit_unchecked(i_row, i_col) {
                    out[[i_row, i_col]] = 1;
                }
            }
        }
        out
    }

    /// Change the logical row width, resizing the word storage and masking truncated tail bits as needed.
    pub fn reformat(&mut self, n_bit: usize) {
        if n_bit >= self.n_bit && divceil(self.n_bit, 64) == divceil(n_bit, 64) {
            // no physical buffer modification needed if the row size stays the same and the bit row is not shortened.
            self.n_bit = n_bit;
            return;
        }

        let mask_tail = |table: &mut Table, n_bit: usize| {
            let rem = n_bit % 64;
            if rem == 0 || table.get_row_size() == 0 {
                return;
            }
            let keep_mask = (1u64 << rem) - 1;
            let i_last_word = table.get_row_size() - 1;
            for i_row in 0..table.len() {
                table[i_row][i_last_word] &= keep_mask;
            }
        };

        mask_tail(&mut self.table, self.n_bit);
        self.table.reformat(divceil(n_bit, 64));
        self.n_bit = n_bit;
        mask_tail(&mut self.table, self.n_bit);
    }

    /// Return the transpose of this bit matrix.
    pub fn transpose(&self) -> Self {
        /// Transpose a `64 x 64` bit block in place, treating each `u64` as one source row.
        fn transpose_block_64x64(block: &mut [u64; 64]) {
            let mut out = [0u64; 64];
            for (i_row, mut word) in block.iter().copied().enumerate() {
                while word != 0 {
                    let i_col = word.trailing_zeros() as usize;
                    out[i_col] |= 1u64 << i_row;
                    word &= word - 1;
                }
            }
            *block = out;
        }

        let n_row_out = self.n_bit;
        let n_bit_out = self.len();
        let mut out = Self::new(n_bit_out);
        out.resize(n_row_out);

        let src_row_blocks = divceil(self.len(), 64);
        for i_row_block in 0..src_row_blocks {
            let row_start = i_row_block * 64;
            for i_word in 0..self.table.get_row_size() {
                let mut block = [0u64; 64];
                for (i, word) in block.iter_mut().enumerate() {
                    let i_row = row_start + i;
                    if i_row < self.len() {
                        *word = self.table[i_row][i_word];
                    }
                }
                transpose_block_64x64(&mut block);

                let bit_start = i_word * 64;
                for (i_bit_offset, word) in block.into_iter().enumerate() {
                    let i_row_out = bit_start + i_bit_offset;
                    if i_row_out < n_row_out {
                        out.table[i_row_out][i_row_block] = word;
                    }
                }
            }
        }

        out
    }
}

impl AsBitMatrix for BitMatrix {
    fn get_table(&self) -> &Table {
        &self.table
    }

    fn get_table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    fn n_bit(&self) -> usize {
        self.n_bit
    }
}

impl Compatible for BitMatrix {
    fn compatible_with(&self, other: &Self) -> bool {
        self.n_bit == other.n_bit
    }
}

impl Elements for BitMatrix {
    fn len(&self) -> usize {
        self.table.len()
    }
}

impl EmptyClone for BitMatrix {
    fn empty_clone(&self) -> Self {
        Self {
            table: Table::new(self.table.get_row_size()),
            n_bit: self.n_bit,
        }
    }
}

impl WordIters for BitMatrix {
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
        format!(
            "[{}]",
            self.get_elem_ref(i)
                .iter()
                .map(|x| if x { "1" } else { "0" })
                .join(", ")
        )
    }

    fn resize(&mut self, n: usize) {
        self.table.resize(n);
    }
}

impl Display for BitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            (0..self.len()).map(|i| self.fmt_elem(i)).join(", ")
        )
    }
}

impl Index<usize> for BitMatrix {
    type Output = [u64];
    fn index(&self, index: usize) -> &Self::Output {
        self.table.index(index)
    }
}

impl IndexMut<usize> for BitMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.table.index_mut(index)
    }
}

/// Borrowed immutable row view into a [`BitMatrix`].
pub type CmpntRef<'a> = word_iters::ElemRef<'a, BitMatrix>;
/// Borrowed mutable row view into a [`BitMatrix`].
pub type CmpntMutRef<'a> = word_iters::ElemMutRef<'a, BitMatrix>;

impl<'a> AsRowRef for CmpntRef<'a> {
    fn bit_mat(&self) -> &impl AsBitMatrix {
        self.word_iters
    }
}

impl<'a> AsRowMutRef for CmpntMutRef<'a> {
    fn bit_mat(&self) -> &impl AsBitMatrix {
        self.word_iters
    }

    fn bit_mat_mut(&mut self) -> &mut impl AsBitMatrix {
        self.word_iters
    }
}

/// A single bit row.
#[derive(Clone)]
pub struct Row(BitMatrix);

impl Row {
    /// Borrow the single stored row as an immutable row view.
    pub fn borrow(&self) -> CmpntRef<'_> {
        CmpntRef {
            word_iters: &self.0,
            index: 0,
        }
    }

    /// Borrow the single stored row as a mutable row view.
    pub fn borrow_mut(&mut self) -> CmpntMutRef<'_> {
        CmpntMutRef {
            word_iters: &mut self.0,
            index: 0,
        }
    }

    /// Create a one-row component initialized to all-zero bits.
    pub fn new(n_bit: usize) -> Self {
        let mut this = Self(BitMatrix::new(n_bit));
        this.0.push_clear();
        this
    }
}

#[cfg(test)]
mod tests {
    use crate::container::traits::MutRefElements;

    use super::*;
    use rstest::rstest;

    #[test]
    fn test_empty() {
        {
            let v = BitMatrix::new(4);
            assert!(v.table.is_empty());
        }
    }

    #[test]
    fn test_get_set() {
        let mut v = BitMatrix::new(10);
        v.resize(1);
        for i in 0..v.n_bit() {
            assert!(!v.get_elem_ref(0).get_bit_unchecked(i));
            v.get_elem_mut_ref(0).set_bit_unchecked(i, true);
            assert!(v.get_elem_ref(0).get_bit_unchecked(i));
        }
    }

    #[rstest]
    #[case("[], [0, 0, 0, 1]", 
        vec![vec![0, 0, 0, 0], vec![0, 0, 0, 1]])]
    #[case("[], [0, 0, 0, 1], []",
        vec![vec![0, 0, 0, 0], vec![0, 0, 0, 1], vec![0, 0, 0, 0]])]
    #[case(
        "[0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0]",
        vec![vec![0, 1, 0, 1, 1, 0], vec![1, 0, 1, 1, 0, 0]]
    )]
    fn test_from_springs(#[case] input: &str, #[case] output: Vec<Vec<i32>>) {
        let springs = BinarySprings::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let cmpnts = BitMatrix::from_springs_default(&springs);
        let vecs = (0..cmpnts.len())
            .map(|i| {
                cmpnts
                    .get_elem_ref(i)
                    .iter()
                    .map(|x| if x { 1 } else { 0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(output, vecs);
    }

    #[test]
    fn test_to_string() {
        let mut v = BitMatrix::new(10);
        v.resize(1);

        v.get_elem_mut_ref(0).assign_vec_unchecked(
            vec![1, 0, 1, 1, 0, 1, 1, 1]
                .into_iter()
                .map(|x| x != 0)
                .collect(),
        );
        assert_eq!(
            v.get_elem_ref(0).to_string(),
            "[1, 0, 1, 1, 0, 1, 1, 1, 0, 0]"
        );
        assert_eq!(v.to_string(), "[[1, 0, 1, 1, 0, 1, 1, 1, 0, 0]]");
    }

    #[rstest]
    #[case(64, [0, 63].into())]
    #[case(100, [].into())]
    #[case(100, [0].into())]
    #[case(100, [0, 1].into())]
    #[case(100, [0, 99].into())]
    #[case(100, [0, 1, 2, 3, 33, 36, 49, 63, 64, 99].into())]
    #[case(1000, [0, 1, 2, 3, 10, 63, 64, 99, 199, 200, 300, 400, 410, 560, 800].into())]
    fn test_bit_iteration(#[case] n_bit: usize, #[case] set_bits: Vec<usize>) {
        let mut cmpnt = Row::new(n_bit);
        cmpnt
            .borrow_mut()
            .assign_set_unchecked(HashSet::from_iter(set_bits.iter().copied()));
        assert_eq!(cmpnt.borrow().hamming_weight(), set_bits.len());
        assert_eq!(
            cmpnt.borrow().to_set(),
            HashSet::from_iter(set_bits.iter().copied())
        );
        assert_eq!(
            set_bits,
            cmpnt.borrow().iter_set_bits_flat().collect::<Vec<_>>()
        );
        assert_eq!(
            set_bits.iter().copied().rev().collect::<Vec<_>>(),
            cmpnt.borrow().iter_set_bits_flat_rev().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_to_matrix() {
        let data: Vec<u8> = vec![1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1];
        let mat = Matrix::from_shape_vec((5, 4), data.clone()).unwrap();
        let mut cmpnt_list = BitMatrix::new(mat.shape()[1]);
        for row in data.chunks(mat.shape()[1]) {
            let vec = row.iter().map(|i| *i == 1).collect::<Vec<_>>();
            cmpnt_list.push_vec(vec).unwrap();
        }
        assert_eq!(cmpnt_list.to_matrix(), mat);
        assert_eq!(cmpnt_list.transpose().to_matrix(), mat.reversed_axes());
    }

    #[test]
    fn test_transpose_multiword() {
        let n_row = 70usize;
        let n_bit = 200usize;
        let mut data = vec![0u8; n_row * n_bit];

        for i_row in 0..n_row {
            for i_col in 0..n_bit {
                let dense_pattern = ((i_row * 17 + i_col * 31) % 11) < 4;
                let boundary_columns = matches!(
                    i_col,
                    0 | 1 | 2 | 62 | 63 | 64 | 65 | 126 | 127 | 128 | 129 | 190 | 198 | 199
                );
                if dense_pattern || boundary_columns {
                    data[i_row * n_bit + i_col] = 1;
                }
            }
        }

        let mat = Matrix::from_shape_vec((n_row, n_bit), data.clone()).unwrap();
        let mut cmpnt_list = BitMatrix::new(n_bit);
        for row in data.chunks(n_bit) {
            let vec = row.iter().map(|i| *i == 1).collect::<Vec<_>>();
            cmpnt_list.push_vec(vec).unwrap();
        }

        let transpose = cmpnt_list.transpose();
        assert_eq!(cmpnt_list.to_matrix(), mat);
        assert_eq!(transpose.to_matrix(), mat.reversed_axes());
        assert_eq!(transpose.transpose(), cmpnt_list);
    }

    #[test]
    fn test_reformat() {
        let mut bit_mat = BitMatrix::new(130);
        bit_mat.resize(2);

        for i_bit in [0, 1, 63, 64, 65, 127, 128, 129] {
            bit_mat.set_bit_unchecked(0, i_bit, true);
        }
        for i_bit in [2, 62, 66, 126] {
            bit_mat.set_bit_unchecked(1, i_bit, true);
        }

        bit_mat.reformat(200);
        assert_eq!(bit_mat.n_bit(), 200);
        assert_eq!(
            bit_mat.get_elem_ref(0).to_set(),
            [0, 1, 63, 64, 65, 127, 128, 129].into()
        );
        assert_eq!(bit_mat.get_elem_ref(1).to_set(), [2, 62, 66, 126].into());

        bit_mat.reformat(65);
        assert_eq!(bit_mat.n_bit(), 65);
        assert_eq!(bit_mat.get_elem_ref(0).to_set(), [0, 1, 63, 64].into());
        assert_eq!(bit_mat.get_elem_ref(1).to_set(), [2, 62].into());
    }
}
