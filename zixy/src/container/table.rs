//! Defines the basic container for many contiguously-stored fixed-width elements.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

use crate::container::{
    traits::{Compatible, Elements, EmptyClone},
    word_iters::{self, WordIters},
};

/// A basic contiguous buffer type for data of a fixed, runtime-determined width
/// `ndarray::Array2<u64>` was considered, but `Vec` is better for dynamic resizing
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Table {
    // buffer storing the rows
    pub buf: Vec<u64>,
    // size of each row in `u64` words, kept constant.
    row_size: usize,
    // number of rows, this is stored in case `row_size` is zero and hence number of rows is not given by buffer size
    n_row: usize,
}

impl Table {
    /// Create a new instance with the given number of rows.
    pub fn new(row_size: usize) -> Self {
        Self {
            buf: Vec::new(),
            row_size,
            n_row: 0,
        }
    }

    /// `Set` all `u64`s in the indexed row to zero.
    pub fn clear_row(&mut self, i_row: usize) {
        let i_begin = i_row * self.row_size;
        self.buf[i_begin..(i_begin + self.row_size)].fill(0);
    }

    /// Resize the table to zero rows, with the same `row_size`.
    pub fn clear(&mut self) {
        *self = Self::new(self.row_size)
    }

    /// Push a clear row to the end of self.
    pub fn push_clear(&mut self) {
        self.buf.resize(self.buf.len() + self.row_size, 0);
        self.n_row += 1;
    }

    /// Assign a new number of rows to self. If `n_row` is larger than `self.n_row`, the added rows will be clear.
    pub fn resize(&mut self, n_row: usize) {
        self.buf.resize(n_row * self.row_size, 0);
        self.n_row = n_row;
    }

    /// Reformat the table so each row has a different word width, preserving the prefix of each row.
    pub fn reformat(&mut self, new_row_size: usize) {
        if self.row_size == new_row_size {
            return;
        }

        let old_row_size = self.row_size;
        if new_row_size > old_row_size {
            self.buf.resize(self.n_row * new_row_size, 0);
            for i_row in (0..self.n_row).rev() {
                let i_src = i_row * old_row_size;
                let i_dst = i_row * new_row_size;
                for i_word in (0..old_row_size).rev() {
                    self.buf[i_dst + i_word] = self.buf[i_src + i_word];
                }
                for i_word in old_row_size..new_row_size {
                    self.buf[i_dst + i_word] = 0;
                }
            }
        } else {
            for i_row in 0..self.n_row {
                let i_src = i_row * old_row_size;
                let i_dst = i_row * new_row_size;
                for i_word in 0..new_row_size {
                    self.buf[i_dst + i_word] = self.buf[i_src + i_word];
                }
            }
            self.buf.truncate(self.n_row * new_row_size);
        }

        self.row_size = new_row_size;
    }

    /// Resize self to have one less row than it currently does if non-empty; else, do nothing.
    pub fn pop(&mut self) {
        self.resize(self.n_row.saturating_sub(1));
    }

    /// Return the row size.
    pub fn get_row_size(&self) -> usize {
        self.row_size
    }

    /// Copy the last element into the indexed row and shorten self by one element.
    pub fn pop_and_swap(&mut self, i_row: usize) {
        if i_row + 1 < self.len() {
            let i_src = self.buf.len() - self.row_size;
            let i_dst = i_row * self.row_size;
            for i in 0..self.row_size {
                self.buf[i_dst + i] = self.buf[i_src + i]
            }
        }
        self.pop();
    }
}

impl Compatible for Table {
    fn compatible_with(&self, other: &Self) -> bool {
        self.get_row_size() == other.get_row_size()
    }
}

impl Elements for Table {
    fn len(&self) -> usize {
        self.n_row
    }
}

impl EmptyClone for Table {
    fn empty_clone(&self) -> Self {
        Self::new(self.get_row_size())
    }
}

impl Index<usize> for Table {
    type Output = [u64];
    fn index(&self, index: usize) -> &Self::Output {
        let i_begin = index * self.row_size;
        &self.buf[i_begin..(i_begin + self.row_size)]
    }
}

impl IndexMut<usize> for Table {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i_begin = index * self.row_size;
        &mut self.buf[i_begin..(i_begin + self.row_size)]
    }
}

impl WordIters for Table {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self[i].iter().copied()
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self[i].iter_mut()
    }

    fn u64it_size(&self) -> usize {
        self.get_row_size()
    }

    fn pop_and_swap(&mut self, i_row: usize) {
        self.pop_and_swap(i_row);
    }

    fn fmt_elem(&self, i: usize) -> String {
        self.elem_u64it(i).join(", ")
    }

    fn resize(&mut self, n: usize) {
        self.resize(n);
    }
}

pub type ElemMutRef<'a> = word_iters::ElemMutRef<'a, Table>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slices() {
        let mut t = Table::new(4);
        t.push_clear();
        assert_eq!(t.len(), 1);
        t.push_clear();
        assert_eq!(t.len(), 2);
        t.push_clear();
        assert_eq!(t.len(), 3);
        t[0].copy_from_slice(vec![0, 1, 2, 3].as_slice());
        t[1].copy_from_slice(vec![4, 5, 6, 7].as_slice());
        t[2].copy_from_slice(vec![8, 9, 10, 11].as_slice());
        assert_eq!(t.buf, (0..12).collect::<Vec<u64>>());
        t.pop_and_swap(1);
        assert_eq!(t[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_serde() {
        let mut t = Table::new(4);
        t.resize(3);
        t[0].copy_from_slice(vec![0, 1, 2, 3].as_slice());
        t[1].copy_from_slice(vec![4, 5, 6, 7].as_slice());
        t[2].copy_from_slice(vec![8, 9, 10, 11].as_slice());
        let t_copy = t.clone();
        let v = bincode::serde::encode_to_vec(t, bincode::config::standard());
        assert!(v.is_ok());
        let v = v.unwrap();
        let t_decoded: Result<(Table, usize), bincode::error::DecodeError> =
            bincode::serde::decode_from_slice(v.as_slice(), bincode::config::standard());
        assert!(t_decoded.is_ok());
        let (t_decoded, size) = t_decoded.unwrap();
        assert_eq!(size, v.len());
        assert_eq!(t_copy, t_decoded);
    }
}
