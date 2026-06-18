use crate::container::table::Table;
use crate::container::traits::Elements;
use crate::utils::arith::divceil;

// A matrix of packed integers stored in a 'Table' buffer.

pub struct PackedIntMatrix {
    table: Table,
    n_bits: usize,
}

impl PackedIntMatrix {
    /// Create a new empty row
    pub fn new(n_bits: usize, max_len: isize) -> Self {
        let row_size = divceil((n_bits as isize) * max_len, 64) as usize;
        Self {
            table: Table::new(row_size),
            n_bits,
        }
    }

    /// Calculate which u64 and bit offset within it corresponds to slot `i_slot`.
    pub fn get_offset(&self, i_slot: usize) -> (usize, usize) {
        let bit_pos = i_slot * self.n_bits;
        let i_u64 = bit_pos / 64;
        let bit_offset = bit_pos % 64;
        (i_u64, bit_offset)
    }

    /// Push a new row of packed integers into the matrix.
    pub fn push_vec(&mut self, values: &[usize]) {
        self.table.push_clear();
        let last_row = self.table.len() - 1;
        for (i, value) in values.iter().enumerate() {
            let (i_u64, bit_offset) = self.get_offset(i);
            self.table[last_row][i_u64] |= (*value as u64) << bit_offset;
        }
    }

    /// Read back integer stored at slot `i_slot` in row `i_row` by reversing the packing from `push_vec`.
    pub fn get_value(&self, i_row: usize, i_slot: usize) -> usize {
        let (i_u64, bit_offset) = self.get_offset(i_slot);
        // Extra safety measure to ensure only the n_bit bits for this slot
        let mask = (1u64 << self.n_bits) - 1;
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
}
