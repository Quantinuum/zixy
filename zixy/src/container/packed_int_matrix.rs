use crate::container::table::Table;
use crate::utils::arith::divceil;

// A matrix of packed integers stored in a 'Table' buffer.

pub struct PackedIntMatrix {
    table: Table,
    n_bits: usize,
}

impl PackedIntMatrix {
    pub fn new(n_bits: usize, max_len: isize) -> Self {
        let row_size = divceil((n_bits as isize) * max_len, 64) as usize;
        Self {
            table: Table::new(row_size),
            n_bits,
        }
    }
}
