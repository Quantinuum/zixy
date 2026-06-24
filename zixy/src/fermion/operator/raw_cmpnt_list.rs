use crate::container::bit_matrix::AsBitMatrix;
use crate::container::bit_matrix::BitMatrix;
use crate::container::packed_int_matrix::PackedIntMatrix;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::word_iters::WordIters;
use crate::fermion::mode::Modes;

/// Contiguous and compact storage for non-normal-ordered fermion operator strings.
#[derive(Clone)]
pub struct RawCmpntList {
    pub mode_part: PackedIntMatrix, // mode index at each operator position
    pub adj_part: BitMatrix,        // cre/ann flag per slot
    pub len_part: Vec<u64>,         // length of each string
    pub modes: Modes,               // list of modes
    pub max_len: usize,             // max operator slots per row
    pub n_bits: usize,              // number of bits per mode index
}

impl RawCmpntList {
    /// Create a new empty `RawCmpntList` with the given mode space and maximum operator string length.
    pub fn new(n_bits: usize, max_len: usize, modes: Modes) -> Self {
        Self {
            mode_part: PackedIntMatrix::new(n_bits, max_len),
            adj_part: BitMatrix::new(max_len),
            len_part: Vec::new(),
            modes,
            max_len,
            n_bits,
        }
    }

    /// Push a new operator string defined by mode indices and cre/ann flags.
    /// `modes` and `adj` must have the same length.
    pub fn push(&mut self, modes: &[usize], adj: &[bool]) {
        self.mode_part.push_vec(modes);
        self.adj_part.push_clear();
        let last_row = self.adj_part.len() - 1;
        for (i, value) in adj.iter().enumerate() {
            self.adj_part.set_bit_unchecked(last_row, i, *value);
        }
        self.len_part.push(modes.len() as u64);
    }

    /// Return true if no operator strings are stored.
    pub fn is_empty(&self) -> bool {
        self.len_part.is_empty()
    }

    /// Read back the operator string at index `i` as a tuple of mode indices and cre/ann flags.
    pub fn get(&self, i: usize) -> (Vec<usize>, Vec<bool>) {
        let length = self.len_part[i] as usize;
        let modes = self.mode_part.read_row(i, length);
        let mut adj = Vec::new();
        for j in 0..length {
            adj.push(self.adj_part.get_bit_unchecked(i, j));
        }
        (modes, adj)
    }
}

impl Elements for RawCmpntList {
    fn len(&self) -> usize {
        self.len_part.len()
    }
}

impl Compatible for RawCmpntList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.modes == other.modes
    }
}

impl EmptyClone for RawCmpntList {
    fn empty_clone(&self) -> Self {
        Self::new(self.n_bits, self.max_len, self.modes.clone())
    }
}

impl WordIters for RawCmpntList {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self.mode_part
            .elem_u64it(i)
            .chain(self.adj_part.elem_u64it(i))
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.mode_part
            .elem_u64it_mut(i)
            .chain(self.adj_part.elem_u64it_mut(i))
    }

    fn u64it_size(&self) -> usize {
        self.mode_part.u64it_size() + self.adj_part.u64it_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_empty() {
        let v = RawCmpntList::new(8, 4, Modes::from_count(8));
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[rstest]
    #[case(&[0,1], &[false, false])]
    #[case(&[3, 1, 2], &[true, false, true])]
    #[case(&[0], &[true])]
    fn test_push_single(#[case] modes: &[usize], #[case] adj: &[bool]) {
        let mut v = RawCmpntList::new(8, 4, Modes::from_count(8));
        v.push(modes, adj);
        assert_eq!(v.len(), 1);
        let (out_modes, out_adj) = v.get(0);
        assert_eq!(out_modes, modes);
        assert_eq!(out_adj, adj);
    }

    #[test]
    fn test_push_multiple() {
        let mut v = RawCmpntList::new(8, 4, Modes::from_count(8));
        let inputs = vec![
            (vec![0, 1], vec![false, false]),
            (vec![3, 1, 2], vec![true, false, true]),
            (vec![0], vec![true]),
        ];
        for (modes, adj) in &inputs {
            v.push(modes, adj);
        }
        assert_eq!(v.len(), inputs.len());
        for (i, (modes, adj)) in inputs.iter().enumerate() {
            let (out_modes, out_adj) = v.get(i);
            assert_eq!(out_modes, *modes);
            assert_eq!(out_adj, *adj);
        }
    }
}
