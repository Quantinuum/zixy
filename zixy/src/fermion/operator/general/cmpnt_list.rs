use crate::container::bit_matrix::AsBitMatrix;
use crate::container::bit_matrix::BitMatrix;
use crate::container::map::Map;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::word_iters::set::{AsView, View};
use crate::container::word_iters::WordIters;
use crate::fermion::mode::ModeInds;
use crate::fermion::mode::Modes;
use crate::fermion::traits::ModesBased;

/// Contiguous and compact storage for non-normal-ordered fermion operator strings.
#[derive(Clone)]
pub struct CmpntList {
    pub mode_part: ModeInds, // mode index at each operator position
    pub adj_part: BitMatrix, // cre/ann flag per slot
    pub len_part: Vec<u64>,  // length of each string, included in packed keys
    pub modes: Modes,        // list of modes
    pub max_len: usize,      // max operator slots per row
    pub n_bits: usize,       // number of bits per mode index
}

impl CmpntList {
    /// Grow to the required string length without changing existing strings.
    pub fn reserve_string_length(&mut self, required: usize) {
        if required <= self.max_len {
            return;
        }
        let mut out = Self::new(required, self.modes.clone());
        for i in 0..self.len() {
            let (modes, adj) = self.get(i);
            out.push(&modes, &adj);
        }
        *self = out;
    }

    /// Replace one string, growing storage only when necessary.
    pub fn set(&mut self, row: usize, modes: &[usize], adj: &[bool]) {
        assert_eq!(modes.len(), adj.len());
        assert!(row < self.len());
        self.reserve_string_length(modes.len());
        self.mode_part.set_row(row, modes);
        self.adj_part.clear_row(row);
        for (i, value) in adj.iter().enumerate() {
            self.adj_part.set_bit_unchecked(row, i, *value);
        }
        self.len_part[row] = modes.len() as u64;
    }

    /// Look up a logical string regardless of the two arrays' storage capacities.
    pub fn lookup(&self, map: &Map, other: &Self, index: usize) -> Option<usize> {
        if other.len_part[index] as usize > self.max_len {
            return None;
        }
        // Normalize padding to our row width without allocating or changing either array.
        let modes = other
            .mode_part
            .elem_u64it(index)
            .chain(std::iter::repeat(0))
            .take(self.mode_part.u64it_size());
        let adj = other
            .adj_part
            .elem_u64it(index)
            .chain(std::iter::repeat(0))
            .take(self.adj_part.u64it_size());
        View {
            word_iters: self,
            map,
        }
        .lookup(
            modes
                .chain(adj)
                .chain(std::iter::once(other.len_part[index])),
        )
    }

    /// Create a new empty non-normal-ordered `CmpntList` with the given mode space and maximum operator string length.
    pub fn new(max_len: usize, modes: Modes) -> Self {
        let n_modes = modes.len();
        let n_bits = if n_modes <= 1 {
            1
        } else {
            (usize::BITS as usize) - (n_modes - 1).leading_zeros() as usize
        };
        Self {
            mode_part: ModeInds::new(n_bits, max_len),
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
        assert_eq!(
            modes.len(),
            adj.len(),
            "modes and adj must have the same length"
        );
        self.reserve_string_length(modes.len());
        self.mode_part.push_vec(modes);
        self.adj_part.push_clear();
        let last_row = self.adj_part.len() - 1;
        for (i, value) in adj.iter().enumerate() {
            self.adj_part.set_bit_unchecked(last_row, i, *value);
        }
        self.len_part.push(modes.len() as u64);
    }

    pub fn push_concat(
        &mut self,
        lhs_modes: &[usize],
        lhs_adj: &[bool],
        rhs_modes: &[usize],
        rhs_adj: &[bool],
    ) {
        assert_eq!(
            lhs_modes.len(),
            lhs_adj.len(),
            "lhs modes and adj must have the same length"
        );
        assert_eq!(
            rhs_modes.len(),
            rhs_adj.len(),
            "rhs modes and adj must have the same length"
        );
        self.reserve_string_length(lhs_modes.len() + rhs_modes.len());
        self.mode_part
            .push_iter(lhs_modes.iter().chain(rhs_modes.iter()).copied());
        self.adj_part.push_clear();
        let last_row = self.adj_part.len() - 1;
        for (i, value) in lhs_adj.iter().chain(rhs_adj.iter()).enumerate() {
            self.adj_part.set_bit_unchecked(last_row, i, *value);
        }
        self.len_part
            .push((lhs_modes.len() + rhs_modes.len()) as u64);
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

impl Elements for CmpntList {
    fn len(&self) -> usize {
        self.len_part.len()
    }
}

impl Compatible for CmpntList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.modes == other.modes
    }
}

impl EmptyClone for CmpntList {
    fn empty_clone(&self) -> Self {
        Self::new(self.max_len, self.modes.clone())
    }
}

impl WordIters for CmpntList {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self.mode_part
            .elem_u64it(i)
            .chain(self.adj_part.elem_u64it(i))
            .chain(std::iter::once(self.len_part[i]))
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.mode_part
            .elem_u64it_mut(i)
            .chain(self.adj_part.elem_u64it_mut(i))
            .chain(std::iter::once(&mut self.len_part[i]))
    }

    fn u64it_size(&self) -> usize {
        self.mode_part.u64it_size() + self.adj_part.u64it_size() + 1
    }

    fn pop_and_swap(&mut self, index: usize) {
        self.mode_part.pop_and_swap(index);
        self.adj_part.pop_and_swap(index);
        let last = self.len_part.len() - 1;
        self.len_part.swap(index, last);
        self.len_part.pop();
    }

    fn resize(&mut self, n: usize) {
        self.mode_part.resize(n);
        self.adj_part.resize(n);
        self.len_part.resize(n, 0);
    }

    /// Format the operator string at index `i` as a human-readable string.
    fn fmt_elem(&self, i: usize) -> String {
        let (modes, adj) = self.get(i);
        modes
            .iter()
            .zip(adj.iter())
            .map(|(mode, is_cre)| format!("F{}{}", mode, if *is_cre { "^" } else { "" }))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl ModesBased for CmpntList {
    fn modes(&self) -> &Modes {
        &self.modes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1)]
    #[case(3)]
    #[case(65)]
    #[case(257)]
    fn test_growth_and_assignment(#[case] n_modes: usize) {
        let mut strings = CmpntList::new(0, Modes::from_count(n_modes));
        strings.push(&[], &[]);
        strings.push(&[0], &[true]);
        for length in [2, 11, 64, 65, 129] {
            let modes = vec![n_modes - 1; length];
            let adj = (0..length).map(|i| i % 2 == 0).collect::<Vec<_>>();
            strings.set(0, &modes, &adj);
            assert_eq!(strings.get(0), (modes, adj));
            assert_eq!(strings.get(1), (vec![0], vec![true]));
            assert_eq!(strings.max_len, length);
        }
        strings.set(0, &[], &[]);
        assert_eq!(strings.max_len, 129);
        assert_eq!(strings.get(0), (vec![], vec![]));
        strings.push_concat(&[0; 100], &[false; 100], &[0; 100], &[true; 100]);
        assert_eq!(strings.max_len, 200);
        assert_eq!(strings.get(2).0.len(), 200);
    }

    #[test]
    fn test_keys_preserve_length_and_capacity_independence() {
        use crate::container::word_iters::set::AsViewMut;
        let mut strings = CmpntList::new(2, Modes::from_count(2));
        strings.push(&[], &[]);
        strings.push(&[0], &[false]);
        strings.push(&[0, 0], &[false, false]);
        let mut map = Map::default();
        map.populate_from(&strings);
        View {
            word_iters: &strings,
            map: &map,
        }
        .consistency_check();
        let mut other = CmpntList::new(129, strings.modes.clone());
        other.push(&[0], &[false]);
        assert_eq!(strings.lookup(&map, &other, 0), Some(1));
        strings.reserve_string_length(65);
        map.populate_from(&strings);
        assert_eq!(strings.lookup(&map, &other, 0), Some(1));
        crate::container::word_iters::set::ViewMut {
            word_iters: &mut strings,
            map: &mut map,
        }
        .drop(1);
        assert_eq!(strings.get(1), (vec![0, 0], vec![false, false]));
        View {
            word_iters: &strings,
            map: &map,
        }
        .consistency_check();
    }

    #[test]
    fn test_empty() {
        let v = CmpntList::new(4, Modes::from_count(8));
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[rstest]
    #[case(&[0,1], &[false, false])]
    #[case(&[3, 1, 2], &[true, false, true])]
    #[case(&[0], &[true])]
    fn test_push_single(#[case] modes: &[usize], #[case] adj: &[bool]) {
        let mut v = CmpntList::new(4, Modes::from_count(8));
        v.push(modes, adj);
        assert_eq!(v.len(), 1);
        let (out_modes, out_adj) = v.get(0);
        assert_eq!(out_modes, modes);
        assert_eq!(out_adj, adj);
    }

    #[test]
    fn test_push_multiple() {
        let mut v = CmpntList::new(4, Modes::from_count(8));
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
