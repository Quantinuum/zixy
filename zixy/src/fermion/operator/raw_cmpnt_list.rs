use crate::container::bit_matrix::BitMatrix;
use crate::container::packed_int_matrix::PackedIntMatrix;
use crate::fermion::mode::Modes;

/// Contiguous and compact storage for non-normal-ordered fermion operator strings.
pub struct RawCmpntList {
    pub mode_part: PackedIntMatrix, // mode index at each operator position
    pub adj_part: BitMatrix,        // cre/ann flag per slot
    pub len_part: Vec<u64>,         // length of each string
    pub modes: Modes,               // list of modes
    pub max_len: usize,             // max operator slots per row
}
