//! Handles the encoding of Pauli words as bitsets within `u64` words.

use crate::qubit::mode::PauliMatrix;
use crate::utils::arith::divrem;

/// Assign the Pauli matrix value at the i_qubit position in the given X and Z bitsets
pub fn set_qubit(x_slice: &mut [u64], z_slice: &mut [u64], i_qubit: usize, pauli: PauliMatrix) {
    let (i_u64, i_bit) = divrem(i_qubit, 64);
    debug_assert!(i_bit < 64);
    let mask = 1u64 << i_bit;
    match pauli {
        PauliMatrix::I => {
            // clear both parts
            x_slice[i_u64] &= !mask;
            z_slice[i_u64] &= !mask;
        }
        PauliMatrix::X => {
            // set x part, clear z part
            x_slice[i_u64] |= mask;
            z_slice[i_u64] &= !mask;
        }
        PauliMatrix::Y => {
            // set both parts
            x_slice[i_u64] |= mask;
            z_slice[i_u64] |= mask;
        }
        PauliMatrix::Z => {
            // clear x part, set z part
            x_slice[i_u64] &= !mask;
            z_slice[i_u64] |= mask;
        }
    }
}

/// Get the Pauli matrix value encoded at the i_qubit position in the given X and Z bitsets
pub fn get_qubit(x_slice: &[u64], z_slice: &[u64], i_qubit: usize) -> PauliMatrix {
    let (i_u64, i_bit) = divrem(i_qubit, 64);
    debug_assert!(i_bit < 64);
    let mask = 1u64 << i_bit;
    if x_slice[i_u64] & mask != 0 {
        // x part set
        if z_slice[i_u64] & mask != 0 {
            // z part set
            PauliMatrix::Y
        } else {
            // z part clear
            PauliMatrix::X
        }
    } else {
        // x part clear
        if z_slice[i_u64] & mask != 0 {
            // z part set
            PauliMatrix::Z
        } else {
            // z part clear
            PauliMatrix::I
        }
    }
}

/// Invert the endianness of a single integer from little to big or big to little.
pub fn invert_endian(i: u64, n_bit: usize) -> u64 {
    i.reverse_bits() >> 64_usize.saturating_sub(n_bit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_endian() {
        for n_bit in [1, 10, 20, 23, 63, 64] {
            // check with individual set bits
            for i_bit in 0..n_bit {
                let inp = 1_u64 << i_bit;
                let out = 1_u64 << (n_bit - 1 - i_bit);
                assert_eq!(invert_endian(inp, n_bit), out);
                assert_eq!(invert_endian(out, n_bit), inp);
                // check with pairs of set bits
                for j_bit in 0..i_bit {
                    let inp = (1_u64 << i_bit) + (1_u64 << j_bit);
                    let out = (1_u64 << (n_bit - 1 - i_bit)) + (1_u64 << (n_bit - 1 - j_bit));
                    assert_eq!(invert_endian(inp, n_bit), out);
                    assert_eq!(invert_endian(out, n_bit), inp);
                }
            }
        }
    }
}
