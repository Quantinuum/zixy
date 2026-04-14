//! Provides multiplication operations for qubit Pauli operator strings and states.
//! In these functions, l and r stand for the left and right hand side operands in the multiplication
//! and x and z stand for the X and Z parts of the symplectic repesentation of the Pauli words.
//!
//! Multiplication operations can be out-of- or in-place and are denoted by:
//! - mul(l, r, o): multiply l by r and store the result in o.
//! - imul(l, r): multiply l by r, storing the result in l.

use itertools::izip;

use crate::container::coeffs::complex_sign::ComplexSign;

/// Compute the phase associated with the multiplication of two u64-pair Pauli strings.
pub fn mul_op_op_phase_u64(lx: u64, lz: u64, rx: u64, rz: u64) -> ComplexSign {
    ComplexSign::from(
        // XY = iZ
        (lx & !lz & rx & rz).count_ones() +
        // YX = -iZ
        3 * (lx & lz & rx & !rz).count_ones() +
        // XZ = -iY
        3 * (lx & !lz & !rx & rz).count_ones() +
        // ZX = iY
        (!lx & lz & rx & !rz).count_ones() +
        // YZ = iX
        (lx & lz & !rx & rz).count_ones() +
        // ZY = -iX
        3 * (!lx & lz & rx & rz).count_ones(),
    )
}

/// Compute the phase associated with the multiplication of two Pauli strings.
pub fn mul_op_op_phase(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    rx: impl Iterator<Item = u64> + Clone,
    rz: impl Iterator<Item = u64> + Clone,
) -> ComplexSign {
    ComplexSign::from(
        izip!(lx, lz, rx, rz)
            .map(|(lx, lz, rx, rz)| mul_op_op_phase_u64(lx, lz, rx, rz).0 as u32)
            .sum::<u32>(),
    )
}

/// Compute the u64-pair Pauli string resulting from the multiplication of two others.
pub fn mul_op_op_matrices_u64(lx: u64, lz: u64, rx: u64, rz: u64) -> (u64, u64) {
    (lx ^ rx, lz ^ rz)
}

/// Compute the Pauli string resulting from the multiplication of two others.
pub fn mul_op_op_matrices<'a>(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    rx: impl Iterator<Item = u64> + Clone,
    rz: impl Iterator<Item = u64> + Clone,
    ox: impl Iterator<Item = &'a mut u64>,
    oz: impl Iterator<Item = &'a mut u64>,
) {
    izip!(lx, rx, ox).for_each(|(l, r, o)| *o = l ^ r);
    izip!(lz, rz, oz).for_each(|(l, r, o)| *o = l ^ r);
}

/// Compute the u64-pair Pauli string resulting from the multiplication of two others along with the phase.
pub fn mul_op_op_u64(lx: u64, lz: u64, rx: u64, rz: u64) -> ((u64, u64), ComplexSign) {
    (
        mul_op_op_matrices_u64(lx, lz, rx, rz),
        mul_op_op_phase_u64(lx, lz, rx, rz),
    )
}

/// Compute the Pauli string resulting from the multiplication of two others along with the phase.
pub fn mul_op_op<'a>(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    rx: impl Iterator<Item = u64> + Clone,
    rz: impl Iterator<Item = u64> + Clone,
    ox: impl Iterator<Item = &'a mut u64>,
    oz: impl Iterator<Item = &'a mut u64>,
) -> ComplexSign {
    mul_op_op_matrices(lx.clone(), lz.clone(), rx.clone(), rz.clone(), ox, oz);
    mul_op_op_phase(lx, lz, rx, rz)
}

/// Compute the Pauli string resulting from the multiplication of two others along with the phase.
pub fn mul_op_ops<'a>(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    rx: impl Iterator<Item = u64> + Clone,
    rz: impl Iterator<Item = u64> + Clone,
    ox: impl Iterator<Item = &'a mut u64>,
    oz: impl Iterator<Item = &'a mut u64>,
) -> ComplexSign {
    mul_op_op_matrices(lx.clone(), lz.clone(), rx.clone(), rz.clone(), ox, oz);
    mul_op_op_phase(lx, lz, rx, rz)
}

/// Right-multiply a Pauli string in-place by another ignoring phase.
pub fn imul_op_op_matrices<'a>(
    lx: impl Iterator<Item = &'a mut u64>,
    lz: impl Iterator<Item = &'a mut u64>,
    rx: impl Iterator<Item = u64>,
    rz: impl Iterator<Item = u64>,
) {
    izip!(lx, lz, rx, rz).for_each(|(lx, lz, rx, rz)| {
        (*lx, *lz) = mul_op_op_matrices_u64(*lx, *lz, rx, rz);
    });
}

/// Right-multiply a Pauli string in-place by another and return the phase.
pub fn imul_op_op<'a>(
    lx: impl Iterator<Item = &'a mut u64>,
    lz: impl Iterator<Item = &'a mut u64>,
    rx: impl Iterator<Item = u64>,
    rz: impl Iterator<Item = u64>,
) -> ComplexSign {
    ComplexSign::from(
        izip!(lx, lz, rx, rz)
            .map(|(lx, lz, rx, rz)| {
                let phase = mul_op_op_phase_u64(*lx, *lz, rx, rz).0 as u32;
                (*lx, *lz) = mul_op_op_matrices_u64(*lx, *lz, rx, rz);
                phase
            })
            .sum::<u32>(),
    )
}

/// Compute the phase associated with the right-multiplication of a u64-pair Pauli string by a single-u64 state.
pub fn mul_op_state_phase_u64(lx: u64, lz: u64, r: u64) -> ComplexSign {
    ComplexSign::from(
        // Y|0> = i|1>
        (lx & lz & !r).count_ones() +
        // Y|1> = -i|0>
        3 * (lx & lz & r).count_ones() +
        // Z|1> = -|1>
        2 * (!lx & lz & r).count_ones(),
    )
}

/// Compute the phase associated with the right-multiplication of a Pauli string by a state.
pub fn mul_op_state_phase(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    r: impl Iterator<Item = u64> + Clone,
) -> ComplexSign {
    ComplexSign::from(
        izip!(lx, lz, r)
            .map(|(lx, lz, r)| mul_op_state_phase_u64(lx, lz, r).0 as u32)
            .sum::<u32>(),
    )
}

/// Compute the single-u64 state resulting from the right-multiplication of a u64-pair Pauli string by a single-u64 state.
pub fn mul_op_state_bits_u64(lx: u64, r: u64) -> u64 {
    lx ^ r
}

/// Compute the state resulting from the right-multiplication of a Pauli string by a state.
pub fn mul_op_state_bits<'a>(
    lx: impl Iterator<Item = u64> + Clone,
    r: impl Iterator<Item = u64> + Clone,
    o: impl Iterator<Item = &'a mut u64>,
) {
    izip!(lx, r, o).for_each(|(l, r, o)| *o = l ^ r)
}

/// Compute the single-u64 state resulting from the right-multiplication of a u64-pair Pauli string by a single-u64 state
/// along with the phase.
pub fn mul_op_state_u64(lx: u64, lz: u64, r: u64) -> (u64, ComplexSign) {
    (
        mul_op_state_bits_u64(lx, r),
        mul_op_state_phase_u64(lx, lz, r),
    )
}

/// Compute the state resulting from the right-multiplication of a Pauli string by a state along with the phase.
pub fn mul_op_state<'a>(
    lx: impl Iterator<Item = u64> + Clone,
    lz: impl Iterator<Item = u64> + Clone,
    r: impl Iterator<Item = u64> + Clone,
    o: impl Iterator<Item = &'a mut u64>,
) -> ComplexSign {
    mul_op_state_bits(lx.clone(), r.clone(), o);
    mul_op_state_phase(lx, lz, r)
}

/// Left-multiply a Pauli string in-place by another and return the phase.
pub fn imul_op_state<'a>(
    lx: impl Iterator<Item = u64>,
    lz: impl Iterator<Item = u64>,
    r: impl Iterator<Item = &'a mut u64>,
) -> ComplexSign {
    ComplexSign::from(
        izip!(lx, lz, r)
            .map(|(lx, lz, r)| {
                let phase = mul_op_state_phase_u64(lx, lz, *r).0 as u32;
                *r = mul_op_state_bits_u64(lx, *r);
                phase
            })
            .sum::<u32>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_op_state_randomized() {
        // result of op * state should agree with op * op with the RHS having X at the set positions in state, followed by
        // projection of the result on the vacuum.
        const N_DRAW: usize = 1 << 10;
        const ROW_SIZE: usize = 6;
        let mut op_x = [0_u64; ROW_SIZE];
        let mut op_z = [0_u64; ROW_SIZE];
        let mut state_in = [0_u64; ROW_SIZE];
        let mut state_out = [0_u64; ROW_SIZE];

        let seed = [45u8; 32];
        let mut rng = StdRng::from_seed(seed);

        for _ in 0..N_DRAW {
            // initialize the op and state with random u64 ints
            op_x.iter_mut().for_each(|x| *x = rng.random::<u64>());
            op_z.iter_mut().for_each(|x| *x = rng.random::<u64>());
            state_in.iter_mut().for_each(|x| *x = rng.random::<u64>());

            op_x.iter_mut().for_each(|x| *x = rng.random::<u8>() as u64);
            op_z.iter_mut().for_each(|x| *x = rng.random::<u8>() as u64);
            state_in
                .iter_mut()
                .for_each(|x| *x = rng.random::<u8>() as u64);

            let phase = mul_op_state(
                op_x.iter().copied(),
                op_z.iter().copied(),
                state_in.iter().copied(),
                state_out.iter_mut(),
            );

            // let state pose as the x channel of a pauli operator string.
            let phase_op_x = imul_op_op(
                op_x.iter_mut(),
                op_z.iter_mut(),
                state_in.iter().copied(),
                std::iter::repeat_n(0_u64, ROW_SIZE),
            );
            assert_eq!(op_x, state_out);
            // then project the result onto the vacuum state to get the other contribution to total phase.
            let phase_op_x_state = mul_op_state_phase(
                op_x.iter().copied(),
                op_z.iter().copied(),
                std::iter::repeat_n(0_u64, ROW_SIZE),
            );
            assert_eq!(phase.0, (phase_op_x.0 + phase_op_x_state.0) & 3);
        }
    }

    #[test]
    fn test_imul_randomized() {
        const N_DRAW: usize = 1 << 10;
        const ROW_SIZE: usize = 4;
        let mut lhs_x = [0_u64; ROW_SIZE];
        let mut lhs_z = [0_u64; ROW_SIZE];
        let mut rhs_x = [0_u64; ROW_SIZE];
        let mut rhs_z = [0_u64; ROW_SIZE];
        let mut out_x = [0_u64; ROW_SIZE];
        let mut out_z = [0_u64; ROW_SIZE];

        let seed = [45u8; 32];
        let mut rng = StdRng::from_seed(seed);

        for _ in 0..N_DRAW {
            // initialize the LHS and RHS with random u64 ints
            lhs_x.iter_mut().for_each(|x| *x = rng.random::<u64>());
            lhs_z.iter_mut().for_each(|x| *x = rng.random::<u64>());
            rhs_x.iter_mut().for_each(|x| *x = rng.random::<u64>());
            rhs_z.iter_mut().for_each(|x| *x = rng.random::<u64>());
            let phase = mul_op_op(
                lhs_x.iter().copied(),
                lhs_z.iter().copied(),
                rhs_x.iter().copied(),
                rhs_z.iter().copied(),
                out_x.iter_mut(),
                out_z.iter_mut(),
            );
            // compute LHS *= RHS and check that the phase is the same as the out-of-place.
            assert_eq!(
                imul_op_op(
                    lhs_x.iter_mut(),
                    lhs_z.iter_mut(),
                    rhs_x.iter().copied(),
                    rhs_z.iter().copied(),
                ),
                phase
            );
        }
    }
}
