//! A raw format for parsing Pauli string input from the user, and for converting between mode orderings and mode spaces.

use itertools::Itertools;
use std::fmt::Display;

use crate::cmpnt::springs::*;
use crate::container::traits::Elements;
use crate::container::two_bit_vec::TwoBitVec;
use crate::qubit::mode::PauliMatrix;

/// Basic storage for a vector of mode indices and the Pauli matrices associated with them.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct Springs {
    /// Indices of the qubit modes.
    mode_inds: ModeInds,
    /// Compact encoding of the settings associated with each mode index, i.e. I, X, Y, Z in this case.
    /// These four matrices are representable in 2-bit integers.
    settings: TwoBitVec,
}

impl Springs {
    /// Convert the `(byte, mode index)` output of `cmpnt::ModeSettings::get_iter` into an iterator over Pauli matrices and mode indices.
    pub fn get_pauli_iter(
        &self,
        i_cmpnt: usize,
    ) -> impl Iterator<Item = (PauliMatrix, ModeInd)> + use<'_> {
        self.get_iter(i_cmpnt).map(|(k, i_mode)| {
            (
                match k {
                    0 => PauliMatrix::I,
                    1 => PauliMatrix::X,
                    2 => PauliMatrix::Y,
                    _ => PauliMatrix::Z,
                },
                i_mode,
            )
        })
    }
}

impl Elements for Springs {
    fn len(&self) -> usize {
        self.mode_inds.len()
    }
}

impl ModeSettings for Springs {
    fn get_mode_inds(&self) -> &ModeInds {
        &self.mode_inds
    }

    fn get_mode_inds_mut(&mut self) -> &mut ModeInds {
        &mut self.mode_inds
    }

    fn push_setting(&mut self, setting: u8) {
        self.settings.push(setting)
    }

    fn get_setting(&self, i: usize) -> u8 {
        self.settings.get_unchecked(i)
    }

    fn push_str(&mut self, s: &str) -> Result<(), BadParse> {
        // validate first
        for s in s.split_whitespace() {
            if s.len() < 2
                || !matches!(&s[..1], "I" | "X" | "Y" | "Z")
                || s[1..].parse::<ModeInd>().is_err()
            {
                return Err(BadParse(format!(
                    "\"{s}\" is not a valid Pauli matrix in a sparse string.",
                )));
            }
        }
        self.push_no_zeros(s.split_whitespace().map(|s| {
            (
                match s.bytes().next().unwrap() as char {
                    'I' => 0,
                    'X' => 1,
                    'Y' => 2,
                    'Z' => 3,
                    _ => unreachable!("verified to be in (I, X, Y, Z)"),
                },
                s[1..].parse::<ModeInd>().unwrap(),
            )
        }));
        Ok(())
    }

    fn to_string_ind(&self, i_cmpnt: usize) -> String {
        self.get_pauli_iter(i_cmpnt)
            .map(|(p, i)| format!("{p}{i}"))
            .join(" ")
    }
}

impl Display for Springs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.len()).map(|i| self.to_string_ind(i)).join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::mode::PauliMatrix::*;
    use rstest::rstest;

    #[rstest]
    #[case("", 0)]
    #[case(" ", 0)]
    #[case(",", 1)]
    #[case(" ,", 1)]
    #[case(",  ", 1)]
    #[case(",, ", 2)]
    #[case(" ,,", 2)]
    #[case(",  ,", 2)]
    #[case(", , ", 2)]
    #[case(", ,, ", 3)]
    #[case(", ,,, ", 4)]
    #[case(" , ,,, ", 4)]
    fn test_empty_lens(#[case] s: &str, #[case] l: usize) {
        let sparse_strings = Springs::from_str(s);
        assert!(sparse_strings.is_ok());
        let sparse_strings = sparse_strings.unwrap();
        assert_eq!(sparse_strings.len(), l);
    }

    #[rstest]
    #[case("-")]
    #[case("X5 X6 Z19 some junk")]
    #[case("12")]
    #[case("W4")]
    #[case("X-1")]
    #[case("X0.324")]
    #[case(" X1 X3 YX")]
    fn test_invalid(#[case] s: &str) {
        assert!(Springs::from_str(s).is_err());
    }

    #[rstest]
    #[case("I1", vec![], 2)]
    #[case("X1 Y5 Z4", vec![(X, 1), (Y, 5), (Z, 4)], 6)]
    #[case("X1 Y5 Z6 I13", vec![(X, 1), (Y, 5), (Z, 6)], 14)]
    #[case("X1 Y5 Z6 I13,", vec![(X, 1), (Y, 5), (Z, 6)], 14)]
    #[case("X1 Y5 Z6 I13, ", vec![(X, 1), (Y, 5), (Z, 6)], 14)]
    fn test_valid_single_part(
        #[case] s: &str,
        #[case] v: Vec<(PauliMatrix, ModeInd)>,
        #[case] default_n_qubit: ModeInd,
    ) {
        let sparse_strings = Springs::from_str(s);
        assert!(sparse_strings.is_ok_and(|x| {
            x.len() == 1
                && x.get_pauli_iter(0).collect::<Vec<_>>() == v
                && x.mode_inds.default_n_mode() == default_n_qubit
        }));
    }

    #[rstest]
    #[case(" ", vec![])]
    #[case(" *", vec![vec![], vec![]])]
    #[case("X1 Y5 Z4", vec![vec![(X, 1), (Y, 5), (Z, 4)]])]
    fn test_valid_multi_part(#[case] s: &str, #[case] v: Vec<Vec<(PauliMatrix, ModeInd)>>) {
        assert!(Springs::all_parts_from_str(s).is_ok_and(|x| {
            x.len() == v.len()
                && (0..v.len()).all(|i| x[i].get_pauli_iter(0).collect::<Vec<_>>() == v[i])
        }));
    }
}
