//! A raw format for parsing Fermion operator string input from the user.

use itertools::Itertools;
use std::fmt::Display;

use crate::cmpnt::springs::*;
use crate::container::traits::Elements;
use crate::container::two_bit_vec::BitVec;

/// Basic storage for a vector of mode indices and the Pauli matrices associated with them.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct Springs {
    /// Indices of the fermion modes.
    mode_inds: ModeInds,
    /// Compact encoding of whether the fermionic operator is a creation operator (1) or an annihilation operator (0).
    cre_flags: BitVec,
}

impl Springs {
    /// Convert the `(byte, mode index)` output of `cmpnt::ModeSettings::get_iter` into an iterator over creation flags and mode indices.
    pub fn get_ladder_op_iter(
        &self,
        i_cmpnt: usize,
    ) -> impl Iterator<Item = (bool, ModeInd)> + use<'_> {
        self.get_iter(i_cmpnt).map(|(k, i_mode)| (k == 1, i_mode))
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
        self.cre_flags.push(setting == 1)
    }

    fn get_setting(&self, i: usize) -> u8 {
        self.cre_flags.get_unchecked(i) as u8
    }

    fn push_str(&mut self, s: &str) -> Result<(), BadParse> {
        // validate first
        for s in s.split_whitespace() {
            let bytes = s.as_bytes();
            let has_caret = bytes.last().is_some_and(|b| *b == b'^');
            let digits = if has_caret {
                &s[1..s.len().saturating_sub(1)]
            } else {
                &s[1..]
            };
            if bytes.is_empty()
                || bytes[0] != b'F'
                || digits.is_empty()
                || !digits.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(BadParse(format!(
                    "\"{s}\" is not a valid fermionic ladder operator in a sparse string.",
                )));
            }
        }
        self.push(
            s.split_whitespace().map(|s| {
                let is_creation = s.ends_with('^');
                let digits = if is_creation {
                    &s[1..s.len() - 1]
                } else {
                    &s[1..]
                };
                (u8::from(is_creation), digits.parse::<ModeInd>().unwrap())
            }),
            true,
        );
        Ok(())
    }

    fn to_string_ind(&self, i_cmpnt: usize) -> String {
        self.get_ladder_op_iter(i_cmpnt)
            .map(|(p, i)| if p { format!("F{i}^") } else { format!("F{i}") })
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
    #[case("F5 F6^ some junk")]
    #[case("12")]
    #[case("W4")]
    #[case("F-1")]
    #[case("F0.324")]
    #[case(" F1 F3^ FX")]
    #[case("F")]
    #[case("F^")]
    #[case("F1^^")]
    fn test_invalid(#[case] s: &str) {
        assert!(Springs::from_str(s).is_err());
    }

    #[rstest]
    #[case("F1", vec![(false, 1)], 2)]
    #[case("F1 F5^ F4", vec![(false, 1), (true, 5), (false, 4)], 6)]
    #[case("F1 F5^ F6 F13^", vec![(false, 1), (true, 5), (false, 6), (true, 13)], 14)]
    #[case("F1 F5^ F6 F13^,", vec![(false, 1), (true, 5), (false, 6), (true, 13)], 14)]
    #[case("F1 F5^ F6 F13^, ", vec![(false, 1), (true, 5), (false, 6), (true, 13)], 14)]
    fn test_valid_single_part(
        #[case] s: &str,
        #[case] v: Vec<(bool, ModeInd)>,
        #[case] default_n_mode: ModeInd,
    ) {
        let sparse_strings = Springs::from_str(s);
        assert!(sparse_strings.is_ok_and(|x| {
            x.len() == 1
                && x.get_ladder_op_iter(0).collect::<Vec<_>>() == v
                && x.mode_inds.default_n_mode() == default_n_mode
        }));
    }

    #[rstest]
    #[case(" ", vec![])]
    #[case(" *", vec![vec![], vec![]])]
    #[case("F1 F5^ F4", vec![vec![(false, 1), (true, 5), (false, 4)]])]
    fn test_valid_multi_part(#[case] s: &str, #[case] v: Vec<Vec<(bool, ModeInd)>>) {
        assert!(Springs::all_parts_from_str(s).is_ok_and(|x| {
            x.len() == v.len()
                && (0..v.len()).all(|i| x[i].get_ladder_op_iter(0).collect::<Vec<_>>() == v[i])
        }));
    }
}
