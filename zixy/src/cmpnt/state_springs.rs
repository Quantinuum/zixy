//! Mode settings that are specific to quantum mechanical basis state vectors.

use std::fmt::Display;

use crate::{cmpnt::springs::*, container::traits::Elements};
use itertools::Itertools;

/// Basic storage for a vector of mode indices and the local state indices associated with them.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct StateSprings {
    /// Mode index strings.
    mode_inds: ModeInds,
    /// Local mode states.
    settings: Vec<u8>,
}

impl StateSprings {
    /// Push from the given string, and check the settings do not exceed max_setting value.
    fn push_str_with_max(&mut self, s: &str, max_setting: u8) -> Result<(), BadParse> {
        let s_trim = s.trim();
        if !s_trim.starts_with('[') || !s_trim.ends_with(']') {
            return Err(BadParse(
                "State string should be between square brackets.".to_string(),
            ));
        }
        let s_trim = s_trim[1..(s_trim.len() - 1)].trim();
        if s_trim.is_empty() {
            self.append_empty(1);
            return Ok(());
        }
        // one trailing comma is allowed
        let s_trim = if let Some(stripped) = s_trim.strip_suffix(',') {
            stripped
        } else {
            s_trim
        };
        // validate first
        for s in s_trim.split(',') {
            if !s.trim().parse::<u8>().is_ok_and(|x| x <= max_setting) {
                return Err(BadParse(format!(
                    "State setting \"{}\" is unparsable or out of bounds.",
                    s.trim()
                )));
            }
        }
        self.push_no_zeros(
            s_trim
                .split(',')
                .enumerate()
                .map(|(i, setting)| (setting.trim().parse::<u8>().unwrap(), i as ModeInd)),
        );
        Ok(())
    }
}

impl Elements for StateSprings {
    fn len(&self) -> usize {
        self.mode_inds.len()
    }
}

impl ModeSettings for StateSprings {
    fn get_mode_inds(&self) -> &ModeInds {
        &self.mode_inds
    }

    fn get_mode_inds_mut(&mut self) -> &mut ModeInds {
        &mut self.mode_inds
    }

    fn push_setting(&mut self, setting: u8) {
        self.settings.push(setting);
    }

    fn get_setting(&self, i: usize) -> u8 {
        self.settings[i]
    }

    fn push_str(&mut self, s: &str) -> Result<(), BadParse> {
        self.push_str_with_max(s, u8::MAX)
    }

    fn to_string_ind(&self, i: usize) -> String {
        format!(
            "[{}]",
            self.get_iter(i).map(|(setting, _)| setting).join(", ")
        )
    }
}

/// Basic storage for a vector of mode indices and the local state indices associated with them.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct BinarySprings(StateSprings);

impl Elements for BinarySprings {
    fn len(&self) -> usize {
        self.0.mode_inds.len()
    }
}

impl Display for BinarySprings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.len()).map(|i| self.to_string_ind(i)).join(", ")
        )
    }
}

impl ModeSettings for BinarySprings {
    fn get_mode_inds(&self) -> &ModeInds {
        self.0.get_mode_inds()
    }

    fn get_mode_inds_mut(&mut self) -> &mut ModeInds {
        self.0.get_mode_inds_mut()
    }

    fn push_setting(&mut self, setting: u8) {
        self.0.push_setting(setting.min(1));
    }

    fn get_setting(&self, i: usize) -> u8 {
        self.0.get_setting(i)
    }

    fn push_str(&mut self, s: &str) -> Result<(), BadParse> {
        self.0.push_str_with_max(s, 1)
    }

    fn to_string_ind(&self, i: usize) -> String {
        self.0.to_string_ind(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("[3, 0, 1, 2]", Ok(vec![(3, 0), (1, 2), (2, 3)]), Some(4))]
    #[case("[1, 0, 1, 2, 0, 0]", Ok(vec![(1, 0), (1, 2), (2, 3)]), Some(6))]
    #[case("[1, 0d, 1, 2]", Err(BadParse("State setting \"0d\" is unparsable or out of bounds.".to_string())), None)]
    #[case("[1, 0, [], 1, 2]", Err(BadParse("State setting \"[]\" is unparsable or out of bounds.".to_string())), None)]
    fn test_state_modes(
        #[case] input: &str,
        #[case] output: Result<Vec<(u8, ModeInd)>, BadParse>,
        #[case] default_n_qubit: Option<ModeInd>,
    ) {
        let mode_settings = StateSprings::from_str(input);
        if let Ok(mode_settings) = &mode_settings {
            assert!(default_n_qubit
                .is_some_and(|x| x == mode_settings.get_mode_inds().default_n_mode()));
        }
        let vec = mode_settings.map(|x| x.get_iter(0).collect::<Vec<_>>());
        assert_eq!(vec, output);
    }

    #[rstest]
    #[case("[1, 0, 1, 1]", Ok(vec![(1, 0), (1, 2), (1, 3)]), Some(4))]
    #[case("[1, 0, 1, 1,]", Ok(vec![(1, 0), (1, 2), (1, 3)]), Some(4))]
    #[case("[1, 0, 1, 1 ,]", Ok(vec![(1, 0), (1, 2), (1, 3)]), Some(4))]
    #[case("[1, 0, 1, 1 , ]", Ok(vec![(1, 0), (1, 2), (1, 3)]), Some(4))]
    #[case("[1, 0, 1, 1, 0, 0]", Ok(vec![(1, 0), (1, 2), (1, 3)]), Some(6))]
    #[case("[1, 0, 1, 2]", Err(BadParse("State setting \"2\" is unparsable or out of bounds.".to_string())), None)]
    #[case("[1, 0, 1, 2, 0, 0]", Err(BadParse("State setting \"2\" is unparsable or out of bounds.".to_string())), None)]
    #[case("[1, 0d, 1, 2]", Err(BadParse("State setting \"0d\" is unparsable or out of bounds.".to_string())), None)]
    #[case("[1, 0, [], 1, 2]", Err(BadParse("State setting \"[]\" is unparsable or out of bounds.".to_string())), None)]
    #[case("[]", Ok(vec![]), Some(0))]
    #[case("[,]", Err(BadParse("State setting \"\" is unparsable or out of bounds.".to_string())), None)]
    fn test_binary_modes(
        #[case] input: &str,
        #[case] output: Result<Vec<(u8, ModeInd)>, BadParse>,
        #[case] default_n_qubit: Option<ModeInd>,
    ) {
        let mode_settings = BinarySprings::from_str(input);
        if let Ok(mode_settings) = &mode_settings {
            assert!(default_n_qubit
                .is_some_and(|x| x == mode_settings.get_mode_inds().default_n_mode()));
        }
        let vec = mode_settings.map(|x| x.get_iter(0).collect::<Vec<_>>());
        assert_eq!(vec, output);
    }
}
