//! Basics for reading data in the component data into a compact, standardised container
//! component data are expressed in the form A * B * ... where each *-delimited string is
//! a component part.

use core::str;
use std::ops::IndexMut;

use crate::container::traits::Elements;

/// Integral type used for storing mode indices.
pub type ModeInd = u16;

/// Error returned when a string cannot be parsed into per-mode settings because it is malformed.
#[derive(Debug, PartialEq)]
pub struct BadParse(pub String);

impl std::fmt::Display for BadParse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bad parse: {}", self.0)
    }
}
impl std::error::Error for BadParse {}

/// Basic storage for a vector of strings representing compositions of quantum mechanical modes and their settings.
#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct ModeInds {
    // Contiguous storage buffer for strings of mode indices.
    buf: Vec<ModeInd>,
    // Positions of the element after the last element of each string.
    end_offsets: Vec<usize>,
    // Maximum value of buf, updated as new elements are pushed.
    max: Option<ModeInd>,
}

impl ModeInds {
    /// `Iterator` over the number of mode indices in each of the strings stored in self.
    pub fn string_len_iter(&self) -> impl Iterator<Item = usize> + use<'_> {
        (0..self.len()).map(|i| {
            self.end_offsets[i].saturating_sub(if i == 0 { 0 } else { self.end_offsets[i - 1] })
        })
    }

    /// Number of modes required in the smallest space of modes that can contain modes indexed in [0, self.max].
    pub fn default_n_mode(&self) -> ModeInd {
        if let Some(max) = self.max {
            max + 1
        } else {
            0
        }
    }

    /// Update the cached maximum mode index if `i_mode` exceeds the current maximum.
    pub fn update_max_index(&mut self, i_mode: ModeInd) {
        match &mut self.max {
            Some(max) => {
                if i_mode > *max {
                    *max = i_mode
                }
            }
            None => self.max = Some(i_mode),
        }
    }

    /// Push a mode index onto the end of the buffer.
    pub fn push(&mut self, i_mode: ModeInd) {
        self.buf.push(i_mode);
        self.update_max_index(i_mode);
    }

    /// End the current string.
    pub fn end_string(&mut self) {
        self.end_offsets.push(self.buf.len());
    }
}

impl Elements for ModeInds {
    fn len(&self) -> usize {
        self.end_offsets.len()
    }
}

/// Defines functions for adding mode indices and settings for some kind of quantum mechanical operator or state.
pub trait ModeSettings: Default + Elements {
    /// Get a ref to the mode indices of the struct `implementing` this trait.
    fn get_mode_inds(&self) -> &ModeInds;

    /// Get a mut ref to the mode indices of the struct `implementing` this trait.
    fn get_mode_inds_mut(&mut self) -> &mut ModeInds;

    /// Unlike mode_inds, the setting can have different storage container type, so a vector-level getter is not defined.
    /// Instead, push a new setting value onto the setting vector (of whatever type) as a byte.
    fn push_setting(&mut self, setting: u8);

    /// Get the indexed setting as a byte.
    fn get_setting(&self, i: usize) -> u8;

    /// Append n empty pairs of settings and mode index strings.
    fn append_empty(&mut self, n: usize) {
        (0..n).for_each(|_| self.get_mode_inds_mut().end_string());
    }

    /// Push a new pair of settings and mode index strings.
    /// Optionally keep the 0_u8 settings in the buffer. If not keep_zeros, still
    /// let the mode indices associated with zeros be respected in the max mode index.
    fn push(&mut self, iter: impl Iterator<Item = (u8, ModeInd)>, keep_zeros: bool) {
        for (setting, i_mode) in iter {
            if !keep_zeros && setting == 0_u8 {
                self.get_mode_inds_mut().update_max_index(i_mode);
                continue;
            }
            self.get_mode_inds_mut().push(i_mode);
            self.push_setting(setting);
        }
        self.get_mode_inds_mut().end_string();
    }

    /// Push a new pair of settings and mode index strings without keeping zeros in the buffer.
    fn push_no_zeros(&mut self, iter: impl Iterator<Item = (u8, ModeInd)>) {
        self.push(iter, false);
    }

    /// Try push one mode settings string extracted from the given string slice.
    fn push_str(&mut self, s: &str) -> Result<(), BadParse>;

    /// Get the settings and mode indices at index i as an iterator.
    fn get_iter(&self, i: usize) -> impl Iterator<Item = (u8, ModeInd)> {
        if i >= self.len() {
            panic!("Index out of bounds");
        }
        let i_begin = if i == 0 {
            0
        } else {
            self.get_mode_inds().end_offsets[i.saturating_sub(1)]
        };
        let i_end = self.get_mode_inds().end_offsets[i];
        (i_begin..i_end).map(move |x| (self.get_setting(x), self.get_mode_inds().buf[x]))
    }

    /// Get a string representation of the settings and mode indices at index i.
    fn to_string_ind(&self, i_cmpnt: usize) -> String;

    /// Parse and append only the `i_part`-th `*`-delimited factor from each comma-delimited entry in `s`.
    fn append_str_part(&mut self, s: &str, i_part: usize) -> Result<(), BadParse> {
        for s in CommaSplitter::trimmed_validated(s)? {
            if let Some(s) = extract_after_coeff(s) {
                let s = s.split('*').nth(i_part).unwrap_or_default();
                self.push_str(s)?
            } else {
                return Err(BadParse(format!("\"{s}\" is ill-formed")));
            }
        }
        Ok(())
    }

    /// Parse and append the first `*`-delimited factor from each comma-delimited entry in `s`.
    fn append_str(&mut self, s: &str) -> Result<(), BadParse> {
        self.append_str_part(s, 0)
    }

    /// Extend `self` with empty entries until it contains at least `n` strings.
    fn pad_to(&mut self, n: usize) {
        self.append_empty(self.len().saturating_sub(n));
    }

    /// Construct an empty instance and populate it by parsing the first part of `s`.
    fn from_str(s: &str) -> Result<Self, BadParse> {
        let mut this = Self::default();
        this.append_str(s).map(|_| this)
    }

    /// Split each comma-delimited entry in `s` into `*`-delimited parts and build one collection per part position.
    fn all_parts_from_str(s: &str) -> Result<Vec<Self>, BadParse> {
        let mut parts = Vec::<Self>::default();
        for (i_cmpnt, s) in CommaSplitter::trimmed_validated(s)?.enumerate() {
            if let Some(s) = extract_after_coeff(s) {
                for (i_part, s) in s.split('*').enumerate() {
                    let part = get_part(&mut parts, i_part, i_cmpnt);
                    part.push_str(s)?;
                }
            } else {
                return Err(BadParse(format!("\"{s}\" is ill-formed")));
            }
        }
        if let Some(n) = parts.iter().map(|x| x.len()).max() {
            for part in parts.iter_mut() {
                part.pad_to(n);
            }
        }
        Ok(parts)
    }
}

fn get_part<T: ModeSettings + Default>(v: &mut Vec<T>, i_part: usize, len: usize) -> &mut T {
    if i_part >= v.len() {
        for j in v.len()..(i_part + 1) {
            v.push(T::default());
            v[j].append_empty(len);
        }
    }
    v.index_mut(i_part)
}

/// Iterator over top-level comma-delimited segments that ignores commas nested inside brackets.
struct CommaSplitter<'a> {
    s: &'a str,
    offset: usize,
    brackets: Vec<char>,
}

impl<'a> CommaSplitter<'a> {
    /// Create a splitter over the top-level comma-delimited segments of `s`.
    pub fn new(s: &'a str) -> Self {
        Self {
            s: s.trim(),
            offset: 0,
            brackets: vec![],
        }
    }

    /// Iterate over comma-delimited segments, trimming whitespace from each yielded slice.
    pub fn trimmed(s: &'a str) -> impl Iterator<Item = Result<&'a str, BadParse>> {
        Self::new(s).map(|x| x.map(|s| s.trim()))
    }

    /// Validate bracket structure while iterating over untrimmed comma-delimited segments.
    pub fn validated(s: &'a str) -> Result<impl Iterator<Item = &'a str>, BadParse> {
        match Self::new(s).find(|x| x.is_err()) {
            Some(x) => Err(x.unwrap_err()),
            None => Ok(Self::new(s).map(|x| x.unwrap())),
        }
    }

    /// Validate bracket structure and yield trimmed comma-delimited segments.
    pub fn trimmed_validated(s: &'a str) -> Result<impl Iterator<Item = &'a str>, BadParse> {
        Self::validated(s).map(|iter| iter.map(|x| x.trim()))
    }

    /// Update the bracket stack for `c`, returning `false` if it closes the wrong bracket type.
    fn push_bracket(&mut self, c: char) -> bool {
        if c == '(' || c == '[' || c == '{' {
            self.brackets.push(c);
            true
        }
        // closing bracket must match the last pushed open bracket.
        else if c == ')' {
            if self.brackets.last().is_some_and(|x| *x == '(') {
                self.brackets.pop();
                true
            } else {
                false
            }
        } else if c == ']' {
            if self.brackets.last().is_some_and(|x| *x == '[') {
                self.brackets.pop();
                true
            } else {
                false
            }
        } else if c == '}' {
            if self.brackets.last().is_some_and(|x| *x == '{') {
                self.brackets.pop();
                true
            } else {
                false
            }
        } else {
            true
        }
    }
}

impl<'a> Iterator for CommaSplitter<'a> {
    type Item = Result<&'a str, BadParse>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.s.len() {
            return None;
        }

        let s = self.s.split_at(self.offset).1;
        for (i, c) in s.as_bytes().iter().enumerate() {
            if !self.push_bracket(*c as char) {
                return Some(Err(BadParse("Mismatched brackets.".to_string())));
            }
            if !self.brackets.is_empty() {
                continue;
            }
            if *c as char == ',' {
                let s = s.split_at(i).0;
                self.offset += i + 1;
                return Some(Ok(s));
            }
        }
        self.offset = self.s.len();
        if self.brackets.is_empty() {
            Some(Ok(s))
        } else {
            Some(Err(BadParse("Mismatched brackets.".to_string())))
        }
    }
}

/// Strip an optional leading `(coeff, ...)` wrapper and return the substring after the first comma.
pub fn extract_after_coeff(s: &str) -> Option<&str> {
    let s_trim = s.trim();
    if s_trim.starts_with('(') {
        s_trim
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split_once(',')
            .map(|x| x.1)
    } else {
        Some(s_trim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("abc, def, ghi", Ok(vec!["abc", "def", "ghi"]))]
    #[case("abc, ( def, ghi", Err(BadParse("Mismatched brackets.".to_string())))]
    #[case("abc, ) def, ghi", Err(BadParse("Mismatched brackets.".to_string())))]
    #[case("abc, [} def, ghi", Err(BadParse("Mismatched brackets.".to_string())))]
    #[case("abc, [} def, ghi", Err(BadParse("Mismatched brackets.".to_string())))]
    #[case("abc, [] def, ghi", Ok(vec!["abc", "[] def", "ghi"]))]
    #[case(",abc, def, ghi", Ok(vec!["", "abc", "def", "ghi"]))]
    #[case(",,abc, def, ghi", Ok(vec!["", "", "abc", "def", "ghi"]))]
    #[case(", ,abc, def, ghi", Ok(vec!["", "", "abc", "def", "ghi"]))]
    #[case(" , , abc, def, ghi", Ok(vec!["", "", "abc", "def", "ghi"]))]
    #[case("abc, [, , ,] def, ghi", Ok(vec!["abc", "[, , ,] def", "ghi"]))]
    #[case("abc, def, ghi, []", Ok(vec!["abc", "def", "ghi", "[]"]))]
    #[case("abc, def, ghi, [,]", Ok(vec!["abc", "def", "ghi", "[,]"]))]
    #[case("abc, def, ghi, [(),{[,,]}]", Ok(vec!["abc", "def", "ghi", "[(),{[,,]}]"]))]
    #[case("abc, def, ghi, [(),{[,,]}], ", Ok(vec!["abc", "def", "ghi", "[(),{[,,]}]"]))]
    #[case("abc, def, ghi, [(),{[,,]}],", Ok(vec!["abc", "def", "ghi", "[(),{[,,]}]"]))]
    #[case("abc, def, ghi, [(),{[,,]}],,", Ok(vec!["abc", "def", "ghi", "[(),{[,,]}]", ""]))]
    fn test_comma_splitter(#[case] input: &str, #[case] output: Result<Vec<&str>, BadParse>) {
        let splitter = CommaSplitter::trimmed_validated(input);
        let vec = splitter.map(|iter| iter.collect::<Vec<_>>());
        assert_eq!(vec, output);
    }
}
