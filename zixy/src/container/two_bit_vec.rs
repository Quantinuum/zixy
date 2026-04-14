//! One- and Two-bit int vectors packed in `u64` word buffers.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::container::traits::{Elements, HasIndex, MutRefElements, NewWithLen, RefElements};
use crate::utils::arith::{divceil, divrem};

pub const EVEN_BIT_MASK: u64 = 0x5555555555555555;
pub const ODD_BIT_MASK: u64 = 0xAAAAAAAAAAAAAAAA;

/// Packed vector of single-bit values stored densely in `u64` words.
#[derive(Default, Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct BitVec {
    buf: Vec<u64>,
    size: usize,
}

impl BitVec {
    /// Resize the vector, adding zeros when the new size is larger than the present.
    pub fn resize(&mut self, n_bit: usize) {
        self.buf.resize(divceil(n_bit, 64), 0);
        if let Some(last) = self.buf.last_mut() {
            let n_bit = 64 - n_bit % 64;
            if n_bit < 64 {
                *last &= !0_u64 >> n_bit;
            }
        }
        self.size = n_bit;
    }

    /// `Set` the value of the i-indexed bit.
    pub fn set_unchecked(&mut self, i: usize, value: bool) {
        let (i_word, i_bit) = divrem(i, 64);
        let mask = 1_u64 << i_bit;
        if value {
            self.buf[i_word] |= mask
        } else {
            self.buf[i_word] &= !mask
        }
    }

    /// `Set` the value of the i-indexed bit with bounds checking.
    pub fn set(&mut self, i: usize, value: bool) -> Option<()> {
        if i > self.size {
            None
        } else {
            self.set_unchecked(i, value);
            Some(())
        }
    }

    /// Get the value of the i-indexed bit.
    pub fn get_unchecked(&self, i: usize) -> bool {
        let (i_word, i_bit) = divrem(i, 64);
        let mask = 1_u64 << i_bit;
        self.buf[i_word] & mask != 0
    }

    /// Get the value of the i-indexed bit with bounds checking.
    pub fn get(&self, i: usize) -> Option<bool> {
        if i > self.size {
            None
        } else {
            Some(self.get_unchecked(i))
        }
    }

    /// Push a bit to the end of this vector.
    pub fn push(&mut self, value: bool) {
        let i = self.len();
        self.resize(i + 1);
        self.set(i, value);
    }

    /// Count the number of set bits
    pub fn n_set(&self) -> usize {
        self.buf.iter().map(|word| word.count_ones()).sum::<u32>() as usize
    }

    /// Count the number of clear bits
    pub fn n_clear(&self) -> usize {
        self.len().saturating_sub(self.n_set())
    }

    /// Return whether all bits are clear.
    pub fn all_clear(&self) -> bool {
        self.buf.iter().all(|i| *i == 0)
    }

    /// Add the scalar to the indexed element in-place.
    pub fn iadd_unchecked(&mut self, i: usize, scalar: bool) {
        self.set_unchecked(i, self.get_unchecked(i) ^ scalar);
    }

    /// Add the scalar to the indexed element in-place with bounds checking.
    pub fn iadd(&mut self, i: usize, scalar: bool) -> Option<()> {
        let value = self.get(i)?;
        self.set_unchecked(i, value ^ scalar);
        Some(())
    }

    /// Get iterator over the contents of this vector as bools.
    pub fn as_bools(&self) -> impl Iterator<Item = bool> + use<'_> {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }

    /// Take the element-wise exclusive or of the two bitvecs in-place.
    /// If one vec is longer than the other, the shorter one is padded with zeros to match the other in length.
    pub fn iadd_elemwise(&mut self, other: &BitVec) {
        self.resize(self.len().max(other.len()));
        self.buf
            .iter_mut()
            .zip(other.buf.iter())
            .for_each(|(dst, src)| *dst ^= src);
    }

    /// Take the element-wise exclusive or of the two bitvecs out-of-place.
    /// If one vec is longer than the other, the function behaves as though the shorter one is padded with zeros
    /// to match the other in length.
    pub fn add_elemwise(&self, other: &BitVec) -> Self {
        let mut out = Self::new_with_len(self.len().max(other.len()));
        out.iadd_elemwise(other);
        out
    }

    /// Invert every stored bit while preserving the logical vector length.
    pub fn flip(&mut self) {
        self.buf.iter_mut().for_each(|x| *x = !*x);
        // clear bits above the last addressable bit.
        let n_in_last = self.len() % 64;
        if n_in_last != 0 {
            if let Some(word) = self.buf.last_mut() {
                *word &= (1 << n_in_last) - 1
            }
        }
    }
}

impl Elements for BitVec {
    fn len(&self) -> usize {
        self.size
    }
}

impl NewWithLen for BitVec {
    fn new_with_len(n_element: usize) -> Self {
        let mut this = Self::default();
        this.resize(n_element);
        this
    }
}

/// Packed vector of 2-bit integers, with 32 values stored in each `u64` word.
#[derive(Default, Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct TwoBitVec(BitVec);

impl TwoBitVec {
    /// Resize the vector, adding zeros when the new size is larger than the present.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n << 1);
    }

    /// `Set` the value of the i-indexed two-bit integer from the least significant two bits in the value byte.
    pub fn set_unchecked(&mut self, i: usize, value: u8) {
        let value = value & 3;
        let (i_word, i_bit_pair) = divrem(i, 32);
        let mask = !(3_u64 << (i_bit_pair * 2));
        let value = (value as u64) << (i_bit_pair * 2);
        self.0.buf[i_word] &= mask;
        self.0.buf[i_word] |= value;
    }

    /// `Set` the value of the i-indexed two-bit integer from the least significant two bits in the value byte with bounds checking.
    pub fn set(&mut self, i: usize, value: u8) -> Option<()> {
        if i < self.len() {
            self.set_unchecked(i, value);
            Some(())
        } else {
            None
        }
    }

    /// Get the value of the i-indexed two-bit integer as a byte.
    pub fn get_unchecked(&self, i: usize) -> u8 {
        let (i_word, i_bit_pair) = divrem(i, 32);
        ((self.0.buf[i_word] >> (i_bit_pair << 1)) & 3) as u8
    }

    /// Get the value of the i-indexed two-bit integer as a byte with bounds checking.
    pub fn get(&mut self, i: usize) -> Option<u8> {
        if i < self.len() {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Push the least significant two bits from the value byte to the end of this vector.
    pub fn push(&mut self, value: u8) {
        let i = self.len();
        self.resize(i + 1);
        self.set(i, value);
    }

    /// Return whether all bits are clear.
    pub fn all_clear(&self) -> bool {
        self.0.all_clear()
    }

    /// Add the scalar to the indexed element in-place.
    pub fn iadd_unchecked(&mut self, i: usize, scalar: u8) {
        self.set_unchecked(i, self.get_unchecked(i) + scalar);
    }

    /// Add the scalar to the indexed element in-place with bounds checking.
    pub fn iadd(&mut self, i: usize, scalar: u8) -> Option<()> {
        let value = self.get(i)?;
        self.set_unchecked(i, value + scalar);
        Some(())
    }

    /// Multiply the indexed element by the scalar in-place.
    pub fn imul_unchecked(&mut self, i: usize, scalar: u8) {
        self.set_unchecked(i, self.get_unchecked(i) * scalar);
    }

    /// Multiply the indexed element by the scalar in-place with bounds checking.
    pub fn imul(&mut self, i: usize, scalar: u8) -> Option<()> {
        let value = self.get(i)?;
        self.set_unchecked(i, value * scalar);
        Some(())
    }

    /// Get iterator over the contents of this vector as bytes.
    pub fn as_bytes(&self) -> impl Iterator<Item = u8> + use<'_> {
        (0..self.len()).map(|i| self.get_unchecked(i))
    }

    /// Take the element-wise sum mod 4 of the two `TwoBitVec`s in-place.
    /// If one vec is longer than the other, the shorter one is padded with zeros to match the other in length.
    pub fn iadd_elemwise(&mut self, other: &TwoBitVec) {
        self.resize(self.len().max(other.len()));
        // mask the odd-position bits out of dst and src, and let the carry be handled by addition,
        // then XOR the result with the odd-position bits.
        self.0
            .buf
            .iter_mut()
            .zip(other.0.buf.iter())
            .for_each(|(dst, src)| {
                let even_pos_sum = (*dst & EVEN_BIT_MASK) + (src & EVEN_BIT_MASK);
                *dst = even_pos_sum ^ ((*dst & ODD_BIT_MASK) + (src & ODD_BIT_MASK));
            })
    }

    /// Take the element-wise sum mod 4 of the two `TwoBitVec`s out-of-place.
    /// If one vec is longer than the other, the function behaves as though the shorter one is padded with zeros
    /// to match the other in length.
    pub fn add_elemwise(&self, other: &TwoBitVec) -> Self {
        let mut out = Self::new_with_len(self.len().max(other.len()));
        out.iadd_elemwise(other);
        out
    }
}

impl Display for TwoBitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_bytes().join(" "))
    }
}

impl Elements for TwoBitVec {
    fn len(&self) -> usize {
        self.0.len() >> 1
    }
}

impl NewWithLen for TwoBitVec {
    fn new_with_len(n_element: usize) -> Self {
        let mut this = Self::default();
        this.resize(n_element);
        this
    }
}

/// Immutable handle to one logical element inside a [`TwoBitVec`].
pub struct TwoBitRef<'a>(pub &'a TwoBitVec, pub usize);

impl<'a> TwoBitRef<'a> {
    /// Return the referenced two-bit value as a byte with value (0, 1, 2, 3).
    pub fn get(&self) -> u8 {
        self.0.get_unchecked(self.1)
    }
}

impl<'a> PartialEq for TwoBitRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<'a> Display for TwoBitRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl<'a> HasIndex for TwoBitRef<'a> {
    fn get_index(&self) -> usize {
        self.1
    }
}

impl<'a> RefElements<'a> for TwoBitVec {
    type Output = TwoBitRef<'a>;

    fn get_elem_ref(&'a self, index: usize) -> TwoBitRef<'a> {
        TwoBitRef(self, index)
    }
}

/// Mutable handle to one logical element inside a [`TwoBitVec`].
pub struct TwoBitMutRef<'a>(pub &'a mut TwoBitVec, pub usize);

impl<'a> TwoBitMutRef<'a> {
    /// Set the referenced two-bit value from the least significant two bits of `value`.
    pub fn set(&mut self, value: u8) {
        self.0.set_unchecked(self.1, value)
    }

    /// Return the referenced two-bit `value` as a byte with value (0, 1, 2, 3).
    pub fn get(&self) -> u8 {
        self.0.get_unchecked(self.1)
    }
}

impl<'a> Display for TwoBitMutRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        TwoBitRef(self.0, self.1).fmt(f)
    }
}

impl<'a> MutRefElements<'a> for TwoBitVec {
    type Output = TwoBitMutRef<'a>;

    fn get_elem_mut_ref(&'a mut self, index: usize) -> <Self as MutRefElements<'a>>::Output {
        TwoBitMutRef(self, index)
    }
}

impl<'a> HasIndex for TwoBitMutRef<'a> {
    fn get_index(&self) -> usize {
        self.1
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, seq::index::sample, Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_bit_vec_resize() {
        let mut v = BitVec::new_with_len(65);
        v.set_unchecked(64, true);
        assert_eq!(v.buf.len(), 2);
        v.resize(64);
        assert_eq!(v.buf.len(), 1);
        v.resize(65);
        assert!(!v.get_unchecked(64));
        v.set_unchecked(64, true);
        v.resize(64);
        assert_eq!(v.buf.len(), 1);
        v.resize(65);
        assert_eq!(v.buf.len(), 2);
        // the downsize should have cleared the set bit
        assert!(!v.get_unchecked(64));
        v.resize(66);
        v.set_unchecked(65, true);
        v.resize(64);
        v.resize(66);
        assert!(!v.get_unchecked(64));
        assert!(!v.get_unchecked(65));
        v.resize(67);
        v.set_unchecked(64, true);
        v.set_unchecked(65, true);
        v.set_unchecked(66, true);
        v.resize(66);
        v.resize(67);
        assert!(v.get_unchecked(64));
        assert!(v.get_unchecked(65));
        assert!(!v.get_unchecked(66));
    }

    #[test]
    fn test_bit_vec_randomized() {
        let seed = [45u8; 32];
        let mut rng = StdRng::from_seed(seed);

        const N_BIT_MAX: usize = 1 << 12;
        const N_DRAW: usize = 1 << 8;
        const N_SET_MAX_RATIO: usize = 2;

        for _ in 0..N_DRAW {
            let n_bit = (rng.random::<u64>() as usize) % N_BIT_MAX;
            let n_set = (rng.random::<u64>() as usize) % (n_bit / N_SET_MAX_RATIO).max(1);
            // sample 10 unique indices from 0..1000
            let set_bits = sample(&mut rng, n_bit, n_set);
            let mut v = BitVec::new_with_len(n_bit);
            for i_bit in set_bits.iter() {
                assert!(v.set(i_bit, true).is_some());
            }
            assert_eq!(v.n_set(), n_set);
            assert_eq!(
                v.as_bools()
                    .map(|x| if x { 0_usize } else { 1_usize })
                    .sum::<usize>(),
                v.n_clear()
            );
            v.flip();
            assert_eq!(v.n_set(), n_bit - n_set);
        }
    }

    #[test]
    fn test_two_bit_vec() {
        let mut v = TwoBitVec::default();
        // make sure there are more than one u64's worth of two-bit ints (>32)
        let items: Vec<u8> = vec![
            0, 2, 3, 1, 0, 1, 3, 2, 3, 3, 2, 0, 0, 2, 3, 3, 3, 3, 2, 1, 0, 2, 3, 3, 3, 2, 2, 1, 0,
            2, 3, 3, 3, 2, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 2, 3, 0,
        ];
        for item in items.iter().copied() {
            v.push(item);
            assert_eq!(v.get_unchecked(v.len() - 1), item);
        }
        for (i, item) in items.iter().copied().enumerate() {
            assert_eq!(v.get_unchecked(i), item);
        }
    }
}
