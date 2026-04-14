//! Arithmetic helper utilities.

use std::mem;

use num_traits::PrimInt;

pub fn divrem<T: PrimInt>(num: T, den: T) -> (T, T) {
    let div = num / den;
    (div, num - div * den)
}

pub fn divceil<T: PrimInt>(num: T, den: T) -> T {
    let (div, rem) = divrem(num, den);
    div + if rem == T::zero() {
        T::zero()
    } else {
        T::one()
    }
}

/// Floor of the binary logarithm of num.
pub fn floor_log2<T: PrimInt>(num: T) -> Option<T> {
    let n_bit = T::from(mem::size_of::<T>() * 8)?;
    let n_lz = T::from(num.leading_zeros())?;
    if num == T::zero() {
        None
    } else {
        Some(n_bit - T::one() - n_lz)
    }
}

/// Whether num is an exact power of 2.
pub fn is_pow2<T: PrimInt>(num: T) -> bool {
    num.count_ones() == 1
}

/// Ceiling of the binary logarithm of num.
pub fn ceil_log2<T: PrimInt>(num: T) -> Option<T> {
    let floor = floor_log2(num)?;
    Some(if is_pow2(num) {
        floor
    } else {
        floor + T::one()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divrem() {
        assert_eq!(divrem(10, 2), (5, 0));
        assert_eq!(divrem(11, 2), (5, 1));
        assert_eq!(divrem(10, 3), (3, 1));
        assert_eq!(divrem(10, 4), (2, 2));
        assert_eq!(divrem(100, 4), (25, 0));
        assert_eq!(divrem(100, 5), (20, 0));
        assert_eq!(divrem(104, 5), (20, 4));
    }

    #[test]
    fn test_divceil() {
        assert_eq!(divceil(10, 2), 5);
        assert_eq!(divceil(10, 3), 4);
        assert_eq!(divceil(11, 3), 4);
        assert_eq!(divceil(12, 3), 4);
        assert_eq!(divceil(13, 3), 5);
        assert_eq!(divceil(99, 3), 33);
        assert_eq!(divceil(100, 3), 34);
        assert_eq!(divceil(101, 3), 34);
        assert_eq!(divceil(102, 3), 34);
        assert_eq!(divceil(103, 3), 35);
    }

    #[test]
    fn test_floor_log2() {
        assert_eq!(floor_log2(0), None);
        assert_eq!(floor_log2(1), Some(0));
        assert_eq!(floor_log2(2), Some(1));
        assert_eq!(floor_log2(3), Some(1));
        assert_eq!(floor_log2(4), Some(2));
        assert_eq!(floor_log2(5), Some(2));
        assert_eq!(floor_log2(6), Some(2));
        assert_eq!(floor_log2(7), Some(2));
        assert_eq!(floor_log2(8), Some(3));
        assert_eq!(floor_log2(9), Some(3));
        assert_eq!(floor_log2(15), Some(3));
        assert_eq!(floor_log2(16), Some(4));
        assert_eq!(floor_log2(17), Some(4));
        assert_eq!(floor_log2(31), Some(4));
        assert_eq!(floor_log2(32), Some(5));
        assert_eq!(floor_log2(33), Some(5));
    }

    #[test]
    fn test_ceil_log2() {
        assert_eq!(ceil_log2(0), None);
        assert_eq!(ceil_log2(1), Some(0));
        assert_eq!(ceil_log2(2), Some(1));
        assert_eq!(ceil_log2(3), Some(2));
        assert_eq!(ceil_log2(4), Some(2));
        assert_eq!(ceil_log2(5), Some(3));
        assert_eq!(ceil_log2(6), Some(3));
        assert_eq!(ceil_log2(7), Some(3));
        assert_eq!(ceil_log2(8), Some(3));
        assert_eq!(ceil_log2(9), Some(4));
        assert_eq!(ceil_log2(15), Some(4));
        assert_eq!(ceil_log2(16), Some(4));
        assert_eq!(ceil_log2(17), Some(5));
        assert_eq!(ceil_log2(31), Some(5));
        assert_eq!(ceil_log2(32), Some(5));
        assert_eq!(ceil_log2(33), Some(6));
    }
}
