//! Generic sorting functions
//!
//! Provides generic implementations of basic sorting functions.  Because Rust's build in sorting
//! is only available on slices and some data structures are not representable using slices, the
//! following sorting functions operate on a generic indexable container.
//!
//! # Examples
//!
//! Sorting a vector
//! ```
//! use index_sort::merge_sort;
//! let mut v : Vec<i32> = (0..1000).rev().collect();
//! let rng = 0..v.len();
//! merge_sort(&mut v[..], rng);
//! ```
//!
//! Sorting a pair of vectors lexicographically
//! ```
//! use index_sort::quick_sort;
//! let mut v1 : Vec<i32> = vec![5, 2, 1, 3, 6, 3];
//! let mut v2 : Vec<i32> = vec![1, 4, 2, 5, 7, 0];
//! let rng = 0..v1.len();
//! let mut v = (v1, v2);
//! quick_sort(&mut v, rng);
//! ```
//!
//! This crate defines generic sorting algorithms that are not tied to Rust slices.  Instead sort
//! is defined on types that provide swap and compare functions with integer indices.  The genesis
//! of this crate was a need to lexicographically sort tuples of made of matched length vectors.
//! Such a data structure cannot be sorted with standard Rust sort functions without creating a
//! permutation vector as an intermediate step.
//!
//!

// Enable `no_std` unless the `std` feature is enabled or the crate is
// compiled in test mode.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

use core::{cmp::Ordering, ops::Range};

/// Sortable is defined for types that should be allowed to be sorted.
pub trait Sortable {
    /// exchanges the data at index `i` with the data at index `j`
    fn swap(&mut self, i: usize, j: usize);

    /// compares the data at index `i` with the data at index `j`
    fn compare(&self, i: usize, j: usize) -> Ordering;
}

/// Slices of ordered elements are sortable
impl<T: Ord> Sortable for [T] {
    fn swap(&mut self, i: usize, j: usize) {
        (*self).swap(i, j);
    }

    fn compare(&self, i: usize, j: usize) -> Ordering {
        self[i].cmp(&self[j])
    }
}

/// Vectors of ordered elements are sortable
#[cfg(feature = "std")]
impl <T: Ord> Sortable for Vec<T> {
    fn swap(&mut self, i: usize, j: usize) {
        self[..].swap(i, j)
    }
    fn compare(&self, i: usize, j: usize) -> Ordering {
        self[i].cmp(&self[j])
    }
}

/// Tuples of sortable elements are sortable within their common range
impl<S: Sortable, T: Sortable> Sortable for (S, T) {
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
        self.1.swap(i, j);
    }

    fn compare(&self, i: usize, j: usize) -> Ordering {
        let res = self.0.compare(i, j);
        if res != Ordering::Equal {
            return res;
        }

        self.1.compare(i, j)
    }
}

/// sort the container in the specified range using the insertion sort algorithm
pub fn insertion_sort<T>(container: &mut T, range: Range<usize>)
where
    T: Sortable + ?Sized,
{
    let mut i = range.start + 1;
    while i < range.end {
        let mut j = i;
        while j > range.start && container.compare(j - 1, j) == Ordering::Greater {
            container.swap(j, j - 1);
            j -= 1;
        }
        i += 1;
    }
}

fn lower_bound(container: &(impl Sortable + ?Sized), range: Range<usize>, pos: usize) -> usize {
    let mut from = range.start;
    let mut len = range.len();
    while len > 0 {
        let half = len / 2;
        let middle = from + half;
        if container.compare(middle, pos) == Ordering::Less {
            from = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    from
}

fn upper_bound(container: &(impl Sortable + ?Sized), range: Range<usize>, pos: usize) -> usize {
    let mut from = range.start;
    let mut len = range.len();
    while len > 0 {
        let half = len / 2;
        let middle = from + half;
        if container.compare(pos, middle) == Ordering::Less {
            len = half;
        } else {
            from = middle + 1;
            len -= half + 1;
        }
    }
    from
}

const MERGESORT_NO_REC: usize = 16;

fn in_place_merge(
    container: &mut (impl Sortable + ?Sized),
    from: usize,
    mut mid: usize,
    to: usize,
) {
    if from >= mid || mid >= to {
        return;
    }
    if to - from == 2 {
        if container.compare(mid, from) == Ordering::Less {
            container.swap(from, mid)
        }
        return;
    }

    let first_cut;
    let second_cut;

    if mid - from > to - mid {
        first_cut = from + (mid - from) / 2;
        second_cut = lower_bound(container, mid..to, first_cut);
    } else {
        second_cut = mid + (to - mid) / 2;
        first_cut = upper_bound(container, from..mid, second_cut);
    }

    let first2 = first_cut;
    let middle2 = mid;
    let last2 = second_cut;
    if middle2 != first2 && middle2 != last2 {
        let mut first1 = first2;
        let mut last1 = middle2 - 1;
        while first1 < last1 {
            container.swap(first1, last1);
            first1 += 1;
            last1 -= 1;
        }
        first1 = middle2;
        last1 = last2 - 1;
        while first1 < last1 {
            container.swap(first1, last1);
            first1 += 1;
            last1 -= 1;
        }
        first1 = first2;
        last1 = last2 - 1;
        while first1 < last1 {
            container.swap(first1, last1);
            first1 += 1;
            last1 -= 1;
        }
    }

    mid = first_cut + (second_cut - mid);
    in_place_merge(container, from, first_cut, mid);
    in_place_merge(container, mid, second_cut, to);
}

/// sort the container in the specified range using the merge sort algorithm
pub fn merge_sort<T>(container: &mut T, range: Range<usize>)
where
    T: Sortable + ?Sized,
{
    let length = range.len();

    // Insertion sort on smallest arrays
    if length < MERGESORT_NO_REC {
        insertion_sort(container, range);
        return;
    }

    // Recursively sort halves
    let mid = range.start + length / 2;
    merge_sort(container, range.start..mid);
    merge_sort(container, mid..range.end);

    // If list is already sorted, nothing left to do. This is an
    // optimization that results in faster sorts for nearly ordered lists.
    if container.compare(mid - 1, mid) != Ordering::Greater {
        return;
    }

    // Merge sorted halves
    in_place_merge(container, range.start, mid, range.end);
}

/// Helper modified from a servo library: https://github.com/servo/rust-quicksort/blob/master/lib.rs
/// Servo library was copied from https://sedgewick.io/wp-content/uploads/2022/03/2002QuicksortIsOptimal.pdf
fn qs_helper<T>(container: &mut T, left: isize, right: isize)
where
    T: Sortable + ?Sized,
{
    if right <= left {
        return;
    }

    // if range is small, use insertion sort instead
    if right - left >= 20 {
        insertion_sort(container, (left as usize)..(right as usize + 1));
        return;
    }

    // pick pivot
    {
        let mid = left + (right - left)/2;
        let pivot = if let Ordering::Equal | Ordering::Less = container.compare(left as usize, mid as usize) {
            if let Ordering::Equal | Ordering::Less = container.compare(mid as usize, right as usize) {
                // left mid right
                mid
            } else {
                if let Ordering::Equal | Ordering::Less = container.compare(left as usize, right as usize) {
                    // left right mid
                    right
                } else {
                    // right left mid
                    left
                }
            }
        } else {
            if let Ordering::Equal | Ordering::Less = container.compare(mid as usize, right as usize) {
                if let Ordering::Equal | Ordering::Less = container.compare(left as usize, right as usize) {
                    // mid left right
                    left
                } else {
                    // mid right left
                    right
                }
            } else {
                // right mid left
                mid
            }
        };

        // swap pivot and right element
        if pivot != right {
            container.swap(pivot as usize, right as usize);
        }
    }

    let mut i = left - 1;
    let mut j = right;
    let mut p = i;
    let mut q = j;

    let v = right;

    loop {
        i += 1;
        while container.compare(i as usize, v as usize) == Ordering::Less {
            i += 1
        }

        j -= 1;
        while container.compare(v as usize, j as usize) == Ordering::Less {
            if j == left {
                break;
            }
            j -= 1;
        }

        if i >= j {
            break;
        }

        container.swap(i as usize, j as usize);
        if container.compare(i as usize, v as usize) == Ordering::Equal {
            p += 1;
            container.swap(p as usize, i as usize);
        }
        if container.compare(v as usize, j as usize) == Ordering::Equal {
            q -= 1;
            container.swap(j as usize, q as usize);
        }
    }

    container.swap(i as usize, right as usize);
    j = i - 1;
    i += 1;
    let mut k = left;
    while k < p {
        container.swap(k as usize, j as usize);
        k += 1;
        j -= 1;
    }

    k = right - 1;
    while k > q {
        container.swap(i as usize, k as usize);
        k -= 1;
        i += 1;
    }

    qs_helper(container, left, j);
    qs_helper(container, i, right);
}

/// sort the container in the specified range using the quicksort algorithm
pub fn quick_sort<T>(container: &mut T, range: Range<usize>)
where
    T: Sortable + ?Sized,
{
    qs_helper(container, range.start as isize, range.end as isize - 1);
}

#[cfg(test)]
mod tests {

    fn ordered_vec(n: i32) -> Vec<i32> {
        (0..n).collect()
    }

    fn rev_ordered_vec(n: i32) -> Vec<i32> {
        (0..n).into_iter().rev().collect()
    }

    fn alternating_vec(n: i32) -> Vec<i32> {
        let mut res = Vec::new();
        for i in 0..n {
            res.push(2 * i);
        }
        for i in 0..n {
            res.push(2 * i + 1);
        }
        res
    }

    fn low_center_vec(n: i32) -> Vec<i32> {
        let mut res = Vec::new();
        res.extend((0..n).into_iter().rev());
        res.extend((0..n).into_iter());
        res
    }

    fn vectors() -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        for i in [10i32, 20, 30, 50, 100, 500, 1000] {
            result.push(ordered_vec(i));
            result.push(rev_ordered_vec(i));
            result.push(alternating_vec(i));
            result.push(low_center_vec(i));
        }
        result
    }

    fn is_sorted<T: Ord>(v: impl AsRef<[T]>) -> bool {
        v.as_ref()
            .iter()
            .zip(v.as_ref().iter().skip(1))
            .all(|(a, b)| a <= b)
    }

    #[test]
    fn quicksort_vectors() {
        for mut v in vectors() {
            let rng = 0..v.len();
            super::quick_sort(v.as_mut_slice(), rng);
            assert!(is_sorted(v));
        }
    }

    #[test]
    fn mergesort_vectors() {
        for mut v in vectors() {
            let rng = 0..v.len();
            super::merge_sort(v.as_mut_slice(), rng);
            assert!(is_sorted(v));
        }
    }

    #[test]
    fn insertionsort_vectors() {
        for mut v in vectors() {
            let rng = 0..v.len();
            super::insertion_sort(v.as_mut_slice(), rng);
            assert!(is_sorted(v));
        }
    }
}
