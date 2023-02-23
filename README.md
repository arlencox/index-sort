index-sort
==========

There are more containers that need sorting besides slices.  This library
provides a way to sort them with a simple API made of two functions: `compare`
and `swap`.  The `compare` function compares the elements at two indexes and
the `swap` function exchanges elements at two indexes.  Any container that
implements the `Sortable` trait by providing these functions can be sorted
with the supplied sorting functions.

Examples
--------

Sorting a vector
```
use index_sort::merge_sort;
let mut v : Vec<i32> = (0..1000).rev().collect();
let rng = 0..v.len();
merge_sort(&mut v[..], rng);
```

Sorting a pair of vectors lexicographically
```
use index_sort::quick_sort;
let mut v1 : Vec<i32> = vec![5, 2, 1, 3, 6, 3];
let mut v2 : Vec<i32> = vec![1, 4, 2, 5, 7, 0];
let rng = 0..v1.len();
let mut v = (v1, v2);
quick_sort(&mut v, rng);
```

Algorithms
----------

The following sorting algorithms provided.  They all have the
same api.

  - `insertion_sort` does the classic N^2 sorting algorithm
  - `merge_sort` performs an in-place stable sort using an adaptation of the merge sort algorithm provided in the fastutil library
  - `quick_sort` performs an unstable sort using a quick sort algorithm based on the implementation used in Servo

Features
--------

The `std` feature (enabled by default) adds a `Sortable` impl for `Vec`.
