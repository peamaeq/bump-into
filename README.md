# bump-into

[![crates.io][crates_io_img]][crates_io_page] [![docs.rs][docs_rs_img]][docs_rs_page] [![CI status][ci_status_img]][ci_status_page]

[crates_io_img]: https://img.shields.io/crates/v/bump-into.svg
[crates_io_page]: https://crates.io/crates/bump-into
[docs_rs_img]: https://docs.rs/bump-into/badge.svg
[docs_rs_page]: https://docs.rs/bump-into
[ci_status_img]: https://img.shields.io/github/workflow/status/autumnontape/bump-into/CI
[ci_status_page]: https://github.com/autumnontape/bump-into/actions

A `no_std` bump allocator sourcing space from a user-provided mutable
slice rather than from a global allocator, making it suitable for use
in embedded applications and tight loops.

## Drop behavior

Values held in `BumpInto` allocations are never dropped. If they must
be dropped, you can use `Option::take`, `core::mem::ManuallyDrop::drop`,
or `core::ptr::drop_in_place` to drop them explicitly.

## Example

```rust
use bump_into::{self, BumpInto};

// allocate 64 bytes of uninitialized space on the stack
let mut bump_into_space = bump_into::space_uninit!(64);
let bump_into = BumpInto::from_slice(&mut bump_into_space[..]);

// allocating an object produces a mutable reference with
// a lifetime borrowed from `bump_into_space`, or gives
// back its argument in `Err` if there isn't enough space
let number: &mut u64 = bump_into
    .alloc_with(|| 123)
    .ok()
    .expect("not enough space");
assert_eq!(*number, 123);
*number = 50000;
assert_eq!(*number, 50000);

// slices can be allocated as well
let slice: &mut [u16] = bump_into
    .alloc_n_with(5, core::iter::repeat(10))
    .expect("not enough space");
assert_eq!(slice, &[10; 5]);
slice[2] = 100;
assert_eq!(slice, &[10, 10, 100, 10, 10]);
```

## Copying

Copyright (c) 2020 autumnontape

This project may be reproduced under the terms of the MIT or
the Apache 2.0 license, at your option. A copy of each license
is included.
