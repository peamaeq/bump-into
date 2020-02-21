# bump-into

A `no_std` bump allocator over an arbitrary region of memory. Can
be used to pass objects up the stack when it would otherwise be
inconvenient; for instance, if the objects are dynamically sized.

## Example

```rust
use bump_into::{self, BumpInto};

// allocate 64 bytes of uninitialized space on the stack
let mut bump_into_space = bump_into::space!(64);
let bump_into = BumpInto::new(&mut bump_into_space[..]);

// allocating an object produces a mutable reference with
// the same lifetime as the `BumpInto` instance, or `None`
// if there isn't enough space
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
