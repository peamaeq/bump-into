/*!

A `no_std` bump allocator sourcing space from a user-provided mutable
slice rather than from a global allocator, making it suitable for use
in embedded applications and tight loops.

## Drop behavior

Values held in `BumpInto` allocations are never dropped. If they must
be dropped, you can use [`core::mem::ManuallyDrop::drop`] or
[`core::ptr::drop_in_place`] to drop them explicitly (and unsafely).
In safe code, you can allocate an [`Option`] and drop the value inside
by overwriting it with `None`.

[`Option`]: core::option::Option

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

*/

#![no_std]

mod size_align;

use core::cell::UnsafeCell;
use core::fmt;
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};

pub use size_align::{AlignOf, SizeOf};

/// A bump allocator sourcing space from an `&mut [MaybeUninit]`.
///
/// Allocation methods produce mutable references with lifetimes
/// tied to the lifetime of the backing slice, meaning the
/// `BumpInto` itself can be freely moved around, including
/// between threads.
///
/// Values held in `BumpInto` allocations are never dropped.
/// If they must be dropped, you can use
/// [`core::mem::ManuallyDrop::drop`] or [`core::ptr::drop_in_place`]
/// to drop them explicitly (and unsafely). In safe code, you can
/// allocate an [`Option`] and drop the value inside by overwriting
/// it with `None`.
///
/// [`Option`]: core::option::Option
pub struct BumpInto<'a> {
    array: UnsafeCell<&'a mut [MaybeUninit<u8>]>,
}

impl<'a> BumpInto<'a> {
    /// Creates a new `BumpInto`, wrapping a slice of `MaybeUninit<S>`.
    #[inline]
    pub fn from_slice<S>(array: &'a mut [MaybeUninit<S>]) -> Self {
        let size = mem::size_of_val(array);
        let ptr = array as *mut [_] as *mut MaybeUninit<u8>;
        let array = unsafe { core::slice::from_raw_parts_mut(ptr, size) };

        BumpInto {
            array: UnsafeCell::new(array),
        }
    }

    /// Returns the number of bytes remaining in the allocator's space.
    #[inline]
    pub fn available_bytes(&self) -> usize {
        unsafe { (*self.array.get()).len() }
    }

    /// Returns the number of spaces of size `size` that could be
    /// allocated in a contiguous region ending at alignment `align`
    /// within the allocator's remaining space.
    ///
    /// Returns `usize::MAX` if `size` is zero.
    ///
    /// # Panics
    ///
    /// Panics if `align` is zero.
    pub fn available_spaces<S: Into<usize>, A: Into<usize>>(&self, size: S, align: A) -> usize {
        let size = size.into();
        let align = align.into();

        if align == 0 {
            panic!("alignment must not be zero");
        }

        if size == 0 {
            return usize::MAX;
        }

        let array = unsafe { &mut *self.array.get() };

        let array_start = *array as *mut [MaybeUninit<u8>] as *mut MaybeUninit<u8> as usize;
        let current_end = array_start + array.len();
        let aligned_end = (current_end / align) * align;

        if aligned_end <= array_start {
            return 0;
        }

        let usable_bytes = aligned_end - array_start;
        usable_bytes / size
    }

    /// Returns the number of `T` that could be allocated in a
    /// contiguous region within the allocator's remaining space.
    ///
    /// Returns `usize::MAX` if `T` is a zero-sized type.
    #[inline]
    pub fn available_spaces_for<T>(&self) -> usize {
        self.available_spaces(SizeOf::<T>::new(), AlignOf::<T>::new())
    }

    /// Tries to allocate `size` bytes with alignment `align`.
    /// Returns a null pointer on failure.
    ///
    /// # Panics
    ///
    /// Panics if `align` is zero.
    pub fn alloc_space<S: Into<usize>, A: Into<usize>>(
        &self,
        size: S,
        align: A,
    ) -> *mut MaybeUninit<u8> {
        let size = size.into();
        let align = align.into();

        if align == 0 {
            panic!("alignment must not be zero");
        }

        if size == 0 {
            // optimization for zero-sized types, as pointers to
            // such types don't have to be distinct in Rust
            return align as *mut MaybeUninit<u8>;
        }

        if size > isize::MAX as usize {
            // Rust doesn't really support allocations this big,
            // and there's no way you'd want to put one in an
            // inline bump heap anyway. it's easier if we put
            // this check here. note that other parts of the code
            // do depend on this check being in this method!
            return ptr::null_mut();
        }

        let array = unsafe { &mut *self.array.get() };

        // since we have to do math to align our output properly,
        // we use `usize` instead of pointers
        let array_start = *array as *mut [MaybeUninit<u8>] as *mut MaybeUninit<u8> as usize;
        let current_end = array_start + array.len();

        // the highest address with enough space to store `size` bytes
        let preferred_ptr = match current_end.checked_sub(size) {
            Some(preferred_ptr) => preferred_ptr,
            None => return ptr::null_mut(),
        };

        // round down to the nearest multiple of `align`
        let aligned_ptr = (preferred_ptr / align) * align;

        if aligned_ptr < array_start {
            // not enough space -- do nothing and return null
            ptr::null_mut()
        } else {
            // bump the bump pointer and return the allocation
            *array = &mut array[..aligned_ptr - array_start];

            aligned_ptr as *mut MaybeUninit<u8>
        }
    }

    /// Tries to allocate enough space to store a `T`.
    /// Returns a properly aligned pointer to uninitialized `T` if
    /// there was enough space; otherwise returns a null pointer.
    #[inline]
    pub fn alloc_space_for<T>(&self) -> *mut T {
        self.alloc_space(SizeOf::<T>::new(), AlignOf::<T>::new()) as *mut T
    }

    /// Tries to allocate enough space to store `count` number of `T`.
    /// Returns a properly aligned pointer to uninitialized `T` if
    /// there was enough space; otherwise returns a null pointer.
    pub fn alloc_space_for_n<T>(&self, count: usize) -> *mut T {
        if mem::size_of::<T>() == 0 {
            return NonNull::dangling().as_ptr();
        }

        let size = mem::size_of::<T>().checked_mul(count);

        match size {
            Some(size) => self.alloc_space(size, AlignOf::<T>::new()) as *mut T,
            None => ptr::null_mut(),
        }
    }

    /// Allocates space for as many aligned `T` as will fit in the
    /// free space of this `BumpInto`. Returns a tuple holding a
    /// pointer to the lowest `T`-space that was just allocated and
    /// the count of `T` that will fit (which may be zero).
    ///
    /// This method will produce a count of `usize::MAX` if `T` is
    /// a zero-sized type.
    pub fn alloc_space_to_limit_for<T>(&self) -> (NonNull<T>, usize) {
        if mem::size_of::<T>() == 0 {
            return (NonNull::dangling(), usize::MAX);
        }

        let count = self.available_spaces(SizeOf::<T>::new(), AlignOf::<T>::new());

        let pointer = self.alloc_space(count * mem::size_of::<T>(), AlignOf::<T>::new());

        match NonNull::new(pointer as *mut T) {
            Some(pointer) => (pointer, count),
            None => (NonNull::dangling(), 0),
        }
    }

    /// Tries to allocate enough space to store a `T` and place `x` there.
    ///
    /// On success (i.e. if there was enough space) produces a mutable
    /// reference to `x` with the lifetime of this `BumpInto`'s backing
    /// slice (`'a`).
    ///
    /// On failure, produces `x`.
    #[inline]
    pub fn alloc<T>(&self, x: T) -> Result<&'a mut T, T> {
        let pointer = self.alloc_space_for::<T>();

        if pointer.is_null() {
            return Err(x);
        }

        unsafe {
            ptr::write(pointer, x);

            Ok(&mut *pointer)
        }
    }

    /// Tries to allocate enough space to store a `T` and place the result
    /// of calling `f` there.
    ///
    /// On success (i.e. if there was enough space) produces a mutable
    /// reference to the stored result with the lifetime of this
    /// `BumpInto`'s backing slice (`'a`).
    ///
    /// On failure, produces `f`.
    ///
    /// Allocating within `f` is allowed; `f` is only called after the
    /// initial allocation succeeds.
    pub fn alloc_with<T, F: FnOnce() -> T>(&self, f: F) -> Result<&'a mut T, F> {
        #[inline(always)]
        unsafe fn eval_and_write<T, F: FnOnce() -> T>(pointer: *mut T, f: F) {
            // this is an optimization borrowed from bumpalo by fitzgen
            // it's meant to help the compiler realize it can avoid a copy by
            // evaluating `f` directly into the allocated space
            ptr::write(pointer, f());
        }

        let pointer = self.alloc_space_for::<T>();

        if pointer.is_null() {
            return Err(f);
        }

        unsafe {
            eval_and_write(pointer, f);

            Ok(&mut *pointer)
        }
    }

    /// Tries to allocate enough space to store a copy of `xs` and copy
    /// `xs` into it.
    ///
    /// On success (i.e. if there was enough space) produces a mutable
    /// reference to the copy with the lifetime of this `BumpInto`'s
    /// backing slice (`'a`).
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, xs: &[T]) -> Option<&'a mut [T]> {
        if mem::size_of::<T>() == 0 {
            unsafe {
                return Some(core::slice::from_raw_parts_mut(
                    NonNull::dangling().as_ptr(),
                    xs.len(),
                ));
            }
        }

        let pointer = self.alloc_space(mem::size_of_val(xs), AlignOf::<T>::new()) as *mut T;

        if pointer.is_null() {
            return None;
        }

        unsafe {
            ptr::copy_nonoverlapping(xs as *const [T] as *const T, pointer, xs.len());

            Some(core::slice::from_raw_parts_mut(pointer, xs.len()))
        }
    }

    /// Tries to allocate enough space to store `count` number of `T` and
    /// fill it with the values produced by `iter.into_iter()`.
    ///
    /// On success (i.e. if there was enough space) produces a mutable
    /// reference to the stored results as a slice, with the lifetime of
    /// this `BumpInto`'s backing slice (`'a`).
    ///
    /// On failure, produces `iter`.
    ///
    /// If the iterator ends before producing enough items to fill the
    /// allocated space, the same amount of space is allocated, but the
    /// returned slice is just long enough to hold the items that were
    /// actually produced.
    ///
    /// Allocating within the iterator's `next` method is allowed;
    /// iteration only begins after the initial allocation succeeds.
    pub fn alloc_n_with<T, I: IntoIterator<Item = T>>(
        &self,
        count: usize,
        iter: I,
    ) -> Result<&'a mut [T], I> {
        #[inline(always)]
        unsafe fn iter_and_write<T, I: Iterator<Item = T>>(pointer: *mut T, mut iter: I) -> bool {
            match iter.next() {
                Some(item) => {
                    ptr::write(pointer, item);

                    false
                }
                None => true,
            }
        }

        let pointer = self.alloc_space_for_n::<T>(count);

        if pointer.is_null() {
            return Err(iter);
        }

        let mut iter = iter.into_iter();

        unsafe {
            for index in 0..count {
                if iter_and_write(pointer.add(index), &mut iter) {
                    // iterator ended before we could fill the whole space.
                    return Ok(core::slice::from_raw_parts_mut(pointer, index));
                }
            }

            Ok(core::slice::from_raw_parts_mut(pointer, count))
        }
    }

    /// Allocates enough space to store as many of the values produced
    /// by `iter.into_iter()` as possible. Produces a mutable reference
    /// to the stored results as a slice, in the opposite order to the
    /// order they were produced in, with the lifetime of this
    /// `BumpInto`'s backing slice (`'a`).
    ///
    /// This method will create a slice of up to `usize::MAX` elements
    /// if `T` is a zero-sized type. This means it technically will not
    /// try to exhaust an infinite iterator, but it may still take much
    /// longer than expected in generic code!
    ///
    /// If you have only a shared reference, and especially if you must
    /// call *non-allocating* methods of `self` within the iterator,
    /// you can use the unsafe [`alloc_down_with_shared`].
    ///
    /// [`alloc_down_with_shared`]: Self::alloc_down_with_shared
    #[inline]
    pub fn alloc_down_with<T, I: IntoIterator<Item = T>>(&mut self, iter: I) -> &'a mut [T] {
        unsafe { self.alloc_down_with_shared(iter) }
    }

    /// Unsafe version of [`alloc_down_with`], taking `self` as a
    /// shared reference instead of a mutable reference.
    ///
    /// [`alloc_down_with`]: Self::alloc_down_with
    ///
    /// # Safety
    ///
    /// Undefined behavior may result if any methods of this `BumpInto`
    /// are called from within the `next` method of the iterator, with
    /// the exception of the `available_bytes`, `available_spaces`,
    /// and `available_spaces_for` methods, which are safe.
    pub unsafe fn alloc_down_with_shared<T, I: IntoIterator<Item = T>>(
        &self,
        iter: I,
    ) -> &'a mut [T] {
        if mem::size_of::<T>() == 0 {
            // this is both meant as an optimization and to bypass
            // the `aligned_end <= array_start` check, since
            // allocating ZSTs shouldn't depend on alignment or
            // remaining space.
            // we also enforce here that an allocation have no more
            // than `usize::MAX` objects, which is obviously implicit
            // on the positive-size path.
            #[allow(clippy::suspicious_map)]
            let count = iter.into_iter().take(usize::MAX).map(mem::forget).count();
            return core::slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), count);
        }

        let (array_start, current_end) = {
            let array = &mut *self.array.get();

            // since we have to do math to align our output properly,
            // we use `usize` instead of pointers
            let array_start = *array as *mut [MaybeUninit<u8>] as *mut MaybeUninit<u8> as usize;
            let current_end = array_start + array.len();

            (array_start, current_end)
        };

        let aligned_end = (current_end / mem::align_of::<T>()) * mem::align_of::<T>();

        if aligned_end <= array_start {
            return &mut [];
        }

        let mut iter = iter.into_iter();

        let mut count = 0;
        let mut cur_space = aligned_end;

        loop {
            let next_space = cur_space.checked_sub(mem::size_of::<T>());

            let finished = match next_space {
                Some(next_space) if next_space >= array_start => match iter.next() {
                    Some(item) => {
                        cur_space = next_space;
                        {
                            let array = &mut *self.array.get();
                            *array = &mut array[..cur_space - array_start];
                        }

                        ptr::write(cur_space as *mut T, item);

                        false
                    }
                    None => true,
                },
                _ => true,
            };

            if finished {
                return core::slice::from_raw_parts_mut(cur_space as *mut T, count);
            }

            count += 1;
        }
    }
}

impl<'a> fmt::Debug for BumpInto<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BumpInto {{ {} bytes free }}", self.available_bytes())
    }
}

/// Creates an uninitialized array of `MaybeUninit` without allocating,
/// suitable for taking a slice of to pass into `BumpInto::from_slice`.
///
/// # Example
///
/// ```rust
/// use bump_into::space_uninit;
/// use core::mem;
///
/// // an array of MaybeUninit<u8> is created by default:
/// let mut space = space_uninit!(64);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 1);
/// // if you need your space to have the alignment of a
/// // particular type, you can use an array-like syntax.
/// // this line will create an array of MaybeUninit<u32>:
/// let mut space = space_uninit!(u32; 16);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 4);
/// ```
#[macro_export]
macro_rules! space_uninit {
    ($capacity:expr) => {
        unsafe {
            extern crate core;

            core::mem::MaybeUninit::<[core::mem::MaybeUninit<u8>; $capacity]>::uninit()
                .assume_init()
        }
    };

    ($like_ty:ty; $capacity:expr) => {
        unsafe {
            extern crate core;

            core::mem::MaybeUninit::<[core::mem::MaybeUninit<$like_ty>; $capacity]>::uninit()
                .assume_init()
        }
    };
}

/// Creates a zeroed array of `MaybeUninit` without allocating,
/// suitable for taking a slice of to pass into `BumpInto::from_slice`.
///
/// # Example
///
/// ```rust
/// use bump_into::space_zeroed;
/// use core::mem;
///
/// // an array of MaybeUninit<u8> is created by default:
/// let mut space = space_zeroed!(64);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 1);
/// // if you need your space to have the alignment of a
/// // particular type, you can use an array-like syntax.
/// // this line will create an array of MaybeUninit<u32>:
/// let mut space = space_zeroed!(u32; 16);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 4);
/// ```
#[macro_export]
macro_rules! space_zeroed {
    ($capacity:expr) => {
        unsafe {
            extern crate core;

            core::mem::MaybeUninit::<[core::mem::MaybeUninit<u8>; $capacity]>::zeroed()
                .assume_init()
        }
    };

    ($like_ty:ty; $capacity:expr) => {
        unsafe {
            extern crate core;

            core::mem::MaybeUninit::<[core::mem::MaybeUninit<$like_ty>; $capacity]>::zeroed()
                .assume_init()
        }
    };
}

/// Creates an uninitialized array of one `MaybeUninit` without
/// allocating, with the given size and alignment, suitable for
/// taking a slice of to pass into `BumpInto::from_slice`.
///
/// The size will be rounded up to the nearest multiple of the
/// given alignment.
///
/// The size must be a const expression of type `usize`.
/// The alignment must be a power-of-two integer literal.
///
/// # Example
///
/// ```rust
/// use bump_into::space_uninit_aligned;
/// use core::mem;
///
/// let mut space = space_uninit_aligned!(size: 64, align: 4);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 4);
/// ```
#[macro_export]
macro_rules! space_uninit_aligned {
    (size: $size:expr, align: $align:literal $(,)?) => {{
        extern crate core;

        #[repr(C, align($align))]
        struct Space {
            _contents: [core::primitive::u8; $size],
        }

        unsafe {
            core::mem::MaybeUninit::<[core::mem::MaybeUninit<Space>; 1]>::uninit().assume_init()
        }
    }};
}

/// Creates a zeroed array of one `MaybeUninit` without allocating,
/// with the given size and alignment, suitable for taking a slice
/// of to pass into `BumpInto::from_slice`.
///
/// The size will be rounded up to the nearest multiple of the
/// given alignment.
///
/// The size must be a const expression of type `usize`.
/// The alignment must be a power-of-two integer literal.
///
/// # Example
///
/// ```rust
/// use bump_into::space_zeroed_aligned;
/// use core::mem;
///
/// let mut space = space_zeroed_aligned!(size: 64, align: 4);
/// assert_eq!(mem::size_of_val(&space), 64);
/// assert_eq!(mem::align_of_val(&space), 4);
/// ```
#[macro_export]
macro_rules! space_zeroed_aligned {
    (size: $size:expr, align: $align:literal $(,)?) => {{
        extern crate core;

        #[repr(C, align($align))]
        struct Space {
            _contents: [core::primitive::u8; $size],
        }

        unsafe {
            core::mem::MaybeUninit::<[core::mem::MaybeUninit<Space>; 1]>::zeroed().assume_init()
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn alloc() {
        let mut space = space_uninit!(64);
        let bump_into = BumpInto::from_slice(&mut space[..]);

        let something1 = bump_into.alloc(123u64).expect("allocation 1 failed");

        assert_eq!(*something1, 123u64);

        let something2 = bump_into.alloc(7775u16).expect("allocation 2 failed");

        assert_eq!(*something1, 123u64);
        assert_eq!(*something2, 7775u16);

        let something3 = bump_into
            .alloc_with(|| 251222u64)
            .ok()
            .expect("allocation 3 failed");

        assert_eq!(*something1, 123u64);
        assert_eq!(*something2, 7775u16);
        assert_eq!(*something3, 251222u64);

        if bump_into.alloc_with(|| [0u32; 128]).is_ok() {
            panic!("allocation 4 succeeded");
        }

        let something5 = bump_into.alloc(123523u32).expect("allocation 5 failed");

        assert_eq!(*something1, 123u64);
        assert_eq!(*something2, 7775u16);
        assert_eq!(*something3, 251222u64);
        assert_eq!(*something5, 123523u32);
    }

    #[test]
    fn alloc_n() {
        let mut space = space_uninit!(192);
        let bump_into = BumpInto::from_slice(&mut space[..]);

        let something1 = bump_into
            .alloc_slice(&[1u32, 258909, 1000][..])
            .expect("allocation 1 failed");

        assert_eq!(something1, &[1u32, 258909, 1000][..]);

        let something2 = bump_into
            .alloc_slice(&[1u64, 258909, 1000, 0][..])
            .expect("allocation 2 failed");

        assert_eq!(something1, &[1u32, 258909, 1000][..]);
        assert_eq!(something2, &[1u64, 258909, 1000, 0][..]);

        let something3 = bump_into
            .alloc_n_with(5, core::iter::repeat(61921u16))
            .expect("allocation 3 failed");

        assert_eq!(something1, &[1u32, 258909, 1000][..]);
        assert_eq!(something2, &[1u64, 258909, 1000, 0][..]);
        assert_eq!(something3, &[61921u16; 5][..]);

        let something4 = bump_into
            .alloc_n_with(6, core::iter::once(71u64))
            .expect("allocation 4 failed");

        assert_eq!(something1, &[1u32, 258909, 1000][..]);
        assert_eq!(something2, &[1u64, 258909, 1000, 0][..]);
        assert_eq!(something3, &[61921u16; 5][..]);
        assert_eq!(something4, &[71u64][..]);

        if bump_into.alloc_n_with::<u64, _>(100, None).is_ok() {
            panic!("allocation 5 succeeded");
        }

        let something6 = bump_into
            .alloc_n_with::<u64, _>(6, None)
            .expect("allocation 6 failed");

        assert_eq!(something1, &[1u32, 258909, 1000][..]);
        assert_eq!(something2, &[1u64, 258909, 1000, 0][..]);
        assert_eq!(something3, &[61921u16; 5][..]);
        assert_eq!(something4, &[71u64][..]);
        assert_eq!(something6, &[]);
    }

    #[test]
    fn available_bytes() {
        let mut space = space_uninit!(32);

        {
            let mut bump_into = BumpInto::from_slice(&mut space[..]);

            assert_eq!(bump_into.available_bytes(), 32);
            assert_eq!(bump_into.available_spaces(1usize, 1usize), 32);
            assert_eq!(bump_into.available_spaces_for::<u8>(), 32);

            bump_into.alloc(0u8).expect("allocation 1 failed");

            assert_eq!(bump_into.available_bytes(), 31);
            assert_eq!(bump_into.available_spaces(1usize, 1usize), 31);
            assert_eq!(bump_into.available_spaces_for::<u8>(), 31);

            let spaces_for_u32 = bump_into.available_spaces_for::<u32>();

            bump_into.alloc(0u32).expect("allocation 2 failed");

            assert_eq!(bump_into.available_spaces_for::<u32>(), spaces_for_u32 - 1);

            {
                let rest = bump_into.alloc_down_with(core::iter::repeat(0u32));

                assert_eq!(rest.len(), spaces_for_u32 - 1);
                assert!(rest.len() >= 6);

                for &x in rest.iter() {
                    assert_eq!(x, 0u32);
                }
            }

            assert_eq!(bump_into.available_spaces_for::<u32>(), 0);
            assert!(bump_into.available_bytes() < 4);
        }

        {
            let bump_into = BumpInto::from_slice(&mut space[..]);

            assert_eq!(bump_into.available_bytes(), 32);
            assert_eq!(bump_into.available_spaces(1usize, 1usize), 32);
            assert_eq!(bump_into.available_spaces_for::<u8>(), 32);

            let something4 = bump_into.alloc(0u8).expect("allocation 4 failed");

            assert_eq!(*something4, 0);

            let (pointer, count) = bump_into.alloc_space_to_limit_for::<i64>();

            assert_eq!(bump_into.available_spaces_for::<i64>(), 0);
            assert!(bump_into.available_bytes() < 8);
            assert!(count >= 3);

            let pointer = pointer.as_ptr();

            for x in 0..count {
                unsafe {
                    core::ptr::write(pointer.add(x), -1);
                }
            }

            assert_eq!(*something4, 0);
        }

        {
            let bump_into = BumpInto::from_slice(&mut space[..]);

            assert_eq!(bump_into.available_bytes(), 32);
            assert_eq!(bump_into.available_spaces(1usize, 1usize), 32);
            assert_eq!(bump_into.available_spaces_for::<u8>(), 32);

            let something6 = bump_into.alloc(0u8).expect("allocation 6 failed");

            assert_eq!(*something6, 0);

            let rest = unsafe {
                let mut count = 0u32;

                bump_into.alloc_down_with_shared(core::iter::from_fn(|| {
                    if bump_into.available_spaces_for::<u32>() > 1 {
                        count += 1;
                        Some(count)
                    } else {
                        None
                    }
                }))
            };

            assert_eq!(bump_into.available_spaces_for::<u32>(), 1);
            assert!(rest.len() >= 6);

            for (a, b) in rest.iter().zip((0..rest.len() as u32).rev()) {
                assert_eq!(*a, b + 1);
            }

            bump_into.alloc(0u32).expect("allocation 8 failed");

            assert_eq!(bump_into.available_spaces_for::<u32>(), 0);
            assert!(bump_into.available_bytes() < 4);
        }
    }

    #[test]
    fn space() {
        {
            let mut space = space_zeroed!(32);
            let bump_into = BumpInto::from_slice(&mut space[..]);

            for _ in 0..32 {
                let something1 = bump_into.alloc_space_for::<u8>();
                if something1.is_null() {
                    panic!("allocation 1 (in loop) failed");
                }
                unsafe {
                    assert_eq!(*something1, 0);
                }
            }
        }

        {
            let mut space = space_uninit!(u32; 32);
            let space_ptr = &space as *const _;
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let (something2_ptr, something2_size) = bump_into.alloc_space_to_limit_for::<u32>();
            let something2_ptr = something2_ptr.as_ptr() as *const u32;
            assert_eq!(space_ptr as *const u32, something2_ptr);
            assert_eq!(something2_size, 32);
        }

        {
            let mut space = space_zeroed!(u32; 32);
            let space_ptr = &space as *const _;
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let (something3_ptr, something3_size) = bump_into.alloc_space_to_limit_for::<u32>();
            let something3_ptr = something3_ptr.as_ptr() as *const u32;
            assert_eq!(space_ptr as *const u32, something3_ptr);
            assert_eq!(something3_size, 32);

            unsafe {
                for x in core::slice::from_raw_parts(something3_ptr, something3_size) {
                    assert_eq!(*x, 0);
                }
            }
        }

        {
            let mut space = space_uninit_aligned!(size: 32 * 4, align: 4);
            let space_ptr = &space as *const _;
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let (something4_ptr, something4_size) = bump_into.alloc_space_to_limit_for::<u32>();
            let something4_ptr = something4_ptr.as_ptr() as *const u32;
            assert_eq!(space_ptr as *const u32, something4_ptr);
            assert_eq!(something4_size, 32);
        }

        {
            let mut space = space_zeroed_aligned!(size: 32 * 4, align: 4);
            let space_ptr = &space as *const _;
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let (something5_ptr, something5_size) = bump_into.alloc_space_to_limit_for::<u32>();
            let something5_ptr = something5_ptr.as_ptr() as *const u32;
            assert_eq!(space_ptr as *const u32, something5_ptr);
            assert_eq!(something5_size, 32);

            unsafe {
                for x in core::slice::from_raw_parts(something5_ptr, something5_size) {
                    assert_eq!(*x, 0);
                }
            }
        }
    }

    #[test]
    fn single() {
        use core::mem::MaybeUninit;

        let mut space = MaybeUninit::<u32>::uninit();

        {
            let bump_into = BumpInto::from_slice(core::slice::from_mut(&mut space));
            let something1 = bump_into.alloc(0x8359u16).expect("allocation 1 failed");
            let something2 = bump_into.alloc(0x1312u16).expect("allocation 2 failed");
            assert_eq!(bump_into.available_bytes(), 0);
            assert_eq!(*something1, 0x8359);
            assert_eq!(*something2, 0x1312);
            *something1 = 0xACAB;
            assert_eq!(*something1, 0xACAB);
            assert_eq!(*something2, 0x1312);
        }

        unsafe {
            #[cfg(target_endian = "little")]
            assert_eq!(space.assume_init(), 0xACAB1312);
            #[cfg(target_endian = "big")]
            assert_eq!(space.assume_init(), 0x1312ACAB);
        }
    }

    #[test]
    fn moving() {
        let mut space = space_uninit!(32);
        let bump_into = BumpInto::from_slice(&mut space[..]);

        let something1 = bump_into.alloc(123u64).expect("allocation 1 failed");

        assert_eq!(*something1, 123u64);

        core::mem::drop(bump_into);

        assert_eq!(*something1, 123u64);
    }

    #[test]
    fn alloc_inside_alloc_with() {
        let mut space = space_uninit!(u32; 8);

        {
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let mut something2: Option<&mut [u32]> = None;
            let something1: &mut u32 = bump_into
                .alloc_with(|| {
                    let inner_something = bump_into
                        .alloc_n_with(bump_into.available_spaces_for::<u32>(), 0u32..)
                        .expect("inner allocation failed");

                    let something1 = inner_something.iter().sum();

                    something2 = Some(inner_something);

                    something1
                })
                .ok()
                .expect("allocation 1 failed");

            assert_eq!(*something1, (0..7).sum());
            assert_eq!(something2, Some(&mut [0, 1, 2, 3, 4, 5, 6][..]));
        }

        {
            let bump_into = BumpInto::from_slice(&mut space[..]);

            let mut something4: Option<&mut [u32]> = None;
            let something3: &mut [u32] = bump_into
                .alloc_n_with(
                    4,
                    core::iter::from_fn(|| {
                        let inner_something = bump_into
                            .alloc_n_with(bump_into.available_spaces_for::<u32>() / 2 + 1, 0u32..);

                        inner_something.ok().map(|inner_something| {
                            let something3 = inner_something.iter().sum();

                            something4 = Some(inner_something);

                            something3
                        })
                    }),
                )
                .ok()
                .expect("allocation 3 failed");

            assert_eq!(something3, &mut [(0..3).sum(), 0]);
            assert_eq!(something4, Some(&mut [0][..]));
        }
    }

    #[derive(Debug)]
    struct ZstWithDrop;

    impl Drop for ZstWithDrop {
        fn drop(&mut self) {
            panic!("ZstWithDrop was dropped!");
        }
    }

    fn zero_sized(space: &mut [MaybeUninit<u8>]) {
        let big_number = if cfg!(miri) { 0x100 } else { 0x10000 };

        let space_len = space.len();
        let bump_into = BumpInto::from_slice(space);

        assert_eq!(bump_into.available_bytes(), space_len);
        assert_eq!(bump_into.available_spaces(0usize, 0x100usize), usize::MAX);
        assert_eq!(bump_into.available_spaces_for::<ZstWithDrop>(), usize::MAX);

        let (nothing1_ptr, nothing1_count) = bump_into.alloc_space_to_limit_for::<ZstWithDrop>();
        assert!(!nothing1_ptr.as_ptr().is_null());
        assert_eq!(nothing1_count, usize::MAX);

        let _nothing2 = bump_into.alloc(ZstWithDrop).expect("allocation 2 failed");

        let _nothing3 = bump_into
            .alloc_with(|| ZstWithDrop)
            .ok()
            .expect("allocation 3 failed");

        let nothing4 = bump_into
            .alloc_slice(&[(), (), (), ()])
            .expect("allocation 4 failed");
        assert_eq!(nothing4, &[(), (), (), ()]);

        let nothing5 = bump_into
            .alloc_n_with(big_number, core::iter::repeat_with(|| ZstWithDrop))
            .ok()
            .expect("allocation 5 failed");
        assert_eq!(nothing5.len(), big_number);

        let nothing6 = unsafe {
            bump_into
                .alloc_down_with_shared(core::iter::repeat_with(|| ZstWithDrop).take(big_number))
        };
        assert_eq!(nothing6.len(), big_number);

        let nothing7_array = [(); usize::MAX];
        let nothing7 = bump_into
            .alloc_slice(&nothing7_array)
            .expect("allocation 7 failed");
        assert_eq!(nothing7.len(), usize::MAX);
    }

    #[test]
    fn zero_sized_0_space() {
        let mut space = space_uninit!(0);
        zero_sized(&mut space[..]);
    }

    #[test]
    fn zero_sized_32_space() {
        let mut space = space_uninit!(32);
        zero_sized(&mut space[..]);
    }

    #[test]
    #[ignore = "hangs when optimizations are off"]
    fn zero_sized_usize_max() {
        let mut space = space_uninit!(0);
        let mut bump_into = BumpInto::from_slice(&mut space[..]);

        let nothing1 = bump_into
            .alloc_n_with(usize::MAX, core::iter::repeat_with(|| ZstWithDrop))
            .ok()
            .expect("allocation 1 failed");
        assert_eq!(nothing1.len(), usize::MAX);

        let nothing2 = bump_into.alloc_down_with(core::iter::repeat_with(|| ZstWithDrop));
        assert_eq!(nothing2.len(), usize::MAX);
    }

    #[test]
    fn iteration_count() {
        let mut space = space_uninit!(u32; 32);
        let mut bump_into = BumpInto::from_slice(&mut space[..]);

        let mut iteration_count = 0u32;
        let something1 = bump_into
            .alloc_n_with(
                16,
                core::iter::repeat_with(|| {
                    iteration_count += 1;
                    iteration_count
                }),
            )
            .ok()
            .expect("allocation 1 failed");
        assert_eq!(
            something1,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        assert_eq!(iteration_count, 16);

        let mut iteration_count = 0usize;
        let nothing2 = bump_into
            .alloc_n_with(
                256,
                core::iter::repeat_with(|| {
                    iteration_count += 1;
                    ZstWithDrop
                }),
            )
            .ok()
            .expect("allocation 2 failed");
        assert_eq!(nothing2.len(), 256);
        assert_eq!(iteration_count, 256);

        let mut iteration_count = 0u32;
        let something3 = bump_into.alloc_down_with(core::iter::repeat_with(|| {
            iteration_count += 1;
            iteration_count
        }));
        assert_eq!(
            something3,
            &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        );
        assert_eq!(iteration_count, 16);
    }

    #[test]
    #[ignore = "hangs when optimizations are off"]
    fn iteration_count_usize_max() {
        let mut space = space_uninit!(u32; 32);
        let mut bump_into = BumpInto::from_slice(&mut space[..]);

        let mut iteration_count = 0u128;
        let nothing1 = bump_into
            .alloc_n_with(
                usize::MAX,
                core::iter::repeat_with(|| {
                    iteration_count += 1;
                    ZstWithDrop
                }),
            )
            .ok()
            .expect("allocation 1 failed");
        assert_eq!(nothing1.len(), usize::MAX);
        assert_eq!(iteration_count, usize::MAX as u128);

        let mut iteration_count = 0u128;
        let nothing2 = bump_into.alloc_down_with(core::iter::repeat_with(|| {
            iteration_count += 1;
            ZstWithDrop
        }));
        assert_eq!(nothing2.len(), usize::MAX);
        assert_eq!(iteration_count, usize::MAX as u128);
    }
}
