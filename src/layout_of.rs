/*!

Types implementing `Into<Option<Layout>>`, for use with
[`BumpInto::available_spaces`](crate::BumpInto::available_spaces)
and [`BumpInto::alloc_space`](crate::BumpInto::alloc_space).

These are mostly an implementation detail, but they may be useful
to users of the crate. By using one of these types, you allow the
conversion to be inlined into the function that calls it, which
may enable significant optimizations by the compiler.

*/

use core::alloc::Layout;
use core::marker::PhantomData;

/// The layout of a single value of type `T`.
///
/// Converting to `Option<Layout>` will always give `Some`.
#[derive(Default, Debug, Copy, Clone)]
pub struct Single<T> {
    _phantom: PhantomData<T>,
}

impl<T> Single<T> {
    /// Create a `Single<T>`.
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> From<Single<T>> for Option<Layout> {
    #[inline]
    fn from(_: Single<T>) -> Option<Layout> {
        Some(Layout::new::<T>())
    }
}

/// The layout of a `[T; len]`, with `len` known at runtime.
///
/// Converting to `Option<Layout>` will give `None` if the length
/// in bytes would overflow a `usize`.
#[derive(Debug, Copy, Clone)]
pub struct Array<T> {
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> Array<T> {
    /// Create an `Array<T>` with the given `len`.
    #[inline]
    pub fn from_len(len: usize) -> Self {
        Self {
            len,
            _phantom: PhantomData,
        }
    }

    /// Get the `len` this `Array<T>` was created with.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> From<Array<T>> for Option<Layout> {
    #[inline]
    fn from(array: Array<T>) -> Option<Layout> {
        Layout::array::<T>(array.len).ok()
    }
}
