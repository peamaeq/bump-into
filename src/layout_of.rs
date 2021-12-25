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

#[cfg(feature = "nightly")]
pub(crate) mod nightly {
    use core::alloc::Layout;
    use core::mem::{align_of, size_of};

    #[derive(Default, Debug, Copy, Clone)]
    pub(crate) struct Single<const S: usize, const A: usize> {
        _sealed: (),
    }

    impl<T> super::Single<T> {
        /// Create a `Single<T>`.
        #[inline]
        pub(crate) fn new_nightly() -> Single<{ size_of::<T>() }, { align_of::<T>() }> {
            Single {
                _sealed: (),
            }
        }
    }

    impl<const S: usize, const A: usize> From<Single<S, A>> for Option<Layout> {
        #[inline]
        fn from(_: Single<S, A>) -> Option<Layout> {
            unsafe {
                Some(Layout::from_size_align_unchecked(S, A))
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub(crate) struct Array<const A: usize> {
        bytes: Option<usize>,
    }

    impl<T> super::Array<T> {
        /// Create an `Array<T>`.
        #[inline]
        pub(crate) fn from_len_nightly(len: usize) -> Array<{ align_of::<T>() }> {
            Array {
                bytes: size_of::<T>().checked_mul(len),
            }
        }
    }

    impl<const A: usize> From<Array<A>> for Option<Layout> {
        #[inline]
        fn from(array: Array<A>) -> Option<Layout> {
            match array.bytes {
                Some(bytes) => unsafe {
                    Some(Layout::from_size_align_unchecked(bytes, A))
                }
                None => None,
            }
        }
    }
}
