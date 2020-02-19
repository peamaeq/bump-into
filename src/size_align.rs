use core::marker::PhantomData;
use core::mem;

/// A zero-sized struct that becomes `core::mem::size_of::<T>::()`
/// when converted using `Into<usize>`.
#[derive(Default, Debug, Copy, Clone)]
pub struct SizeOf<T> {
    _phantom: PhantomData<T>,
}

impl<T> SizeOf<T> {
    #[inline]
    pub fn new() -> SizeOf<T> {
        SizeOf {
            _phantom: PhantomData,
        }
    }
}

impl<T> From<SizeOf<T>> for usize {
    #[inline]
    fn from(_: SizeOf<T>) -> usize {
        mem::size_of::<T>()
    }
}

/// A zero-sized struct that becomes `core::mem::align_of::<T>::()`
/// when converted using `Into<usize>`.
#[derive(Default, Debug, Copy, Clone)]
pub struct AlignOf<T> {
    _phantom: PhantomData<T>,
}

impl<T> AlignOf<T> {
    #[inline]
    pub fn new() -> AlignOf<T> {
        AlignOf {
            _phantom: PhantomData,
        }
    }
}

impl<T> From<AlignOf<T>> for usize {
    #[inline]
    fn from(_: AlignOf<T>) -> usize {
        mem::align_of::<T>()
    }
}
