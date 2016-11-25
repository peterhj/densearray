#![feature(zero_one)]

//extern crate densearray_kernels;

extern crate byteorder;
extern crate cblas_ffi;
#[cfg(feature = "mkl_parallel")]
extern crate mkl_ffi;
extern crate openblas_ffi;

extern crate libc;

use std::marker::{PhantomData};
use std::mem::{size_of};
use std::num::{Zero};
use std::ops::{Deref, DerefMut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod kernels;
pub mod linalg;
pub mod parallel_linalg;
pub mod prelude;
pub mod serial;

pub trait ArrayIndex: Copy {
  fn least_stride(&self) -> Self;
  fn flat_len(&self) -> usize;
  fn offset(&self, stride: Self) -> usize;
  fn diff(&self, rhs: Self) -> Self;
}

impl ArrayIndex for usize {
  fn least_stride(&self) -> Self {
    1
  }

  fn flat_len(&self) -> usize {
    *self
  }

  fn offset(&self, stride: Self) -> usize {
    *self * stride
  }

  fn diff(&self, rhs: Self) -> Self {
    *self - rhs
  }
}

impl ArrayIndex for (usize, usize) {
  fn least_stride(&self) -> Self {
    (1, self.0)
  }

  fn flat_len(&self) -> usize {
    self.0 * self.1
  }

  fn offset(&self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1
  }

  fn diff(&self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1)
  }
}

impl ArrayIndex for (usize, usize, usize) {
  fn least_stride(&self) -> Self {
    (1, self.0, self.0 * self.1)
  }

  fn flat_len(&self) -> usize {
    self.0 * self.1 * self.2
  }

  fn offset(&self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1 + stride.2 * self.2
  }

  fn diff(&self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
  }
}

impl ArrayIndex for (usize, usize, usize, usize) {
  fn least_stride(&self) -> Self {
    (1, self.0, self.0 * self.1, self.0 * self.1 * self.2)
  }

  fn flat_len(&self) -> usize {
    self.0 * self.1 * self.2 * self.3
  }

  fn offset(&self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1 + stride.2 * self.2 + stride.3 * self.3
  }

  fn diff(&self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2, self.3 - rhs.3)
  }
}

pub trait Reshape<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape(self, dim: Idx) -> Target;
  //fn reshape_stride(self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait ReshapeMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape_mut(self, dim: Idx) -> Target;
  //fn reshape_mut_stride(self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait AsView<'a, Target> {
  fn as_view(&'a self) -> Target;
}

pub trait AsViewMut<'a, Target> {
  fn as_view_mut(&'a mut self) -> Target;
}

pub trait View<'a, Idx, Target> where Idx: ArrayIndex {
  fn view(self, lo: Idx, hi: Idx) -> Target;
}

pub trait ViewMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn view_mut(self, lo: Idx, hi: Idx) -> Target;
}

pub trait AliasBytes<'a, Target: ?Sized> {
  fn alias_bytes(self) -> Target;
}

pub trait AliasBytesMut<'a, Target: ?Sized> {
  fn alias_bytes_mut(self) -> Target;
}

// FIXME(20161104): this should be removed in favor of impl functions.
pub trait CastFrom<'a, Target: ?Sized> {
  fn cast_from(self, target: Target);
}

impl<'a> CastFrom<'a, &'a [u8]> for &'a mut [f32] {
  fn cast_from(self, xs: &'a [u8]) {
    for (x, y) in xs.iter().zip(self.iter_mut()) {
      *y = *x as f32;
    }
  }
}

impl<'a> AliasBytes<'a, &'a [f32]> for &'a [u8] {
  fn alias_bytes(self) -> &'a [f32] {
    let bytes_sz = self.len();
    let new_sz = bytes_sz / size_of::<f32>();
    assert_eq!(0, bytes_sz % size_of::<f32>());
    unsafe { from_raw_parts(self.as_ptr() as *const f32, new_sz) }
  }
}

impl<'a> AliasBytesMut<'a, &'a mut [f32]> for &'a mut [u8] {
  fn alias_bytes_mut(self) -> &'a mut [f32] {
    let bytes_sz = self.len();
    let new_sz = bytes_sz / size_of::<f32>();
    assert_eq!(0, bytes_sz % size_of::<f32>());
    unsafe { from_raw_parts_mut(self.as_mut_ptr() as *mut f32, new_sz) }
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize, usize, usize), Array4dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: (usize, usize, usize, usize)) -> Array4dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array4dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize, usize, usize), Array4dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: (usize, usize, usize, usize)) -> Array4dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array4dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array1dView<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array1dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array2dView<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dView{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

#[derive(Clone)]
pub struct Array1d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      usize,
  stride:   usize,
  _marker:  PhantomData<T>,
}

impl<T> Array1d<T> where T: Copy + Zero {
  pub fn zeros(dim: usize) -> Array1d<T> {
    let mut data = Vec::with_capacity(dim);
    data.resize(dim, T::zero());
    Array1d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array1d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array1dView<'a, T>> for Array1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array1dView<'a, T> {
    Array1dView{
      buf:      self.buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array1dViewMut<'a, T>> for Array1d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array1dViewMut<'a, T> {
    Array1dViewMut{
      buf:      self.buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct Array1dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      usize,
  stride:   usize,
}

impl<'a, T> Array1dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

impl<'a> Array1dView<'a, f32> {
}

impl<'a, T> View<'a, usize, Array1dView<'a, T>> for Array1dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: usize, hi: usize) -> Array1dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array1dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct Array1dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      usize,
  stride:   usize,
}

impl<'a, T> ViewMut<'a, usize, Array1dViewMut<'a, T>> for Array1dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: usize, hi: usize) -> Array1dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array1dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Array1dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == 1 {
      for i in 0 .. self.dim {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct Array2d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize),
  stride:   (usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array2d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize)) -> Array2d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array2d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array2d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize), buf: S) -> Array2d<T, S> {
    assert_eq!(dim.flat_len(), buf.len());
    Array2d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array2d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array2dView<'a, T>> for Array2d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array2dView<'a, T> {
    Array2dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array2dViewMut<'a, T>> for Array2d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array2dViewMut<'a, T> {
    Array2dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct Array2dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> Array2dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

impl<'a> Array2dView<'a, f32> {
}

impl<'a, T> View<'a, (usize, usize), Array2dView<'a, T>> for Array2dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct Array2dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> ViewMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Array2dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct Array3d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array3d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize, usize)) -> Array3d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array3d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize, usize), buf: S) -> Array3d<T, S> {
    assert_eq!(dim.flat_len(), buf.as_ref().len());
    Array3d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn into_storage(self) -> S {
    self.buf
  }

  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array3d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

#[derive(Clone)]
pub struct Array4d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array4d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize, usize, usize)) -> Array4d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array4d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize, usize, usize), buf: S) -> Array4d<T, S> {
    assert_eq!(dim.flat_len(), buf.as_ref().len());
    Array4d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array4dView<'a, T>> for Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array4dView<'a, T> {
    Array4dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array4dViewMut<'a, T>> for Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array4dViewMut<'a, T> {
    Array4dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct Array4dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

pub struct Array4dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}
