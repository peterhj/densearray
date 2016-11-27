//extern crate libc;

use libc::*;

#[cfg(not(feature = "knl"))]
#[link(name = "gomp")]
extern "C" {}

#[cfg(feature = "knl")]
#[link(name = "iomp5")]
extern "C" {}

#[link(name = "densearray_kernels", kind = "static")]
extern "C" {
  pub fn densearray_copy_f32(
      dst: *mut f32,
      dim: size_t,
      src: *const f32);
  pub fn densearray_square_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_sqrt_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_reciprocal_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_add_scalar_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_scale_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_div_scalar_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_elem_mult_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_elem_div_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_elem_ldiv_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_vector_add_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32,
      alpha: f32);
  pub fn densearray_vector_average_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32,
      alpha: f32);
}

#[link(name = "densearray_omp_kernels", kind = "static")]
extern "C" {
  pub fn densearray_omp_copy_f32(
      dst: *mut f32,
      dim: size_t,
      src: *const f32);
  pub fn densearray_omp_square_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_omp_sqrt_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_omp_reciprocal_f32(
      dst: *mut f32,
      dim: size_t);
  pub fn densearray_omp_add_scalar_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_omp_scale_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_omp_div_scalar_f32(
      dst: *mut f32,
      dim: size_t,
      c: f32);
  pub fn densearray_omp_elem_mult_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_omp_elem_div_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_omp_elem_ldiv_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32);
  pub fn densearray_omp_vector_add_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32,
      alpha: f32);
  pub fn densearray_omp_vector_average_f32(
      dst: *mut f32,
      dim: size_t,
      xs: *const f32,
      alpha: f32);
}
