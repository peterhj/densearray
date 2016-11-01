extern crate libc;

use libc::*;

#[link(name = "densearray_kernels", kind = "static")]
extern "C" {
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
