use super::{Array1dView, Array1dViewMut, Array2dView, Array2dViewMut};
use kernels::*;

use cblas_ffi::*;
use openblas_ffi::*;

#[derive(Clone, Copy)]
pub enum Transpose {
  N,
  T,
}

impl<'a> Array1dView<'a, f32> {
  pub fn l1_norm(&'a self) -> f32 {
    let x_n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    let mut x_sum = 0.0;
    for _ in 0 .. x_n {
      let x_i = self.buf[p].abs();
      x_sum += x_i;
      p += incx;
    }
    x_sum
  }

  pub fn l2_norm(&'a self) -> f32 {
    let n = self.dim();
    let incx = self.stride();
    unsafe { openblas_sequential_cblas_snrm2(
        n as _,
        self.buf.as_ptr(),
        incx as _,
    ) }
  }

  pub fn elem_sum(&'a self) -> f32 {
    let x_n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    let mut x_sum = 0.0;
    for _ in 0 .. x_n {
      let x_i = self.buf[p];
      x_sum += x_i;
      p += incx;
    }
    x_sum
  }

  pub fn inner_prod(&'a self, alpha: f32, y: Array1dView<'a, f32>) -> f32 {
    let x_n = self.dim();
    let y_n = y.dim();
    assert_eq!(x_n, y_n);
    let incx = self.stride();
    let incy = y.stride();
    unsafe { openblas_sequential_cblas_sdot(
        x_n as _,
        alpha,
        self.buf.as_ptr(),
        incx as _,
        y.as_ptr(),
        incy as _,
    ) }
  }
}

impl<'a> Array1dViewMut<'a, f32> {
  pub fn copy(&'a mut self, src: Array1dView<'a, f32>) {
    assert_eq!(self.dim(), src.dim());
    if self.stride() == 1 {
      unsafe { densearray_copy_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          src.buf.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn cast(&'a mut self, src: Array1dView<'a, u8>) {
    assert_eq!(self.dim(), src.dim());
    if self.stride() == 1 {
      unsafe { densearray_cast_u8_to_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          src.buf.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn add_scalar(&'a mut self, c: f32) {
    if self.stride() == 1 {
      unsafe { densearray_add_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          c,
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_add_scalar(&'a mut self, c: f32) {
    self.add_scalar(c);
  }

  pub fn scale(&'a mut self, alpha: f32) {
    if self.stride() == 1 {
      unsafe { densearray_scale_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          alpha,
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_scale(&'a mut self, alpha: f32) {
    self.scale(alpha);
  }

  pub fn div_scalar(&'a mut self, c: f32) {
    if self.stride() == 1 {
      unsafe { densearray_div_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          c,
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn square(&'a mut self) {
    if self.stride() == 1 {
      unsafe { densearray_square_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_square(&'a mut self) {
    self.square();
  }

  pub fn cube(&'a mut self) {
    if self.stride() == 1 {
      unsafe { densearray_cube_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn sqrt(&'a mut self) {
    if self.stride() == 1 {
      unsafe { densearray_sqrt_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_sqrt(&'a mut self) {
    self.sqrt();
  }

  pub fn reciprocal(&'a mut self) {
    if self.stride() == 1 {
      unsafe { densearray_reciprocal_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_recip(&'a mut self) {
    self.reciprocal();
  }

  pub fn exp(&mut self) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      let x_i = self.buf[p];
      self.buf[p] = x_i.exp();
      p += incx;
    }
  }

  pub fn add(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    if self.stride() == 1 {
      unsafe { densearray_vector_add_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          x.as_ptr(),
          alpha,
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_add(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    self.add(alpha, x);
  }

  pub fn average(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    if self.stride() == 1 {
      unsafe { densearray_vector_average_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          x.as_ptr(),
          alpha,
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn elem_mult(&'a mut self, /*alpha: f32,*/ x: Array1dView<'a, f32>) {
    if self.stride() == 1 {
      unsafe { densearray_elem_mult_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          x.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn vector_elem_mult(&'a mut self, x: Array1dView<'a, f32>) {
    self.elem_mult(x);
  }

  pub fn elem_div(&'a mut self, x: Array1dView<'a, f32>) {
    if self.stride() == 1 {
      unsafe { densearray_elem_div_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          x.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn elem_ldiv(&'a mut self, x: Array1dView<'a, f32>) {
    if self.stride() == 1 {
      unsafe { densearray_elem_ldiv_f32(
          self.buf.as_mut_ptr(),
          self.dim(),
          x.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn matrix_vector_prod(&'a mut self, alpha: f32, a: Array2dView<'a, f32>, a_trans: Transpose, x: Array1dView<'a, f32>, beta: f32) {
    let (a_m, a_n) = a.dim();
    let x_n = x.dim();
    let y_m = self.dim();
    let (at_m, at_n) = match a_trans {
      Transpose::N => (a_m, a_n),
      Transpose::T => (a_n, a_m),
    };
    assert_eq!(y_m, at_m);
    assert_eq!(x_n, at_n);
    let k = at_n;
    let (a_inc, lda) = a.stride();
    let x_inc = x.stride();
    let y_inc = self.stride();
    assert_eq!(1, a_inc);
    unsafe { openblas_sequential_cblas_sgemv(
        CblasOrder::ColMajor,
        match a_trans {
          Transpose::N => CblasTranspose::NoTrans,
          Transpose::T => CblasTranspose::Trans,
        },
        a_m as _, a_n as _,
        alpha,
        a.buf.as_ptr(), lda as _,
        x.buf.as_ptr(), x_inc as _,
        beta,
        self.buf.as_mut_ptr(), y_inc as _,
    ) };
  }

  pub fn symm_linear_solve(&'a mut self, a: Array2dViewMut<'a, f32>, b: Array1dView<'a, f32>) {
    let (a_m, n) = a.dim();
    let b_n = b.dim();
    let x_n = self.dim();
    assert_eq!(a_m, n);
    assert_eq!(b_n, n);
    assert_eq!(x_n, n);
    let (a_inc, lda) = a.stride();
    let b_inc = b.stride();
    let x_inc = self.stride();
    assert_eq!(1, a_inc);
    assert_eq!(1, b_inc);
    assert_eq!(1, x_inc);
    unsafe { openblas_sequential_LAPACKE_spotrf(
        CblasOrder::ColMajor as i32,
        'L' as i8,
        n as _,
        a.buf.as_mut_ptr(), lda as _,
    ) };
    { self.buf.copy_from_slice(b.buf) };
    unsafe { openblas_sequential_LAPACKE_spotrs(
        CblasOrder::ColMajor as i32,
        'L' as i8,
        n as _, 1,
        a.buf.as_ptr(), lda as _,
        self.buf.as_mut_ptr(), n as _,
    ) };
  }
}

impl<'a> Array2dView<'a, f32> {
  pub fn matrix_diag(&'a self, y: Array1dViewMut<'a, f32>) {
    let (a_m, a_n) = self.dim();
    let y_m = y.dim();
    assert_eq!(a_m, a_n);
    assert_eq!(a_m, y_m);
    let (a_inc, lda) = self.stride();
    let incy = y.stride();
    assert_eq!(1, a_inc);
    let mut p = 0;
    let mut q = 0;
    for _ in 0 .. y_m {
      y.buf[q] = self.buf[p];
      p += a_inc + lda;
      q += incy;
    }
  }
}

impl<'a> Array2dViewMut<'a, f32> {
  pub fn matrix_add(&'a mut self, alpha: f32, x: Array2dView<'a, f32>) {
    let (x_m, x_n) = x.dim();
    let (y_m, y_n) = self.dim();
    assert_eq!(x_m, y_m);
    assert_eq!(x_n, y_n);
    let (incx, ldx) = x.stride();
    let (incy, ldy) = self.stride();
    if x_n == 1 {
      unsafe { openblas_sequential_cblas_saxpy(
          x_m as _,
          alpha,
          x.buf.as_ptr(),
          incx as _,
          self.buf.as_mut_ptr(),
          incy as _,
      ) };
    } else if x_m == 1 {
      unimplemented!();
    } else {
      unimplemented!();
    }
  }

  pub fn matrix_prod(&'a mut self, alpha: f32, a: Array2dView<'a, f32>, a_trans: Transpose, b: Array2dView<'a, f32>, b_trans: Transpose, beta: f32) {
    let (a_m, a_n) = a.dim();
    let (b_m, b_n) = b.dim();
    let (c_m, c_n) = self.dim();
    let (at_m, at_n) = match a_trans {
      Transpose::N => (a_m, a_n),
      Transpose::T => (a_n, a_m),
    };
    let (bt_m, bt_n) = match b_trans {
      Transpose::N => (b_m, b_n),
      Transpose::T => (b_n, b_m),
    };
    assert_eq!(c_m, at_m);
    assert_eq!(c_n, bt_n);
    assert_eq!(at_n, bt_m);
    let k = at_n;
    let (a_inc, lda) = a.stride();
    let (b_inc, ldb) = b.stride();
    let (c_inc, ldc) = self.stride();
    assert_eq!(1, a_inc);
    assert_eq!(1, b_inc);
    assert_eq!(1, c_inc);
    unsafe { openblas_sequential_cblas_sgemm(
        CblasOrder::ColMajor,
        match a_trans {
          Transpose::N => CblasTranspose::NoTrans,
          Transpose::T => CblasTranspose::Trans,
        },
        match b_trans {
          Transpose::N => CblasTranspose::NoTrans,
          Transpose::T => CblasTranspose::Trans,
        },
        c_m as _, c_n as _, k as _,
        alpha,
        a.buf.as_ptr(), lda as _,
        b.buf.as_ptr(), ldb as _,
        beta,
        self.buf.as_mut_ptr(), ldc as _,
    ) };
  }
}

impl<'a> Array2dViewMut<'a, f64> {
  pub fn matrix_prod(&'a mut self) {
    unimplemented!();
  }
}

/*pub fn symmetric_tridiagonal_eigenvalues_workspace_size(dim: usize) -> usize {
  // TODO: LAPACK `sstebz`.
  unimplemented!();
}*/

pub fn solve_symmetric_tridiagonal_eigenvalues(
    diag: &[f32],
    offdiag: &[f32],
    eigenvals: &mut [f32],
    //workspace: &mut [u8],
    abs_tol: f32,
) -> Result<usize, ()>
{
  let n = diag.len();
  assert_eq!(n - 1, offdiag.len());
  assert_eq!(n, eigenvals.len());
  for i in 0 .. n {
    eigenvals[i] = 0.0;
  }
  let mut m: i32 = 0;
  let mut nsplit: i32 = 0;
  /*let mut w: Vec<f32> = Vec::with_capacity(n);
  w.resize(n, 0.0);*/
  let mut iblock: Vec<i32> = Vec::with_capacity(n);
  iblock.resize(n, 0);
  let mut isplit: Vec<i32> = Vec::with_capacity(n);
  isplit.resize(n, 0);
  let status = unsafe { openblas_sequential_LAPACKE_sstebz(
      'A' as i8,
      'E' as i8,
      n as _,
      0.0, 0.0,
      0, 0,
      abs_tol,
      diag.as_ptr(),
      offdiag.as_ptr(),
      &mut m as *mut _,
      &mut nsplit as *mut _,
      eigenvals.as_mut_ptr(),
      iblock.as_mut_ptr(),
      isplit.as_mut_ptr(),
  ) };
  if status < 0 {
    return Err(());
  }
  Ok(m as usize)
}
