/*use super::{Array2dView, Array2dViewMut};

pub trait MatrixOps<T> {
  fn matrix_prod(&mut self, alpha: T, a: Array2dView, b: (), beta: T); 
}*/

#[derive(Clone, Copy)]
pub enum Transpose {
  N,
  T,
}
