use super::{Array1d, Array2d};

use std::io::{Read, Write};

pub trait NdArraySerialize {
  fn deserialize(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn serialize(&self, writer: &mut Write) -> Result<(), ()>;
}

impl<T> NdArraySerialize for Array1d<T> where T: Copy {
  fn deserialize(reader: &mut Read) -> Result<Array1d<T>, ()> {
    unimplemented!();
  }

  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    unimplemented!();
  }
}

impl<T> NdArraySerialize for Array2d<T> where T: Copy {
  fn deserialize(reader: &mut Read) -> Result<Array2d<T>, ()> {
    unimplemented!();
  }

  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    unimplemented!();
  }
}
