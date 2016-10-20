use super::{Array1d, Array2d, Array3d};

use byteorder::{ReadBytesExt, LittleEndian};

use std::io::{Read, Write};
use std::mem::{size_of};
use std::num::{Zero};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait NdArrayDtype: Copy {
  fn dtype_id() -> u8;
}

impl NdArrayDtype for u8 {
  fn dtype_id() -> u8 { 0 }
}

impl NdArrayDtype for f32 {
  fn dtype_id() -> u8 { 1 }
}

pub trait NdArrayDeserialize {
  fn deserialize(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
}

pub trait NdArraySerialize {
  fn serialize(&self, writer: &mut Write) -> Result<(), ()>;
}

impl<T> NdArrayDeserialize for Array1d<T> where T: NdArrayDtype {
  fn deserialize(reader: &mut Read) -> Result<Array1d<T>, ()> {
    unimplemented!();
  }
}

impl<T> NdArraySerialize for Array1d<T> where T: NdArrayDtype {
  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    unimplemented!();
  }
}

impl<T> NdArrayDeserialize for Array2d<T> where T: NdArrayDtype {
  fn deserialize(reader: &mut Read) -> Result<Array2d<T>, ()> {
    unimplemented!();
  }
}

impl<T> NdArraySerialize for Array2d<T> where T: NdArrayDtype {
  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    unimplemented!();
  }
}

impl<T> NdArrayDeserialize for Array3d<T> where T: NdArrayDtype + Zero {
  fn deserialize(reader: &mut Read) -> Result<Array3d<T>, ()> {
    let magic0 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let magic1 = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = reader.read_u8()
      .ok().expect("failed to deserialize!");
    assert_eq!(version, 0);
    let data_ty = reader.read_u8()
      .ok().expect("failed to deserialize!");
    let ndim = reader.read_u32::<LittleEndian>()
      .ok().expect("failed to deserialize!");
    let expected_data_ty = T::dtype_id();
    assert_eq!(data_ty, expected_data_ty);
    assert_eq!(ndim, 3);
    let bound0 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound1 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let bound2 = reader.read_u64::<LittleEndian>()
      .ok().expect("failed to deserialize!") as usize;
    let dims = (bound0, bound1, bound2);
    //let mut arr = unsafe { Array3d::new(dims) };
    let mut arr = Array3d::zeros(dims);
    {
      let mut arr = arr.as_mut_slice();
      let mut data_bytes = unsafe { from_raw_parts_mut(arr.as_mut_ptr() as *mut u8, size_of::<T>() * arr.len()) };
      let mut read_idx: usize = 0;
      loop {
        match reader.read(&mut data_bytes[read_idx ..]) {
          Ok(n) => {
            read_idx += n;
            if n == 0 {
              break;
            }
          }
          Err(e) => panic!("failed to deserialize: {:?}", e),
        }
      }
      assert_eq!(read_idx, data_bytes.len());
    }
    Ok(arr)
  }
}

impl<T> NdArraySerialize for Array3d<T> where T: NdArrayDtype + Zero {
  fn serialize(&self, writer: &mut Write) -> Result<(), ()> {
    unimplemented!();
  }
}
