use super::*;

use byteorder::*;
use sharedmem::*;

use std::cmp::{max};
//use std::collections::{VecDeque};
use std::io::{Read, Write};
use std::mem::{size_of};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub fn write_flat_buf<T, W>(writer: &mut W, buf: &[T]) -> Result<(), ()> where T: Copy, W: Write {
  let num_bytes = buf.len() * size_of::<T>();
  let byte_buf = unsafe { from_raw_parts(buf.as_ptr() as *const u8, num_bytes) };
  writer.write_all(byte_buf).unwrap();
  Ok(())
}

pub fn read_flat_buf<T, R>(num_bytes: usize, reader: &mut R) -> Result<Vec<T>, ()> where T: Copy, R: Read {
  let cache_sz = max(16, (size_of::<T>() + 4096 - 1) / 4096) * 4096;
  let num_elems = num_bytes / size_of::<T>();
  assert_eq!(0, num_bytes % size_of::<T>());
  let mut buf = Vec::with_capacity(num_elems);
  let mut cache = Vec::with_capacity(cache_sz);
  cache.resize(cache_sz, 0);
  let mut cache_count = 0;
  let mut total_count = 0;
  loop {
    match reader.read(&mut cache) {
      Err(_) => return Err(()),
      Ok(count) => {
        cache_count += count;
        total_count += count;
        //assert_eq!(cache.len(), cache_count);
        if cache_count >= cache_sz {
          let cache_elems = cache_count / size_of::<T>();
          let num_leftover_bytes = cache_count - cache_elems * size_of::<T>();
          let elem_buf = unsafe { from_raw_parts(cache.as_ptr() as *const T, cache_elems) };
          buf.extend_from_slice(elem_buf);
          if num_leftover_bytes > 0 {
            // FIXME
            unimplemented!();
          }
          cache.clear();
          cache.resize(cache_sz, 0);
          cache_count = 0;
        }
        if count == 0 || total_count >= num_bytes {
          break;
        }
      }
    }
  }
  if cache_count > 0 {
    let cache_elems = cache_count / size_of::<T>();
    let num_leftover_bytes = cache_count - cache_elems * size_of::<T>();
    let elem_buf = unsafe { from_raw_parts(cache.as_ptr() as *const T, cache_elems) };
    buf.extend_from_slice(elem_buf);
    if num_leftover_bytes > 0 {
      // FIXME
      unimplemented!();
    }
    cache.clear();
    cache_count = 0;
  }
  assert_eq!(total_count, num_bytes);
  assert_eq!(buf.len(), num_elems);
  Ok(buf)
}

pub trait NdReader {
  fn deserialize(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
}

pub trait NdDatatype: Send + Sync + Copy {
  fn nd_type_id() -> u8;
}

impl NdDatatype for u8 {
  fn nd_type_id() -> u8 { 0 }
}

impl NdDatatype for f32 {
  fn nd_type_id() -> u8 { 1 }
}

impl<T> NdReader for Array2d<T, SharedMem<T>> where T: 'static + NdDatatype {
  fn deserialize(reader: &mut Read) -> Result<Self, ()> {
    let magic0 = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad magic: {:?}", e),
      Ok(x) => x,
    };
    let magic1 = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad magic: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad version: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(version, 0);
    let data_ty = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad data type id: {:?}", e),
      Ok(x) => x,
    };
    let ndim = match reader.read_u32::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad num dims: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(data_ty, T::nd_type_id());
    assert_eq!(ndim, 2);
    let dim0 = match reader.read_u64::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad dim 0: {:?}", e),
      Ok(x) => x as usize,
    };
    let dim1 = match reader.read_u64::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad dim 1: {:?}", e),
      Ok(x) => x as usize,
    };
    let dim = (dim0, dim1);
    let arr_len = dim.flat_len();
    let mut inner_buf: Vec<T> = Vec::with_capacity(arr_len);
    unsafe { inner_buf.set_len(arr_len); }
    {
      let mut data_bytes = unsafe { from_raw_parts_mut(inner_buf.as_mut_ptr() as *mut u8, size_of::<T>() * arr_len) };
      let mut read_ptr: usize = 0;
      loop {
        assert!(read_ptr <= data_bytes.len());
        if read_ptr == data_bytes.len() {
          break;
        }
        match reader.read(&mut data_bytes[read_ptr ..]) {
          Err(e) => panic!("failed to deserialize: read error: {:?}", e),
          Ok(count) => {
            if count == 0 {
              break;
            }
            read_ptr += count;
          }
        }
      }
      assert_eq!(read_ptr, data_bytes.len());
    }
    let buf = SharedMem::new(inner_buf);
    let mut arr = Array2d::from_storage(dim, buf);
    Ok(arr)
  }
}

impl<T> NdReader for Array3d<T, SharedMem<T>> where T: 'static + NdDatatype {
  fn deserialize(reader: &mut Read) -> Result<Self, ()> {
    let magic0 = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad magic: {:?}", e),
      Ok(x) => x,
    };
    let magic1 = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad magic: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(magic0, b'N');
    assert_eq!(magic1, b'D');
    let version = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad version: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(version, 0);
    let data_ty = match reader.read_u8() {
      Err(e) => panic!("failed to deserialize: bad data type id: {:?}", e),
      Ok(x) => x,
    };
    let ndim = match reader.read_u32::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad num dims: {:?}", e),
      Ok(x) => x,
    };
    assert_eq!(data_ty, T::nd_type_id());
    assert_eq!(ndim, 3);
    let dim0 = match reader.read_u64::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad dim 0: {:?}", e),
      Ok(x) => x as usize,
    };
    let dim1 = match reader.read_u64::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad dim 1: {:?}", e),
      Ok(x) => x as usize,
    };
    let dim2 = match reader.read_u64::<LittleEndian>() {
      Err(e) => panic!("failed to deserialize: bad dim 2: {:?}", e),
      Ok(x) => x as usize,
    };
    let dim = (dim0, dim1, dim2);
    let arr_len = dim.flat_len();
    let mut inner_buf: Vec<T> = Vec::with_capacity(arr_len);
    unsafe { inner_buf.set_len(arr_len); }
    {
      let mut data_bytes = unsafe { from_raw_parts_mut(inner_buf.as_mut_ptr() as *mut u8, size_of::<T>() * arr_len) };
      let mut read_ptr: usize = 0;
      loop {
        assert!(read_ptr <= data_bytes.len());
        if read_ptr == data_bytes.len() {
          break;
        }
        match reader.read(&mut data_bytes[read_ptr ..]) {
          Err(e) => panic!("failed to deserialize: read error: {:?}", e),
          Ok(count) => {
            if count == 0 {
              break;
            }
            read_ptr += count;
          }
        }
      }
      assert_eq!(read_ptr, data_bytes.len());
    }
    let buf = SharedMem::new(inner_buf);
    let mut arr = Array3d::from_storage(dim, buf);
    Ok(arr)
  }
}
