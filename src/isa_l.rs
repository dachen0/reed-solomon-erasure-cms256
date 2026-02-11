extern crate alloc;

use alloc::alloc::{alloc_zeroed, dealloc, Layout};
use alloc::vec::Vec;
use core::fmt;
use core::ptr::NonNull;
#[cfg(not(feature = "avx512"))]
use isa_l_rust::ec_encode_data_avx2_gfni;
#[cfg(feature = "avx512")]
use isa_l_rust::ec_encode_data_avx512_gfni;
use isa_l_rust::ec_init_tables_gfni;
use std::os::raw::c_int;
use std::sync::OnceLock;

const ISA_L_TABLE_BYTES_PER_COEFF: usize = 32;
const ISA_L_ALIGN_BYTES: usize = 32;
const ISA_L_MIN_SHARDS_DEFAULT: usize = 16;

pub(crate) fn isal_min_shards() -> usize {
    static MIN_SHARDS: OnceLock<usize> = OnceLock::new();
    *MIN_SHARDS.get_or_init(|| {
        std::env::var("RSE_ISA_L_MIN_SHARDS")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(ISA_L_MIN_SHARDS_DEFAULT)
    })
}

struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Option<Self> {
        let layout = Layout::from_size_align(len, align).ok()?;
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)?;
        Some(Self { ptr, layout })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("size", &self.layout.size())
            .field("align", &self.layout.align())
            .finish()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

pub struct IsaLTables {
    k: usize,
    rows: usize,
    gftbls: AlignedBuf,
}

impl fmt::Debug for IsaLTables {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IsaLTables")
            .field("k", &self.k)
            .field("rows", &self.rows)
            .field("gftbls", &self.gftbls)
            .finish()
    }
}

pub(crate) fn try_code_some_slices<T: AsRef<[u8]>, U: AsMut<[u8]>>(
    matrix_rows: &[&[u8]],
    inputs: &[T],
    outputs: &mut [U],
    aligned: bool,
) -> bool {
    let k = inputs.len();
    let rows = outputs.len();

    if k == 0 || rows == 0 {
        return true;
    }

    debug_assert!(aligned);
    debug_assert_eq!(matrix_rows.len(), rows);

    let len = inputs[0].as_ref().len();
    if len == 0 {
        return true;
    }
    debug_assert!(len <= c_int::MAX as usize);
    debug_assert!(k <= c_int::MAX as usize);
    debug_assert!(rows <= c_int::MAX as usize);

    for row in matrix_rows.iter() {
        debug_assert!(row.len() >= k);
    }

    debug_assert!(len
        .checked_mul(k)
        .and_then(|v| v.checked_mul(rows))
        .is_some());

    let mut data_ptrs: Vec<*mut u8> = Vec::with_capacity(k);
    for input in inputs.iter() {
        let input_slice = input.as_ref();
        debug_assert_eq!(input_slice.len(), len);
        data_ptrs.push(input_slice.as_ptr() as *mut u8);
    }

    let mut coding_ptrs: Vec<*mut u8> = Vec::with_capacity(rows);
    for output in outputs.iter_mut() {
        let output_slice = output.as_mut();
        debug_assert_eq!(output_slice.len(), len);
        coding_ptrs.push(output_slice.as_mut_ptr());
    }

    debug_assert!(k.checked_mul(rows).is_some());
    let coeffs_len = k * rows;
    debug_assert!(coeffs_len
        .checked_mul(ISA_L_TABLE_BYTES_PER_COEFF)
        .is_some());
    let gftbls_len = coeffs_len * ISA_L_TABLE_BYTES_PER_COEFF;

    let mut coeffs: Vec<u8> = Vec::with_capacity(coeffs_len);
    for row in matrix_rows.iter() {
        coeffs.extend_from_slice(&row[..k]);
    }

    let gftbls = match AlignedBuf::new(gftbls_len, ISA_L_ALIGN_BYTES) {
        Some(buf) => buf,
        None => return false,
    };

    unsafe {
        ec_init_tables_gfni(
            k as c_int,
            rows as c_int,
            coeffs.as_ptr() as *mut u8,
            gftbls.as_ptr(),
        );
        #[cfg(feature = "avx512")]
        ec_encode_data_avx512_gfni(
            len as c_int,
            k as c_int,
            rows as c_int,
            gftbls.as_ptr(),
            data_ptrs.as_mut_ptr(),
            coding_ptrs.as_mut_ptr(),
        );
        #[cfg(not(feature = "avx512"))]
        ec_encode_data_avx2_gfni(
            len as c_int,
            k as c_int,
            rows as c_int,
            gftbls.as_ptr(),
            data_ptrs.as_mut_ptr(),
            coding_ptrs.as_mut_ptr(),
        );
    }

    true
}
