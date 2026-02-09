//! Raw FFI bindings to the cm256 Cauchy MDS Reed-Solomon C library.

use std::os::raw::{c_int, c_uchar, c_void};

pub const CM256_VERSION: c_int = 2;

/// Encoder / decoder parameters.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CM256EncoderParams {
    /// Number of original (data) blocks, < 256.
    pub original_count: c_int,
    /// Number of recovery (parity) blocks, < 256.
    pub recovery_count: c_int,
    /// Bytes per block (all blocks same size).
    pub block_bytes: c_int,
}

/// Block descriptor.
#[repr(C)]
pub struct CM256Block {
    /// Pointer to the block data.
    pub block: *mut c_void,
    /// Block index.
    /// For originals: 0..originalCount-1
    /// For recovery:  originalCount..originalCount+recoveryCount-1
    pub index: c_uchar,
}

extern "C" {
    /// Initialize cm256. Returns 0 on success.
    pub fn cm256_init_(version: c_int) -> c_int;

    /// Encode: produce recovery blocks end-to-end.
    /// Returns 0 on success.
    pub fn cm256_encode(
        params: CM256EncoderParams,
        originals: *mut CM256Block,
        recovery_blocks: *mut c_void,
    ) -> c_int;

    /// Encode a single recovery block.
    /// `recovery_block_index` should be `original_count + i` for the i-th
    /// recovery block (use `cm256_get_recovery_block_index`).
    /// This does NOT validate input — use with care.
    pub fn cm256_encode_block(
        params: CM256EncoderParams,
        originals: *mut CM256Block,
        recovery_block_index: c_int,
        recovery_block: *mut c_void,
    );

    /// Decode: recover missing originals in-place.
    /// `blocks` is an array of `original_count` blocks, some of which may be
    /// recovery blocks (identified by `index >= original_count`). Recovery
    /// blocks are replaced with the reconstructed original data and their
    /// index is updated.
    /// Returns 0 on success.
    pub fn cm256_decode(
        params: CM256EncoderParams,
        blocks: *mut CM256Block,
    ) -> c_int;
}

/// Initialize cm256. Must be called once before any other cm256 function.
/// Returns `true` on success.
pub fn init() -> bool {
    unsafe { cm256_init_(CM256_VERSION) == 0 }
}
