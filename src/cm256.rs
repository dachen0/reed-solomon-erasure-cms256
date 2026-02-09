//! `CM256ReedSolomon` — an MDS Reed-Solomon erasure codec backed by the
//! cm256 C++ library (Cauchy MDS GF(256) erasure code by Christopher A. Taylor).
//!
//! This provides the same high-level API as `crate::core::ReedSolomon<galois_8::Field>`
//! (`encode`, `verify`, `reconstruct`, etc.) but delegates the heavy lifting
//! to cm256's SIMD-accelerated encoder/decoder.
//!
//! cm256 is a true MDS code — decoding is guaranteed to succeed
//! whenever at least `data_shard_count` shards (any mix of data + parity) are
//! available.  The constraint is `data_shards + parity_shards <= 256`.

use std::os::raw::c_void;
use std::sync::Once;

use smallvec::SmallVec;

use crate::cm256_ffi::{self, CM256Block, CM256EncoderParams};
use crate::errors::Error;

static CM256_INIT: Once = Once::new();

fn ensure_init() {
    CM256_INIT.call_once(|| {
        if !cm256_ffi::init() {
            panic!("cm256_init() failed — unsupported platform");
        }
    });
}

/// Reed-Solomon erasure encoder/decoder backed by cm256.
#[derive(Debug, Clone, PartialEq)]
pub struct CM256ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
}

impl CM256ReedSolomon {
    /// Create a new encoder/decoder.
    ///
    /// Constraints: `data_shards >= 1`, `parity_shards >= 1`,
    /// `data_shards + parity_shards <= 256`.
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > 256 {
            return Err(Error::TooManyShards);
        }

        ensure_init();

        Ok(CM256ReedSolomon {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: data_shards + parity_shards,
        })
    }

    pub fn data_shard_count(&self) -> usize {
        self.data_shard_count
    }

    pub fn parity_shard_count(&self) -> usize {
        self.parity_shard_count
    }

    pub fn total_shard_count(&self) -> usize {
        self.total_shard_count
    }

    fn params(&self, block_bytes: usize) -> CM256EncoderParams {
        CM256EncoderParams {
            original_count: self.data_shard_count as _,
            recovery_count: self.parity_shard_count as _,
            block_bytes: block_bytes as _,
        }
    }

    // ------------------------------------------------------------------
    // Encoding
    // ------------------------------------------------------------------

    /// Encode parity shards from data shards (combined slice).
    ///
    /// `shards` must have `total_shard_count` entries, all the same length.
    /// The first `data_shard_count` are read; the remaining parity slots
    /// are overwritten.
    pub fn encode<T, U>(&self, mut shards: T) -> Result<(), Error>
    where
        T: AsRef<[U]> + AsMut<[U]>,
        U: AsRef<[u8]> + AsMut<[u8]>,
    {
        let slices: &mut [U] = shards.as_mut();
        self.check_all(slices.len())?;
        Self::check_slices_uniform(slices)?;

        let shard_size = slices[0].as_ref().len();
        let (data, parity) = slices.split_at_mut(self.data_shard_count);
        self.encode_raw(data, parity, shard_size)
    }

    /// Encode with separate data / parity references.
    pub fn encode_sep<T: AsRef<[u8]>, U: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        if data.len() != self.data_shard_count {
            return if data.len() < self.data_shard_count {
                Err(Error::TooFewDataShards)
            } else {
                Err(Error::TooManyDataShards)
            };
        }
        if parity.len() != self.parity_shard_count {
            return if parity.len() < self.parity_shard_count {
                Err(Error::TooFewParityShards)
            } else {
                Err(Error::TooManyParityShards)
            };
        }

        let shard_size = data[0].as_ref().len();
        if shard_size == 0 {
            return Err(Error::EmptyShard);
        }
        for d in data.iter() {
            if d.as_ref().len() != shard_size {
                return Err(Error::IncorrectShardSize);
            }
        }
        for p in parity.iter() {
            if p.as_ref().len() != shard_size {
                return Err(Error::IncorrectShardSize);
            }
        }

        self.encode_raw(data, parity, shard_size)
    }

    /// Core encode — writes each recovery block directly into its parity
    /// shard using `cm256_encode_block` (zero-copy, no temp buffer).
    fn encode_raw<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
        shard_size: usize,
    ) -> Result<(), Error> {
        let params = self.params(shard_size);

        // Stack-allocated block descriptors (avoids heap alloc for ≤32 shards).
        let mut blocks: SmallVec<[CM256Block; 32]> = SmallVec::with_capacity(self.data_shard_count);
        for (i, d) in data.iter().enumerate() {
            blocks.push(CM256Block {
                block: d.as_ref().as_ptr() as *mut c_void,
                index: i as u8,
            });
        }

        // Encode each recovery block directly into its parity shard.
        for (i, p) in parity.iter_mut().enumerate() {
            let recovery_index = self.data_shard_count + i;
            unsafe {
                cm256_ffi::cm256_encode_block(
                    params,
                    blocks.as_mut_ptr(),
                    recovery_index as _,
                    p.as_mut().as_mut_ptr() as *mut c_void,
                );
            }
        }

        Ok(())
    }

    /// Encode only specific parity blocks (by recovery index) directly into
    /// the provided destination pointers.  Re-uses a pre-built blocks array
    /// to avoid rebuilding it.
    fn encode_blocks_into(
        params: CM256EncoderParams,
        blocks: &mut SmallVec<[CM256Block; 32]>,
        data_shard_count: usize,
        targets: &[(usize, *mut u8)], // (parity_index, dst_ptr)
    ) {
        for &(parity_idx, dst) in targets {
            let recovery_index = data_shard_count + parity_idx;
            unsafe {
                cm256_ffi::cm256_encode_block(
                    params,
                    blocks.as_mut_ptr(),
                    recovery_index as _,
                    dst as *mut c_void,
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Verification
    // ------------------------------------------------------------------

    /// Verify that the parity shards are consistent with the data shards.
    pub fn verify<T: AsRef<[u8]>>(&self, slices: &[T]) -> Result<bool, Error> {
        self.check_all(slices.len())?;
        Self::check_slices_uniform(slices)?;

        let shard_size = slices[0].as_ref().len();
        let data = &slices[..self.data_shard_count];
        let existing_parity = &slices[self.data_shard_count..];

        // Re-encode into a temporary buffer and compare.
        let mut buf: Vec<Vec<u8>> = (0..self.parity_shard_count)
            .map(|_| vec![0u8; shard_size])
            .collect();

        self.encode_raw(data, &mut buf, shard_size)?;

        for (expected, actual) in buf.iter().zip(existing_parity.iter()) {
            if expected.as_slice() != actual.as_ref() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // ------------------------------------------------------------------
    // Reconstruction
    // ------------------------------------------------------------------

    /// Reconstruct all missing shards in-place.
    pub fn reconstruct<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [(T, bool)],
    ) -> Result<(), Error> {
        self.reconstruct_internal(shards, false)
    }

    /// Reconstruct only missing data shards.
    pub fn reconstruct_data<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [(T, bool)],
    ) -> Result<(), Error> {
        self.reconstruct_internal(shards, true)
    }

    fn reconstruct_internal<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [(T, bool)],
        data_only: bool,
    ) -> Result<(), Error> {
        self.check_all(shards.len())?;

        // Determine shard_size from the first present shard.
        let mut shard_size: Option<usize> = None;
        let mut present = 0usize;
        for (s, exists) in shards.iter() {
            if *exists {
                if s.as_ref().is_empty() {
                    return Err(Error::EmptyShard);
                }
                if let Some(sz) = shard_size {
                    if s.as_ref().len() != sz {
                        return Err(Error::IncorrectShardSize);
                    }
                }
                shard_size = Some(s.as_ref().len());
                present += 1;
            }
        }

        if present == self.total_shard_count {
            return Ok(()); // nothing missing
        }
        if present < self.data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }

        let shard_size = shard_size.expect("at least one shard present");
        let params = self.params(shard_size);

        // Collect missing indices on the stack.
        let mut data_missing: SmallVec<[usize; 32]> =
            SmallVec::with_capacity(self.data_shard_count);
        for i in 0..self.data_shard_count {
            if !shards[i].1 {
                data_missing.push(i);
            }
        }

        // ---- Recover missing DATA shards via cm256_decode ----
        if !data_missing.is_empty() {
            let mut blocks: SmallVec<[CM256Block; 32]> =
                SmallVec::with_capacity(self.data_shard_count);

            // Split into data / parity halves so we can simultaneously read
            // recovery (parity) shards and write into missing data shards
            // without conflicting borrows.
            let (data_shards, parity_shards) = shards.split_at_mut(self.data_shard_count);

            let mut recovery_iter = 0..self.parity_shard_count;
            for i in 0..self.data_shard_count {
                if data_shards[i].1 {
                    blocks.push(CM256Block {
                        block: data_shards[i].0.as_ref().as_ptr() as *mut c_void,
                        index: i as u8,
                    });
                } else {
                    let parity_local = loop {
                        let pi = recovery_iter.next().expect("not enough recovery shards");
                        if parity_shards[pi].1 {
                            break pi;
                        }
                    };
                    let recov_global = self.data_shard_count + parity_local;

                    // Copy recovery data directly into the target shard's
                    // pre-allocated buffer.  cm256_decode will then decode
                    // in-place.
                    data_shards[i]
                        .0
                        .as_mut()
                        .copy_from_slice(parity_shards[parity_local].0.as_ref());

                    blocks.push(CM256Block {
                        block: data_shards[i].0.as_mut().as_mut_ptr() as *mut c_void,
                        index: recov_global as u8,
                    });
                }
            }

            let rc = unsafe { cm256_ffi::cm256_decode(params, blocks.as_mut_ptr()) };
            if rc != 0 {
                return Err(Error::TooFewShardsPresent);
            }
            // Decoded data is already in the correct shard buffers.
        }

        // ---- Recompute only the missing PARITY shards ----
        if !data_only {
            let mut parity_missing: SmallVec<[usize; 32]> =
                SmallVec::with_capacity(self.parity_shard_count);
            for i in self.data_shard_count..self.total_shard_count {
                if !shards[i].1 {
                    parity_missing.push(i);
                }
            }

            if !parity_missing.is_empty() {
                // Build block descriptors from the (now-complete) data shards.
                let mut blocks: SmallVec<[CM256Block; 32]> =
                    SmallVec::with_capacity(self.data_shard_count);
                for i in 0..self.data_shard_count {
                    blocks.push(CM256Block {
                        block: shards[i].0.as_ref().as_ptr() as *mut c_void,
                        index: i as u8,
                    });
                }

                // Encode directly into each missing parity shard's buffer.
                for &idx in &parity_missing {
                    unsafe {
                        cm256_ffi::cm256_encode_block(
                            params,
                            blocks.as_mut_ptr(),
                            idx as _,
                            shards[idx].0.as_mut().as_mut_ptr() as *mut c_void,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn check_all(&self, count: usize) -> Result<(), Error> {
        if count < self.total_shard_count {
            Err(Error::TooFewShards)
        } else if count > self.total_shard_count {
            Err(Error::TooManyShards)
        } else {
            Ok(())
        }
    }

    fn check_slices_uniform<T: AsRef<[u8]>>(slices: &[T]) -> Result<(), Error> {
        if slices.is_empty() {
            return Ok(());
        }
        let first_len = slices[0].as_ref().len();
        if first_len == 0 {
            return Err(Error::EmptyShard);
        }
        for s in &slices[1..] {
            if s.as_ref().len() != first_len {
                return Err(Error::IncorrectShardSize);
            }
        }
        Ok(())
    }
}
