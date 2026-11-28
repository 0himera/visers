use core::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::{NanoEncoder, NanoWorkspace, RGB_BYTES};

struct SpinPoolShared {
    encoder: Arc<NanoEncoder>,
    worker_count: usize,
    epoch: AtomicU64,
    shutdown: AtomicBool,
    batch_active: AtomicBool,
    next_idx: AtomicUsize,
    done: AtomicUsize,
    images_ptr: AtomicPtr<()>,
    images_len: AtomicUsize,
    out_ptr: AtomicPtr<f32>,
}

struct ActiveBatch<'a>(&'a SpinPoolShared);

impl Drop for ActiveBatch<'_> {
    fn drop(&mut self) {
        // Every worker must stop using the borrowed buffers before this call can return or unwind.
        while self.0.done.load(Ordering::Acquire) < self.0.worker_count {
            spin_loop();
        }
        self.0.batch_active.store(false, Ordering::Release);
    }
}

/// Persistent spin-wait thread pool for multi-core `nano-v1` batch encoding.
pub struct NanoSpinPool {
    shared: Arc<SpinPoolShared>,
    workers: Vec<JoinHandle<()>>,
}

impl NanoSpinPool {
    /// Creates a persistent pool with `threads` total participants (`threads - 1` background
    /// workers + the calling thread). Never allocates or locks OS mutexes during `encode_batch`.
    pub fn new(encoder: Arc<NanoEncoder>, threads: usize) -> Self {
        let total = threads.max(1);
        let worker_count = total - 1;
        let shared = Arc::new(SpinPoolShared {
            encoder,
            worker_count,
            epoch: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            batch_active: AtomicBool::new(false),
            next_idx: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            images_ptr: AtomicPtr::new(core::ptr::null_mut()),
            images_len: AtomicUsize::new(0),
            out_ptr: AtomicPtr::new(core::ptr::null_mut()),
        });

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let s = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                let mut ws = s.encoder.workspace();
                let dim = s.encoder.embedding_dim();
                let mut seen_epoch = 0u64;
                loop {
                    let mut spins = 0u32;
                    let curr = loop {
                        let e = s.epoch.load(Ordering::Acquire);
                        if e != seen_epoch {
                            break e;
                        }
                        if s.shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        if spins < 4096 {
                            spin_loop();
                            spins += 1;
                        } else {
                            thread::yield_now();
                        }
                    };
                    if s.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    seen_epoch = curr;

                    let len = s.images_len.load(Ordering::Relaxed);
                    let img_base = s.images_ptr.load(Ordering::Relaxed).cast::<&[u8]>();
                    let out_base = s.out_ptr.load(Ordering::Relaxed);
                    loop {
                        let idx = s.next_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= len {
                            break;
                        }
                        // SAFETY: the active batch keeps both buffers alive until all workers
                        // finish. Each index has one owner and a disjoint output slice.
                        unsafe {
                            let img = *img_base.add(idx);
                            let dst = core::slice::from_raw_parts_mut(out_base.add(idx * dim), dim);
                            let _ = s.encoder.encode_rgb(img, &mut ws, dst);
                        }
                    }
                    s.done.fetch_add(1, Ordering::Release);
                }
            }));
        }

        Self { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.shared.worker_count + 1
    }

    /// Encodes `images` into `out` (`images.len() * dim`) with zero heap allocations and zero mutexes.
    /// With background workers, concurrent callers wait for the active batch to finish.
    pub fn encode_batch(
        &self,
        images: &[&[u8]],
        ws_main: &mut NanoWorkspace,
        out: &mut [f32],
    ) -> Result<(), &'static str> {
        let dim = self.shared.encoder.embedding_dim();
        let needed = images
            .len()
            .checked_mul(dim)
            .ok_or("batch output overflow")?;
        if out.len() < needed {
            return Err("batch output slice too small");
        }
        if images.is_empty() {
            return Ok(());
        }
        for &im in images {
            if im.len() != RGB_BYTES {
                return Err("nano-v1 expects 128x128x3 RGB bytes");
            }
        }
        if ws_main.head_in != self.shared.encoder.head_in {
            return Err("nano-v1 output/workspace mismatch");
        }

        if self.shared.worker_count == 0 {
            for (i, &im) in images.iter().enumerate() {
                self.shared
                    .encoder
                    .encode_rgb(im, ws_main, &mut out[i * dim..(i + 1) * dim])?;
            }
            return Ok(());
        }

        let shared = &*self.shared;
        while shared
            .batch_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            thread::yield_now();
        }

        let len = images.len();
        let out_ptr = out.as_mut_ptr();
        shared.next_idx.store(0, Ordering::Relaxed);
        shared.done.store(0, Ordering::Relaxed);
        shared
            .images_ptr
            .store(images.as_ptr().cast_mut().cast(), Ordering::Relaxed);
        shared.images_len.store(len, Ordering::Relaxed);
        shared.out_ptr.store(out_ptr, Ordering::Relaxed);

        // Release all workers simultaneously
        shared.epoch.fetch_add(1, Ordering::Release);
        let _batch = ActiveBatch(shared);

        // Calling thread participates in dynamic work stealing
        loop {
            let idx = shared.next_idx.fetch_add(1, Ordering::Relaxed);
            if idx >= len {
                break;
            }
            // SAFETY: each `idx` is claimed exclusively via atomic `fetch_add`.
            unsafe {
                let dst = core::slice::from_raw_parts_mut(out_ptr.add(idx * dim), dim);
                shared.encoder.encode_rgb(images[idx], ws_main, dst)?;
            }
        }

        Ok(())
    }
}

impl Drop for NanoSpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}
