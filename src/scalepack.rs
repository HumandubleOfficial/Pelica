//! Packed expert scales: a lossless side file that makes every expert ~9%
//! smaller on disk, in RAM and in VRAM.
//!
//! An int4 expert matrix has one f32 scale per 64 weights, 1.6 MB per expert,
//! but each matrix only uses a few dozen distinct scale values. So the side
//! file stores, per matrix, a table of its distinct values (up to 256) and a
//! one-byte index per group. Looking the index up in the table gives back the
//! exact f32, so the arithmetic, and the tokens, do not change.
//!
//! Layout: a 4096-byte header, then one record per routed expert in (layer,
//! expert) order. A record is three blocks (gate, up, down), each a 256-entry
//! f32 table followed by the group indices, padded to a whole number of
//! sectors so it can be read unbuffered.

use crate::config::Config;
use crate::quant::Q4_GROUP;
use crate::st::Shards;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const FILE: &str = "pelica_scales.bin";
const MAGIC: &[u8; 8] = b"PELSCL01";
const HEADER: u64 = 4096;
const TABLE: usize = 256;

/// Where each expert's packed scales live.
#[derive(Clone, Debug)]
pub struct Pack {
    pub path: PathBuf,
    first: usize,
    n_experts: usize,
    /// groups per matrix (gate, up and down all have inter * hidden / 64)
    pub groups: usize,
    pub record: u64,
}

impl Pack {
    /// Bytes of one matrix's block: table, then indices.
    pub fn block(&self) -> usize {
        TABLE * 4 + self.groups
    }

    pub fn offset(&self, layer: usize, e: usize) -> u64 {
        HEADER + ((layer - self.first) * self.n_experts + e) as u64 * self.record
    }

    fn layout(c: &Config) -> (usize, usize, u64) {
        let groups = c.moe_inter * c.hidden / Q4_GROUP;
        let record = ((3 * (TABLE * 4 + groups)) as u64).div_ceil(4096) * 4096;
        (c.first_dense, groups, record)
    }

    /// The side file in `dir` if it exists and matches this model.
    pub fn open(dir: &Path, c: &Config) -> Option<Pack> {
        let path = dir.join(FILE);
        let (first, groups, record) = Self::layout(c);
        let mut head = [0u8; 40];
        let f = std::fs::File::open(&path).ok()?;
        read_at(&f, &mut head, 0).ok()?;
        let num = |i: usize| u64::from_le_bytes(head[8 + 8 * i..16 + 8 * i].try_into().unwrap());
        let n_records = (c.n_layers - first) * c.n_experts;
        let ok = &head[..8] == MAGIC
            && num(0) == first as u64
            && num(1) == c.n_experts as u64
            && num(2) == groups as u64
            && num(3) == record
            && f.metadata().ok()?.len() == HEADER + n_records as u64 * record;
        ok.then(|| Pack { path, first, n_experts: c.n_experts, groups, record })
    }

    /// Open the side file, building it first if needed (a one-off pass over
    /// the experts' scales). None if it cannot be built; experts then use
    /// their f32 scales as stored.
    pub fn ensure(dir: &Path, c: &Config, shards: &Shards, prefix: &str) -> Option<Pack> {
        if let Some(p) = Self::open(dir, c) {
            return Some(p);
        }
        match build(dir, c, shards, prefix) {
            Ok(()) => Self::open(dir, c),
            Err(e) => {
                eprintln!("[pelica] not packing expert scales ({e:#}); using them as stored");
                None
            }
        }
    }
}

fn build(dir: &Path, c: &Config, shards: &Shards, prefix: &str) -> Result<()> {
    let (first, groups, record) = Pack::layout(c);
    let n_records = (c.n_layers - first) * c.n_experts;
    let total = HEADER + n_records as u64 * record;
    if let Some(free) = free_disk(dir) {
        if free < total + (4 << 30) {
            bail!("needs {:.1} GB and only {:.1} GB is free", total as f64 / 1e9, free as f64 / 1e9);
        }
    }
    eprintln!("[pelica] packing expert scales into {FILE} (one-off, {:.1} GB)", total as f64 / 1e9);
    let t0 = std::time::Instant::now();
    let tmp = dir.join(format!("{FILE}.tmp"));
    let mut out = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp).context("creating the side file")?);
    let mut head = vec![0u8; HEADER as usize];
    head[..8].copy_from_slice(MAGIC);
    for (i, v) in [first as u64, c.n_experts as u64, groups as u64, record].iter().enumerate() {
        head[8 + 8 * i..16 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    out.write_all(&head)?;

    let keys: Vec<(usize, usize)> = (first..c.n_layers).flat_map(|l| (0..c.n_experts).map(move |e| (l, e))).collect();
    let mut widest = 0;
    let mut done = 0;
    for batch in keys.chunks(256) {
        let recs: Vec<Result<(Vec<u8>, usize)>> = batch
            .par_iter()
            .map(|&(l, e)| {
                let mut rec = vec![0u8; record as usize];
                let mut most = 0;
                for (m, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let name = format!("{prefix}layers.{l}.mlp.experts.{e}.{proj}.weight.qs");
                    let info = shards.info(&name)?;
                    if info.len != groups * 4 {
                        bail!("{name}: {} bytes, expected {}", info.len, groups * 4);
                    }
                    let mut raw = vec![0u8; info.len];
                    read_at(&shards.files[info.shard], &mut raw, info.offset)?;
                    let bits: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
                    let mut table = bits.clone();
                    table.sort_unstable();
                    table.dedup();
                    if table.len() > TABLE {
                        bail!("{name} has {} distinct scales (more than {TABLE})", table.len());
                    }
                    most = most.max(table.len());
                    let block = &mut rec[m * (TABLE * 4 + groups)..(m + 1) * (TABLE * 4 + groups)];
                    for (i, v) in table.iter().enumerate() {
                        block[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    for (g, b) in bits.iter().enumerate() {
                        block[TABLE * 4 + g] = table.binary_search(b).unwrap() as u8;
                    }
                }
                Ok((rec, most))
            })
            .collect();
        for r in recs {
            let (rec, most) = r?;
            widest = widest.max(most);
            out.write_all(&rec)?;
        }
        done += batch.len();
        eprint!("\r[pelica] packing expert scales: {done}/{}   ", keys.len());
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp, dir.join(FILE))?;
    eprintln!(
        "\r[pelica] packed {} experts' scales in {:.0}s (at most {widest} distinct values per matrix)",
        keys.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(windows)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut got = 0;
    while got < buf.len() {
        let n = f.seek_read(&mut buf[got..], off + got as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        got += n;
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

/// Free bytes on the volume holding `dir`.
#[cfg(windows)]
fn free_disk(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let w: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
    (unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut a, &mut t, &mut f) } != 0).then_some(a)
}

#[cfg(not(windows))]
fn free_disk(_dir: &Path) -> Option<u64> {
    None
}
