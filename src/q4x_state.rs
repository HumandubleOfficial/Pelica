//! Prompt states on disk: everything the model keeps about a prompt's first
//! `pos` tokens (the DeltaNet states and conv histories from the GPU, the
//! attention layers' caches, the n-gram embedding's history), in one file
//! per state in `<model>.pelica-state/`, named by position and a hash of the
//! tokens. A later prompt that starts with those tokens (a new chat with the
//! same system prompt and tools, or an old chat opened again) loads the file
//! instead of reading them again: about 0.2 s for a state the model takes a
//! minute to rebuild.
//!
//! States are saved where the system prompt ends (the second `<|im_start|>`
//! when the first message is the system's) and where a prompt ends, once
//! enough new tokens went in since the last one. The folder keeps to a size
//! (`PELICA_STATE_GB`, default 8), the states used longest ago going first;
//! `PELICA_STATE_GB=0` turns it off.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::{Att, Engine, Snap, State};

const MAGIC: &[u8; 4] = b"PQST";
const VERSION: u32 = 1;
/// fewer new positions than this aren't worth a file: reading them again
/// takes a few seconds at most
pub(super) const MIN_SAVE: usize = 1024;
/// GPU readbacks at most this many words at a time
const READ_WORDS: usize = 16 << 20;

/// FNV-1a over the tokens
pub(super) fn hash(toks: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in toks {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn budget() -> u64 {
    std::env::var("PELICA_STATE_GB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(8 << 30, |g| (g.max(0.0) * 1e9) as u64)
}

fn file_name(pos: usize, h: u64) -> String {
    format!("{pos}-{h:016x}.pst")
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numbers (f32, u16, u32); any bytes are fine to write
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at + n > self.b.len() {
            bail!("the saved state is cut short");
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn u16s(&mut self) -> Result<Vec<u16>> {
        let n = self.u32()? as usize;
        Ok(self.take(n * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

impl Engine {
    /// What a saved state must agree with to be of use to this model.
    fn sig(&self, st: &State) -> u64 {
        let hp = &self.hp;
        hash(&[hp.n_layer, hp.d, self.gst_len() as usize, self.gcv_len() as usize, st.att.len(), hp.nkv * hp.hd, hp.ihd, hp.kpool, st.ple.len()].map(|v| v as u32))
    }

    /// The saved states: (position, token hash, file).
    fn saved(&self) -> Vec<(usize, u64, PathBuf)> {
        let Ok(rd) = std::fs::read_dir(&self.state_dir) else { return Vec::new() };
        rd.flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let (p, h) = name.strip_suffix(".pst")?.split_once('-')?;
                Some((p.parse().ok()?, u64::from_str_radix(h, 16).ok()?, e.path()))
            })
            .collect()
    }

    /// The furthest saved state that `prompt` starts with, past position
    /// `beyond` and short of the prompt's last token.
    pub(super) fn best_saved(&self, prompt: &[u32], beyond: usize) -> Option<(usize, PathBuf)> {
        if budget() == 0 {
            return None;
        }
        let lim = prompt.len().saturating_sub(1);
        self.saved().into_iter().filter(|(p, h, _)| *p > beyond && *p <= lim && hash(&prompt[..*p]) == *h).max_by_key(|x| x.0).map(|(p, _, f)| (p, f))
    }

    pub(super) fn has_saved(&self, toks: &[u32]) -> bool {
        self.state_dir.join(file_name(toks.len(), hash(toks))).exists()
    }

    /// Where the system prompt ends: the second message's `<|im_start|>`,
    /// when the first message is the system's.
    pub(super) fn sys_end(&self, prompt: &[u32]) -> Option<usize> {
        let (start, system) = self.chat_marks?;
        if prompt.first() != Some(&start) || prompt.get(1) != Some(&system) {
            return None;
        }
        prompt.iter().skip(1).position(|&t| t == start).map(|i| i + 1)
    }

    /// Words [at, at + n) of the GPU activations.
    fn read_region(&self, at: u32, n: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let m = (n - i).min(READ_WORDS);
            let mut b = self.gpu.batch();
            b.read(at + i as u32, m);
            out.extend(b.run()?.pop().unwrap_or_default());
            i += m;
        }
        Ok(out)
    }

    /// Saves the cache (its st.pos positions) to disk; the file is written
    /// in the background.
    pub(super) fn save_state(&self, st: &mut State) -> Result<()> {
        if budget() == 0 || st.pos == 0 {
            return Ok(());
        }
        let r = &self.r;
        let gst = self.read_region(r.gst, self.gst_len() as usize)?;
        let gcv = self.read_region(r.gcv, self.gcv_len() as usize)?;
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.sig(st).to_le_bytes());
        b.extend_from_slice(&(st.pos as u32).to_le_bytes());
        b.extend_from_slice(bytes_of(&st.toks[..st.pos]));
        b.extend_from_slice(bytes_of(&gst));
        b.extend_from_slice(bytes_of(&gcv));
        b.extend_from_slice(bytes_of(&st.ple));
        for a in &st.att {
            for v in [&a.k, &a.v, &a.ik, &a.pk] {
                b.extend_from_slice(&(v.len() as u32).to_le_bytes());
                b.extend_from_slice(bytes_of(v));
            }
        }
        st.saved_pos = st.pos;
        let dir = self.state_dir.clone();
        let name = file_name(st.pos, hash(&st.toks[..st.pos]));
        std::thread::spawn(move || {
            if let Err(e) = write_state(&dir, &name, &b) {
                eprintln!("[pelica] couldn't save the prompt state: {e:#}");
            }
        });
        Ok(())
    }

    /// Loads a saved state that `prompt` starts with into the cache;
    /// returns its position. The cache is left as it was if the file won't do.
    pub(super) fn load_state(&self, st: &mut State, file: &Path, prompt: &[u32]) -> Result<usize> {
        let data = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
        let mut c = Rd { b: &data, at: 0 };
        if c.take(4)? != MAGIC || c.u32()? != VERSION || c.u64()? != self.sig(st) {
            bail!("saved by another version or for another model");
        }
        let pos = c.u32()? as usize;
        let toks = c.u32s(pos)?;
        if pos > prompt.len() || toks[..] != prompt[..pos] {
            bail!("saved for other tokens");
        }
        let gst = c.f32s(self.gst_len() as usize)?;
        let gcv = c.f32s(self.gcv_len() as usize)?;
        let ple = c.f32s(st.ple.len())?;
        let mut att = Vec::with_capacity(st.att.len());
        for _ in 0..st.att.len() {
            att.push(Att { k: c.u16s()?, v: c.u16s()?, ik: c.u16s()?, pk: c.u16s()? });
        }
        let r = &self.r;
        let mut b = self.gpu.batch();
        b.upload(r.gst, &gst);
        b.upload(r.gcv, &gcv);
        b.run()?;
        // the prompt-end copies too: a prompt that stops here carries on from it
        self.fill(&[(r.gst, r.sgst, self.gst_len(), 0), (r.gcv, r.sgcv, self.gcv_len(), 0)])?;
        st.att = att;
        st.ple = ple;
        st.toks = toks;
        st.pos = pos;
        st.snap = Some(Snap { pos, ple: st.ple.clone() });
        st.saved_pos = pos;
        // used now: the last to go when the folder is pruned
        if let Ok(f) = std::fs::File::options().write(true).open(file) {
            let _ = f.set_modified(std::time::SystemTime::now());
        }
        Ok(pos)
    }
}

/// Writes a state file (through a temporary name, so a half-written file
/// is never taken for a state), then keeps the folder within its budget.
fn write_state(dir: &Path, name: &str, b: &[u8]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{name}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(b)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(name))?;
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".pst"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    files.sort_by_key(|f| f.0);
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    for (_, len, p) in &files {
        if total <= budget() || p.file_name().is_some_and(|n| n == name) {
            break;
        }
        if std::fs::remove_file(p).is_ok() {
            total -= len;
        }
    }
    Ok(())
}
