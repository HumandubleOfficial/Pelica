//! Saved chats: a long chat's cache kept on the drive, so reopening it (after
//! another chat, a model switch, or a restart) loads the rows it already
//! computed instead of reading the whole conversation again. Lossless: the
//! rows are exactly the ones the model wrote.
//!
//! Each model has a folder under the cache directory (`moe::kv_dir`)/saved.
//! A saved chat is two files: `<id>.tok` (its tokens, u32) and `<id>.kv`
//! (their rows, in blocks of BLOCK positions laid out as `Engine::read_rows`
//! does). After a reply only the new blocks are written. The oldest saved
//! chats go once they take more than CAP_GB (or a tenth of the free space).

use crate::moe::Any;
use anyhow::{Context, Result};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// positions per block of a `.kv` file
const BLOCK: usize = 256;
/// chats shorter than this aren't saved (they're read again in a moment)
pub const MIN: usize = 1024;
/// what loading must save over the cache already in memory to be worth it
const GAIN: usize = 256;
const CAP_GB: u64 = 8;

struct Entry {
    id: String,
    toks: Vec<u32>,
    used: SystemTime,
}

pub struct Saved {
    dir: PathBuf,
    pos: usize,
    list: Vec<Entry>,
    /// the entry whose files match the cache in memory (up to its tokens)
    pub live: Option<String>,
}

fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn toks_of(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn toks_bytes(t: &[u32]) -> Vec<u8> {
    t.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Saved {
    /// The saved chats for this model (None: its engine can't save).
    pub fn open(model: &Path, eng: &Any) -> Option<Saved> {
        let pos = eng.pos_bytes()?;
        // the model's name, size and date and the cache's shape name its folder
        let meta = fs::metadata(model).ok()?;
        let when = meta.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let mut h = fnv(model.file_name()?.to_string_lossy().as_bytes(), 0xcbf29ce484222325);
        h = fnv(&meta.len().to_le_bytes(), h);
        h = fnv(&when.to_le_bytes(), h);
        h = fnv(&(pos as u64).to_le_bytes(), h);
        let dir = crate::moe::kv_dir().join("saved").join(format!("{h:016x}"));
        let mut list = Vec::new();
        if let Ok(rd) = fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "tok") {
                    let id = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    let kv = p.with_extension("kv");
                    let toks = fs::read(&p).map(|b| toks_of(&b)).unwrap_or_default();
                    // (rows for every block its tokens need, or it's half written)
                    let need = (toks.len().div_ceil(BLOCK) * BLOCK * pos) as u64;
                    if toks.is_empty() || fs::metadata(&kv).map_or(true, |m| m.len() < need) {
                        let _ = fs::remove_file(&p);
                        let _ = fs::remove_file(&kv);
                        continue;
                    }
                    let used = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
                    list.push(Entry { id, toks, used });
                }
            }
        }
        Some(Saved { dir, pos, list, live: None })
    }

    fn block_bytes(&self) -> usize {
        BLOCK * self.pos
    }

    /// Before a reply to `ids` whose cache in memory holds `reuse` of them:
    /// loads the saved chat that covers more of them, if one does. Returns
    /// how many positions the cache now holds and their tokens.
    pub fn restore(&mut self, eng: &Any, ids: &[u32], reuse: usize) -> Result<Option<(usize, Vec<u32>)>> {
        let top = ids.len().saturating_sub(1);
        let Some((i, n)) = self.list.iter().enumerate().map(|(i, e)| (i, lcp(&e.toks, ids).min(top))).max_by_key(|x| x.1) else { return Ok(None) };
        if n < reuse + GAIN {
            return Ok(None);
        }
        let e = &self.list[i];
        let mut f = fs::File::open(self.dir.join(format!("{}.kv", e.id)))?;
        let mut buf = vec![0u8; self.block_bytes()];
        for b in 0..n.div_ceil(BLOCK) {
            f.read_exact(&mut buf).context("reading a saved chat")?;
            let (a, z) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
            eng.write_rows(a, z, BLOCK, &buf)?;
        }
        let toks = e.toks[..n].to_vec();
        let id = e.id.clone();
        self.touch(&id);
        // (its files hold more than the cache now does: keep them as they are)
        self.live = (n == self.list[i].toks.len()).then_some(id);
        Ok(Some((n, toks)))
    }

    fn touch(&mut self, id: &str) {
        if let Some(e) = self.list.iter_mut().find(|e| e.id == id) {
            e.used = SystemTime::now();
            let _ = fs::File::options().append(true).open(self.dir.join(format!("{id}.tok"))).and_then(|f| f.set_modified(e.used));
        }
    }

    /// After a reply: the cache holds `toks`; saves what isn't saved yet.
    pub fn keep(&mut self, eng: &Any, toks: &[u32]) -> Result<()> {
        if toks.len() < MIN {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)?;
        // carry on the live entry when the cache still starts with it
        let from = match self.live.as_ref().and_then(|id| self.list.iter().position(|e| &e.id == id)) {
            Some(i) if toks.starts_with(&self.list[i].toks) => {
                let had = self.list[i].toks.len();
                if had == toks.len() {
                    return Ok(());
                }
                had
            }
            _ => {
                let id = format!("{:016x}", fnv(&toks_bytes(&toks[..MIN]), SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)));
                self.list.push(Entry { id: id.clone(), toks: Vec::new(), used: SystemTime::now() });
                self.live = Some(id);
                0
            }
        };
        let id = self.live.clone().unwrap();
        let kv = self.dir.join(format!("{id}.kv"));
        let tok = self.dir.join(format!("{id}.tok"));
        let mut f = fs::File::options().create(true).write(true).truncate(false).open(&kv)?;
        let mut buf = vec![0u8; self.block_bytes()];
        let b0 = from / BLOCK;
        f.seek(SeekFrom::Start((b0 * self.block_bytes()) as u64))?;
        for b in b0..toks.len().div_ceil(BLOCK) {
            let (a, z) = (b * BLOCK, ((b + 1) * BLOCK).min(toks.len()));
            eng.read_rows(a, z, BLOCK, &mut buf)?;
            f.write_all(&buf)?;
        }
        f.sync_data()?;
        // the tokens last: a crash before this leaves the old, still true, list
        let tmp = tok.with_extension("tmp");
        fs::write(&tmp, toks_bytes(toks))?;
        fs::rename(&tmp, &tok)?;
        let e = self.list.iter_mut().find(|e| e.id == id).unwrap();
        e.toks = toks.to_vec();
        e.used = SystemTime::now();
        if from == 0 {
            self.drop_older(&id);
        }
        self.trim();
        Ok(())
    }

    fn remove(&mut self, id: &str) {
        let _ = fs::remove_file(self.dir.join(format!("{id}.tok")));
        let _ = fs::remove_file(self.dir.join(format!("{id}.kv")));
        self.list.retain(|e| e.id != id);
        if self.live.as_deref() == Some(id) {
            self.live = None;
        }
    }

    /// A new save that repeats most of an older one (the same chat before a
    /// retry or an edit) makes the older one useless.
    fn drop_older(&mut self, id: &str) {
        let new = self.list.iter().find(|e| e.id == id).map(|e| e.toks.clone()).unwrap_or_default();
        let old: Vec<String> = self.list.iter().filter(|e| e.id != id && lcp(&e.toks, &new) * 10 >= e.toks.len() * 9).map(|e| e.id.clone()).collect();
        for o in old {
            self.remove(&o);
        }
    }

    /// The oldest saved chats go past the cap.
    fn trim(&mut self) {
        let free = crate::moe::disk_free(&self.dir).unwrap_or(u64::MAX);
        let cap = (CAP_GB << 30).min(free / 10 + self.bytes());
        while self.bytes() > cap && self.list.len() > 1 {
            let oldest = self.list.iter().filter(|e| Some(&e.id) != self.live.as_ref()).min_by_key(|e| e.used).map(|e| e.id.clone());
            match oldest {
                Some(id) => self.remove(&id),
                None => break,
            }
        }
    }

    /// Bytes on the drive (an estimate from the token counts).
    pub fn bytes(&self) -> u64 {
        self.list.iter().map(|e| (e.toks.len().div_ceil(BLOCK) * self.block_bytes()) as u64).sum()
    }

    pub fn clear(&mut self) {
        let ids: Vec<String> = self.list.iter().map(|e| e.id.clone()).collect();
        for id in ids {
            self.remove(&id);
        }
    }
}

/// Every model's saved chats: bytes on the drive (for Settings, with no model loaded).
pub fn all_bytes() -> u64 {
    fn walk(d: &Path) -> u64 {
        fs::read_dir(d).map_or(0, |rd| rd.flatten().map(|e| if e.path().is_dir() { walk(&e.path()) } else { e.metadata().map_or(0, |m| m.len()) }).sum())
    }
    walk(&crate::moe::kv_dir().join("saved"))
}

/// Forgets every model's saved chats.
pub fn clear_all() -> Result<()> {
    let d = crate::moe::kv_dir().join("saved");
    if d.exists() {
        fs::remove_dir_all(&d).with_context(|| format!("removing {}", d.display()))?;
    }
    Ok(())
}
