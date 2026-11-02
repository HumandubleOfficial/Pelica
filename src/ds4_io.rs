//! Expert reads for the DeepSeek V4 engine (`ds4`): a RAM cache of expert
//! slots kept by heat (use counts), filled from the GGUF by reader threads
//! with unbuffered reads, so the OS file cache doesn't take the RAM.
//!
//! An expert is three pieces (its rows of the gate, up and down tensors),
//! each contiguous in the file. A slot holds the three, each read as the
//! 4 KiB-aligned span around it. Slots in use are pinned; a miss takes the
//! coldest unpinned slot. Heat is recent use: each use adds 1 and it halves
//! every `half_life()` tokens (replays of decode traces: 72% hits against 54%
//! for lifetime counts). A slower heat (half-life `slow_half_life()`) adds a
//! share of longer-run use, so a steady expert outlasts one used in a burst.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{bail, Result};

const ALIGN: usize = 4096;
/// reader threads (PELICA_READERS). Two whole-expert reads in flight fill an
/// NVMe drive (16 MB reads: 5.45 GB/s with 2, 5.15 with 4), and with fewer
/// guesses queued an urgent read waits less: 2 decoded 4% faster than 4
fn readers() -> usize {
    std::env::var("PELICA_READERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}
/// tokens over which an expert's heat halves (PELICA_HALF_LIFE). Alone, 16
/// did best; with the slower heat below, 8 (replays of decode picks: 3% fewer
/// reads than 16 alone; the engine read 3% less and decoded 3% faster)
fn half_life() -> f64 {
    std::env::var("PELICA_HALF_LIFE").ok().and_then(|v| v.parse().ok()).unwrap_or(8.0)
}
/// the slower heat's half-life in tokens and its weight (PELICA_SLOW_HL,
/// PELICA_SLOW_MIX; mix 0 = recent heat only)
fn slow_half_life() -> f64 {
    std::env::var("PELICA_SLOW_HL").ok().and_then(|v| v.parse().ok()).unwrap_or(128.0)
}
fn slow_mix() -> f32 {
    std::env::var("PELICA_SLOW_MIX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25)
}
/// heat a read-ahead guess starts with, so it isn't the next one evicted
const GUESS_HEAT: f32 = 0.5;

/// Where expert pieces sit: per layer, per piece (gate, up, down), expert
/// 0's file offset and the bytes one expert's piece takes, and the bytes from
/// one expert to the next. A plain GGUF keeps each piece in its own tensor
/// (stride 0: each piece steps by its own size). A layer `pelica repack` has
/// gone over keeps each expert's three pieces together (stride = all three):
/// one read gets it.
/// A model split over several files (`-00002-of-00003.gguf`) names, per
/// layer and piece, the file holding it (`file`, empty: all in the first).
#[derive(Clone)]
pub struct Layout {
    pub pieces: Vec<[(u64, usize); 3]>,
    pub stride: Vec<usize>,
    pub n_exp: usize,
    pub file: Vec<[u8; 3]>,
}

impl Layout {
    /// the file (of those `Store::new_multi` opened) holding piece p
    pub fn file_of(&self, key: u32, p: usize) -> usize {
        self.file.get(key as usize / self.n_exp).map_or(0, |f| f[p] as usize)
    }

    pub fn span(&self, key: u32, p: usize) -> (u64, usize) {
        let (l, e) = (key as usize / self.n_exp, key as usize % self.n_exp);
        let (off, len) = self.pieces[l][p];
        (off + (e * if self.stride[l] == 0 { len } else { self.stride[l] }) as u64, len)
    }

    /// Where piece p of a layer-l expert starts in its slot (p = 3: the
    /// bytes a slot needs for the expert): each piece has a region of its
    /// own, with room to read whole aligned blocks around it.
    pub fn region(&self, l: usize, p: usize) -> usize {
        self.pieces[l][..p].iter().map(|x| (x.1 + 2 * ALIGN).div_ceil(ALIGN) * ALIGN).sum()
    }

    /// The expert's three pieces sit back to back in the file.
    pub fn merged(&self, l: usize) -> bool {
        let p = &self.pieces[l];
        let one_file = self.file.get(l).map_or(true, |f| f[0] == f[1] && f[1] == f[2]);
        one_file && self.stride[l] != 0 && p[1].0 == p[0].0 + p[0].1 as u64 && p[2].0 == p[1].0 + p[1].1 as u64 && self.stride[l] == p[0].1 + p[1].1 + p[2].1
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Fill {
    Empty,
    Loading,
    Ready,
    Failed,
    /// given back (`Store::retire`): no memory, never used again
    Retired,
}

struct Slot {
    key: u32,
    pins: u32,
    fill: Fill,
    done: u8,
    /// pieces a reader has taken up
    started: u8,
    /// where each piece's data starts in the slot's memory
    moff: [u32; 3],
    /// asked for by a pass (not only guessed)
    used: bool,
    /// the prompt pass that last asked for it
    pass: u32,
}

struct St {
    slots: Vec<Slot>,
    map: HashMap<u32, usize>,
    heat: Vec<f32>,
    /// the slower heat, and its weight in eff
    slow: Vec<f32>,
    mix: f32,
    /// token count when each heat was last brought up to date
    last: Vec<u64>,
    tick: u64,
    /// decay per token, by tokens elapsed
    pow: Vec<f32>,
    pow2: Vec<f32>,
    /// reads for experts wanted now, then read-ahead guesses
    urgent: VecDeque<(usize, u8)>,
    jobs: VecDeque<(usize, u8)>,
    /// unsure guesses, read only while nothing else is queued, one at a time
    spec: VecDeque<(usize, u8)>,
    spec_on: u32,
    reads: u64,
    cancelled: u64,
    hits: u64,
    guesses: u64,
    /// drive time with at least one read in flight, bytes read, guesses
    /// evicted without ever being used
    inflight: u32,
    busy_from: Option<std::time::Instant>,
    busy_ns: u64,
    bytes: u64,
    wasted: u64,
    /// drive idle time by what the engine was doing (mark): layer, or the
    /// last bucket for between tokens
    cur: usize,
    idle_from: Option<std::time::Instant>,
    idle_ns: Vec<u64>,
    /// drive time by reads in flight (0..4, 4 = four or more), since the last change
    qd_ns: [u64; 5],
    qd_from: std::time::Instant,
    /// the layer a prompt pass is at: its later layers' experts are wanted
    /// in this pass, its earlier ones' not again until the next, so slots
    /// go to those first (heat alone throws out what the pass needs next)
    sweep: Option<usize>,
    /// prompt passes so far (`Slot::pass`)
    pass: u32,
}

impl St {
    /// Books the time since the last change of reads in flight.
    fn qd_tick(&mut self) {
        let now = std::time::Instant::now();
        let i = (self.inflight as usize).min(4);
        self.qd_ns[i] += (now - self.qd_from).as_nanos() as u64;
        self.qd_from = now;
    }
}

impl St {
    fn eff(&self, k: usize) -> f32 {
        let dt = (self.tick - self.last[k]) as usize;
        if dt >= self.pow.len() {
            return 0.0;
        }
        let h = self.heat[k] * self.pow[dt];
        if self.mix == 0.0 {
            h
        } else {
            h + self.mix * self.slow[k] * self.pow2[dt]
        }
    }
    fn bump(&mut self, k: usize, by: f32) {
        let dt = (self.tick - self.last[k]) as usize;
        let (a, b) = if dt < self.pow.len() { (self.pow[dt], self.pow2[dt]) } else { (0.0, 0.0) };
        self.heat[k] = self.heat[k] * a + by;
        if self.mix != 0.0 {
            self.slow[k] = self.slow[k] * b + by;
        }
        self.last[k] = self.tick;
    }
}

struct Shared {
    /// one handle per reader (and file of the model): Windows runs reads on
    /// a single synchronous handle one at a time (one handle: 3.8 GB/s at
    /// 4 MB, one each: 4.5)
    files: Vec<Vec<File>>,
    lay: Layout,
    /// bytes of the largest slot
    cap: usize,
    mem: Vec<usize>,
    /// each slot's bytes and size class, and each layer's class: a layer's
    /// experts go only in slots of its class (experts differ in size by
    /// layer; one size for all wastes the difference)
    caps: Vec<usize>,
    scls: Vec<u8>,
    lcls: Vec<u8>,
    st: Mutex<St>,
    work: Condvar,
    ready: Condvar,
    stop: AtomicBool,
}

pub struct Store {
    sh: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

/// An expert's bytes, held in place until dropped.
pub struct Pin {
    sh: Arc<Shared>,
    slot: usize,
    moff: [u32; 3],
    lens: [usize; 3],
}

impl Pin {
    /// piece p (0 gate, 1 up, 2 down)
    pub fn piece(&self, p: usize) -> &[u8] {
        // SAFETY: the slot is pinned and fully read; nothing writes it until
        // the pin is dropped
        unsafe { std::slice::from_raw_parts((self.sh.mem[self.slot] + self.moff[p] as usize) as *const u8, self.lens[p]) }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let mut st = self.sh.st.lock().unwrap();
        st.slots[self.slot].pins -= 1;
    }
}

fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
    }
    #[cfg(not(windows))]
    {
        File::open(path)
    }
}

fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, off)
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
}

/// Random unbuffered reads of `chunk` bytes by `threads` threads over the
/// file's first `span` bytes for `secs`: GB/s.
pub fn bench(path: &std::path::Path, chunk: usize, threads: usize, span: u64, secs: f64) -> Result<f64> {
    let shared = Arc::new(open_unbuffered(path)?);
    let own = std::env::var_os("PELICA_SHARED_HANDLE").is_none();
    let t0 = std::time::Instant::now();
    let total: u64 = (0..threads)
        .map(|i| {
            // a handle per thread: Windows serializes reads on one synchronous handle
            let f = if own { Arc::new(open_unbuffered(path).unwrap()) } else { shared.clone() };
            std::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(chunk, ALIGN).unwrap();
                let p = unsafe { std::alloc::alloc(layout) };
                let buf = unsafe { std::slice::from_raw_parts_mut(p, chunk) };
                let mut s = 0x9e3779b97f4a7c15u64 ^ i as u64;
                let mut n = 0u64;
                while t0.elapsed().as_secs_f64() < secs {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let off = (s % (span / ALIGN as u64 - chunk as u64 / ALIGN as u64)) * ALIGN as u64;
                    if let Ok(k) = read_at(&f, buf, off) {
                        n += k as u64;
                    }
                }
                unsafe { std::alloc::dealloc(p, layout) };
                n
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap_or(0))
        .sum();
    Ok(total as f64 / t0.elapsed().as_secs_f64() / 1e9)
}

impl Store {
    /// A cache of `slots` experts; `heat` seeds each expert's use count.
    pub fn new(path: &std::path::Path, lay: Layout, slots: usize, heat: Vec<f32>) -> Result<Store> {
        Self::new_multi(&[path.to_path_buf()], lay, slots, heat)
    }

    /// `new` for a model in several files (the layout's `file` indexes `paths`).
    pub fn new_multi(paths: &[std::path::PathBuf], lay: Layout, slots: usize, heat: Vec<f32>) -> Result<Store> {
        // bytes per slot: the largest expert's three regions
        let cap = (0..lay.pieces.len()).map(|l| lay.region(l, 3)).max().unwrap_or(ALIGN);
        let lcls = vec![0u8; lay.pieces.len()];
        Self::build(paths, lay, vec![(cap, 0); slots], lcls, heat)
    }

    /// `new_multi` with `bytes` of RAM, in slots sized for each layer's
    /// experts: every layer gets the same number.
    pub fn new_sized(paths: &[std::path::PathBuf], lay: Layout, bytes: u64, heat: Vec<f32>) -> Result<Store> {
        let need: Vec<usize> = (0..lay.pieces.len()).map(|l| lay.region(l, 3)).collect();
        let mut sizes = need.clone();
        sizes.sort_unstable();
        sizes.dedup();
        let lcls: Vec<u8> = need.iter().map(|n| sizes.iter().position(|s| s == n).unwrap() as u8).collect();
        let per_layer = ((bytes / need.iter().sum::<usize>().max(1) as u64).max(1) as usize).min(lay.n_exp);
        let mut caps = Vec::new();
        for (c, &sz) in sizes.iter().enumerate() {
            let n = lcls.iter().filter(|&&x| x as usize == c).count() * per_layer;
            caps.extend(std::iter::repeat_n((sz, c as u8), n.max(16)));
        }
        Self::build(paths, lay, caps, lcls, heat)
    }

    fn build(paths: &[std::path::PathBuf], lay: Layout, want: Vec<(usize, u8)>, lcls: Vec<u8>, heat: Vec<f32>) -> Result<Store> {
        let files = if std::env::var_os("PELICA_SHARED_HANDLE").is_some() {
            // the old way, for comparison: duplicates share one file object
            let fs = paths.iter().map(|p| open_unbuffered(p)).collect::<std::io::Result<Vec<File>>>()?;
            (0..readers()).map(|_| fs.iter().map(|f| f.try_clone()).collect()).collect::<std::io::Result<Vec<Vec<File>>>>()?
        } else {
            (0..readers()).map(|_| paths.iter().map(|p| open_unbuffered(p)).collect()).collect::<std::io::Result<Vec<Vec<File>>>>()?
        };
        let cap = want.iter().map(|w| w.0).max().unwrap_or(ALIGN);
        let mut mem = Vec::with_capacity(want.len());
        let (mut caps, mut scls) = (Vec::new(), Vec::new());
        for &(c, k) in &want {
            let layout = std::alloc::Layout::from_size_align(c, ALIGN).unwrap();
            // SAFETY: non-zero size; freed in Drop with the same layout
            let p = unsafe { std::alloc::alloc(layout) };
            if p.is_null() {
                break;
            }
            mem.push(p as usize);
            caps.push(c);
            scls.push(k);
        }
        if mem.len() < 16 || mem.len() < want.len() && lcls.iter().any(|&c| !scls.contains(&c)) {
            for (&p, &c) in mem.iter().zip(&caps) {
                unsafe { std::alloc::dealloc(p as *mut u8, std::alloc::Layout::from_size_align(c, ALIGN).unwrap()) };
            }
            bail!("not enough RAM for the expert cache");
        }
        let n = mem.len();
        let sh = Arc::new(Shared {
            files,
            lay,
            cap,
            mem,
            caps,
            scls,
            lcls,
            st: Mutex::new(St {
                slots: (0..n).map(|_| Slot { key: u32::MAX, pins: 0, fill: Fill::Empty, done: 0, started: 0, moff: [0; 3], used: false, pass: 0 }).collect(),
                map: HashMap::new(),
                last: vec![0; heat.len()],
                slow: heat.clone(),
                mix: slow_mix(),
                heat,
                tick: 0,
                pow: (0..8192).map(|t| 0.5f64.powf(t as f64 / half_life()) as f32).collect(),
                pow2: (0..8192).map(|t| 0.5f64.powf(t as f64 / slow_half_life()) as f32).collect(),
                urgent: VecDeque::new(),
                jobs: VecDeque::new(),
                spec: VecDeque::new(),
                spec_on: 0,
                reads: 0,
                cancelled: 0,
                hits: 0,
                guesses: 0,
                inflight: 0,
                busy_from: None,
                busy_ns: 0,
                bytes: 0,
                wasted: 0,
                cur: 0,
                idle_from: None,
                idle_ns: vec![0; 256],
                qd_ns: [0; 5],
                qd_from: std::time::Instant::now(),
                sweep: None,
                pass: 0,
            }),
            work: Condvar::new(),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let threads = (0..sh.files.len())
            .map(|i| {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    reader_priority();
                    reader(sh, i)
                })
            })
            .collect();
        Ok(Store { sh, threads })
    }

    pub fn slots(&self) -> usize {
        self.sh.mem.len()
    }

    pub fn slot_bytes(&self) -> usize {
        self.sh.cap
    }

    /// RAM the slots take in all
    pub fn bytes(&self) -> usize {
        self.sh.caps.iter().sum()
    }

    /// layer l's slot size class
    pub fn class_of(&self, l: usize) -> usize {
        self.sh.lcls.get(l).copied().unwrap_or(0) as usize
    }

    /// slots that can hold layer l's experts
    pub fn class_slots(&self, l: usize) -> usize {
        let c = self.sh.lcls.get(l).copied().unwrap_or(0);
        let st = self.sh.st.lock().unwrap();
        self.sh.scls.iter().zip(&st.slots).filter(|(&x, s)| x == c && s.fill != Fill::Retired).count()
    }

    /// Gives back about `bytes` of the cache's memory (for a chat's growing
    /// attention cache): the coldest slots nobody holds, each class keeping
    /// a quarter of its slots and at least 16. Returns the bytes freed.
    pub fn retire(&self, bytes: u64) -> u64 {
        let sh = &self.sh;
        let mut st = self.sh.st.lock().unwrap();
        let n_cls = sh.scls.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        let mut live = vec![0usize; n_cls];
        let mut total = vec![0usize; n_cls];
        for (i, s) in st.slots.iter().enumerate() {
            total[sh.scls[i] as usize] += 1;
            if s.fill != Fill::Retired {
                live[sh.scls[i] as usize] += 1;
            }
        }
        let queued: std::collections::HashSet<usize> = st.urgent.iter().chain(&st.jobs).chain(&st.spec).map(|j| j.0).collect();
        let mut cand: Vec<(f32, usize)> = st
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| s.pins == 0 && s.fill != Fill::Loading && s.fill != Fill::Retired && !queued.contains(i))
            .map(|(i, s)| (if s.fill == Fill::Ready { st.eff(s.key as usize) } else { f32::NEG_INFINITY }, i))
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut freed = 0u64;
        for (_, i) in cand {
            if freed >= bytes {
                break;
            }
            let c = sh.scls[i] as usize;
            if live[c] <= (total[c] / 4).max(16) {
                continue;
            }
            let old = st.slots[i].key;
            if old != u32::MAX && st.map.get(&old) == Some(&i) {
                st.map.remove(&old);
            }
            st.slots[i].key = u32::MAX;
            st.slots[i].fill = Fill::Retired;
            live[c] -= 1;
            // SAFETY: allocated in build with this layout; no pin, read or
            // queued read refers to the slot, and it's never handed out again
            unsafe { std::alloc::dealloc(sh.mem[i] as *mut u8, std::alloc::Layout::from_size_align(sh.caps[i], ALIGN).unwrap()) };
            freed += sh.caps[i] as u64;
        }
        freed
    }

    /// Counts a use of each expert (key = layer * n_exp + expert).
    pub fn touch(&self, keys: &[u32], by: f32) {
        let mut st = self.sh.st.lock().unwrap();
        for &k in keys {
            st.bump(k as usize, by);
        }
    }

    /// `n` more tokens went through the model (heat decays by tokens).
    pub fn advance(&self, n: usize) {
        self.sh.st.lock().unwrap().tick += n as u64;
    }

    /// Every expert's heat now.
    pub fn heat(&self) -> Vec<f32> {
        let st = self.sh.st.lock().unwrap();
        (0..st.heat.len()).map(|k| st.eff(k)).collect()
    }

    /// Starts reading experts likely to be needed soon (those not in RAM),
    /// behind the reads already queued; nothing is pinned.
    pub fn guess(&self, keys: &[u32]) {
        self.guess_q(keys, 1);
    }

    /// Like guess, for unsure guesses: read only while the drive has nothing
    /// else queued, one at a time, so real reads never wait behind them.
    pub fn guess_idle(&self, keys: &[u32]) {
        self.guess_q(keys, 2);
    }

    fn guess_q(&self, keys: &[u32], q: u8) {
        let miss: Vec<u32> = {
            let mut st = self.sh.st.lock().unwrap();
            let m: Vec<u32> = keys.iter().copied().filter(|k| !st.map.contains_key(k)).collect();
            for &k in &m {
                st.bump(k as usize, if q == 2 { GUESS_HEAT / 2.0 } else { GUESS_HEAT });
            }
            st.guesses += m.len() as u64;
            m
        };
        if !miss.is_empty() {
            let _ = self.request_p(&miss, q);
        }
    }

    /// Frees the RAM copies of experts now held in VRAM (unless someone
    /// holds them), so the slots cache other experts.
    pub fn forget(&self, keys: &[u32]) {
        let mut st = self.sh.st.lock().unwrap();
        for k in keys {
            if let Some(&s) = st.map.get(k) {
                if st.slots[s].pins == 0 && st.slots[s].fill == Fill::Ready {
                    st.map.remove(k);
                    st.slots[s].fill = Fill::Empty;
                    st.slots[s].key = u32::MAX;
                }
            }
        }
    }

    /// Whether the expert is in RAM now (read or being read).
    pub fn has(&self, key: u32) -> bool {
        self.sh.st.lock().unwrap().map.contains_key(&key)
    }

    /// (reads from the drive, requests served from RAM, read-ahead guesses)
    pub fn counts(&self) -> (u64, u64, u64) {
        let st = self.sh.st.lock().unwrap();
        (st.reads - st.cancelled, st.hits, st.guesses - st.cancelled)
    }

    /// (seconds the drive was busy, bytes read, guesses never used)
    pub fn io_prof(&self) -> (f64, u64, u64) {
        let st = self.sh.st.lock().unwrap();
        (st.busy_ns as f64 / 1e9, st.bytes, st.wasted)
    }

    /// What the engine is doing now (for drive idle time): layer * 4 + phase
    /// (attention, routing, experts, after), or 255 for between tokens.
    /// A prompt pass is at layer `l` (None: no pass, or a small one).
    /// A pass's layer 0 starts a new pass.
    pub fn sweep(&self, l: Option<usize>) {
        let mut st = self.sh.st.lock().unwrap();
        if l.is_some() && (st.sweep.is_none() || l == Some(0)) {
            st.pass = st.pass.wrapping_add(1);
        }
        st.sweep = l;
    }

    pub fn mark(&self, what: usize) {
        let mut st = self.sh.st.lock().unwrap();
        if let Some(t) = st.idle_from {
            let now = std::time::Instant::now();
            let c = st.cur;
            st.idle_ns[c] += (now - t).as_nanos() as u64;
            st.idle_from = Some(now);
        }
        st.cur = what.min(255);
    }

    /// Drive seconds by reads in flight (0, 1, 2, 3, 4 or more).
    pub fn io_qd(&self) -> [f64; 5] {
        let mut st = self.sh.st.lock().unwrap();
        st.qd_tick();
        st.qd_ns.map(|n| n as f64 / 1e9)
    }

    /// Drive idle seconds by mark.
    pub fn idle_by_mark(&self) -> Vec<f64> {
        self.sh.st.lock().unwrap().idle_ns.iter().map(|&n| n as f64 / 1e9).collect()
    }

    /// Starts the io_prof counters over (decode only).
    pub fn io_prof_reset(&self) {
        let mut st = self.sh.st.lock().unwrap();
        st.busy_ns = 0;
        st.bytes = 0;
        st.wasted = 0;
        st.idle_ns.iter_mut().for_each(|x| *x = 0);
        st.qd_ns = [0; 5];
        st.qd_from = std::time::Instant::now();
        if st.idle_from.is_some() {
            st.idle_from = Some(std::time::Instant::now());
        }
        if st.busy_from.is_some() {
            st.busy_from = Some(std::time::Instant::now());
        }
    }

    /// Pins the experts, starting reads for those not in RAM (in order).
    /// Fails only if every slot is pinned.
    pub fn request(&self, keys: &[u32]) -> Result<Vec<Pin>> {
        self.request_p(keys, 0)
    }

    /// Drops queued read-ahead guesses for layer `layer` that aren't among
    /// `keep` (and that no reader has started).
    pub fn cancel_guesses(&self, layer: usize, keep: &[u32]) {
        let n_exp = self.sh.lay.n_exp;
        let mut st = self.sh.st.lock().unwrap();
        let stale: Vec<usize> = st
            .jobs
            .iter()
            .chain(st.spec.iter())
            .map(|&(s, _)| s)
            .filter(|&s| {
                let sl = &st.slots[s];
                sl.key as usize / n_exp == layer && sl.pins == 0 && sl.started == 0 && sl.fill == Fill::Loading && !keep.contains(&sl.key)
            })
            .collect();
        if stale.is_empty() {
            return;
        }
        st.jobs.retain(|(s, _)| !stale.contains(s));
        st.spec.retain(|(s, _)| !stale.contains(s));
        for &s in &stale {
            if st.slots[s].fill == Fill::Loading {
                let k = st.slots[s].key;
                st.map.remove(&k);
                st.slots[s].fill = Fill::Empty;
                st.slots[s].key = u32::MAX;
                st.cancelled += 1;
            }
        }
    }

    /// `q`: 0 real reads (ahead of everything; guesses already queued for
    /// these experts move up), 1 guesses, 2 unsure guesses (idle drive only).
    fn request_p(&self, keys: &[u32], q: u8) -> Result<Vec<Pin>> {
        let urgent = q == 0;
        let sh = &self.sh;
        let mut st = sh.st.lock().unwrap();
        let mut out = Vec::with_capacity(keys.len());
        for &key in keys {
            let lens = [sh.lay.span(key, 0).1, sh.lay.span(key, 1).1, sh.lay.span(key, 2).1];
            let slot = match st.map.get(&key).copied() {
                Some(s) if st.slots[s].fill != Fill::Failed => {
                    st.hits += 1;
                    if urgent {
                        st.slots[s].used = true;
                        st.slots[s].pass = st.pass;
                    }
                    if urgent && st.slots[s].fill == Fill::Loading && st.jobs.iter().chain(st.spec.iter()).any(|j| j.0 == s) {
                        let mine: Vec<(usize, u8)> = st.jobs.iter().chain(st.spec.iter()).copied().filter(|j| j.0 == s).collect();
                        st.jobs.retain(|j| j.0 != s);
                        st.spec.retain(|j| j.0 != s);
                        st.urgent.extend(mine);
                    } else if q == 1 && st.slots[s].fill == Fill::Loading && st.spec.iter().any(|j| j.0 == s) {
                        // a surer guess for an unsure one: it joins the guesses
                        let mine: Vec<(usize, u8)> = st.spec.iter().copied().filter(|j| j.0 == s).collect();
                        st.spec.retain(|j| j.0 != s);
                        st.jobs.extend(mine);
                    }
                    s
                }
                _ => {
                    // the coldest slot of the layer's size nobody holds
                    let cls = sh.lcls.get(key as usize / sh.lay.n_exp).copied().unwrap_or(0);
                    let mut best: Option<(usize, f32)> = None;
                    for (i, s) in st.slots.iter().enumerate() {
                        if s.pins > 0 || s.fill == Fill::Loading || s.fill == Fill::Retired || sh.scls[i] != cls {
                            continue;
                        }
                        let mut h = if s.fill == Fill::Empty || s.fill == Fill::Failed { f32::NEG_INFINITY } else { st.eff(s.key as usize) };
                        // in a prompt pass, as one sweep over the layers wants
                        // them: used already (an earlier layer, or this one's
                        // asked for this pass) first, by heat; then later
                        // layers', the furthest first; this layer's still to
                        // come last
                        if let (Some(at), true) = (st.sweep, s.key != u32::MAX && s.fill == Fill::Ready) {
                            let l = s.key as usize / sh.lay.n_exp;
                            if l > at {
                                h = 1e9 + (sh.lay.pieces.len() - l) as f32 * 1e4;
                            } else if l == at && s.pass != st.pass {
                                h = 2e9;
                            }
                        }
                        if best.is_none_or(|b| h < b.1) {
                            best = Some((i, h));
                        }
                    }
                    let Some((s, _)) = best else {
                        // unlock first: dropping the pins taken so far locks st
                        drop(st);
                        drop(out);
                        bail!("every expert slot is in use")
                    };
                    let old = st.slots[s].key;
                    if old != u32::MAX && st.map.get(&old) == Some(&s) {
                        st.map.remove(&old);
                        if !st.slots[s].used {
                            st.wasted += 1;
                        }
                    }
                    // pieces apart: one read each, into a third of the slot;
                    // pieces together: one read (job 3) for the whole expert
                    let merged = sh.lay.merged(key as usize / sh.lay.n_exp);
                    let mut moff = [0u32; 3];
                    for (p, m) in moff.iter_mut().enumerate() {
                        *m = if merged {
                            (sh.lay.span(key, 0).0 % ALIGN as u64) as u32 + lens[..p].iter().sum::<usize>() as u32
                        } else {
                            sh.lay.region(key as usize / sh.lay.n_exp, p) as u32 + (sh.lay.span(key, p).0 % ALIGN as u64) as u32
                        };
                    }
                    let pass = if urgent { st.pass } else { 0 };
                    st.slots[s] = Slot { key, pins: 0, fill: Fill::Loading, done: 0, started: 0, moff, used: urgent, pass };
                    st.map.insert(key, s);
                    let jobs: &[u8] = if merged { &[3] } else { &[0, 1, 2] };
                    for &p in jobs {
                        match q {
                            0 => st.urgent.push_back((s, p)),
                            1 => st.jobs.push_back((s, p)),
                            _ => st.spec.push_back((s, p)),
                        }
                    }
                    st.reads += 1;
                    sh.work.notify_all();
                    s
                }
            };
            st.slots[slot].pins += 1;
            out.push(Pin { sh: sh.clone(), slot, moff: st.slots[slot].moff, lens });
        }
        Ok(out)
    }

    /// Waits until the pinned expert's bytes are all in.
    pub fn wait(&self, pin: &Pin) -> Result<()> {
        let mut st = self.sh.st.lock().unwrap();
        loop {
            match st.slots[pin.slot].fill {
                Fill::Ready => return Ok(()),
                Fill::Failed => bail!("reading an expert from the drive failed"),
                _ => st = self.sh.ready.wait(st).unwrap(),
            }
        }
    }

    /// Waits until one of the experts not yet `done` is in; returns its index.
    pub fn wait_any(&self, pins: &[Pin], done: &[bool]) -> Result<usize> {
        let mut st = self.sh.st.lock().unwrap();
        loop {
            for (i, p) in pins.iter().enumerate() {
                if done[i] {
                    continue;
                }
                match st.slots[p.slot].fill {
                    Fill::Ready => return Ok(i),
                    Fill::Failed => bail!("reading an expert from the drive failed"),
                    _ => {}
                }
            }
            st = self.sh.ready.wait(st).unwrap();
        }
    }

    /// Whether the pinned expert is ready (without waiting).
    pub fn is_ready(&self, pin: &Pin) -> bool {
        self.sh.st.lock().unwrap().slots[pin.slot].fill == Fill::Ready
    }


}

impl Drop for Store {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        {
            let _st = self.sh.st.lock().unwrap();
            self.sh.work.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let st = self.sh.st.lock().unwrap();
        for ((&p, &c), s) in self.sh.mem.iter().zip(&self.sh.caps).zip(&st.slots) {
            if s.fill == Fill::Retired {
                continue;
            }
            // SAFETY: allocated in build with this layout; the readers are gone
            unsafe { std::alloc::dealloc(p as *mut u8, std::alloc::Layout::from_size_align(c, ALIGN).unwrap()) };
        }
    }
}

/// Reader threads wake ahead of the CPU expert work that keeps every core
/// busy, so the next read starts at once (PELICA_READER_PRIO: Windows thread
/// priority, 0 normal .. 2 highest; default 0).
fn reader_priority() {
    let p: i32 = std::env::var("PELICA_READER_PRIO").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    #[cfg(windows)]
    if p != 0 {
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        // SAFETY: the pseudo handle of this thread
        unsafe { SetThreadPriority(GetCurrentThread(), p.clamp(-2, 2)) };
    }
    let _ = p;
}

fn reader(sh: Arc<Shared>, me: usize) {
    loop {
        let (slot, p, key, moff, is_spec) = {
            let mut st = sh.st.lock().unwrap();
            loop {
                if sh.stop.load(Ordering::Relaxed) {
                    return;
                }
                let mut is_spec = false;
                let job = match st.urgent.pop_front().or_else(|| st.jobs.pop_front()) {
                    Some(j) => Some(j),
                    None if st.spec_on == 0 => {
                        is_spec = true;
                        st.spec.pop_front()
                    }
                    None => None,
                };
                if let Some((s, p)) = job {
                    if is_spec {
                        st.spec_on += 1;
                    }
                    st.slots[s].started += 1;
                    st.qd_tick();
                    if st.inflight == 0 {
                        let now = std::time::Instant::now();
                        st.busy_from = Some(now);
                        if let Some(t) = st.idle_from.take() {
                            let c = st.cur;
                            st.idle_ns[c] += (now - t).as_nanos() as u64;
                        }
                    }
                    st.inflight += 1;
                    break (s, p as usize, st.slots[s].key, st.slots[s].moff, is_spec);
                }
                st = sh.work.wait(st).unwrap();
            }
        };
        // job 3: a repacked expert's three pieces in one read, from the
        // slot's start; jobs 0..2: one piece into its third of the slot
        let file = &sh.files[me][sh.lay.file_of(key, if p == 3 { 0 } else { p })];
        let (off, len, at, skip) = if p == 3 {
            let (o, _) = sh.lay.span(key, 0);
            let all = (0..3).map(|q| sh.lay.span(key, q).1).sum::<usize>();
            (o, all, 0, moff[0] as usize)
        } else {
            let (o, l) = sh.lay.span(key, p);
            let at = sh.lay.region(key as usize / sh.lay.n_exp, p);
            (o, l, at, moff[p] as usize - at)
        };
        let start = off - skip as u64;
        let want = (skip + len).div_ceil(ALIGN) * ALIGN;
        // SAFETY: this slot's piece is written only by this job while the
        // slot is Loading (pinned by its requester, never handed out)
        let buf = unsafe { std::slice::from_raw_parts_mut((sh.mem[slot] + at) as *mut u8, want) };
        let mut got = 0usize;
        let mut ok = true;
        while got < skip + len {
            match read_at(file, &mut buf[got..], start + got as u64) {
                Ok(0) => {
                    ok = false;
                    break;
                }
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    eprintln!("[pelica] expert read failed: {e}");
                    ok = false;
                    break;
                }
            }
        }
        let mut st = sh.st.lock().unwrap();
        st.qd_tick();
        st.inflight -= 1;
        if is_spec {
            st.spec_on -= 1;
            // the next unsure guess may go now
            sh.work.notify_one();
        }
        st.bytes += got as u64;
        if st.inflight == 0 {
            if let Some(t) = st.busy_from.take() {
                st.busy_ns += t.elapsed().as_nanos() as u64;
            }
            st.idle_from = Some(std::time::Instant::now());
        }
        let s = &mut st.slots[slot];
        if !ok {
            s.fill = Fill::Failed;
        } else {
            s.done += if p == 3 { 3 } else { 1 };
            if s.done == 3 && s.fill == Fill::Loading {
                s.fill = Fill::Ready;
            }
        }
        sh.ready.notify_all();
    }
}
