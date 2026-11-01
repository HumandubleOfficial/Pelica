//! Routed experts: where they live on disk and the RAM tier that caches them.
//!
//! An expert is three int4 matrices (gate, up, down) plus their group scales,
//! about 14 MB. The checkpoint stores the three weight blocks back to back and
//! the three scale blocks back to back, so one expert is two positioned reads.
//!
//! The RAM tier is a byte-budgeted LRU shared by every layer. Two ways in:
//! * `get` — the experts a layer needs now; misses are read in parallel.
//! * `prefetch` — experts a layer will probably need soon; queued to a pool
//!   of reader threads so the disk works while the CPU computes. A `get` that
//!   finds its expert still in flight waits for that read instead of
//!   issuing a second one.
//!
//! Prefetches are read in chunks, nearest layer and most confident guess
//! first, so the disk always works on the bytes needed soonest. A better
//! guess made later can `regress` the queue (drop guesses not yet started),
//! and once a layer has routed, `retire` stops the reads of guesses it did
//! not pick.
//!
//! Eviction is by heat rather than recency: experts come round once per
//! token in layer order, and plain LRU throws out exactly the ones whose
//! layer is up next. Heat is mostly a short memory of how highly the router
//! ranked each expert, over all of a layer's experts and not just the chosen
//! top-k, so a near-miss is warm before it gets picked; a little of a long
//! memory of actual selections keeps the steady favourites. Replaying
//! recorded router traces, that reads ~4% less from disk than counting uses.

use crate::quant::{Kind, MRef, Q4_GROUP};
use crate::st::Shards;
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const PIECES: usize = 6; // gate.q, gate.s, up.q, up.s, down.q, down.s

#[derive(Clone, Copy, Debug)]
struct Piece {
    shard: usize,
    offset: u64,
    len: usize,
}

/// A contiguous file range covering one or more pieces.
#[derive(Clone, Debug)]
struct Span {
    shard: usize,
    offset: u64,
    len: usize,
}

/// Unbuffered reads must start, end and land on sector boundaries.
const SECTOR: usize = 4096;
/// Readers that only take blocking reads (an expert is about seven chunks).
const DEMAND_THREADS: usize = 4;
/// Reads go to the disk in pieces of at most this many bytes. This drive is
/// fastest with few, large requests, so 16 MB keeps each weight span whole.
const CHUNK: usize = 16 << 20;
/// The same for blocking reads.
const DEMAND_CHUNK: usize = 16 << 20;

/// Experiment switches (PELICA_CHUNK=<bytes>, PELICA_EVICT=lru, PELICA_RETIRE=0).
fn knob(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
fn chunk_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| knob("PELICA_CHUNK").and_then(|v| v.parse().ok()).unwrap_or(CHUNK))
}
fn demand_chunk_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| knob("PELICA_DCHUNK").and_then(|v| v.parse().ok()).unwrap_or(DEMAND_CHUNK))
}
#[derive(Clone, Copy, PartialEq)]
enum Evict {
    /// short rank heat + a little long selection heat (the default)
    Rank,
    /// decayed selection counts (PELICA_EVICT=uses)
    Uses,
    /// least recently used (PELICA_EVICT=lru)
    Lru,
}
fn evict_by() -> Evict {
    static V: std::sync::OnceLock<Evict> = std::sync::OnceLock::new();
    *V.get_or_init(|| match knob("PELICA_EVICT").as_deref() {
        Some("lru") => Evict::Lru,
        Some("uses") => Evict::Uses,
        _ => Evict::Rank,
    })
}
fn retire_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| knob("PELICA_RETIRE").as_deref() != Some("0"))
}
/// Heat halves after this many expert selections (8 tokens of 42 layers x top-8).
const HEAT_HALF: f64 = 8.0 * 42.0 * 8.0;
/// Rank heat: router rank r adds exp(-r / RANK_TEMP), halving every
/// RANK_HALF tokens; plus PICK_WEIGHT x selections halving every PICK_HALF.
const RANK_TEMP: f64 = 8.0;
const RANK_HALF: f64 = 4.0;
const PICK_HALF: f64 = 64.0;
const PICK_WEIGHT: f64 = 0.1;
/// Router ranks that count towards heat (past this, exp(-r / 8) < 0.003).
pub const RANKS: usize = 48;

#[derive(Clone, Debug)]
struct Loc {
    spans: Vec<Span>,
    /// byte offset of each piece inside the expert buffer
    at: [usize; PIECES],
    total: usize,
    /// the same spans widened to whole sectors, each at a sector-aligned
    /// offset in the buffer: (span, buffer offset)
    dspans: Vec<(Span, usize)>,
    dat: [usize; PIECES],
    dtotal: usize,
}

/// A sector-aligned heap buffer.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

/// Recently freed expert buffers. Every expert has the same size, and a fresh
/// 14 MB allocation costs thousands of page faults while the disk fills it.
static POOL: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new()); // (ptr, len)
const POOL_MAX: usize = 12;

impl AlignedBuf {
    /// A sector-aligned buffer; contents are unspecified (reads overwrite them).
    fn new(len: usize) -> Self {
        let len = len.div_ceil(SECTOR) * SECTOR;
        {
            let mut pool = POOL.lock().unwrap();
            if let Some(i) = pool.iter().position(|&(_, l)| l == len) {
                let (ptr, len) = pool.swap_remove(i);
                return AlignedBuf { ptr: ptr as *mut u8, len };
            }
        }
        let layout = std::alloc::Layout::from_size_align(len, SECTOR).unwrap();
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBuf { ptr, len }
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        {
            let mut pool = POOL.lock().unwrap();
            if pool.len() < POOL_MAX {
                pool.push((self.ptr as usize, self.len));
                return;
            }
        }
        let layout = std::alloc::Layout::from_size_align(self.len, SECTOR).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

pub struct ExpertBuf {
    data: AlignedBuf,
    at: [usize; PIECES],
    inter: usize,
    hidden: usize,
    /// scale pieces are packed blocks (table + indices), not f32 arrays
    packed: bool,
}

impl ExpertBuf {
    fn bytes(&self) -> &[u8] {
        self.data.bytes()
    }

    fn q4(&self, piece: usize, rows: usize, cols: usize) -> MRef<'_> {
        let b = self.bytes();
        let q = &b[self.at[piece]..self.at[piece] + rows * cols / 2];
        let s_at = self.at[piece + 1];
        let n_s = rows * cols / Q4_GROUP;
        if self.packed {
            // SAFETY: blocks start 4-byte aligned in the sector-aligned buffer.
            let table = unsafe { std::slice::from_raw_parts(b[s_at..].as_ptr() as *const f32, 256) };
            let idx = &b[s_at + 1024..s_at + 1024 + n_s];
            return MRef { kind: Kind::Q4T(q, idx, table), rows, cols };
        }
        // SAFETY: s_at is 4-byte aligned (checked when the table is built).
        let s = unsafe { std::slice::from_raw_parts(b[s_at..].as_ptr() as *const f32, n_s) };
        MRef { kind: Kind::Q4(q, s), rows, cols }
    }

    pub fn gate(&self) -> MRef<'_> {
        self.q4(0, self.inter, self.hidden)
    }
    pub fn up(&self) -> MRef<'_> {
        self.q4(2, self.inter, self.hidden)
    }
    pub fn down(&self) -> MRef<'_> {
        self.q4(4, self.hidden, self.inter)
    }
    pub fn size(&self) -> usize {
        self.data.len
    }

    /// Raw bytes of one piece: 0/2/4 = gate/up/down int4 weights, 1/3/5 = their
    /// scales (f32 array, or a packed block: 256-entry table then indices).
    pub fn piece(&self, i: usize) -> &[u8] {
        &self.bytes()[self.at[i]..self.at[i] + piece_len(self.inter * self.hidden, i, self.packed)]
    }
}

/// Bytes of piece `i` of an expert whose matrices have `n` weights.
pub fn piece_len(n: usize, i: usize, packed: bool) -> usize {
    match (i % 2, packed) {
        (0, _) => n / 2,
        (_, false) => n / Q4_GROUP * 4,
        (_, true) => 1024 + n / Q4_GROUP,
    }
}

/// A read in flight; whoever needs the expert waits on it.
struct Pending {
    done: Mutex<Option<Result<Arc<ExpertBuf>, String>>>,
    cv: Condvar,
    /// set by `retire`: the rest of this read is not worth the disk time
    cancelled: AtomicBool,
    /// a reader has started on it
    begun: AtomicBool,
}

impl Pending {
    fn new() -> Arc<Self> {
        Arc::new(Pending {
            done: Mutex::new(None),
            cv: Condvar::new(),
            cancelled: AtomicBool::new(false),
            begun: AtomicBool::new(false),
        })
    }
    fn finish(&self, r: Result<Arc<ExpertBuf>, String>) {
        *self.done.lock().unwrap() = Some(r);
        self.cv.notify_all();
    }
    fn wait(&self) -> Result<Arc<ExpertBuf>> {
        let mut g = self.done.lock().unwrap();
        while g.is_none() {
            g = self.cv.wait(g).unwrap();
        }
        g.as_ref().unwrap().clone().map_err(|e| anyhow!(e))
    }
}

enum Slot {
    Ready { buf: Arc<ExpertBuf>, last: u64, prefetched: bool },
    Loading { pending: Arc<Pending>, prefetched: bool },
}

#[derive(Default, Debug, Clone, Copy)]
pub struct CacheStats {
    /// expert was resident when asked for
    pub hits: u64,
    /// expert arrived (or was arriving) through a prefetch
    pub prefetch_hits: u64,
    /// expert was read on demand
    pub misses: u64,
    pub prefetch_issued: u64,
    /// prefetched experts read in full but not picked by their layer
    pub prefetch_wasted: u64,
    /// prefetches stopped part way because their layer did not pick them
    pub prefetch_cancelled: u64,
    pub bytes_read: u64,
    /// everything the disk delivered, including chunks of cancelled reads
    pub disk_bytes: u64,
    /// seconds with at least one read under way
    pub disk_busy_secs: f64,
    /// seconds with no read under way, at token boundaries and elsewhere
    pub idle_boundary_secs: f64,
    pub idle_inner_secs: f64,
    /// time the compute path spent blocked on expert reads
    pub wait_secs: f64,
    /// ... of which on its own on-demand reads
    pub wait_demand_secs: f64,
    pub resident: usize,
    pub resident_bytes: usize,
}

struct State {
    map: HashMap<u32, Slot>,
    used: usize,
    stats: CacheStats,
    /// per expert key: sum over its uses of 2^(tick_at_use / HEAT_HALF), so
    /// values compare as decayed use counts without a decay pass
    heat: Vec<f64>,
    tick: f64,
    /// rank heat and selection heat, scaled the same way by 2^(step / half)
    rank: Vec<f64>,
    pick: Vec<f64>,
    /// one step per layer routing of one token
    step: f64,
    steps_per_token: f64,
}

impl State {
    /// Halving periods of the two heats, in steps.
    fn halves(&self) -> (f64, f64) {
        (RANK_HALF * self.steps_per_token, PICK_HALF * self.steps_per_token)
    }

    /// One token's routing of one layer: its experts best first, the first
    /// `k` chosen.
    fn observe(&mut self, base: usize, ranked: &[usize], k: usize) {
        self.step += 1.0;
        let (hr, hp) = self.halves();
        if self.step > 800.0 * hr {
            let (fr, fp) = ((-self.step / hr).exp2(), (-self.step / hp).exp2());
            self.rank.iter_mut().for_each(|h| *h *= fr);
            self.pick.iter_mut().for_each(|h| *h *= fp);
            self.step = 0.0;
        }
        let (gr, gp) = ((self.step / hr).exp2(), (self.step / hp).exp2());
        for (r, &e) in ranked.iter().enumerate() {
            self.rank[base + e] += gr * (-(r as f64) / RANK_TEMP).exp();
        }
        for &e in &ranked[..k.min(ranked.len())] {
            self.pick[base + e] += gp;
        }
    }

    /// Factors that turn the stored heats into current values.
    fn now(&self) -> (f64, f64) {
        let (hr, hp) = self.halves();
        ((-self.step / hr).exp2(), PICK_WEIGHT * (-self.step / hp).exp2())
    }
    fn heat_up(&mut self, key: u32) {
        self.tick += 1.0;
        if self.tick > 900.0 * HEAT_HALF {
            let f = (-self.tick / HEAT_HALF).exp2();
            self.heat.iter_mut().for_each(|h| *h *= f);
            self.tick = 0.0;
        }
        self.heat[key as usize] += (self.tick / HEAT_HALF).exp2();
    }
}

struct Shared {
    locs: Vec<Option<Loc>>, // [layer * n_experts + e]
    /// routing counts: history loaded at start plus this run's selections
    usage: Vec<AtomicU64>,
    files: Vec<File>,
    direct: bool,
    packed: bool,
    n_experts: usize,
    inter: usize,
    hidden: usize,
    budget: std::sync::atomic::AtomicUsize,
    state: Mutex<State>,
    clock: AtomicU64,
    disk_bytes: AtomicU64,
    /// reads under way, and the time at least one was (ns), for utilisation
    active: Mutex<(usize, Option<Instant>, u64)>,
    /// when the disk last went idle; idle time ending in a token's first
    /// MoE layer (the token boundary) and elsewhere (ns)
    idle: Mutex<(Option<Instant>, u64, u64)>,
    /// the layer that routed last, and the first MoE layer
    phase: AtomicUsize,
    first_moe: usize,
}

/// One expert being read chunk by chunk by the reader threads.
struct ChunkedRead {
    key: u32,
    data: Mutex<Option<AlignedBuf>>,
    ptr: *mut u8,
    left: AtomicUsize,
    failed: Mutex<Option<String>>,
    pending: Arc<Pending>,
}
// SAFETY: chunks write disjoint ranges of `data` through `ptr`; the buffer is
// only taken out after the last chunk is done.
unsafe impl Send for ChunkedRead {}
unsafe impl Sync for ChunkedRead {}

/// One chunk read. The queue serves the smallest `prio` first: lookahead
/// guesses by (layer, guess rank), then warm-up fills.
struct Job {
    prio: (usize, usize, u64),
    read: Arc<ChunkedRead>,
    shard: usize,
    offset: u64,
    at: usize,
    len: usize,
}
impl PartialEq for Job {
    fn eq(&self, o: &Self) -> bool {
        self.prio == o.prio
    }
}
impl Eq for Job {}
impl PartialOrd for Job {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Job {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.prio.cmp(&self.prio) // BinaryHeap is a max-heap
    }
}

struct Queue {
    jobs: Mutex<BinaryHeap<Job>>,
    cv: Condvar,
    seq: AtomicU64,
    /// chunks of blocking (demand) reads queued or under way; while any are,
    /// prefetch readers hold off so the disk serves the demand first
    demand: AtomicUsize,
    /// set when the store is dropped: the readers finish what is under way
    /// and exit, letting go of the cache
    quit: AtomicBool,
    /// the async engine's completion port, rung when jobs are queued
    #[cfg(windows)]
    port: Option<Arc<crate::aio::Port>>,
}

impl Queue {
    /// Tell whoever serves the queue that there is new work.
    fn wake(&self) {
        self.cv.notify_all();
        #[cfg(windows)]
        if let Some(p) = &self.port {
            p.ring();
        }
    }
}

fn is_demand(j: &Job) -> bool {
    j.prio.0 == 0
}

/// A `get` in progress: see `ExpertStore::start`.
pub struct Fetch {
    layer: usize,
    ids: Vec<usize>,
    out: Vec<Option<Arc<ExpertBuf>>>,
    mine: Vec<(usize, Arc<Pending>)>,
    theirs: Vec<(usize, Arc<Pending>)>,
}

impl Fetch {
    /// (position in `ids`, buffer) for the experts that were resident.
    pub fn ready(&self) -> Vec<(usize, Arc<ExpertBuf>)> {
        self.out.iter().enumerate().filter_map(|(i, b)| b.clone().map(|b| (i, b))).collect()
    }
}

pub struct ExpertStore {
    shared: Arc<Shared>,
    queue: Arc<Queue>,
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        self.queue.quit.store(true, Ordering::Release);
        self.queue.wake();
    }
}

impl ExpertStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shards: &Shards,
        prefix: &str,
        n_layers: usize,
        moe_layers: impl Iterator<Item = usize>,
        n_experts: usize,
        inter: usize,
        hidden: usize,
        budget_bytes: usize,
        io_threads: usize,
        direct: bool,
        pack: Option<&crate::scalepack::Pack>,
    ) -> Result<Self> {
        let side = shards.paths.len(); // the side file's index in `files`
        let mut locs = vec![None; n_layers * n_experts];
        let moe_layers: Vec<usize> = moe_layers.collect();
        let n_moe = moe_layers.len();
        let first_moe = moe_layers.first().copied().unwrap_or(0);
        for layer in moe_layers {
            for e in 0..n_experts {
                let base = format!("{prefix}layers.{layer}.mlp.experts.{e}.");
                let mut pieces = [Piece { shard: 0, offset: 0, len: 0 }; PIECES];
                for (i, (m, sfx)) in [
                    ("gate_proj", "weight"),
                    ("gate_proj", "weight.qs"),
                    ("up_proj", "weight"),
                    ("up_proj", "weight.qs"),
                    ("down_proj", "weight"),
                    ("down_proj", "weight.qs"),
                ]
                .iter()
                .enumerate()
                {
                    let info = shards.info(&format!("{base}{m}.{sfx}"))?;
                    let want = if sfx.ends_with("qs") { inter * hidden / Q4_GROUP * 4 } else { inter * hidden / 2 };
                    if info.len != want {
                        bail!("{base}{m}.{sfx}: {} bytes, expected {want} (int4 gs64 container?)", info.len);
                    }
                    pieces[i] = Piece { shard: info.shard, offset: info.offset, len: info.len };
                }
                if let Some(p) = pack {
                    // scales come from the side file's record instead
                    for m in 0..3 {
                        let offset = p.offset(layer, e) + (m * p.block()) as u64;
                        pieces[2 * m + 1] = Piece { shard: side, offset, len: p.block() };
                    }
                }
                locs[layer * n_experts + e] = Some(plan(&pieces)?);
            }
        }
        let mut files = if direct {
            shards.paths.iter().map(|p| open_unbuffered(p)).collect::<std::io::Result<Vec<_>>>()?
        } else {
            shards.files.iter().map(File::try_clone).collect::<std::io::Result<Vec<_>>>()?
        };
        if let Some(p) = pack {
            files.push(if direct { open_unbuffered(&p.path)? } else { File::open(&p.path)? });
        }
        let usage = (0..locs.len()).map(|_| AtomicU64::new(0)).collect();
        let shared = Arc::new(Shared {
            locs,
            usage,
            files,
            direct,
            packed: pack.is_some(),
            n_experts,
            inter,
            hidden,
            budget: std::sync::atomic::AtomicUsize::new(budget_bytes),
            state: Mutex::new(State {
                map: HashMap::new(),
                used: 0,
                stats: CacheStats::default(),
                heat: vec![0.0; n_layers * n_experts],
                tick: 0.0,
                rank: vec![0.0; n_layers * n_experts],
                pick: vec![0.0; n_layers * n_experts],
                step: 0.0,
                steps_per_token: n_moe.max(1) as f64,
            }),
            clock: AtomicU64::new(0),
            disk_bytes: AtomicU64::new(0),
            active: Mutex::new((0, None, 0)),
            idle: Mutex::new((None, 0, 0)),
            phase: AtomicUsize::new(0),
            first_moe,
        });

        // Unbuffered reads on Windows go through one async engine that keeps
        // the drive at a fixed depth; otherwise, reader threads.
        #[cfg(windows)]
        let engine = if direct && knob("PELICA_AIO").as_deref() != Some("0") {
            let port = Arc::new(crate::aio::Port::new()?);
            let mut afiles = Vec::new();
            for (i, p) in shards.paths.iter().chain(pack.map(|p| &p.path)).enumerate() {
                let f = crate::aio::open(p)?;
                port.attach(&f, i)?;
                afiles.push(f);
            }
            Some((port, afiles))
        } else {
            None
        };
        let queue = Arc::new(Queue {
            jobs: Mutex::new(BinaryHeap::new()),
            cv: Condvar::new(),
            seq: AtomicU64::new(0),
            demand: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            #[cfg(windows)]
            port: engine.as_ref().map(|e| e.0.clone()),
        });
        #[cfg(windows)]
        if let Some((port, afiles)) = engine {
            let (sh, q) = (shared.clone(), queue.clone());
            std::thread::Builder::new().name("pelica-aio".into()).spawn(move || aio_engine(sh, q, port, afiles))?;
            return Ok(ExpertStore { shared, queue });
        }
        // one set of readers for anything, one that only takes demand reads,
        // so a blocked layer never waits for a prefetch chunk to free a thread
        let demand_threads = knob("PELICA_DEMAND_THREADS").and_then(|v| v.parse().ok()).unwrap_or(DEMAND_THREADS);
        for (demand_only, n) in [(false, io_threads.max(1)), (true, demand_threads)] {
            for _ in 0..n {
                let (sh, q) = (shared.clone(), queue.clone());
                std::thread::Builder::new().name("pelica-io".into()).spawn(move || io_worker(sh, q, demand_only))?;
            }
        }
        Ok(ExpertStore { shared, queue })
    }

    pub fn n_experts(&self) -> usize {
        self.shared.n_experts
    }

    pub fn set_budget(&self, bytes: usize) {
        self.shared.budget.store(bytes, Ordering::Relaxed);
        let mut g = self.shared.state.lock().unwrap();
        let now = self.shared.clock.load(Ordering::Relaxed);
        self.shared.evict(&mut g, now);
    }

    /// Drop an expert from the RAM tier (it moved up to VRAM).
    pub fn forget(&self, layer: usize, e: usize) {
        let mut g = self.shared.state.lock().unwrap();
        if let Some(Slot::Ready { buf, .. }) = g.map.get(&self.shared.key(layer, e)) {
            let size = buf.size();
            g.map.remove(&self.shared.key(layer, e));
            g.used -= size;
        }
    }

    /// Read one expert straight from disk, bypassing the RAM tier.
    pub fn read_now(&self, layer: usize, e: usize) -> Result<ExpertBuf> {
        self.shared.read(layer, e)
    }

    /// Experts ranked by routing history, hottest first: (layer, expert, count).
    pub fn ranked_by_usage(&self) -> Vec<(usize, usize, u64)> {
        let ne = self.shared.n_experts;
        let mut v: Vec<(usize, usize, u64)> = self
            .shared
            .usage
            .iter()
            .enumerate()
            .map(|(i, u)| (i / ne, i % ne, u.load(Ordering::Relaxed)))
            .filter(|t| t.2 > 0)
            .collect();
        v.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        v
    }

    /// Read random experts with `threads` readers for `secs`; returns GB/s.
    pub fn bench_reads(&self, threads: usize, secs: f64) -> f64 {
        let sh = &self.shared;
        let keys: Vec<usize> = (0..sh.locs.len()).filter(|&i| sh.locs[i].is_some()).collect();
        let bytes = AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for th in 0..threads {
                let (keys, bytes) = (&keys, &bytes);
                sc.spawn(move || {
                    let mut r = 0x2545_F491_4F6C_DD1Du64 ^ ((th as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    while t0.elapsed().as_secs_f64() < secs {
                        r ^= r << 13;
                        r ^= r >> 7;
                        r ^= r << 17;
                        let k = keys[(r % keys.len() as u64) as usize];
                        if let Ok(b) = sh.read(k / sh.n_experts, k % sh.n_experts) {
                            bytes.fetch_add(b.size() as u64, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        bytes.load(Ordering::Relaxed) as f64 / 1e9 / t0.elapsed().as_secs_f64()
    }

    /// Like `bench_reads`, but each reader fetches its experts in `chunk`-byte
    /// requests (unbuffered stores only); returns GB/s.
    pub fn bench_chunks(&self, threads: usize, chunk: usize, secs: f64) -> f64 {
        let sh = &self.shared;
        let keys: Vec<usize> = (0..sh.locs.len()).filter(|&i| sh.locs[i].is_some()).collect();
        let bytes = AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for th in 0..threads {
                let (keys, bytes) = (&keys, &bytes);
                sc.spawn(move || {
                    let mut r = 0x9E37_79B9_7F4A_7C15u64 ^ (th as u64 + 7).wrapping_mul(0x2545_F491_4F6C_DD1D);
                    let mut buf = AlignedBuf::new(chunk);
                    while t0.elapsed().as_secs_f64() < secs {
                        r ^= r << 13;
                        r ^= r >> 7;
                        r ^= r << 17;
                        let loc = sh.locs[keys[(r % keys.len() as u64) as usize]].as_ref().unwrap();
                        for (s, _) in loc.read_plan(sh.direct) {
                            for o in (0..s.len).step_by(chunk) {
                                let n = chunk.min(s.len - o);
                                if sh.read_span(s.shard, &mut buf.bytes_mut()[..n], s.offset + o as u64).is_ok() {
                                    bytes.fetch_add(n as u64, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                });
            }
        });
        bytes.load(Ordering::Relaxed) as f64 / 1e9 / t0.elapsed().as_secs_f64()
    }

    /// Whether experts carry packed scales.
    pub fn packed(&self) -> bool {
        self.shared.packed
    }

    pub fn expert_bytes(&self) -> usize {
        self.shared.locs.iter().flatten().next().map(|l| l.total).unwrap_or(0)
    }

    pub fn stats(&self) -> CacheStats {
        let g = self.shared.state.lock().unwrap();
        let mut s = g.stats;
        s.disk_bytes = self.shared.disk_bytes.load(Ordering::Relaxed);
        s.disk_busy_secs = self.shared.active.lock().unwrap().2 as f64 / 1e9;
        let idle = self.shared.idle.lock().unwrap();
        s.idle_boundary_secs = idle.1 as f64 / 1e9;
        s.idle_inner_secs = idle.2 as f64 / 1e9;
        s.resident = g.map.values().filter(|v| matches!(v, Slot::Ready { .. })).count();
        s.resident_bytes = g.used;
        s
    }

    /// Add routing history from a `layer expert count` file (Pelica's own
    /// `.pelica_usage`, or Colibrì's `.coli_usage`, which has the same lines).
    /// Returns the number of selections read.
    pub fn load_usage(&self, path: &std::path::Path) -> u64 {
        let Ok(text) = std::fs::read_to_string(path) else { return 0 };
        let mut total = 0;
        for line in text.lines() {
            let mut f = line.split_whitespace().map(|x| x.parse::<i64>());
            let (Some(Ok(l)), Some(Ok(e)), Some(Ok(n))) = (f.next(), f.next(), f.next()) else { continue };
            if l < 0 || e < 0 || n <= 0 || e as usize >= self.shared.n_experts {
                continue; // header lines use negative ids
            }
            let i = l as usize * self.shared.n_experts + e as usize;
            if i < self.shared.usage.len() && self.shared.locs[i].is_some() {
                self.shared.usage[i].fetch_add(n as u64, Ordering::Relaxed);
                total += n as u64;
            }
        }
        total
    }

    pub fn save_usage(&self, path: &std::path::Path) -> std::io::Result<()> {
        let ne = self.shared.n_experts;
        let mut out = String::new();
        for (i, u) in self.shared.usage.iter().enumerate() {
            let n = u.load(Ordering::Relaxed);
            if n > 0 {
                out.push_str(&format!("{} {} {}\n", i / ne, i % ne, n));
            }
        }
        std::fs::write(path, out)
    }

    /// Queue the historically most-used experts into the RAM tier, filling up
    /// to `fraction` of the budget and passing over the keys `skip` says live
    /// elsewhere (VRAM). Returns how many were queued.
    pub fn warm(&self, fraction: f64, skip: impl Fn(usize) -> bool) -> usize {
        let per = self.expert_bytes().max(1);
        let n = ((self.shared.budget() as f64 * fraction) as usize / per).min(self.shared.usage.len());
        let mut order: Vec<(u64, usize)> = self
            .shared
            .usage
            .iter()
            .enumerate()
            .map(|(i, u)| (u.load(Ordering::Relaxed), i))
            .filter(|&(c, i)| c > 0 && !skip(i))
            .collect();
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        order.truncate(n);
        let ne = self.shared.n_experts;
        // queue layer by layer in rank order so the hottest arrive first
        for &(_, i) in &order {
            self.enqueue(i / ne, &[i % ne], false);
        }
        order.len()
    }

    /// Feed the router's verdicts to the heat that decides eviction: for each
    /// token, `layer`'s experts best first (the first `k` were chosen).
    pub fn observe(&self, layer: usize, ranked: &[Vec<usize>], k: usize) {
        let sh = &self.shared;
        sh.phase.store(layer, Ordering::Relaxed);
        let mut g = sh.state.lock().unwrap();
        for r in ranked {
            g.observe(layer * sh.n_experts, r, k);
        }
    }

    /// Queue reads for experts of `layer` that are not resident or in flight;
    /// `ids` go most confident first.
    pub fn prefetch(&self, layer: usize, ids: &[usize]) {
        self.enqueue(layer, ids, true);
    }

    /// Lookahead guesses (`urgent`) are held for their layer and served
    /// nearest layer first; warm-up fills wait behind them and are plain
    /// cache entries.
    fn enqueue(&self, layer: usize, ids: &[usize], urgent: bool) {
        let sh = &self.shared;
        if sh.budget() == 0 {
            return;
        }
        let mut reads = Vec::new();
        {
            let mut g = sh.state.lock().unwrap();
            for &e in ids {
                let key = sh.key(layer, e);
                if !g.map.contains_key(&key) {
                    let p = Pending::new();
                    g.map.insert(key, Slot::Loading { pending: p.clone(), prefetched: urgent });
                    reads.push((key, p));
                }
            }
            if urgent {
                g.stats.prefetch_issued += reads.len() as u64;
            }
        }
        if reads.is_empty() {
            return;
        }
        let mut q = self.queue.jobs.lock().unwrap();
        for (rank, (key, pending)) in reads.into_iter().enumerate() {
            let prio = if urgent { (layer + 1, rank) } else { (usize::MAX, 0) };
            self.submit(&mut q, key, pending, prio);
        }
        drop(q);
        self.queue.wake();
    }

    /// Split one expert read into chunk jobs at `prio`. (0, _) is reserved
    /// for reads a layer is blocked on, so they jump the queue and every
    /// reader thread works on them at once.
    fn submit(&self, q: &mut BinaryHeap<Job>, key: u32, pending: Arc<Pending>, prio: (usize, usize)) {
        let sh = &self.shared;
        let loc = sh.locs[key as usize].as_ref().unwrap();
        let mut data = AlignedBuf::new(if sh.direct { loc.dtotal } else { loc.total });
        let ptr = data.bytes_mut().as_mut_ptr();
        let c = if prio.0 == 0 { demand_chunk_bytes() } else { chunk_bytes() };
        let chunks: Vec<(usize, u64, usize, usize)> = loc
            .read_plan(sh.direct)
            .into_iter()
            .flat_map(|(s, at)| (0..s.len).step_by(c).map(move |o| (s.shard, s.offset + o as u64, at + o, c.min(s.len - o))))
            .collect();
        let read = Arc::new(ChunkedRead {
            key,
            data: Mutex::new(Some(data)),
            ptr,
            left: AtomicUsize::new(chunks.len()),
            failed: Mutex::new(None),
            pending,
        });
        if prio.0 == 0 {
            self.queue.demand.fetch_add(chunks.len(), Ordering::AcqRel);
        }
        for (shard, offset, at, len) in chunks {
            let seq = self.queue.seq.fetch_add(1, Ordering::Relaxed);
            q.push(Job { prio: (prio.0, prio.1, seq), read: read.clone(), shard, offset, at, len });
        }
    }

    /// A better guess for `layer` ranks `keep` highest: drop the earlier
    /// guesses for it that no reader has started on and that it left out.
    pub fn regress(&self, layer: usize, keep: &[usize]) {
        let sh = &self.shared;
        let ne = sh.n_experts;
        let mut g = sh.state.lock().unwrap();
        let stale: Vec<u32> = g
            .map
            .iter()
            .filter(|(key, slot)| {
                let (l, e) = (**key as usize / ne, **key as usize % ne);
                l == layer
                    && !keep.contains(&e)
                    && matches!(slot, Slot::Loading { pending, prefetched: true } if !pending.begun.load(Ordering::Relaxed))
            })
            .map(|(k, _)| *k)
            .collect();
        for key in &stale {
            if let Some(Slot::Loading { pending, .. }) = g.map.remove(key) {
                pending.cancelled.store(true, Ordering::Relaxed);
            }
        }
        g.stats.prefetch_cancelled += stale.len() as u64;
    }

    /// `layer` has routed to `keep`. Guesses for it (or earlier layers) that
    /// were not picked stop reading, or, if already read, lose the protection
    /// they had while waiting for their layer.
    pub fn retire(&self, layer: usize, keep: &[usize]) {
        if !retire_on() {
            return;
        }
        let sh = &self.shared;
        let ne = sh.n_experts;
        let mut g = sh.state.lock().unwrap();
        let mut stale = Vec::new();
        let mut wasted = 0;
        for (key, slot) in g.map.iter_mut() {
            let (l, e) = (*key as usize / ne, *key as usize % ne);
            if l > layer || (l == layer && keep.contains(&e)) {
                continue;
            }
            match slot {
                Slot::Loading { pending, prefetched: true } => {
                    pending.cancelled.store(true, Ordering::Relaxed);
                    stale.push(*key);
                }
                Slot::Ready { prefetched, .. } if *prefetched => {
                    *prefetched = false;
                    wasted += 1;
                }
                _ => {}
            }
        }
        // Nobody waits on these (a waiter clears `prefetched`), so they can
        // go; their reader finishes the chunk it is on and drops the buffer.
        for key in &stale {
            g.map.remove(key);
        }
        g.stats.prefetch_cancelled += stale.len() as u64;
        if !stale.is_empty() {
            self.queue.wake();
        }
        g.stats.prefetch_wasted += wasted;
    }

    /// Fetch the given experts of one layer: resident ones at once, in-flight
    /// ones by waiting, the rest read now in parallel.
    pub fn get(&self, layer: usize, ids: &[usize]) -> Result<Vec<Arc<ExpertBuf>>> {
        let f = self.start(layer, ids);
        self.finish(f)
    }

    /// First half of `get`: takes what is resident and starts reading the
    /// rest, so the caller can compute with `Fetch::ready` while it arrives.
    pub fn start(&self, layer: usize, ids: &[usize]) -> Fetch {
        let sh = &self.shared;
        let now = sh.clock.fetch_add(1, Ordering::Relaxed);
        for &e in ids {
            sh.usage[layer * sh.n_experts + e].fetch_add(1, Ordering::Relaxed);
        }
        let mut out: Vec<Option<Arc<ExpertBuf>>> = vec![None; ids.len()];
        let mut mine: Vec<(usize, Arc<Pending>)> = Vec::new();
        let mut theirs: Vec<(usize, Arc<Pending>)> = Vec::new();
        {
            let mut g = sh.state.lock().unwrap();
            for &e in ids {
                g.heat_up(sh.key(layer, e));
            }
            for (i, &e) in ids.iter().enumerate() {
                let key = sh.key(layer, e);
                match g.map.get_mut(&key) {
                    Some(Slot::Ready { buf, last, prefetched }) => {
                        *last = now;
                        out[i] = Some(buf.clone());
                        if std::mem::take(prefetched) {
                            g.stats.prefetch_hits += 1;
                        } else {
                            g.stats.hits += 1;
                        }
                    }
                    Some(Slot::Loading { pending, prefetched }) => {
                        *prefetched = false;
                        theirs.push((i, pending.clone()));
                        g.stats.prefetch_hits += 1;
                    }
                    None => {
                        let p = Pending::new();
                        g.map.insert(key, Slot::Loading { pending: p.clone(), prefetched: false });
                        mine.push((i, p));
                        g.stats.misses += 1;
                    }
                }
            }
        }

        if !mine.is_empty() {
            let mut q = self.queue.jobs.lock().unwrap();
            for (i, p) in &mine {
                self.submit(&mut q, sh.key(layer, ids[*i]), p.clone(), (0, *i));
            }
            drop(q);
            self.queue.wake();
        }
        Fetch { layer, ids: ids.to_vec(), out, mine, theirs }
    }

    /// Second half of `get`: wait for the reads `start` began.
    pub fn finish(&self, f: Fetch) -> Result<Vec<Arc<ExpertBuf>>> {
        let sh = &self.shared;
        let Fetch { layer, ids, mut out, mine, theirs } = f;
        let t0 = Instant::now();
        for (i, p) in &mine {
            out[*i] = Some(p.wait()?);
        }
        let demand = t0.elapsed().as_secs_f64();
        for (i, p) in &theirs {
            out[*i] = Some(match p.wait() {
                Ok(b) => b,
                // cancelled just before this layer asked for it: read it here
                Err(_) if p.cancelled.load(Ordering::Relaxed) => Arc::new(sh.read(layer, ids[*i])?),
                Err(e) => return Err(e),
            });
        }
        let waited = t0.elapsed().as_secs_f64();

        let mut g = sh.state.lock().unwrap();
        g.stats.wait_secs += waited;
        g.stats.wait_demand_secs += demand;
        // touch everything this call used so eviction leaves it alone
        let now = sh.clock.fetch_add(1, Ordering::Relaxed);
        for &e in &ids {
            if let Some(Slot::Ready { last, .. }) = g.map.get_mut(&sh.key(layer, e)) {
                *last = now;
            }
        }
        sh.evict(&mut g, now);
        drop(g);
        Ok(out.into_iter().map(Option::unwrap).collect())
    }
}

impl Shared {
    fn key(&self, layer: usize, e: usize) -> u32 {
        (layer * self.n_experts + e) as u32
    }

    fn budget(&self) -> usize {
        self.budget.load(Ordering::Relaxed)
    }

    /// Publish a finished read: wake waiters and make it resident.
    fn complete(&self, key: u32, p: &Arc<Pending>, r: Result<Arc<ExpertBuf>, String>, from_prefetch: bool) {
        let mut g = self.state.lock().unwrap();
        let still_prefetched = match g.map.get(&key) {
            Some(Slot::Loading { pending, prefetched }) if Arc::ptr_eq(pending, p) => *prefetched && from_prefetch,
            // cancelled and asked for again since: the newer read owns the slot
            Some(Slot::Loading { .. }) => {
                drop(g);
                p.finish(r);
                return;
            }
            _ => false,
        };
        match &r {
            Ok(buf) => {
                g.used += buf.size();
                g.stats.bytes_read += buf.size() as u64;
                let last = self.clock.fetch_add(1, Ordering::Relaxed);
                g.map.insert(key, Slot::Ready { buf: buf.clone(), last, prefetched: still_prefetched });
            }
            Err(_) => {
                g.map.remove(&key);
            }
        }
        if from_prefetch {
            let now = self.clock.load(Ordering::Relaxed);
            self.evict(&mut g, now.saturating_sub(1));
        }
        drop(g);
        p.finish(r);
    }

    /// Drop the coldest resident experts until under budget. Entries touched
    /// at or after `protect`, and guesses still waiting for their layer, stay;
    /// buffers still held by a caller live on until it drops them.
    fn evict(&self, g: &mut State, protect: u64) {
        let by = evict_by();
        while g.used > self.budget() {
            let (fr, fp) = g.now();
            let heat = |k: usize| match by {
                Evict::Rank => g.rank[k] * fr + g.pick[k] * fp,
                Evict::Uses => g.heat[k],
                Evict::Lru => 0.0,
            };
            let victim = g
                .map
                .iter()
                .filter_map(|(k, s)| match s {
                    Slot::Ready { last, prefetched, .. } if *last < protect && (!*prefetched || !retire_on()) => {
                        Some((*k, heat(*k as usize), *last))
                    }
                    _ => None,
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
            let Some((k, ..)) = victim else { break };
            if let Some(Slot::Ready { buf, .. }) = g.map.remove(&k) {
                g.used -= buf.size();
            }
        }
    }

    fn read(&self, layer: usize, e: usize) -> Result<ExpertBuf> {
        let loc = self.locs[layer * self.n_experts + e]
            .as_ref()
            .ok_or_else(|| anyhow!("layer {layer} has no routed experts"))?;
        let mut data = AlignedBuf::new(if self.direct { loc.dtotal } else { loc.total });
        let bytes = data.bytes_mut();
        for (span, at) in loc.read_plan(self.direct) {
            self.read_span(span.shard, &mut bytes[at..at + span.len], span.offset)?;
        }
        Ok(self.wrap(data, loc))
    }

    fn read_span(&self, shard: usize, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        self.io_begin(buf.len());
        let r = if self.direct { read_at_least(&self.files[shard], buf, offset) } else { read_exact_at(&self.files[shard], buf, offset) };
        self.io_end();
        r
    }

    /// Account for a read going to the disk (bytes, busy and idle time).
    fn io_begin(&self, len: usize) {
        self.disk_bytes.fetch_add(len as u64, Ordering::Relaxed);
        {
            let mut a = self.active.lock().unwrap();
            if a.0 == 0 {
                let now = Instant::now();
                a.1 = Some(now);
                let mut i = self.idle.lock().unwrap();
                if let Some(t) = i.0.take() {
                    let ns = (now - t).as_nanos() as u64;
                    if self.phase.load(Ordering::Relaxed) == self.first_moe {
                        i.1 += ns;
                    } else {
                        i.2 += ns;
                    }
                }
            }
            a.0 += 1;
        }
    }

    /// A read accounted by `io_begin` is over.
    fn io_end(&self) {
        let mut a = self.active.lock().unwrap();
        a.0 -= 1;
        if a.0 == 0 {
            a.2 += a.1.take().map_or(0, |t| t.elapsed().as_nanos() as u64);
            self.idle.lock().unwrap().0 = Some(Instant::now());
        }
    }

    fn wrap(&self, data: AlignedBuf, loc: &Loc) -> ExpertBuf {
        ExpertBuf { data, at: if self.direct { loc.dat } else { loc.at }, inter: self.inter, hidden: self.hidden, packed: self.packed }
    }

    /// The last chunk of a prefetch is in: publish it, or drop it if cancelled.
    fn finish_chunked(&self, r: &ChunkedRead) {
        let data = r.data.lock().unwrap().take().unwrap();
        if r.pending.cancelled.load(Ordering::Relaxed) {
            r.pending.finish(Err("cancelled".into()));
            return;
        }
        let res = match r.failed.lock().unwrap().take() {
            Some(e) => Err(e),
            None => Ok(Arc::new(self.wrap(data, self.locs[r.key as usize].as_ref().unwrap()))),
        };
        self.complete(r.key, &r.pending, res, true);
    }
}

impl Loc {
    /// (file span, offset in the expert buffer) for each contiguous read.
    fn read_plan(&self, direct: bool) -> Vec<(Span, usize)> {
        if direct {
            return self.dspans.clone();
        }
        let mut at = 0;
        self.spans
            .iter()
            .map(|s| {
                at += s.len;
                (s.clone(), at - s.len)
            })
            .collect()
    }
}

/// Requests the async engine keeps at the drive. This drive is fastest with
/// two large requests in flight (5.8 GB/s at 16 MB); more only queue up
/// inside it, where they can no longer be reordered or cancelled.
const AIO_DEPTH: usize = 4;
/// Prefetches wait while this many blocking reads are in flight.
const AIO_HOLD: usize = 1;
/// Of those, at most this many guesses.
const AIO_PREFETCH_DEPTH: usize = 2;

/// The async reader: issues queued chunks in priority order, never more than
/// AIO_DEPTH at once, so everything else stays in the queue where a demand
/// read jumps ahead and a cancelled guess costs nothing.
#[cfg(windows)]
fn aio_engine(sh: Arc<Shared>, q: Arc<Queue>, port: Arc<crate::aio::Port>, files: Vec<File>) {
    use crate::aio::Overlapped;
    let depth = knob("PELICA_AIO_DEPTH").and_then(|v| v.parse().ok()).unwrap_or(AIO_DEPTH).max(1);
    // prefetches wait while this many blocking reads are at the drive
    // (PELICA_AIO_HOLD=0 never waits; 2 lets one guess keep a lone blocking
    // read company, since this drive is faster at depth 2 than at 1)
    let hold: usize = knob("PELICA_AIO_HOLD").and_then(|v| v.parse().ok()).unwrap_or(AIO_HOLD);
    // guesses at the drive at once: each one in flight when its layer
    // routes is read in full even if wrong
    let pdepth = knob("PELICA_AIO_PDEPTH").and_then(|v| v.parse().ok()).unwrap_or(AIO_PREFETCH_DEPTH).max(1);
    let cancel = knob("PELICA_AIO_CANCEL").as_deref() != Some("0");
    // in flight: (request, the job it serves, cancel already asked)
    let mut slots: Vec<Option<(Box<Overlapped>, Job, bool)>> = (0..depth).map(|_| None).collect();
    let mut demand_in_flight = 0usize;
    let mut prefetch_in_flight = 0usize;
    let finish_job = |job: &Job, ok: Result<(), String>| {
        let r = &job.read;
        if let Err(e) = ok {
            *r.failed.lock().unwrap() = Some(e);
        }
        if r.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            sh.finish_chunked(r);
        }
        if is_demand(job) {
            q.demand.fetch_sub(1, Ordering::AcqRel);
        }
    };
    loop {
        if q.quit.load(Ordering::Acquire) && slots.iter().all(Option::is_none) {
            return;
        }
        // stop requests whose guess was dropped (retire rings the port)
        if cancel {
            for (ov, job, asked) in slots.iter_mut().flatten() {
                if !*asked && job.read.pending.cancelled.load(Ordering::Relaxed) {
                    port.cancel(&files[job.shard], &mut **ov);
                    *asked = true;
                }
            }
        }
        // fill free slots, best job first
        {
            let mut jobs = q.jobs.lock().unwrap();
            while let Some(i) = slots.iter().position(Option::is_none) {
                let Some(top) = jobs.peek() else { break };
                if !is_demand(top) && ((hold > 0 && demand_in_flight >= hold) || prefetch_in_flight >= pdepth) {
                    break; // prefetches wait while a blocked layer is served
                }
                let job = jobs.pop().unwrap();
                if job.read.pending.cancelled.load(Ordering::Relaxed) {
                    drop(jobs);
                    finish_job(&job, Ok(()));
                    jobs = q.jobs.lock().unwrap();
                    continue;
                }
                job.read.pending.begun.store(true, Ordering::Relaxed);
                let mut ov = Box::new(Overlapped::at(job.offset));
                sh.io_begin(job.len);
                // SAFETY: the buffer range belongs to this chunk and lives
                // until its last chunk completes; `ov` is kept in `slots`.
                let r = unsafe { port.read(&files[job.shard], job.read.ptr.add(job.at), job.len, &mut *ov) };
                if let Err(e) = r {
                    sh.io_end();
                    drop(jobs);
                    finish_job(&job, Err(e.to_string()));
                    jobs = q.jobs.lock().unwrap();
                    continue;
                }
                if is_demand(&job) {
                    demand_in_flight += 1;
                } else {
                    prefetch_in_flight += 1;
                }
                slots[i] = Some((ov, job, false));
            }
        }
        // wait for a completion (or a doorbell for new work)
        let Some(done) = port.wait() else { continue };
        let Some(i) = slots.iter().position(|s| s.as_ref().is_some_and(|(ov, ..)| std::ptr::eq(&**ov, done.ov))) else { continue };
        let (_, job, _) = slots[i].take().unwrap();
        sh.io_end();
        if is_demand(&job) {
            demand_in_flight -= 1;
        } else {
            prefetch_in_flight -= 1;
        }
        // unbuffered reads of a span widened past the end of the file come
        // back short; the bytes that hold data are in once at most a sector is missing
        let ok = if done.ok && done.bytes + SECTOR > job.len {
            Ok(())
        } else {
            Err(format!("read of {} bytes at {} returned {} ({})", job.len, job.offset, done.bytes, if done.ok { "short" } else { "failed" }))
        };
        finish_job(&job, ok);
    }
}

fn io_worker(sh: Arc<Shared>, q: Arc<Queue>, demand_only: bool) {
    loop {
        let job = {
            let mut g = q.jobs.lock().unwrap();
            loop {
                let take = match g.peek() {
                    Some(j) if is_demand(j) => true,
                    Some(_) => !demand_only && q.demand.load(Ordering::Acquire) == 0,
                    None => false,
                };
                if take {
                    break g.pop().unwrap();
                }
                if q.quit.load(Ordering::Acquire) {
                    return;
                }
                g = q.cv.wait(g).unwrap();
            }
        };
        let r = &job.read;
        if !r.pending.cancelled.load(Ordering::Relaxed) {
            r.pending.begun.store(true, Ordering::Relaxed);
            // SAFETY: each chunk owns a disjoint range of the buffer, which
            // stays alive until the last chunk takes it out.
            let buf = unsafe { std::slice::from_raw_parts_mut(r.ptr.add(job.at), job.len) };
            if let Err(e) = sh.read_span(job.shard, buf, job.offset) {
                *r.failed.lock().unwrap() = Some(e.to_string());
            }
        }
        if r.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            sh.finish_chunked(r);
        }
        if is_demand(&job) && q.demand.fetch_sub(1, Ordering::AcqRel) == 1 {
            // last demand chunk: let the prefetch readers go again
            let _g = q.jobs.lock().unwrap();
            q.cv.notify_all();
        }
    }
}
/// Merge the pieces into as few contiguous reads as possible and record where
/// each piece lands in the expert buffer.
fn plan(pieces: &[Piece; PIECES]) -> Result<Loc> {
    let mut order: Vec<usize> = (0..PIECES).collect();
    order.sort_by_key(|&i| (pieces[i].shard, pieces[i].offset));
    let mut spans: Vec<Span> = Vec::new();
    let mut span_of = [0usize; PIECES];
    let mut at = [0usize; PIECES];
    let mut total = 0;
    for &i in &order {
        let p = pieces[i];
        if p.offset % 4 != 0 {
            bail!("expert piece at unaligned offset {}", p.offset);
        }
        at[i] = total;
        match spans.last_mut() {
            Some(s) if s.shard == p.shard && s.offset + s.len as u64 == p.offset => s.len += p.len,
            _ => spans.push(Span { shard: p.shard, offset: p.offset, len: p.len }),
        }
        span_of[i] = spans.len() - 1;
        total += p.len;
    }
    // unbuffered layout: each span widened to whole sectors
    let sector = SECTOR as u64;
    let mut dspans = Vec::new();
    let mut first = Vec::new(); // where each span's first real byte lands
    let mut dtotal = 0;
    for s in &spans {
        let a0 = s.offset / sector * sector;
        let a1 = (s.offset + s.len as u64).div_ceil(sector) * sector;
        let len = (a1 - a0) as usize;
        dspans.push((Span { shard: s.shard, offset: a0, len }, dtotal));
        first.push(dtotal + (s.offset - a0) as usize);
        dtotal += len;
    }
    let mut dat = [0usize; PIECES];
    for i in 0..PIECES {
        let k = span_of[i];
        dat[i] = first[k] + (pieces[i].offset - spans[k].offset) as usize;
    }
    Ok(Loc { spans, at, total, dspans, dat, dtotal })
}

#[cfg(windows)]
fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
}

#[cfg(unix)]
fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    File::open(path) // O_DIRECT needs per-platform flags; buffered for now
}

/// Fill a sector-widened span. Its tail may run past the end of the file, so
/// a short read is fine once the bytes that hold data have arrived (the last
/// sector of the span always does).
fn read_at_least(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut got = 0;
    while got < buf.len() {
        let n = read_some_at(file, &mut buf[got..], offset + got as u64)?;
        got += n;
        // Unbuffered reads come back in whole sectors except at end of file,
        // and a retry from the resulting unaligned offset would be rejected.
        if n == 0 || n % SECTOR != 0 {
            break;
        }
    }
    if got + SECTOR <= buf.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
    }
    Ok(())
}

#[cfg(windows)]
fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(unix)]
fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}
