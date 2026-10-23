//! Pelica's engine for mixture-of-experts GGUF models in general (Qwen3-MoE,
//! OLMoE, Qwen2-MoE, Mixtral-style llama, ...): the architecture is read
//! from the file's metadata and tensors, the weights stay in their stored
//! quant formats (`moe_kernels::deq4` decodes them), and the whole token
//! runs on the GPU, routing and argmax included, so decoding never waits on
//! the CPU.
//!
//! Experts that don't fit in VRAM live in host RAM that the GPU reads in
//! place; which ones get VRAM is decided by how often each was used before
//! (counted on the GPU, saved next to the model), the hottest first.
//!
//! The KV cache is split by position: the first positions in VRAM, the next
//! in RAM, the rest in a scratch file on the drive mapped into memory. The
//! CPU holds the RAM and drive parts: decoding sends each layer's queries to
//! the CPU, which attends over them (`cpu_attn`, at DDR speed) while the GPU
//! does VRAM's; prompt passes stage them into VRAM a round (`k::RK`
//! positions) at a time. Attention is computed in fixed key ranges merged in
//! a fixed order, with arithmetic the CPU reproduces bit for bit, so the
//! output doesn't depend on where the cache lives.

use crate::gguf::{gpu_layout, Gguf, Tensor, Ty};
use crate::moe_kernels as k;
use crate::oss::Stats;
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Percent of layers uploaded by the `Engine::load` in progress.
pub static LOAD_PROGRESS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// passes whose tokens are not yet read back when the next one is recorded
const LAG: usize = 2;
/// prompt passes of fewer tokens use the vector kernels (faster there)
const TC_MIN: usize = 32;

/// Architectures whose output was compared with llama.cpp's.
const CHECKED: [&str; 2] = ["olmoe", "qwen3moe"];
/// Architectures whose RoPE rotates adjacent pairs (the rest: NeoX halves).
const ROPE_NORM: [&str; 8] = ["llama", "llama4", "deepseek", "granitemoe", "bailingmoe", "ernie4_5-moe", "arctic", "minicpm"];

/// Model shape, from the GGUF metadata.
#[derive(Debug, Clone)]
pub struct Cfg {
    pub arch: String,
    pub n_layer: usize,
    pub d: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub n_exp: usize,
    pub top_k: usize,
    pub ff_exp: usize,
    pub vocab: usize,
    pub eps: f32,
    pub ctx: usize,
    /// rotated dimensions per head
    pub rot: usize,
    pub neox: bool,
    /// expert weights divided by their sum
    pub norm_w: bool,
    pub w_scale: f32,
    pub sigmoid: bool,
}

/// A byte range of one of the weight buffers.
#[derive(Clone, Copy)]
struct Reg {
    buf: usize,
    off: u64,
    size: u64,
}

/// A matrix in VRAM, in its GPU layout: `rows` rows of `wpr` words.
#[derive(Clone, Copy)]
struct Mat {
    reg: Reg,
    ty: Ty,
    rows: usize,
    cols: usize,
    wpr: usize,
}

/// One kind of expert matrix (gate, up or down) of a layer.
#[derive(Clone, Copy)]
struct Exp {
    ty: Ty,
    rows: usize,
    cols: usize,
    wpr: usize,
    /// words per expert
    wpe: usize,
}

struct Shared {
    gate: Mat,
    up: Mat,
    down: Mat,
    /// a gate logit per token (sigmoid) scaling the shared expert, if any
    inp: Option<Mat>,
}

enum Ffn {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe { router: Mat, sel_b: Option<u32>, gate: Exp, up: Exp, down: Exp, shared: Option<Shared> },
}

struct Layer {
    attn_norm: u32,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    bias: [Option<u32>; 4],
    /// RMS norms of q and k: weight and length (a head, or all of them)
    q_norm: Option<(u32, usize)>,
    k_norm: Option<(u32, usize)>,
    ffn_norm: u32,
    ffn: Ffn,
}

/// Host-side staging of the pieces that become one VRAM buffer.
struct Blob {
    data: Vec<u8>,
    buf: usize,
}

impl Blob {
    fn add(&mut self, bytes: &[u8]) -> Reg {
        let off = self.data.len().div_ceil(256) * 256;
        self.data.resize(off, 0);
        self.data.extend_from_slice(bytes);
        let size = (bytes.len() as u64).div_ceil(16) * 16;
        Reg { buf: self.buf, off: off as u64, size }
    }
}

/// A prompt-attention round's dispatches: the pass's rows into staging,
/// attention over the round's keys, folding it into the running result.
struct RoundPlan {
    stage: usize,
    pre: usize,
    fold: usize,
}

/// A layer's dispatches: up to the cache writes, decode attention over the
/// resident cache, prompt rounds, merging decode attention, the rest.
struct LayerPlan {
    before: std::ops::Range<usize>,
    dec_res: usize,
    rounds: Vec<RoundPlan>,
    merge: usize,
    after: std::ops::Range<usize>,
}

/// The cache positions on the drive: a scratch file (deleted when closed),
/// mapped into memory, holding from the first round with drive positions on
/// a block per (round, layer) of keys then values.
struct Drive {
    map: memmap2::MmapRaw,
    _file: std::fs::File,
    path: PathBuf,
    r_first: usize,
}

/// Zeroed memory of our own (the OS commits pages as they're touched).
struct Mem {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain bytes; `Engine` orders the writes (between passes and
// before the reads of each layer's attention)
unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

impl Mem {
    fn zeroed(len: usize) -> Option<Mem> {
        let layout = std::alloc::Layout::from_size_align(len.max(1), 64).ok()?;
        // SAFETY: nonzero size
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        (!ptr.is_null()).then_some(Mem { ptr, len })
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align_unchecked(self.len.max(1), 64)) };
    }
}

/// The cache positions past VRAM's, which the CPU holds and attends over
/// (`cpu_attn`): the next `kv_ram` in RAM (per layer: keys, then values),
/// the rest on the drive.
struct HostKv {
    ram: Vec<Mem>,
    drive: Option<Drive>,
}

struct Disp {
    pipe: usize,
    /// for debugging: the kernel and its arguments
    what: String,
    bind: wgpu::BindGroup,
    /// workgroups in a prompt pass (up to PT tokens) and a decode pass
    pre: [u32; 3],
    dec: [u32; 3],
    /// prompt passes' tensor-core kernel instead, if any (grid: `pre`;
    /// `pre_v` for the vector kernel in short prompt passes)
    tc: Option<(usize, wgpu::BindGroup)>,
    pre_v: [u32; 3],
    /// decode passes' kernel instead, if any (grid: `dec`)
    dv: Option<(usize, wgpu::BindGroup)>,
    /// 0: every pass; 1: prompt passes only; 2: decode passes only
    mode: u8,
}

impl Disp {
    /// The pipeline, bind group and grid of a decode pass, or of a prompt
    /// pass of `t` tokens (tensor cores from TC_MIN tokens on).
    /// (None: the dispatch isn't part of such passes.)
    fn pick(&self, decode: bool, t: usize) -> Option<(usize, &wgpu::BindGroup, [u32; 3])> {
        if self.mode == if decode { 1 } else { 2 } {
            return None;
        }
        if let (Some((p, b)), true) = (&self.dv, decode) {
            return Some((*p, b, self.dec));
        }
        Some(match (&self.tc, decode) {
            (Some((p, b)), false) if t >= TC_MIN => (*p, b, self.pre),
            (Some(_), false) => (self.pipe, &self.bind, self.pre_v),
            _ => (self.pipe, &self.bind, if decode { self.dec } else { self.pre }),
        })
    }
}

/// Which experts sit in VRAM follows what the model uses now. The use
/// counters (`usage`, counted by the routing kernel) are read back every few
/// tokens without waiting; each expert keeps a decaying score; per layer, the
/// host experts used more than the least used VRAM ones change places with
/// them: GPU copies through `tmp`, the slot table (`loc`) rewritten in the
/// same submission, so passes already queued see the old places and later
/// ones the new. The prompt says what the reply will be about (code, a
/// story, another language), so the first round after it moves the most.
struct Adapt {
    /// per layer: (VRAM region, host region, bytes per expert) of gate, up, down
    regs: Vec<Option<[(Reg, Reg, u64); 3]>>,
    vbufs: Vec<wgpu::Buffer>,
    hbufs: Vec<wgpu::Buffer>,
    loc_buf: wgpu::Buffer,
    tmp: wgpu::Buffer,
    rb: [wgpu::Buffer; 2],
    ready: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2],
    st: std::sync::Mutex<AdaptState>,
}

struct AdaptState {
    /// each (layer, expert)'s slot; the top bit set: in host RAM
    loc: Vec<u32>,
    score: Vec<f32>,
    /// the counters as last read
    prev: Vec<u32>,
    /// the readback buffer on its way, if any
    pending: Option<usize>,
    turn: usize,
    /// experts moved so far
    moved: usize,
}

/// Tokens between two reads of the use counters, and how fast old use
/// fades (per read: half of it after about 100 tokens).
const ADAPT_EVERY: usize = 8;
const ADAPT_DECAY: f32 = 0.944;
/// Bytes moved per round (both ways), and in the first round after a prompt.
const ADAPT_BYTES: u64 = 64 << 20;
const ADAPT_FIRST: u64 = 1 << 30;
const ADAPT_TMP: u64 = 32 << 20;

pub struct Engine {
    pub cfg: Cfg,
    /// the GPU and API, for display
    pub name: String,
    /// the model's own context length
    pub max_ctx: usize,
    /// experts in VRAM and in host RAM, over all layers
    pub vram_experts: usize,
    pub host_experts: usize,
    pub host_bytes: usize,
    /// tokens that end a reply
    pub eos: Vec<u32>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Vec<wgpu::ComputePipeline>,
    /// every dispatch, recorded once; a pass picks from them
    disps: Vec<Disp>,
    embed: usize,
    plans: Vec<LayerPlan>,
    head: std::ops::Range<usize>,
    /// cache positions in VRAM ([0, kv_vram)) and in host RAM (the next
    /// kv_ram); the rest are on the drive
    pub kv_vram: usize,
    pub kv_ram: usize,
    /// bytes of a position's keys (or values) in a layer
    rowb: usize,
    host: HostKv,
    /// address ranges for the prefetch thread to bring in (drive rows)
    prefetcher: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>,
    /// a decode pass's row past VRAM; per layer, its queries and that row
    /// read back for the CPU
    krow: wgpu::Buffer,
    lreads: Vec<wgpu::Buffer>,
    /// the VRAM part of the cache, per layer
    kv: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// decode attention partials, per (chunk, head)
    part: wgpu::Buffer,
    /// the queries (word offset in `act`) and their scale
    rq: u32,
    scale: f32,
    stage: (wgpu::Buffer, wgpu::Buffer),
    /// a prompt pass's new rows past VRAM on their way to the CPU
    kv_out: wgpu::Buffer,
    st: wgpu::Buffer,
    act: wgpu::Buffer,
    usage: wgpu::Buffer,
    reads: Vec<wgpu::Buffer>,
    usage_path: PathBuf,
    usage_base: Vec<u32>,
    /// experts moving between VRAM and host RAM as their use changes
    adapt: Option<Adapt>,
    /// where in `st` the routing kernel writes each prompt pass's routes
    trace_at: usize,
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as u32;
    if e == 0 {
        let v = m as f32 * (1.0 / (1u32 << 24) as f32);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}

fn floats(g: &Gguf, t: &Tensor) -> Result<Vec<f32>> {
    let b = g.bytes(t);
    Ok(match t.ty {
        Ty::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Ty::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        Ty::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        t => bail!("expected a float tensor, got {}", t.name()),
    })
}

/// A tensor's bytes, read with large reads (much faster than faulting in a
/// memory map on slow or USB drives).
fn read_tensor(f: &std::fs::File, t: &Tensor) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; t.bytes()];
    let mut done = 0usize;
    while done < buf.len() {
        let end = (done + (64 << 20)).min(buf.len());
        let off = (t.offset + done) as u64;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..end], off)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..end], off)?;
        if n == 0 {
            bail!("unexpected end of file");
        }
        done += n;
    }
    Ok(buf)
}

/// A thread that asks the OS to bring address ranges (of the mapped drive
/// file) into memory, so the reads go on while the GPU and CPU work.
fn prefetch_thread() -> std::sync::mpsc::Sender<Vec<(usize, usize)>> {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
    std::thread::spawn(move || {
        while let Ok(ranges) = rx.recv() {
            #[cfg(windows)]
            {
                #[repr(C)]
                struct Range {
                    addr: usize,
                    len: usize,
                }
                extern "system" {
                    fn GetCurrentProcess() -> isize;
                    fn PrefetchVirtualMemory(process: isize, n: usize, ranges: *const Range, flags: u32) -> i32;
                }
                let r: Vec<Range> = ranges.iter().map(|&(addr, len)| Range { addr, len }).collect();
                // SAFETY: ranges inside the live mapping (a hint: a range
                // unmapped meanwhile only makes the call fail)
                unsafe { PrefetchVirtualMemory(GetCurrentProcess(), r.len(), r.as_ptr(), 0) };
            }
            #[cfg(not(windows))]
            let _ = ranges;
        }
    });
    tx
}

/// Bytes free for this user on the drive holding `dir`.
fn disk_free(dir: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        extern "system" {
            fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
        }
        let w: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
        if unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut a, &mut t, &mut f) } != 0 {
            return Some(a);
        }
    }
    let _ = dir;
    None
}

/// Where the drive part of the cache goes: PELICA_KV_DIR, else
/// %LOCALAPPDATA%\Pelica\kv (the temp directory elsewhere).
fn kv_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PELICA_KV_DIR") {
        return PathBuf::from(d);
    }
    match std::env::var_os("LOCALAPPDATA") {
        Some(d) => PathBuf::from(d).join("Pelica").join("kv"),
        None => std::env::temp_dir().join("pelica-kv"),
    }
}

/// A new scratch file that the OS deletes once it's closed (Windows), or
/// that is unlinked at once (elsewhere).
fn scratch_file(dir: &Path) -> Result<(std::fs::File, PathBuf)> {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("cache-{}-{}.kv", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    let mut oo = std::fs::OpenOptions::new();
    oo.read(true).write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        oo.custom_flags(0x0400_0000); // FILE_FLAG_DELETE_ON_CLOSE
    }
    let f = oo.open(&path).with_context(|| format!("creating {}", path.display()))?;
    #[cfg(unix)]
    let _ = std::fs::remove_file(&path);
    Ok((f, path))
}

fn bytes_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn decodable(name: &str, t: &Tensor) -> Result<()> {
    if t.ty.gpu_words().is_none() {
        bail!("{name}: {} weights are not supported yet", t.ty.name());
    }
    if t.dims[0] % t.ty.gpu_values() != 0 {
        bail!("{name}: rows of {} are not whole {} blocks", t.dims[0], t.ty.name());
    }
    Ok(())
}

fn words_per_row(ty: Ty, cols: usize) -> usize {
    cols / ty.gpu_values() * ty.gpu_words().unwrap()
}

/// cos/sin per rotated pair (pair m turns by pos * base^(-2m/rot)), with
/// linear or YaRN scaling as the file says, computed in f64: rows for the
/// positions 256*i (with YaRN's magnitude scale), then rows for 0..256; the
/// kernel composes a position's angle from the two (a row per position
/// would be large at long contexts).
fn rope_table(g: &Gguf, c: &Cfg) -> Vec<f32> {
    let a = &c.arch;
    let mf = |k: &str| g.meta.get(&format!("{a}.{k}")).and_then(|v| v.as_f64());
    let base = mf("rope.freq_base").unwrap_or(10000.0);
    let kind = g.s(&format!("{a}.rope.scaling.type")).unwrap_or("none").to_string();
    let factor = mf("rope.scaling.factor").filter(|f| *f > 0.0).unwrap_or(1.0);
    let orig = mf("rope.scaling.original_context_length").unwrap_or(4096.0);
    let n_dims = c.rot as f64;
    let half = c.rot / 2;
    let (freq_scale, yarn) = match kind.as_str() {
        "linear" => (1.0 / factor, false),
        "yarn" if factor != 1.0 => (1.0 / factor, true),
        _ => (1.0, false),
    };
    let corr = |rot: f64| n_dims * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let (lo, hi) = (corr(mf("rope.scaling.yarn_beta_fast").unwrap_or(32.0)).floor().max(0.0), corr(mf("rope.scaling.yarn_beta_slow").unwrap_or(1.0)).ceil().min(n_dims - 1.0));
    let mscale = if yarn { 1.0 + 0.1 * (1.0 / freq_scale).ln() } else { 1.0 };
    let n_hi = c.ctx.div_ceil(256);
    let mut t = vec![0f32; (n_hi + 256) * c.rot];
    for row in 0..n_hi + 256 {
        let (pos, ms) = if row < n_hi { (row * 256, mscale) } else { (row - n_hi, 1.0) };
        for j in 0..half {
            let extrap = pos as f64 * base.powf(-((2 * j) as f64) / n_dims);
            let interp = freq_scale * extrap;
            let theta = if yarn {
                let ramp = 1.0 - ((j as f64 - lo) / (hi - lo).max(0.001)).clamp(0.0, 1.0);
                interp * (1.0 - ramp) + extrap * ramp
            } else {
                interp
            };
            t[(row * half + j) * 2] = (theta.cos() * ms) as f32;
            t[(row * half + j) * 2 + 1] = (theta.sin() * ms) as f32;
        }
    }
    t
}

/// Kernel sources, each compiled once however many dispatches use it.
#[derive(Default)]
struct Kernels {
    srcs: Vec<String>,
    names: Vec<&'static str>,
    by: HashMap<String, usize>,
}

impl Kernels {
    fn get(&mut self, name: &'static str, s: String) -> usize {
        if let Some(&i) = self.by.get(&s) {
            return i;
        }
        self.names.push(name);
        self.srcs.push(s.clone());
        self.by.insert(s, self.srcs.len() - 1);
        self.srcs.len() - 1
    }
}

/// Is this GGUF file a model this engine runs (a MoE that isn't gpt-oss, or
/// DeepSeek V4, which has an engine of its own in `ds4`)?
pub fn handles(g: &Gguf) -> bool {
    !matches!(g.arch(), "gpt-oss" | "deepseek4") && g.tensors.keys().any(|n| n.ends_with("ffn_gate_exps.weight"))
}

/// The tokens that end a reply: the file's end-of-sequence and end-of-turn
/// tokens, and the usual end-of-turn strings if the vocabulary has them.
pub fn stop_tokens(g: &Gguf) -> Vec<u32> {
    let mut v = Vec::new();
    for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
        if let Some(id) = g.meta.get(key).and_then(|x| x.as_u64()) {
            v.push(id as u32);
        }
    }
    if let Ok(toks) = g.strs("tokenizer.ggml.tokens") {
        for s in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "<|return|>", "<end_of_turn>", "<|end_of_text|>", "</s>"] {
            if let Some(i) = toks.iter().position(|t| *t == s) {
                v.push(i as u32);
            }
        }
    }
    v.sort();
    v.dedup();
    v
}

impl Engine {
    pub fn load(path: &Path, ctx: usize) -> Result<Engine> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let file = std::fs::File::open(path)?;
        let a = g.arch().to_string();
        if !handles(&g) {
            bail!("{}: not a mixture-of-experts model this engine runs (architecture {a})", path.display());
        }
        let key = |s: &str| format!("{a}.{s}");
        let n_head = g.u(&key("attention.head_count"))?;
        let d = g.u(&key("embedding_length"))?;
        let hd = g.u(&key("attention.key_length")).unwrap_or(d / n_head);
        if g.u(&key("attention.value_length")).unwrap_or(hd) != hd {
            bail!("{a}: keys and values of different sizes are not supported");
        }
        let max_ctx = g.u(&key("context_length")).unwrap_or(4096);
        let ctx = ctx.min(max_ctx).div_ceil(k::CHUNK) * k::CHUNK;
        let tok_embd = g.tensor("token_embd.weight")?;
        let gating = g.u(&key("expert_gating_func")).unwrap_or(1);
        let cfg = Cfg {
            arch: a.clone(),
            n_layer: g.u(&key("block_count"))?,
            d,
            n_head,
            n_kv: g.u(&key("attention.head_count_kv")).unwrap_or(n_head),
            hd,
            n_exp: g.u(&key("expert_count"))?,
            top_k: g.u(&key("expert_used_count"))?,
            ff_exp: g.u(&key("expert_feed_forward_length")).or_else(|_| g.u(&key("feed_forward_length")))?,
            vocab: tok_embd.dims[1],
            eps: g.f(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            ctx,
            rot: g.u(&key("rope.dimension_count")).unwrap_or(hd),
            neox: !ROPE_NORM.contains(&a.as_str()),
            norm_w: g.b(&key("expert_weights_norm")).unwrap_or(matches!(a.as_str(), "qwen3moe" | "llama" | "mixtral")),
            w_scale: g.f(&key("expert_weights_scale")).unwrap_or(0.0),
            sigmoid: gating == 2,
        };
        if cfg.hd % 64 != 0 || cfg.n_head % cfg.n_kv != 0 || cfg.n_exp > k::NEMAX || cfg.top_k > k::KMAX || cfg.d % 8 != 0 || cfg.rot > cfg.hd {
            bail!("unsupported model shape: {cfg:?}");
        }
        if !CHECKED.contains(&a.as_str()) {
            eprintln!("[pelica] note: the {a} architecture has not been checked against llama.cpp; output may be off");
        }
        let ctx = cfg.ctx;

        // device with 32-wide subgroups
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if !adapter.features().contains(wgpu::Features::SUBGROUP) || info.subgroup_min_size > 32 || info.subgroup_max_size < 32 {
            bail!("{} lacks 32-wide subgroups, which the GPU engine needs", info.name);
        }
        let limits = adapter.limits();
        let mappable = adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        // tensor cores for the prompt passes' products, if offered
        // (PELICA_TC=0: not)
        let tcf = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        let tc = adapter.features().contains(tcf)
            && adapter.cooperative_matrix_properties().iter().any(|c| c.m_size == 16 && c.n_size == 16 && c.k_size == 16 && c.ab_type == wgpu::CooperativeScalarType::F16 && c.cr_type == wgpu::CooperativeScalarType::F32)
            && std::env::var("PELICA_TC").map_or(true, |v| v != "0");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pelica-moe"),
            required_features: wgpu::Features::SUBGROUP | mappable | (adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)) | if tc { tcf } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            // SAFETY: cooperative matrices are only used as wgpu documents
            // them (16x16 f16 x f16 + f32, a configuration the device lists)
            experimental_features: if tc { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;
        let name = format!("{} ({:?})", info.name, info.backend);

        let vram = std::cell::Cell::new(0usize);
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: 64 << 20, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| -> Result<wgpu::Buffer> {
            let size = size.max(16).div_ceil(16) * 16;
            let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let b = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
            if let Some(e) = pollster::block_on(scope.pop()) {
                bail!("allocating {label} ({size} bytes, {:.2} GB so far): {e}", vram.get() as f64 / 1e9);
            }
            Ok(b)
        };
        let upload = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| -> Result<wgpu::Buffer> {
            let b = make(label, bytes.len() as u64, usage | wgpu::BufferUsages::COPY_DST)?;
            vram.set(vram.get() + b.size() as usize);
            let n4 = bytes.len() / 4 * 4;
            crate::gpu::write_through(&device, &queue, &staging, &[(&b, 0, &bytes[..n4])]);
            if n4 < bytes.len() {
                let mut tail = [0u8; 4];
                tail[..bytes.len() - n4].copy_from_slice(&bytes[n4..]);
                crate::gpu::write_through(&device, &queue, &staging, &[(&b, n4 as u64, &tail)]);
            }
            Ok(b)
        };
        let ro = wgpu::BufferUsages::STORAGE;
        let mut wbufs: Vec<wgpu::Buffer> = Vec::new();
        let finish = |mut blob: Blob, wbufs: &mut Vec<wgpu::Buffer>| -> Result<()> {
            assert_eq!(blob.buf, wbufs.len());
            let n = blob.data.len().div_ceil(16) * 16;
            blob.data.resize(n, 0);
            wbufs.push(upload("weights", &blob.data, ro)?);
            Ok(())
        };
        let mat = |name: &str, blob: &mut Blob| -> Result<Mat> {
            let t = g.tensor(name)?;
            decodable(name, t)?;
            let (cols, rows) = (t.dims[0], t.dims.get(1).copied().unwrap_or(1));
            Ok(Mat { reg: blob.add(&gpu_layout(t.ty, &read_tensor(&file, t)?)), ty: t.ty, rows, cols, wpr: words_per_row(t.ty, cols) })
        };
        let exp = |name: &str| -> Result<Exp> {
            let t = g.tensor(name)?;
            decodable(name, t)?;
            if t.dims.len() != 3 || t.dims[2] != cfg.n_exp {
                bail!("{name}: unexpected shape {:?}", t.dims);
            }
            let wpr = words_per_row(t.ty, t.dims[0]);
            Ok(Exp { ty: t.ty, rows: t.dims[1], cols: t.dims[0], wpr, wpe: wpr * t.dims[1] })
        };

        // small f32 weights (norms, biases), all in one buffer
        let mut fw: Vec<f32> = Vec::new();
        let mut put = |v: Vec<f32>| -> u32 {
            let at = fw.len() as u32;
            fw.extend(v);
            while fw.len() % 4 != 0 {
                fw.push(0.0);
            }
            at
        };
        let has = |n: &str| g.tensors.contains_key(n);

        let mut blob = Blob { data: Vec::new(), buf: 0 };
        let embd = mat("token_embd.weight", &mut blob)?;
        let out = if has("output.weight") { mat("output.weight", &mut blob)? } else { embd };
        finish(blob, &mut wbufs)?;
        let out_norm = put(floats(&g, g.tensor("output_norm.weight")?)?);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut ff_dense = 0usize;
        let mut ff_sh = 0usize;
        for l in 0..cfg.n_layer {
            let t = |s: &str| format!("blk.{l}.{s}");
            let opt = |s: &str, put: &mut dyn FnMut(Vec<f32>) -> u32| -> Result<Option<u32>> {
                if has(&t(s)) { Ok(Some(put(floats(&g, g.tensor(&t(s))?)?))) } else { Ok(None) }
            };
            let mut blob = Blob { data: Vec::new(), buf: wbufs.len() };
            if !has(&t("attn_q.weight")) {
                bail!("{a}: layer {l} has no separate q/k/v projections (fused or latent attention is not supported yet)");
            }
            let (q, kk, v, o) = (mat(&t("attn_q.weight"), &mut blob)?, mat(&t("attn_k.weight"), &mut blob)?, mat(&t("attn_v.weight"), &mut blob)?, mat(&t("attn_output.weight"), &mut blob)?);
            if q.rows != cfg.n_head * hd || kk.rows != cfg.n_kv * hd || v.rows != cfg.n_kv * hd || o.cols != cfg.n_head * hd {
                bail!("{a}: layer {l}: attention shapes don't match the metadata");
            }
            let bias = [opt("attn_q.bias", &mut put)?, opt("attn_k.bias", &mut put)?, opt("attn_v.bias", &mut put)?, opt("attn_output.bias", &mut put)?];
            let mut norm = |s: &str| -> Result<Option<(u32, usize)>> {
                if !has(&t(s)) {
                    return Ok(None);
                }
                let w = floats(&g, g.tensor(&t(s))?)?;
                let n = w.len();
                Ok(Some((put(w), n)))
            };
            let (q_norm, k_norm) = (norm("attn_q_norm.weight")?, norm("attn_k_norm.weight")?);
            let ffn = if has(&t("ffn_gate_exps.weight")) {
                let router = mat(&t("ffn_gate_inp.weight"), &mut blob)?;
                let sel_b = opt("exp_probs_b.bias", &mut put)?;
                let shared = if has(&t("ffn_up_shexp.weight")) {
                    let s = Shared {
                        gate: mat(&t("ffn_gate_shexp.weight"), &mut blob)?,
                        up: mat(&t("ffn_up_shexp.weight"), &mut blob)?,
                        down: mat(&t("ffn_down_shexp.weight"), &mut blob)?,
                        inp: if has(&t("ffn_gate_inp_shexp.weight")) { Some(mat(&t("ffn_gate_inp_shexp.weight"), &mut blob)?) } else { None },
                    };
                    ff_sh = ff_sh.max(s.gate.rows);
                    Some(s)
                } else {
                    None
                };
                let (gate, up, down) = (exp(&t("ffn_gate_exps.weight"))?, exp(&t("ffn_up_exps.weight"))?, exp(&t("ffn_down_exps.weight"))?);
                if gate.rows != cfg.ff_exp || up.rows != cfg.ff_exp || down.cols != cfg.ff_exp || gate.cols != d || down.rows != d {
                    bail!("{a}: layer {l}: expert shapes don't match the metadata");
                }
                Ffn::Moe { router, sel_b, gate, up, down, shared }
            } else {
                let f = Ffn::Dense { gate: mat(&t("ffn_gate.weight"), &mut blob)?, up: mat(&t("ffn_up.weight"), &mut blob)?, down: mat(&t("ffn_down.weight"), &mut blob)? };
                if let Ffn::Dense { gate, .. } = &f {
                    ff_dense = ff_dense.max(gate.rows);
                }
                f
            };
            let ffn_norm = if has(&t("ffn_norm.weight")) { t("ffn_norm.weight") } else { t("post_attention_norm.weight") };
            layers.push(Layer { attn_norm: put(floats(&g, g.tensor(&t("attn_norm.weight"))?)?), q, k: kk, v, o, bias, q_norm, k_norm, ffn_norm: put(floats(&g, g.tensor(&ffn_norm)?)?), ffn });
            finish(blob, &mut wbufs)?;
        }
        let fw_buf = upload("fw", bytes_f32(&fw), ro)?;
        let rope_buf = upload("rope", bytes_f32(&rope_table(&g, &cfg)), ro)?;
        // VRAM left once the weights are in (buffers below are counted by
        // size: the driver only reports them once they're touched)
        let free_now = crate::free_vram();

        // activations: regions of one buffer, each a prompt pass's worth
        let (pt, kx) = (k::PT, cfg.top_k);
        let (qd, kvd, ne) = (cfg.n_head * hd, cfg.n_kv * hd, cfg.n_exp);
        let mut asize = 0usize;
        let mut region = |n: usize| -> u32 {
            let at = asize;
            asize += n.div_ceil(64) * 64;
            at as u32
        };
        let (rx, rxn, rq, rk, rv, ratt) = (region(pt * d), region(pt * d), region(pt * qd), region(pt * kvd), region(pt * kvd), region(pt * qd));
        let (rlg, rwt, rh, reo) = (region(pt * ne), region(pt * kx), region(pt * kx * cfg.ff_exp), region(pt * kx * d));
        let (rfg, rfu) = (region(pt * ff_dense), region(pt * ff_dense));
        let (rsg, rsu, rso, rsl) = (region(pt * ff_sh), region(pt * ff_sh), region(pt * d), region(pt));
        let (rlog, rarg) = (region(cfg.vocab), region(2 * k::ARGG));
        let racc = region(pt * cfg.n_head * (hd + 2));
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let act = make("activations", asize as u64 * 4, rw)?;
        // state: the pass plan, tokens, then the routes of the last prompt pass
        let trace_at = k::HIST + ctx + LAG + 64;
        let st = make("state", (trace_at + cfg.n_layer * pt * kx * 2) as u64 * 4, rw)?;
        let usage = make("expert use", (cfg.n_layer * ne) as u64 * 4, rw)?;
        // attention partials: per (head, chunk of positions) for decoding,
        // per (token, head, split) for prompt passes
        let n_ch = ctx / k::DCHUNK;
        let part = make("decode partials", (cfg.n_head * n_ch * (hd + 2)) as u64 * 4, rw)?;
        let ppart = make("prompt partials", (pt * cfg.n_head * k::RSPL * (hd + 2)) as u64 * 4, rw)?;
        // the cache, f16, a buffer per layer for keys and for values: its
        // first positions in VRAM (all of them if that takes at most 40% of
        // what's free, else a quarter of it, so the experts keep the rest)
        let rowb = kvd * 2;
        let perpos = rowb * 2 * cfg.n_layer;
        let env_n = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let kv_vram = match env_n("PELICA_KV_VRAM") {
            Some(n) => n,
            None => match free_now {
                Some(f) if ctx * perpos > f * 2 / 5 => f / 4 / perpos,
                _ => ctx,
            },
        };
        let kv_vram = (kv_vram.min(ctx) / k::CHUNK * k::CHUNK).max(k::CHUNK);
        let mut kv = Vec::new();
        for _ in 0..cfg.n_layer {
            kv.push((make("k cache", (kv_vram * rowb) as u64, rw)?, make("v cache", (kv_vram * rowb) as u64, rw)?));
        }
        // a round of positions beyond VRAM's, staged for attention
        let srows = if kv_vram < ctx { k::RK } else { 1 };
        let stage = (make("k staging", (srows * rowb) as u64, rw)?, make("v staging", (srows * rowb) as u64, rw)?);
        let others = act.size() as usize + st.size() as usize + part.size() as usize + ppart.size() as usize + usage.size() as usize + 2 * cfg.n_layer * kv_vram * rowb + 2 * srows * rowb;
        vram.set(vram.get() + others);

        // experts: the hottest in VRAM, as many as fit, the rest in host RAM
        let usage_path = PathBuf::from(format!("{}.pelica-usage", path.display()));
        let usage_base: Vec<u32> = std::fs::read(&usage_path)
            .ok()
            .filter(|b| b.len() == cfg.n_layer * ne * 4)
            .map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
            .unwrap_or_else(|| vec![0; cfg.n_layer * ne]);
        let ebytes = |l: usize| match &layers[l].ffn {
            Ffn::Moe { gate, up, down, .. } => (gate.wpe + up.wpe + down.wpe) * 4,
            _ => 0,
        };
        let total: usize = (0..cfg.n_layer).map(|l| ebytes(l) * ne).sum();
        let budget = match std::env::var("PELICA_EXPERTS_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(gb) => (gb * 1e9) as usize,
            None => match free_now {
                // (and per layer: the expert places and params, small)
                Some(f) => f.saturating_sub(crate::vram_margin() + others + (16 << 20)),
                None => total,
            },
        };
        // (layer, expert) hottest first; ties: lower expert index, so that
        // without history every layer gets the same experts
        let mut order: Vec<(usize, usize)> = (0..cfg.n_layer).filter(|&l| ebytes(l) > 0).flat_map(|l| (0..ne).map(move |e| (l, e))).collect();
        order.sort_by_key(|&(l, e)| (std::cmp::Reverse(usage_base[l * ne + e]), e, l));
        if std::env::var_os("PELICA_DEBUG").is_some() {
            eprintln!("[pelica] free {:?} after weights {:.2} GB, others {:.2} GB, expert budget {:.2} GB of {:.2} GB", free_now.map(|f| f as f64 / 1e9), vram.get() as f64 / 1e9, others as f64 / 1e9, budget as f64 / 1e9, total as f64 / 1e9);
        }
        // (when some experts stay in RAM: room for the buffer they change places through)
        let budget = if total > budget { budget.saturating_sub(ADAPT_TMP as usize) } else { budget };
        let mut in_vram = vec![false; cfg.n_layer * ne];
        let mut used = 0usize;
        for &(l, e) in &order {
            if used + ebytes(l) > budget {
                break;
            }
            used += ebytes(l);
            in_vram[l * ne + e] = true;
        }
        let host_bytes = total - used;
        if host_bytes > 0 {
            if mappable.is_empty() {
                bail!("the experts need {:.1} GB more VRAM than is free, and this GPU cannot read host memory", host_bytes as f64 / 1e9);
            }
            let ram = crate::available_ram() as usize;
            if host_bytes + (4usize << 30) > ram {
                bail!("the experts that don't fit in VRAM need {:.1} GB of RAM ({:.1} GB free)", host_bytes as f64 / 1e9, ram as f64 / 1e9);
            }
        }
        let host_buf = |label: &str, bytes: &[u8]| -> Result<wgpu::Buffer> {
            let b = make(label, bytes.len() as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC)?;
            crate::gpu::write_through(&device, &queue, &staging, &[(&b, 0, bytes)]);
            Ok(b)
        };
        let dummy = make("none", 16, rw)?;
        let mut loc = vec![0u32; cfg.n_layer * ne];
        // the experts' bytes, packed into buffers of about 1 GB (the
        // allocator wastes much of a block on each medium-sized buffer): in
        // VRAM and in host RAM
        let arena = (limits.max_storage_buffer_binding_size as usize).min(1 << 30);
        let (mut vbufs, mut hbufs): (Vec<wgpu::Buffer>, Vec<wgpu::Buffer>) = (Vec::new(), Vec::new());
        let (mut vblob, mut hblob) = (Blob { data: Vec::new(), buf: 0 }, Blob { data: Vec::new(), buf: 0 });
        let flush = |blob: &mut Blob, bufs: &mut Vec<wgpu::Buffer>, host: bool| -> Result<()> {
            if blob.data.is_empty() {
                return Ok(());
            }
            let n = blob.data.len().div_ceil(16) * 16;
            blob.data.resize(n, 0);
            bufs.push(if host { host_buf("experts (RAM)", &blob.data)? } else { upload("experts", &blob.data, ro | wgpu::BufferUsages::COPY_SRC)? });
            blob.data.clear();
            blob.buf = bufs.len();
            Ok(())
        };
        // per layer: (VRAM, host) places of gate, up, down
        let mut eregs: Vec<Option<[(Option<Reg>, Option<Reg>); 3]>> = Vec::new();
        let (mut nv, mut nh) = (0usize, 0usize);
        for l in 0..cfg.n_layer {
            let Ffn::Moe { gate, up, down, .. } = &layers[l].ffn else {
                eregs.push(None);
                continue;
            };
            let (mut sv, mut sh) = (0u32, 0u32);
            for e in 0..ne {
                loc[l * ne + e] = if in_vram[l * ne + e] {
                    sv += 1;
                    sv - 1
                } else {
                    sh += 1;
                    (sh - 1) | 0x8000_0000
                };
            }
            nv += sv as usize;
            nh += sh as usize;
            let mut set = [(None, None); 3];
            for (i, (s, x)) in [("ffn_gate_exps", gate), ("ffn_up_exps", up), ("ffn_down_exps", down)].into_iter().enumerate() {
                let t = g.tensor(&format!("blk.{l}.{s}.weight"))?;
                let tm = Instant::now();
                let lay = gpu_layout(t.ty, &read_tensor(&file, t)?);
                let eb = x.wpe * 4;
                let t1 = tm.elapsed().as_secs_f64();
                for (vr, blob, bufs, n) in [(true, &mut vblob, &mut vbufs, sv), (false, &mut hblob, &mut hbufs, sh)] {
                    if n == 0 {
                        continue;
                    }
                    if blob.data.len() + n as usize * eb + 256 > arena {
                        flush(blob, bufs, !vr)?;
                    }
                    let at = blob.data.len().div_ceil(256) * 256;
                    blob.data.resize(at, 0);
                    for e in (0..ne).filter(|&e| in_vram[l * ne + e] == vr) {
                        blob.data.extend_from_slice(&lay[e * eb..(e + 1) * eb]);
                    }
                    let reg = Reg { buf: blob.buf, off: at as u64, size: ((n as usize * eb) as u64).div_ceil(16) * 16 };
                    if vr { set[i].0 = Some(reg) } else { set[i].1 = Some(reg) }
                }
                if std::env::var_os("PELICA_DEBUG").is_some() {
                    eprintln!("[pelica] layer {l} {s}: read+layout {t1:.2}s, packing and uploads {:.2}s", tm.elapsed().as_secs_f64() - t1);
                }
            }
            eregs.push(Some(set));
            eprint!("\r[pelica] uploading experts of layer {}/{} ({:.2} GB in VRAM)   ", l + 1, cfg.n_layer, vram.get() as f64 / 1e9);
            LOAD_PROGRESS.store((l + 1) * 100 / cfg.n_layer, std::sync::atomic::Ordering::Relaxed);
        }
        flush(&mut vblob, &mut vbufs, false)?;
        flush(&mut hblob, &mut hbufs, true)?;
        drop((vblob, hblob));
        let loc_buf = upload("expert places", bytes_u32(&loc), ro)?;
        let adapt = if nh > 0 && std::env::var("PELICA_ADAPT").map_or(true, |v| v != "0") {
            let regs = eregs
                .iter()
                .enumerate()
                .map(|(l, set)| {
                    let set = set.as_ref()?;
                    let Ffn::Moe { gate, up, down, .. } = &layers[l].ffn else { return None };
                    let wpe = [gate.wpe, up.wpe, down.wpe];
                    let mut out = [(Reg { buf: 0, off: 0, size: 0 }, Reg { buf: 0, off: 0, size: 0 }, 0u64); 3];
                    for i in 0..3 {
                        out[i] = (set[i].0?, set[i].1?, wpe[i] as u64 * 4);
                    }
                    Some(out)
                })
                .collect();
            let rbuf = || device.create_buffer(&wgpu::BufferDescriptor { label: Some("expert use readback"), size: usage.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            Some(Adapt {
                regs,
                vbufs: vbufs.clone(),
                hbufs: hbufs.clone(),
                loc_buf: loc_buf.clone(),
                tmp: make("expert swap", ADAPT_TMP, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)?,
                rb: [rbuf(), rbuf()],
                ready: Default::default(),
                st: std::sync::Mutex::new(AdaptState { loc: loc.clone(), score: vec![0.0; cfg.n_layer * ne], prev: vec![0; cfg.n_layer * ne], pending: None, turn: 0, moved: 0 }),
            })
        } else {
            None
        };
        queue.write_buffer(&usage, 0, &vec![0u8; usage.size() as usize]);

        // the cache positions after VRAM's, held by the CPU: RAM as far as
        // it goes (leaving 6 GB free), then a scratch file on the drive
        let rest = ctx - kv_vram;
        let mut kv_ram = if rest == 0 {
            0
        } else {
            match env_n("PELICA_KV_RAM") {
                Some(n) => n,
                None => (crate::available_ram() as usize).saturating_sub(6 << 30) / perpos,
            }
            .min(rest)
                / k::CHUNK
                * k::CHUNK
        };
        let mut ram = Vec::new();
        while kv_ram > 0 {
            match (0..cfg.n_layer).map(|_| Mem::zeroed(2 * kv_ram * rowb)).collect::<Option<Vec<_>>>() {
                Some(v) => {
                    ram = v;
                    break;
                }
                None => {
                    eprintln!("[pelica] no room for {:.1} GB of cache in RAM; trying half as much", (kv_ram * perpos) as f64 / 1e9);
                    kv_ram = kv_ram / 2 / k::CHUNK * k::CHUNK;
                }
            }
        }
        let d0 = kv_vram + kv_ram;
        let mut ctx_final = ctx;
        let mut drive = None;
        if d0 < ctx {
            // whole rounds from the first one holding drive positions,
            // leaving 5 GB free on the drive
            let dir = kv_dir();
            std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let r_first = d0 / k::RK;
            if let Some(free) = disk_free(&dir) {
                let end = (r_first * k::RK + (free as usize).saturating_sub(5 << 30) / perpos) / k::CHUNK * k::CHUNK;
                if end < ctx {
                    ctx_final = end.max(d0);
                    eprintln!("[pelica] {} has room for the cache up to position {ctx_final}: context {ctx} reduced", dir.display());
                }
            }
            if ctx_final > d0 {
                let (file, path) = scratch_file(&dir)?;
                // up to the last position's values, in the layout of `drive_off`
                let r_last = (ctx_final - 1) / k::RK;
                let size = (((r_last - r_first) * cfg.n_layer + cfg.n_layer - 1) * 2 + 1) * k::RK + ctx_final - r_last * k::RK;
                file.set_len((size * rowb) as u64).with_context(|| format!("sizing {}", path.display()))?;
                let map = memmap2::MmapRaw::map_raw(&file).with_context(|| format!("mapping {}", path.display()))?;
                drive = Some(Drive { _file: file, path, map, r_first });
            }
        }
        let prefetcher = drive.as_ref().map(|_| prefetch_thread());
        let host = HostKv { ram, drive };
        let kv_out = make("new cache rows", if kv_vram < ctx_final { (pt * cfg.n_layer * 2 * rowb) as u64 } else { 16 }, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
        // a decode pass's row past VRAM (keys, values), and per layer its
        // queries and that row on their way to the CPU
        let krow = make("new row", (2 * rowb) as u64, rw)?;
        let lreads = (0..cfg.n_layer)
            .map(|_| make("queries to the CPU", (qd * 4 + 2 * rowb) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST))
            .collect::<Result<Vec<_>>>()?;

        // every dispatch, recorded once
        let mut kern = Kernels::default();
        struct Op<'a> {
            pipe: usize,
            binds: Vec<(u32, wgpu::BufferBinding<'a>)>,
            pre: [u32; 3],
            dec: [u32; 3],
            /// prompt passes' tensor-core kernel (grid: `pre`; the vector
            /// kernel's: `pre_v`)
            tc: Option<usize>,
            pre_v: [u32; 3],
            /// decode passes' kernel (grid: `dec`)
            dv: Option<usize>,
            /// as `Disp::mode`
            mode: u8,
            args: Vec<u32>,
        }
        let mut ops: Vec<Op> = Vec::new();
        let w = |r: Reg| wgpu::BufferBinding { buffer: &wbufs[r.buf], offset: r.off, size: wgpu::BufferSize::new(r.size) };
        fn all(b: &wgpu::Buffer) -> wgpu::BufferBinding<'_> {
            b.as_entire_buffer_binding()
        }
        let (pt32, none) = (pt as u32, k::NONE);
        let eps = cfg.eps.to_bits();
        let d32 = d as u32;
        // per-token kernels: x workgroups, a row per token
        let tok = |x: usize| ([x as u32, pt32, 1], [x as u32, 1, 1]);
        let mv_bind = |m: &Mat| vec![(1, all(&st)), (2, all(&act)), (3, w(m.reg)), (5, all(&fw_buf))];
        // y = W x (+ bias) (+ y): x region and stride, y region and stride
        // decoding: kernels reading whole quant blocks (PELICA_DEC=0: the
        // prompt passes' vector kernels instead)
        let dec_on = std::env::var("PELICA_DEC").map_or(true, |v| v != "0");
        let matvec = |kern: &mut Kernels, m: &Mat, x: u32, xs: usize, y: u32, ys: usize, bias: Option<u32>, add: bool, head: bool| {
            let (pre, dec) = tok(m.rows.div_ceil(8));
            // tensor cores: tiles of 128 or (fewer rows) 64 rows by 64 tokens
            let (tcp, pre) = if tc && !head && m.cols % 32 == 0 {
                let (wm, tr) = if m.rows >= 2048 { (2, 128) } else { (1, 64) };
                (Some(kern.get("mm_tc", k::mm_tc(m.ty, None, false, 2, wm, 2))), [m.rows.div_ceil(tr) as u32, pt.div_ceil(64) as u32, 1])
            } else {
                (None, [pre[0], 1, 1])
            };
            Op { pipe: kern.get("matvec", k::matvec(m.ty)), binds: mv_bind(m), pre, dec, tc: tcp, pre_v: [dec[0], 1, 1], dv: dec_on.then(|| kern.get("matvec_dec", k::matvec_dec(m.ty, m.cols))), mode: 0, args: vec![m.rows as u32, m.cols as u32, x, xs as u32, y, ys as u32, 0, m.wpr as u32, bias.unwrap_or(none), add as u32, head as u32] }
        };
        // routed experts on tensor cores: 128 rows by 16 or 32 of an
        // expert's tokens (as many as it gets on average)
        let ewn = if pt * kx / ne.max(1) > 24 { 2 } else { 1 };
        let norm_k = kern.get("rmsnorm", k::rmsnorm());
        let norm = |n: usize, per: usize, x: u32, y: u32, ts: usize, rs: usize, wt: u32, _head: bool| {
            let (pre, dec) = tok(per);
            Op { pipe: norm_k, binds: vec![(1, all(&st)), (2, all(&act)), (5, all(&fw_buf))], pre, dec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![n as u32, x, y, ts as u32, rs as u32, wt, eps] }
        };
        let hpk = cfg.n_head / cfg.n_kv;
        let rope_k = kern.get("rope", k::rope());
        let home_k = kern.get("kv_home", k::kv_home());
        let stage_k = kern.get("kv_stage", k::kv_stage());
        // flash-decoding attention where the shape allows (PELICA_ATTN=0:
        // the kernel cpu_attn reproduces bit for bit)
        let fast = k::attn_dec_fast_ok(hd, hpk) && std::env::var("PELICA_ATTN").map_or(true, |v| v != "0");
        let dec_k = if fast { kern.get("attn_dec", k::attn_dec_fast(hd, hpk)) } else { kern.get("attn_dec", k::attn_dec(hd, hpk)) };
        let mdec_k = if fast { kern.get("merge_dec", k::merge_dec_fast(hd)) } else { kern.get("merge_dec", k::merge_dec(hd)) };
        let pre_k = kern.get("attn_pre", k::attn_pre(hd, hpk));
        let fold_k = kern.get("attn_fold", k::attn_fold(hd));
        let plan_k = kern.get("plan", k::plan());
        let comb_k = kern.get("combine", k::combine());
        let swiglu_k = kern.get("swiglu", k::swiglu());
        let scale_f = 1.0f32 / (hd as f32).sqrt();
        let scale = scale_f.to_bits();
        let eg = (ne.min(pt * kx) as u32, kx as u32);
        let emb_scale = g.f(&key("embedding_scale")).unwrap_or(1.0);
        ops.push(Op {
            pipe: kern.get("embed", k::embed(embd.ty)),
            binds: vec![(1, all(&st)), (2, all(&act)), (3, w(embd.reg))],
            pre: [(d / 4).div_ceil(64) as u32, pt32, 1],
            dec: [(d / 4).div_ceil(64) as u32, 1, 1],
            tc: None, pre_v: [0; 3], dv: None, mode: 0,
            args: vec![d32, rx, embd.wpr as u32, 0, emb_scale.to_bits()],
        });
        let mut plans = Vec::new();
        let (nv32, rw32, nkv32, nh32) = (kv_vram as u32, (kvd / 2) as u32, cfg.n_kv as u32, cfg.n_head as u32);
        // decode attention's chunks in VRAM
        let ncv = (kv_vram / k::DCHUNK) as u32;
        for (l, ly) in layers.iter().enumerate() {
            let (kc, vc) = &kv[l];
            let l0 = ops.len();
            ops.push(norm(d, 1, rx, rxn, d, 0, ly.attn_norm, false));
            ops.push(matvec(&mut kern, &ly.q, rxn, d, rq, qd, ly.bias[0], false, false));
            ops.push(matvec(&mut kern, &ly.k, rxn, d, rk, kvd, ly.bias[1], false, false));
            ops.push(matvec(&mut kern, &ly.v, rxn, d, rv, kvd, ly.bias[2], false, false));
            // decoding: norm and the three projections in one dispatch
            if dec_on && ly.bias[..3].iter().all(|b| b.is_none()) && ly.q.reg.buf == ly.k.reg.buf && ly.k.reg.buf == ly.v.reg.buf {
                let n = ops.len();
                for op in &mut ops[n - 4..] {
                    op.mode = 1;
                }
                let rows = qd + 2 * kvd;
                let whole = wgpu::BufferBinding { buffer: &wbufs[ly.q.reg.buf], offset: 0, size: None };
                let wo = |m: &Mat| (m.reg.off / 4) as u32;
                ops.push(Op {
                    pipe: kern.get("qkv_dec", k::qkv_dec(ly.q.ty, ly.k.ty, ly.v.ty, d)),
                    binds: vec![(2, all(&act)), (3, whole), (5, all(&fw_buf))],
                    pre: [0; 3],
                    dec: [rows.div_ceil(8) as u32, 1, 1],
                    tc: None,
                    pre_v: [0; 3],
                    dv: None,
                    mode: 2,
                    args: vec![qd as u32, kvd as u32, kvd as u32, d32, rx, ly.attn_norm, eps, rq, rk, rv, wo(&ly.q), wo(&ly.k), wo(&ly.v), ly.q.wpr as u32, ly.k.wpr as u32, ly.v.wpr as u32],
                });
            }
            let n_post = ops.len();
            for (nm, r, heads, width) in [(ly.q_norm, rq, cfg.n_head, qd), (ly.k_norm, rk, cfg.n_kv, kvd)] {
                if let Some((wt, n)) = nm {
                    if n == hd {
                        ops.push(norm(hd, heads, r, r, width, hd, wt, false));
                    } else if n == width {
                        ops.push(norm(width, 1, r, r, width, 0, wt, false));
                    } else {
                        bail!("{a}: layer {l}: q/k norm of {n} values");
                    }
                }
            }
            let (pre, dec) = tok(((cfg.n_head + cfg.n_kv) * cfg.rot / 2).div_ceil(256));
            ops.push(Op { pipe: rope_k, binds: vec![(1, all(&st)), (2, all(&act)), (6, all(&rope_buf))], pre, dec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![rq, qd as u32, rk, kvd as u32, nh32, nkv32, hd as u32, cfg.rot as u32, cfg.neox as u32, ctx.div_ceil(256) as u32] });
            let (spre, sdec) = tok((kvd / 2).div_ceil(256));
            ops.push(Op { pipe: home_k, binds: vec![(1, all(&st)), (2, all(&act)), (11, all(kc)), (12, all(vc)), (13, all(&krow))], pre: spre, dec: sdec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![rk, rv, kvd as u32, rw32, nv32] });
            // decoding: those (head norms, rope, the cache row) in one dispatch,
            // when the norms are per head
            let per_head = |nm: Option<(u32, usize)>| nm.is_none_or(|(_, n)| n == hd);
            if dec_on && hd <= 256 && cfg.rot <= hd && per_head(ly.q_norm) && per_head(ly.k_norm) && std::env::var("PELICA_QKFUSE").map_or(true, |v| v != "0") {
                for op in &mut ops[n_post..] {
                    op.mode = 1;
                }
                let w = |nm: Option<(u32, usize)>| nm.map_or(none, |(w, _)| w);
                ops.push(Op {
                    pipe: kern.get("qk_dec", k::qk_dec()),
                    binds: vec![(1, all(&st)), (2, all(&act)), (5, all(&fw_buf)), (6, all(&rope_buf)), (11, all(kc)), (12, all(vc)), (13, all(&krow))],
                    pre: [0; 3],
                    dec: [nh32 + 2 * nkv32, 1, 1],
                    tc: None, pre_v: [0; 3], dv: None,
                    mode: 2,
                    args: vec![rq, rk, rv, nh32, nkv32, hd as u32, cfg.rot as u32, cfg.neox as u32, ctx.div_ceil(256) as u32, w(ly.q_norm), w(ly.k_norm), eps, rw32, nv32],
                });
            }
            let before = l0..ops.len();
            let dec_res = ops.len();
            ops.push(Op { pipe: dec_k, binds: vec![(1, all(&st)), (2, all(&act)), (11, all(kc)), (12, all(vc)), (17, all(&part))], pre: [nkv32, ncv, 1], dec: [nkv32, ncv, 1], tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![rq, rw32, nv32, scale, nh32] });
            let mut rounds = Vec::new();
            for r in 0..ctx.div_ceil(k::RK) {
                let (r0, r1) = ((r * k::RK) as u32, ((r + 1) * k::RK) as u32);
                let stage_i = ops.len();
                ops.push(Op { pipe: stage_k, binds: vec![(1, all(&st)), (2, all(&act)), (15, all(&stage.0)), (16, all(&stage.1))], pre: spre, dec: sdec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![rk, rv, kvd as u32, rw32, r0, r1] });
                let pre_i = ops.len();
                let g = [pt.div_ceil(k::attn_tq(hpk)) as u32, nkv32, k::RSPL as u32];
                ops.push(Op {
                    pipe: pre_k,
                    binds: vec![(1, all(&st)), (2, all(&act)), (11, all(kc)), (12, all(vc)), (15, all(&stage.0)), (16, all(&stage.1)), (18, all(&ppart))],
                    pre: g,
                    dec: g,
                    tc: None, pre_v: [0; 3], dv: None, mode: 0,
                    args: vec![rq, qd as u32, rw32, nv32, r0, scale, nh32],
                });
                let fold_i = ops.len();
                ops.push(Op { pipe: fold_k, binds: vec![(1, all(&st)), (2, all(&act)), (18, all(&ppart))], pre: [nh32, pt32, 1], dec: [nh32, 1, 1], tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![ratt, qd as u32, racc, r0, (r == 0) as u32, nh32] });
                rounds.push(RoundPlan { stage: stage_i, pre: pre_i, fold: fold_i });
            }
            let merge = ops.len();
            ops.push(Op { pipe: mdec_k, binds: vec![(1, all(&st)), (2, all(&act)), (17, all(&part))], pre: [nh32, 1, 1], dec: [nh32, 1, 1], tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![ratt, nh32] });
            let a0 = ops.len();
            ops.push(matvec(&mut kern, &ly.o, ratt, qd, rx, d, ly.bias[3], true, false));
            ops.push(norm(d, 1, rx, rxn, d, 0, ly.ffn_norm, false));
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    let f = gate.rows;
                    ops.push(matvec(&mut kern, gate, rxn, d, rfg, f, None, false, false));
                    ops.push(matvec(&mut kern, up, rxn, d, rfu, f, None, false, false));
                    let (pre, dec) = tok(f.div_ceil(256));
                    ops.push(Op { pipe: swiglu_k, binds: vec![(1, all(&st)), (2, all(&act))], pre, dec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![f as u32, rfg, rfu, rfg] });
                    ops.push(matvec(&mut kern, down, rfg, f, rx, d, None, true, false));
                }
                Ffn::Moe { router, sel_b, gate, up, down, shared } => {
                    ops.push(matvec(&mut kern, router, rxn, d, rlg, ne, None, false, false));
                    ops.push(Op {
                        pipe: plan_k,
                        binds: vec![(1, all(&st)), (2, all(&act)), (5, all(&fw_buf)), (10, all(&usage))],
                        pre: [1, 1, 1],
                        dec: [1, 1, 1],
                        tc: None, pre_v: [0; 3], dv: None, mode: 0,
                        args: vec![rlg, ne as u32, kx as u32, cfg.norm_w as u32, cfg.w_scale.to_bits(), cfg.sigmoid as u32, sel_b.unwrap_or(none), l as u32, rwt, trace_at as u32],
                    });
                    let regs = eregs[l].unwrap();
                    let vb = |i: usize| match regs[i].0 {
                        Some(r) => wgpu::BufferBinding { buffer: &vbufs[r.buf], offset: r.off, size: wgpu::BufferSize::new(r.size) },
                        None => all(&dummy),
                    };
                    let hb = |i: usize| match regs[i].1 {
                        Some(r) => wgpu::BufferBinding { buffer: &hbufs[r.buf], offset: r.off, size: wgpu::BufferSize::new(r.size) },
                        None => all(&dummy),
                    };
                    let ff = cfg.ff_exp as u32;
                    let etc = tc && d % 32 == 0 && cfg.ff_exp % 32 == 0;
                    let ez = pt.div_ceil(16 * ewn) as u32;
                    ops.push(Op {
                        pipe: kern.get("expert", k::expert(gate.ty, Some(up.ty))),
                        binds: vec![(1, all(&st)), (2, all(&act)), (6, vb(0)), (7, vb(1)), (8, hb(0)), (9, hb(1)), (10, all(&loc_buf))],
                        pre: if etc { [ff.div_ceil(128), eg.0, ez] } else { [ff.div_ceil(8), eg.0, 1] },
                        dec: [ff.div_ceil(8 * k::exp_rps(true)), eg.1, 1],
                        tc: etc.then(|| kern.get("expert_tc", k::mm_tc(gate.ty, Some(up.ty), true, 1, 1, ewn))),
                        pre_v: [ff.div_ceil(8), eg.0, 1],
                        dv: dec_on.then(|| kern.get("expert_dec", k::expert_dec(gate.ty, Some(up.ty), d))),
                        mode: 0,
                        args: vec![ff, d32, rxn, d32, rh, l as u32, ne as u32, gate.wpe as u32, gate.wpr as u32, up.wpe as u32, up.wpr as u32, 0],
                    });
                    ops.push(Op {
                        pipe: kern.get("expert down", k::expert(down.ty, None)),
                        binds: vec![(1, all(&st)), (2, all(&act)), (6, vb(2)), (8, hb(2)), (10, all(&loc_buf))],
                        pre: if etc { [d32.div_ceil(128), eg.0, ez] } else { [d32.div_ceil(8), eg.0, 1] },
                        dec: [d32.div_ceil(8 * k::exp_rps(false)), eg.1, 1],
                        tc: etc.then(|| kern.get("expert_tc", k::mm_tc(down.ty, None, true, 1, 1, ewn))),
                        pre_v: [d32.div_ceil(8), eg.0, 1],
                        dv: dec_on.then(|| kern.get("expert_dec down", k::expert_dec(down.ty, None, cfg.ff_exp))),
                        mode: 0,
                        args: vec![d32, ff, rh, ff, reo, l as u32, ne as u32, down.wpe as u32, down.wpr as u32, 0, 0, 1],
                    });
                    let (mut so, mut sl) = (none, none);
                    if let Some(s) = shared {
                        let f = s.gate.rows;
                        ops.push(matvec(&mut kern, &s.gate, rxn, d, rsg, f, None, false, false));
                        ops.push(matvec(&mut kern, &s.up, rxn, d, rsu, f, None, false, false));
                        let (pre, dec) = tok(f.div_ceil(256));
                        ops.push(Op { pipe: swiglu_k, binds: vec![(1, all(&st)), (2, all(&act))], pre, dec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![f as u32, rsg, rsu, rsg] });
                        ops.push(matvec(&mut kern, &s.down, rsg, f, rso, d, None, false, false));
                        so = rso;
                        if let Some(inp) = &s.inp {
                            ops.push(matvec(&mut kern, inp, rxn, d, rsl, 1, None, false, false));
                            sl = rsl;
                        }
                    }
                    let (pre, dec) = tok(d.div_ceil(256));
                    ops.push(Op { pipe: comb_k, binds: vec![(1, all(&st)), (2, all(&act))], pre, dec, tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![d32, rx, d32, reo, rwt, kx as u32, so, sl] });
                }
            }
            plans.push(LayerPlan { before, dec_res, rounds, merge, after: a0..ops.len() });
        }
        // the head: the last token's logits and argmax
        let h0 = ops.len();
        ops.push(norm(d, 1, rx, rxn, d, 0, out_norm, true));
        ops.push(matvec(&mut kern, &out, rxn, d, rlog, cfg.vocab, None, false, true));
        ops.push(Op { pipe: kern.get("argmax1", k::argmax1()), binds: vec![(2, all(&act))], pre: [k::ARGG as u32, 1, 1], dec: [k::ARGG as u32, 1, 1], tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![cfg.vocab as u32, rlog, rarg] });
        ops.push(Op { pipe: kern.get("argmax2", k::argmax2()), binds: vec![(1, all(&st)), (2, all(&act))], pre: [1, 1, 1], dec: [1, 1, 1], tc: None, pre_v: [0; 3], dv: None, mode: 0, args: vec![rarg] });

        let pipes: Vec<wgpu::ComputePipeline> = kern
            .srcs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("moe kernel {i}")), source: wgpu::ShaderSource::Wgsl(s.as_str().into()) });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&format!("moe kernel {i}")), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
            })
            .collect();
        let stride = 256usize;
        let mut pbytes = vec![0u8; ops.len() * stride];
        for (i, op) in ops.iter().enumerate() {
            assert!(op.args.len() <= 16);
            for (j, v) in op.args.iter().enumerate() {
                pbytes[i * stride + j * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let pbuf = upload("params", &pbytes, wgpu::BufferUsages::UNIFORM)?;
        let head = h0..ops.len();
        let mut disps = Vec::with_capacity(ops.len());
        for (i, op) in ops.into_iter().enumerate() {
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &pbuf, offset: (i * stride) as u64, size: wgpu::BufferSize::new(64) }) }];
            for (b, buf) in op.binds {
                entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::Buffer(buf) });
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipes[op.pipe].get_bind_group_layout(0), entries: &entries });
            let tcb = op.tc.map(|p| (p, device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipes[p].get_bind_group_layout(0), entries: &entries })));
            let dvb = op.dv.map(|p| (p, device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipes[p].get_bind_group_layout(0), entries: &entries })));
            disps.push(Disp { pipe: op.pipe, what: format!("{} {:?}", kern.names[op.pipe], op.args), bind, pre: op.pre, dec: op.dec, tc: tcb, pre_v: op.pre_v, dv: dvb, mode: op.mode });
        }
        let reads = (0..LAG + 2).map(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("token readback"), size: 256, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })).collect();
        let eos = stop_tokens(&g);
        drop((layers, wbufs, fw_buf, rope_buf, vbufs, hbufs, loc_buf, ppart, pbuf, dummy));
        eprintln!(
            "\r[pelica] {} ({a}) on {name}: {:.2} GB in VRAM; experts: {nv} in VRAM, {nh} in RAM ({:.2} GB); loaded in {:.1}s   ",
            path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
            vram.get() as f64 / 1e9,
            host_bytes as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
        if kv_vram < ctx_final {
            let gb = |n: usize| (n * perpos) as f64 / 1e9;
            let nd = ctx_final - d0;
            eprintln!(
                "[pelica] context {ctx_final}: cache positions 0..{kv_vram} in VRAM ({:.1} GB), {kv_ram} in RAM ({:.1} GB), {nd} on the drive ({:.1} GB{})",
                gb(kv_vram),
                gb(kv_ram),
                gb(nd),
                host.drive.as_ref().map(|d| format!(", {}", d.path.display())).unwrap_or_default()
            );
        }
        let mut cfg = cfg;
        cfg.ctx = ctx_final;
        let embed = 0;
        Ok(Engine { cfg, name, max_ctx, vram_experts: nv, host_experts: nh, host_bytes, eos, device, queue, pipes, disps, embed, plans, head, kv_vram, kv_ram, rowb, prefetcher, host, krow, lreads, kv, part, rq, scale: scale_f, stage, kv_out, st, act, usage, reads, usage_path, usage_base, adapt, trace_at })
    }

    /// The dispatches of a pass whose keys all sit in VRAM, in order:
    /// decoding, or prompt passes with `rounds` rounds.
    fn plain_order(&self, decode: bool, rounds: usize, head: bool) -> Vec<usize> {
        let mut v = vec![self.embed];
        for lp in &self.plans {
            v.extend(lp.before.clone());
            if decode {
                v.extend([lp.dec_res, lp.merge]);
            } else {
                for rp in &lp.rounds[..rounds] {
                    v.extend([rp.pre, rp.fold]);
                }
            }
            v.extend(lp.after.clone());
        }
        if head {
            v.extend(self.head.clone());
        }
        v
    }

    /// Byte offset in the drive file of row `p` (of round `r`) of layer
    /// `l`'s keys (half 0) or values (half 1).
    fn drive_off(&self, d: &Drive, l: usize, r: usize, half: usize, p: usize) -> u64 {
        ((((r - d.r_first) * self.cfg.n_layer + l) * 2 + half) * k::RK + (p - r * k::RK)) as u64 * self.rowb as u64
    }

    /// Where the CPU keeps layer `l`'s keys (half 0) or values (half 1) of
    /// positions [a, b) past VRAM, which lie in one tier and (on the drive)
    /// one round.
    fn rows(&self, l: usize, half: usize, a: usize, b: usize) -> *mut u8 {
        let (nv, d0) = (self.kv_vram, self.kv_vram + self.kv_ram);
        assert!(nv <= a && a < b && (b <= d0 || (a >= d0 && (b - 1) / k::RK == a / k::RK)), "cache rows {a}..{b} span tiers");
        if b <= d0 {
            // SAFETY: inside the layer's allocation of 2 * kv_ram rows
            unsafe { self.host.ram[l].ptr.add((half * self.kv_ram + a - nv) * self.rowb) }
        } else {
            let d = self.host.drive.as_ref().expect("cache rows past the drive file");
            // SAFETY: the file is sized to hold every position of the context
            unsafe { d.map.as_mut_ptr().add(self.drive_off(d, l, a / k::RK, half, a) as usize) }
        }
    }

    /// Positions [a, b) past VRAM split at the RAM/drive border and rounds.
    fn pieces(&self, a: usize, b: usize) -> Vec<(usize, usize)> {
        let d0 = self.kv_vram + self.kv_ram;
        let mut v = Vec::new();
        let mut x = a;
        while x < b {
            let y = if x < d0 { b.min(d0) } else { b.min((x / k::RK + 1) * k::RK) };
            v.push((x, y));
            x = y;
        }
        v
    }

    /// Layer `l`'s new rows from position `a` past VRAM: `n` rows of keys
    /// then `n` of values from `src`, to where the CPU keeps them.
    fn put_rows(&self, l: usize, a: usize, n: usize, src: &[u8]) {
        let nb = n * self.rowb;
        for half in 0..2 {
            for (x, y) in self.pieces(a, a + n) {
                let from = &src[half * nb + (x - a) * self.rowb..][..(y - x) * self.rowb];
                // SAFETY: `rows` is in bounds; nothing reads it meanwhile
                unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), self.rows(l, half, x, y), from.len()) };
            }
        }
    }

    /// Records and submits a pass of `t` tokens from position `c` (decoding
    /// if `decode`: the token is already in place), with the head if `head`,
    /// then copies the state's first words to readback buffer `read`.
    /// Decode attention past VRAM runs on the CPU (`cpu_layer`); prompt
    /// passes stage the rows the CPU holds a round at a time between
    /// submissions, and the pass's new rows past VRAM go to the CPU before
    /// this returns.
    fn pass(&self, c: usize, t: usize, decode: bool, head: bool, read: Option<usize>) -> Result<wgpu::SubmissionIndex> {
        let (nv, end, rk, rowb) = (self.kv_vram, c + t, k::RK, self.rowb);
        let mut rec = Rec::new(self, decode, t);
        // the pass's new rows past VRAM: (layer, first position, count, offset in kv_out)
        let mut persist: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut out_at = 0u64;
        let mut keep = |rec: &mut Rec, l: usize, r: usize, a: usize, b: usize| {
            let n = ((b - a) * rowb) as u64;
            let so = ((a - r * rk) * rowb) as u64;
            rec.copy(&self.stage.0, so, &self.kv_out, out_at, n);
            rec.copy(&self.stage.1, so, &self.kv_out, out_at + n, n);
            persist.push((l, a, b - a, out_at));
            out_at += 2 * n;
        };
        rec.disp(self.embed);
        for (l, lp) in self.plans.iter().enumerate() {
            for i in lp.before.clone() {
                rec.disp(i);
            }
            if decode {
                if c >= nv {
                    self.cpu_layer(&mut rec, l, c)?;
                } else {
                    rec.disp(lp.dec_res);
                }
                rec.disp(lp.merge);
            } else {
                self.prefetch((l + 1) % self.plans.len(), nv, c);
                for r in 0..=(end - 1) / rk {
                    let (r0, r1) = (r * rk, (r + 1) * rk);
                    let rp = &lp.rounds[r];
                    if r1 > nv {
                        // rows the CPU holds, before the pass's
                        let (a, b) = (r0.max(nv), r1.min(c));
                        if a < b {
                            rec.submit();
                            self.fill(l, r, a, b);
                        }
                        rec.disp(rp.stage);
                        let (a, b) = (r0.max(nv).max(c), r1.min(end));
                        if a < b {
                            keep(&mut rec, l, r, a, b);
                        }
                    }
                    rec.disp(rp.pre);
                    rec.disp(rp.fold);
                }
            }
            for i in lp.after.clone() {
                rec.disp(i);
            }
            // prompt passes go to the GPU a layer at a time: one submission
            // running for seconds gets the device reset by Windows
            if !decode {
                rec.submit();
            }
        }
        if head {
            for i in self.head.clone() {
                rec.disp(i);
            }
        }
        if let Some(slot) = read {
            rec.copy(&self.st, 0, &self.reads[slot], 0, 16);
        }
        let idx = rec.submit();
        if !persist.is_empty() {
            let slice = self.kv_out.slice(..out_at);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.wait(idx.clone())?;
            {
                let v = slice.get_mapped_range().context("mapping new cache rows")?;
                for &(l, a, n, off) in &persist {
                    self.put_rows(l, a, n, &v[off as usize..off as usize + 2 * n * rowb]);
                }
            }
            self.kv_out.unmap();
        }
        Ok(idx)
    }

    /// Decode attention of layer `l` at position `c`, past VRAM: the GPU
    /// sends the queries and the new row to the CPU, which keeps the row and
    /// computes the partials of the chunks past VRAM (`cpu_attn`, a thread
    /// per chunk) while the GPU does VRAM's; they go back to the GPU before
    /// the next submission.
    fn cpu_layer(&self, rec: &mut Rec, l: usize, c: usize) -> Result<()> {
        use rayon::prelude::*;
        let (hd, nh, rowb, nv) = (self.cfg.hd, self.cfg.n_head, self.rowb, self.kv_vram);
        let qb = nh * hd * 4;
        let buf = &self.lreads[l];
        rec.copy(&self.act, self.rq as u64 * 4, buf, 0, qb as u64);
        rec.copy(&self.krow, 0, buf, qb as u64, 2 * rowb as u64);
        let sent = rec.submit();
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        rec.disp(self.plans[l].dec_res);
        rec.submit();
        self.wait(sent)?;
        let q: Vec<f32> = {
            let v = buf.slice(..).get_mapped_range().context("mapping queries")?;
            self.put_rows(l, c, 1, &v[qb..]);
            v[..qb].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) * self.scale).collect()
        };
        buf.unmap();
        // the next layer's drive rows on their way in while this one runs
        self.prefetch((l + 1) % self.cfg.n_layer, nv, c + 1);
        let sh = crate::cpu_attn::Shape { hd, n_kv: self.cfg.n_kv, hpk: nh / self.cfg.n_kv };
        let (per, hpk, n_kv) = (nh * (hd + 2), sh.hpk, sh.n_kv);
        let c0 = nv / k::CHUNK;
        let nc = c / k::CHUNK + 1 - c0;
        let mut out = vec![0f32; nc * per];
        // tasks: (chunk, kv heads), split by heads too when chunks are few,
        // so every thread gets work
        let ng = (2 * rayon::current_num_threads()).div_ceil(nc).clamp(1, n_kv);
        let mut tasks = Vec::with_capacity(nc * ng);
        for (i, o) in out.chunks_mut(per).enumerate() {
            let mut rest = o;
            for gi in 0..ng {
                let kv = gi * n_kv / ng..(gi + 1) * n_kv / ng;
                let (mine, more) = rest.split_at_mut(kv.len() * hpk * (hd + 2));
                rest = more;
                tasks.push((c0 + i, kv, mine));
            }
        }
        tasks.into_par_iter().for_each(|(ch, kv, o)| {
            let a = ch * k::CHUNK;
            let b = (a + k::CHUNK).min(c + 1);
            let n = (b - a) * rowb / 2;
            // SAFETY: rows in bounds, written only between attentions
            let (ks, vs) = unsafe { (std::slice::from_raw_parts(self.rows(l, 0, a, b) as *const u16, n), std::slice::from_raw_parts(self.rows(l, 1, a, b) as *const u16, n)) };
            crate::cpu_attn::chunk(&sh, kv.clone(), ks, vs, &q[kv.start * hpk * hd..kv.end * hpk * hd], o);
        });
        // into the partials' DCHUNK slots: each chunk in the first of its
        // slots, the others empty (no weight in the merge)
        let r = k::CHUNK / k::DCHUNK;
        let mut slots = vec![0f32; nc * r * per];
        for (i, o) in out.chunks(per).enumerate() {
            for s in 0..r {
                let dst = &mut slots[(i * r + s) * per..(i * r + s + 1) * per];
                if s == 0 {
                    dst.copy_from_slice(o);
                } else {
                    for h in dst.chunks_mut(hd + 2) {
                        h[0] = -3.0e38;
                    }
                }
            }
        }
        self.queue.write_buffer(&self.part, (c0 * r * per * 4) as u64, bytes_f32(&slots));
        Ok(())
    }

    /// Asks the OS to read layer `l`'s drive rows of positions [a, b) into
    /// memory, if they aren't (on the prefetch thread, without waiting).
    fn prefetch(&self, l: usize, a: usize, b: usize) {
        let (Some(tx), d0) = (&self.prefetcher, self.kv_vram + self.kv_ram) else { return };
        let a = a.max(d0);
        if a >= b {
            return;
        }
        let ranges: Vec<(usize, usize)> = self.pieces(a, b).into_iter().flat_map(|(x, y)| (0..2).map(move |h| (h, x, y))).map(|(h, x, y)| (self.rows(l, h, x, y) as usize, (y - x) * self.rowb)).collect();
        let _ = tx.send(ranges);
    }

    /// Stages layer `l`'s rows [a, b) of round `r`, which the CPU holds
    /// (written to the GPU before the next submission).
    fn fill(&self, l: usize, r: usize, a: usize, b: usize) {
        for (x, y) in self.pieces(a, b) {
            let at = ((x - r * k::RK) * self.rowb) as u64;
            for (half, sb) in [(0, &self.stage.0), (1, &self.stage.1)] {
                // SAFETY: rows in bounds, not written during a pass
                let src = unsafe { std::slice::from_raw_parts(self.rows(l, half, x, y) as *const u8, (y - x) * self.rowb) };
                self.queue.write_buffer(sb, at, src);
            }
        }
    }

    /// GPU time per kernel of a pass of `t` tokens at position `pos` (prompt
    /// pass if t > 1), averaged over `steps` passes. The cache holds whatever
    /// it holds: only timing matters. Needs timestamp queries.
    pub fn profile(&self, pos: usize, t: usize, steps: usize, real: Option<&[u32]>) -> Result<String> {
        let f = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        if !self.device.features().contains(f) {
            bail!("this GPU/driver does not offer timestamp queries inside passes");
        }
        let t = t.clamp(1, k::PT);
        // real tokens (one position per step, so real routing), else made-up ones
        let toks: Vec<u32> = match real {
            Some(r) => r.to_vec(),
            None => (0..(pos + t + 1) as u32).map(|i| (i * 7919 + 13) % (self.cfg.vocab as u32 - 300) + 200).collect(),
        };
        self.queue.write_buffer(&self.st, (k::HIST * 4) as u64, bytes_u32(&toks));
        let list: Vec<&Disp> = self.plain_order(t == 1, (pos + t - 1) / k::RK + 1, true).into_iter().map(|i| &self.disps[i]).filter(|d| d.pick(t == 1, t).is_some()).collect();
        let n = list.len() + 1;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: n as u32 });
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let mb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let period = self.queue.get_timestamp_period() as f64;
        let mut by: Vec<(String, f64)> = Vec::new();
        let mut total = 0f64;
        for step in 0..steps {
            let pos = if real.is_some() { (pos + step).min(toks.len() - t - 1) } else { pos };
            self.queue.write_buffer(&self.st, 0, bytes_u32(&[pos as u32, t as u32]));
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.write_timestamp(&qs, 0);
                for (q, d) in list.iter().enumerate() {
                    let Some((p, b, g)) = d.pick(t == 1, t) else { continue };
                    pass.set_pipeline(&self.pipes[p]);
                    pass.set_bind_group(0, b, &[]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                    pass.write_timestamp(&qs, q as u32 + 1);
                }
            }
            enc.resolve_query_set(&qs, 0..n as u32, &rb, 0);
            enc.copy_buffer_to_buffer(&rb, 0, &mb, 0, n as u64 * 8);
            let idx = self.queue.submit([enc.finish()]);
            mb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.wait(idx)?;
            {
                let v = mb.slice(..).get_mapped_range().context("mapping timestamps")?;
                let ts: Vec<u64> = v.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                for (i, d) in list.iter().enumerate() {
                    let dt = ts[i + 1].saturating_sub(ts[i]) as f64 * period / 1e6;
                    let mut name = d.what.split(" [").next().unwrap_or("").to_string();
                    if t >= TC_MIN && d.tc.is_some() {
                        name += " (tc)";
                    } else if t == 1 && d.dv.is_some() {
                        name += " (dec)";
                    }
                    if std::env::var_os("PELICA_PROF_SHAPES").is_some() {
                        // (rows x cols, from the arguments)
                        let a: Vec<&str> = d.what.split(['[', ',', ']']).map(|x| x.trim()).collect();
                        name += &format!(" {}x{}", a.get(1).unwrap_or(&""), a.get(2).unwrap_or(&""));
                    }
                    match by.iter_mut().find(|(n, _)| *n == name) {
                        Some(e) => e.1 += dt,
                        None => by.push((name, dt)),
                    }
                    total += dt;
                }
            }
            mb.unmap();
        }
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        let st = steps as f64;
        let mut out = format!("GPU time per pass of {t} token(s) at position {pos} (avg of {steps}): {:.3} ms ({:.0} tok/s)\n", total / st, t as f64 * 1e3 / (total / st));
        for (name, x) in by {
            out += &format!("  {:12} {:8.3} ms  {:4.1}%\n", name, x / st, x / total * 100.0);
        }
        Ok(out)
    }

    /// Debug: the dispatches of a pass, and the activations after a prompt
    /// pass of `toks` (at most PT, from position 0) through the first `upto`.
    pub fn dump(&self, toks: &[u32], upto: usize) -> Result<(Vec<String>, Vec<f32>)> {
        self.queue.write_buffer(&self.st, (k::HIST * 4) as u64, bytes_u32(toks));
        self.queue.write_buffer(&self.st, 0, bytes_u32(&[0, toks.len() as u32]));
        let order = self.plain_order(false, 1, true);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            for d in order.iter().map(|&i| &self.disps[i]).take(upto) {
                let Some((p, b, g)) = d.pick(false, toks.len()) else { continue };
                pass.set_pipeline(&self.pipes[p]);
                pass.set_bind_group(0, b, &[]);
                pass.dispatch_workgroups(g[0], g[1], g[2]);
            }
        }
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: self.act.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        enc.copy_buffer_to_buffer(&self.act, 0, &rb, 0, self.act.size());
        let idx = self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait(idx)?;
        let v = rb.slice(..).get_mapped_range().context("mapping")?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok((order.iter().map(|&i| self.disps[i].what.clone()).collect(), v))
    }

    /// Debug (PELICA_ATTN_CHECK): layer `l`'s decode attention partials at
    /// position `pos` (the cache in VRAM up to it), from the GPU and from
    /// `cpu_attn`, compared bit for bit.
    pub fn check_attn(&self, pos: usize, l: usize) -> Result<String> {
        let (hd, nh, rowb) = (self.cfg.hd, self.cfg.n_head, self.rowb);
        let sh = crate::cpu_attn::Shape { hd, n_kv: self.cfg.n_kv, hpk: nh / self.cfg.n_kv };
        if pos >= self.kv_vram {
            bail!("position {pos} is not in VRAM");
        }
        let nch = pos / k::DCHUNK + 1;
        self.queue.write_buffer(&self.st, 0, bytes_u32(&[pos as u32, 1]));
        let mut rec = Rec::new(self, true, 1);
        rec.disp(self.embed);
        for (i, lp) in self.plans[..=l].iter().enumerate() {
            for d in lp.before.clone() {
                rec.disp(d);
            }
            rec.disp(lp.dec_res);
            if i < l {
                rec.disp(lp.merge);
                for d in lp.after.clone() {
                    rec.disp(d);
                }
            }
        }
        let (qb, pb, kb) = ((nh * hd * 4) as u64, (nch * nh * (hd + 2) * 4) as u64, ((pos + 1) * rowb) as u64);
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: qb + pb + 2 * kb, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        rec.copy(&self.act, self.rq as u64 * 4, &rb, 0, qb);
        rec.copy(&self.part, 0, &rb, qb, pb);
        rec.copy(&self.kv[l].0, 0, &rb, qb + pb, kb);
        rec.copy(&self.kv[l].1, 0, &rb, qb + pb + kb, kb);
        let idx = rec.submit();
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait(idx)?;
        let v = rb.slice(..).get_mapped_range().context("mapping")?;
        let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let u16s = |b: &[u8]| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect() };
        let q: Vec<f32> = f32s(&v[..qb as usize]).iter().map(|x| x * self.scale).collect();
        let gpu = f32s(&v[qb as usize..(qb + pb) as usize]);
        let kc = u16s(&v[(qb + pb) as usize..(qb + pb + kb) as usize]);
        let vc = u16s(&v[(qb + pb + kb) as usize..]);
        drop(v);
        rb.unmap();
        let (rl, per) = (sh.row(), nh * (hd + 2));
        let mut cpu = vec![0f32; nch * per];
        let mut scalar = vec![0f32; nch * per];
        for c in 0..nch {
            let (a, b) = (c * k::DCHUNK, (c * k::DCHUNK + k::DCHUNK).min(pos + 1));
            crate::cpu_attn::chunk(&sh, 0..sh.n_kv, &kc[a * rl..b * rl], &vc[a * rl..b * rl], &q, &mut cpu[c * per..(c + 1) * per]);
            crate::cpu_attn::chunk_scalar(&sh, 0..sh.n_kv, &kc[a * rl..b * rl], &vc[a * rl..b * rl], &q, &mut scalar[c * per..(c + 1) * per]);
        }
        let diff = |x: &[f32]| x.iter().zip(&gpu).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| ((a - b) / b.abs().max(1e-30)).abs()).fold(0f32, f32::max);
        let mut out = format!("layer {l} at {pos}: {} partial values; differing bits: cpu {} (worst rel {worst:.2e}), scalar {}", gpu.len(), diff(&cpu), diff(&scalar));
        for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate().filter(|(_, (a, b))| a.to_bits() != b.to_bits()).take(8) {
            let (c, h, x) = (i / per, i % per / (hd + 2), i % (hd + 2));
            out += &format!("\n  chunk {c} head {h} value {x}: cpu {a:e} ({:#x}) gpu {b:e} ({:#x})", a.to_bits(), b.to_bits());
        }
        Ok(out)
    }

    /// Debug (PELICA_KV_DUMP): layer `l`'s cached keys and values of
    /// positions 0..n, wherever they live.
    pub fn dump_kv(&self, l: usize, n: usize) -> Result<Vec<u8>> {
        let (rowb, nv) = (self.rowb, self.kv_vram);
        let mut out = Vec::new();
        for half in 0..2 {
            let m = n.min(nv);
            let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * rowb) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut rec = Rec::new(self, true, 1);
            rec.copy(if half == 0 { &self.kv[l].0 } else { &self.kv[l].1 }, 0, &rb, 0, (m * rowb) as u64);
            let idx = rec.submit();
            rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.wait(idx)?;
            out.extend_from_slice(&rb.slice(..).get_mapped_range().context("mapping")?);
            rb.unmap();
            if n > nv {
                for (x, y) in self.pieces(nv, n) {
                    // SAFETY: rows in bounds, no pass running
                    out.extend_from_slice(unsafe { std::slice::from_raw_parts(self.rows(l, half, x, y) as *const u8, (y - x) * rowb) });
                }
            }
        }
        Ok(out)
    }

    fn wait(&self, idx: wgpu::SubmissionIndex) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
        Ok(())
    }

    /// Whether expert `e` of layer `l` sits in host RAM now.
    pub fn in_host(&self, l: usize, e: usize) -> bool {
        let ne = self.expert_grid().1;
        self.adapt.as_ref().is_some_and(|a| a.st.lock().unwrap().loc[l * ne + e] >> 31 != 0)
    }

    /// Experts moved between VRAM and host RAM since loading.
    pub fn experts_moved(&self) -> usize {
        self.adapt.as_ref().map_or(0, |a| a.st.lock().unwrap().moved)
    }

    /// One round of expert placement (see `Adapt`): takes in the counters
    /// asked for last time, if they have arrived, moves up to `budget` bytes
    /// of experts, and asks for the counters again. Never waits on the GPU.
    fn adapt(&self, budget: u64) -> Result<()> {
        use std::sync::atomic::Ordering;
        let Some(a) = &self.adapt else { return Ok(()) };
        let mut s = a.st.lock().unwrap();
        if let Some(i) = s.pending {
            if !a.ready[i].load(Ordering::Acquire) {
                return Ok(());
            }
            let now: Vec<u32> = a.rb[i].slice(..).get_mapped_range().context("mapping expert use")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            a.rb[i].unmap();
            s.pending = None;
            let st = &mut *s;
            for ((sc, p), &x) in st.score.iter_mut().zip(st.prev.iter_mut()).zip(&now) {
                *sc = *sc * ADAPT_DECAY + x.wrapping_sub(*p) as f32;
                *p = x;
            }
            self.swap_round(a, &mut s, budget);
        }
        let i = s.turn % 2;
        s.turn += 1;
        a.ready[i].store(false, Ordering::Release);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("expert use") });
        enc.copy_buffer_to_buffer(&self.usage, 0, &a.rb[i], 0, self.usage.size());
        self.queue.submit([enc.finish()]);
        let flag = a.ready[i].clone();
        a.rb[i].slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::Release));
        s.pending = Some(i);
        Ok(())
    }

    /// Per layer, the host experts used more than the least used VRAM ones
    /// change places with them, the biggest gains first, up to `budget`
    /// bytes moved.
    fn swap_round(&self, a: &Adapt, s: &mut AdaptState, budget: u64) {
        let ne = self.expert_grid().1;
        let mut pairs: Vec<(f32, usize, usize, usize)> = Vec::new();
        for (l, r) in a.regs.iter().enumerate() {
            if r.is_none() {
                continue;
            }
            let sc = |e: &usize| s.score[l * ne + e];
            let (mut host, mut vram): (Vec<usize>, Vec<usize>) = (0..ne).partition(|&e| s.loc[l * ne + e] >> 31 != 0);
            host.sort_by(|x, y| sc(y).total_cmp(&sc(x)));
            vram.sort_by(|x, y| sc(x).total_cmp(&sc(y)));
            for (&h, &v) in host.iter().zip(&vram) {
                let (sh, sv) = (sc(&h), sc(&v));
                // (a margin, so experts don't go back and forth)
                if sh < sv * 1.25 + 1.5 {
                    break;
                }
                pairs.push((sh - sv, l, h, v));
            }
        }
        if pairs.is_empty() {
            return;
        }
        pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        let ebytes = |l: usize| a.regs[l].as_ref().map_or(0, |r| r.iter().map(|x| x.2).sum::<u64>());
        let mut moved = 0u64;
        let mut chosen = Vec::new();
        for &(_, l, h, v) in &pairs {
            if moved + 2 * ebytes(l) > budget {
                break;
            }
            moved += 2 * ebytes(l);
            chosen.push((l, h, v));
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("expert swap") });
        // in batches that fit `tmp`: VRAM experts out to it, host experts
        // into their slots, then `tmp` out to the host slots
        let mut i = 0;
        while i < chosen.len() {
            let mut batch = Vec::new();
            let mut at = 0u64;
            while i < chosen.len() && at + ebytes(chosen[i].0) <= a.tmp.size() {
                let (l, h, v) = chosen[i];
                let (hs, vs) = ((s.loc[l * ne + h] & 0x7fff_ffff) as u64, s.loc[l * ne + v] as u64);
                for &(vr, hr, eb) in a.regs[l].as_ref().unwrap() {
                    batch.push((vr, vr.off + vs * eb, hr, hr.off + hs * eb, eb, at));
                    at += eb;
                }
                i += 1;
            }
            if batch.is_empty() {
                break;
            }
            for &(vr, vo, _, _, eb, t) in &batch {
                enc.copy_buffer_to_buffer(&a.vbufs[vr.buf], vo, &a.tmp, t, eb);
            }
            for &(vr, vo, hr, ho, eb, _) in &batch {
                enc.copy_buffer_to_buffer(&a.hbufs[hr.buf], ho, &a.vbufs[vr.buf], vo, eb);
            }
            for &(_, _, hr, ho, eb, t) in &batch {
                enc.copy_buffer_to_buffer(&a.tmp, t, &a.hbufs[hr.buf], ho, eb);
            }
        }
        for &(l, h, v) in &chosen[..i] {
            s.loc.swap(l * ne + h, l * ne + v);
        }
        s.moved += i;
        // (applied at the start of this submission, before the copies; the
        // passes queued before it still read the old places)
        self.queue.write_buffer(&a.loc_buf, 0, bytes_u32(&s.loc));
        self.queue.submit([enc.finish()]);
    }

    pub fn generate(&self, prompt: &[u32], n: usize, reuse: usize, on_token: impl FnMut(u32) -> bool) -> Result<Stats> {
        self.generate_with(prompt, n, reuse, on_token, |_, _| true)
    }

    /// Greedy decoding of up to `n` tokens after `prompt`, whose first
    /// `reuse` positions are still in the cache. Prompt passes report
    /// progress to `on_prefill` (done, total); if it returns false, stops
    /// with nothing generated and `Stats::prompt` the tokens now cached.
    pub fn generate_with(&self, prompt: &[u32], n: usize, reuse: usize, mut on_token: impl FnMut(u32) -> bool, mut on_prefill: impl FnMut(usize, usize) -> bool) -> Result<Stats> {
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        if prompt.len() + n + LAG > self.cfg.ctx {
            bail!("prompt ({}) + tokens ({n}) exceed the context ({})", prompt.len(), self.cfg.ctx);
        }
        self.queue.write_buffer(&self.st, (k::HIST * 4) as u64, bytes_u32(prompt));
        let t0 = Instant::now();
        let mut c = reuse.min(prompt.len() - 1);
        let mut prev: Option<wgpu::SubmissionIndex> = None;
        let mut slot = 0usize;
        let mut pending: VecDeque<(usize, wgpu::SubmissionIndex)> = VecDeque::new();
        while c < prompt.len() {
            let t = (prompt.len() - c).min(k::PT);
            let last = c + t == prompt.len();
            self.queue.write_buffer(&self.st, 0, bytes_u32(&[c as u32, t as u32]));
            let idx = self.pass(c, t, false, last, last.then_some(slot))?;
            c += t;
            if last {
                pending.push_back((slot, idx));
                slot = (slot + 1) % self.reads.len();
            } else {
                // one pass in flight while the next is recorded
                if let Some(p) = prev.replace(idx) {
                    self.wait(p)?;
                }
                if !on_prefill(c, prompt.len()) {
                    if let Some(p) = prev {
                        self.wait(p)?;
                    }
                    return Ok(Stats { prompt: c, generated: 0, prefill_s: t0.elapsed().as_secs_f64(), decode_s: 0.0, passes: 0, log: Vec::new() });
                }
            }
        }
        let mut produced = 0usize;
        let mut stop = false;
        let mut t_first = None;
        let mut take = |slot: usize, idx: wgpu::SubmissionIndex, stop: &mut bool, produced: &mut usize| -> Result<()> {
            let buf = &self.reads[slot];
            buf.slice(..16).map_async(wgpu::MapMode::Read, |_| {});
            self.wait(idx)?;
            let tok = {
                let v = buf.slice(..16).get_mapped_range().context("mapping token readback")?;
                u32::from_le_bytes(v[8..12].try_into().unwrap())
            };
            buf.unmap();
            if *stop {
                return Ok(());
            }
            if t_first.is_none() {
                t_first = Some(Instant::now());
            }
            *produced += 1;
            if !on_token(tok) || *produced >= n {
                *stop = true;
            }
            Ok(())
        };
        // decode passes: each the token the last one produced, which the GPU
        // already put in place, so they go back to back
        let mut recorded = 1usize;
        // expert places: counters asked for after the prompt, the first round
        // (which moves the most) when they're in
        self.adapt(ADAPT_FIRST)?;
        let mut first_round = true;
        loop {
            while pending.len() > LAG || (stop && !pending.is_empty()) || (recorded >= n && !pending.is_empty()) {
                let (sl, ix) = pending.pop_front().unwrap();
                take(sl, ix, &mut stop, &mut produced)?;
            }
            if stop || recorded >= n {
                break;
            }
            if recorded % ADAPT_EVERY == 0 {
                let was = self.adapt.as_ref().map(|a| a.st.lock().unwrap().turn);
                self.adapt(if first_round { ADAPT_FIRST } else { ADAPT_BYTES })?;
                // (the first round is over once a read came back and was used)
                if self.adapt.as_ref().map(|a| a.st.lock().unwrap().turn) != was {
                    first_round = false;
                }
            }
            let pos = prompt.len() + recorded - 1;
            pending.push_back((slot, self.pass(pos, 1, true, true, Some(slot))?));
            slot = (slot + 1) % self.reads.len();
            recorded += 1;
        }
        drop(take);
        let end = Instant::now();
        let first = t_first.unwrap_or(end);
        Ok(Stats { prompt: prompt.len(), generated: produced, prefill_s: (first - t0).as_secs_f64(), decode_s: (end - first).as_secs_f64(), passes: produced.saturating_sub(1), log: Vec::new() })
    }

    /// The experts each token of `toks` goes to in every layer, with their
    /// weights: [token][layer][(expert, weight)]. Runs prompt passes from
    /// position 0, so the cache no longer holds what it held before.
    pub fn trace(&self, toks: &[u32]) -> Result<Vec<Vec<Vec<(u32, f32)>>>> {
        if toks.is_empty() {
            bail!("nothing to trace");
        }
        if toks.len() + LAG > self.cfg.ctx {
            bail!("{} tokens are more than the context holds", toks.len());
        }
        let (nl, kx) = (self.cfg.n_layer, self.cfg.top_k);
        let bytes = (nl * k::PT * kx * 2 * 4) as u64;
        self.queue.write_buffer(&self.st, (k::HIST * 4) as u64, bytes_u32(toks));
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("routes"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut out = Vec::with_capacity(toks.len());
        let mut c = 0;
        while c < toks.len() {
            let t = (toks.len() - c).min(k::PT);
            self.queue.write_buffer(&self.st, 0, bytes_u32(&[c as u32, t as u32]));
            self.pass(c, t, false, false, None)?;
            let mut rec = Rec::new(self, true, 1);
            rec.copy(&self.st, (self.trace_at * 4) as u64, &rb, 0, bytes);
            let idx = rec.submit();
            rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.wait(idx)?;
            {
                let v = rb.slice(..).get_mapped_range().context("mapping the routes")?;
                let w: Vec<u32> = v.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
                for i in 0..t {
                    out.push((0..nl).map(|l| (0..kx).map(|j| {
                        let b = ((l * k::PT + i) * kx + j) * 2;
                        (w[b], f32::from_bits(w[b + 1]))
                    }).collect()).collect());
                }
            }
            rb.unmap();
            c += t;
        }
        Ok(out)
    }

    /// How much of `prompt` the cache still holds after a generation that
    /// ran `cached` through the model.
    pub fn reusable(&self, cached: &[u32], prompt: &[u32]) -> usize {
        cached.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1))
    }

    /// Adds this session's expert use to the counts next to the model (the
    /// next load puts the hottest experts in VRAM).
    pub fn save_usage(&self) -> Result<()> {
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: self.usage.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.usage, 0, &rb, 0, self.usage.size());
        let idx = self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait(idx)?;
        let now: Vec<u32> = rb.slice(..).get_mapped_range().context("mapping")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let sum: Vec<u32> = self.usage_base.iter().zip(&now).map(|(a, b)| a.saturating_add(*b)).collect();
        std::fs::write(&self.usage_path, bytes_u32(&sum)).with_context(|| format!("writing {}", self.usage_path.display()))
    }

    /// The experts picked since the last call, as (layer, expert, times)
    /// triples, flattened; `prev` keeps the counters as last read (the first
    /// call, with `prev` empty, only fills it).
    pub fn activity(&self, prev: &mut Vec<u32>) -> Result<Vec<u32>> {
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: self.usage.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.usage, 0, &rb, 0, self.usage.size());
        let idx = self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.wait(idx)?;
        let now: Vec<u32> = rb.slice(..).get_mapped_range().context("mapping")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let ne = now.len() / self.cfg.n_layer;
        let mut out = Vec::new();
        if prev.len() == now.len() {
            for (i, (a, b)) in now.iter().zip(prev.iter()).enumerate() {
                if a > b {
                    out.extend_from_slice(&[(i / ne) as u32, (i % ne) as u32, a - b]);
                }
            }
        }
        *prev = now;
        Ok(out)
    }

    /// (layers, routed experts per layer)
    pub fn expert_grid(&self) -> (usize, usize) {
        (self.cfg.n_layer, self.usage.size() as usize / 4 / self.cfg.n_layer)
    }
}

/// Records dispatches and copies into command buffers, submitting them when
/// asked.
struct Rec<'a> {
    e: &'a Engine,
    decode: bool,
    /// the pass's tokens
    t: usize,
    enc: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl<'a> Rec<'a> {
    fn new(e: &'a Engine, decode: bool, t: usize) -> Self {
        Rec { e, decode, t, enc: e.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pass") }), pass: None }
    }

    fn disp(&mut self, i: usize) {
        let Some((p, b, g)) = self.e.disps[i].pick(self.decode, self.t) else { return };
        let pass = self.pass.get_or_insert_with(|| self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime());
        pass.set_pipeline(&self.e.pipes[p]);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(g[0], g[1], g[2]);
    }

    fn copy(&mut self, src: &wgpu::Buffer, so: u64, dst: &wgpu::Buffer, dof: u64, n: u64) {
        self.pass = None;
        self.enc.copy_buffer_to_buffer(src, so, dst, dof, n);
    }

    fn submit(&mut self) -> wgpu::SubmissionIndex {
        self.pass = None;
        let enc = std::mem::replace(&mut self.enc, self.e.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pass") }));
        self.e.queue.submit([enc.finish()])
    }
}

/// `pelica -m model.gguf ...` for a MoE model: load, generate, report.
pub fn run(path: &Path, prompt: &str, ids: Option<&[u32]>, raw: bool, n: usize, ctx: usize, teacher: bool, ignore_eos: bool) -> Result<()> {
    use std::io::Write;
    let g = Gguf::open(path)?;
    let tok = crate::gguf::tokenizer(&g)?;
    let template = crate::chat_template::ChatTemplate::from_gguf(&g);
    drop(g);
    let eng = Engine::load(path, ctx)?;
    let prompt_ids: Vec<u32> = match ids {
        Some(v) => v.to_vec(),
        None => {
            let text = match (&template, raw) {
                (Some(t), false) => t.render(&[serde_json::json!({"role": "user", "content": prompt})], &Default::default())?,
                _ => prompt.to_string(),
            };
            tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec()
        }
    };
    if std::env::var_os("PELICA_PROF").is_some() {
        let ev = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        if std::env::var_os("PELICA_PROF_REAL").is_some() {
            // after a real generation: its tokens, so routing is real (and wall time to compare)
            let mut ids = prompt_ids.clone();
            let mut before = Vec::new();
            eng.activity(&mut before)?;
            let st = eng.generate(&prompt_ids, n, 0, |t| (ids.push(t), true).1)?;
            let picks = eng.activity(&mut before)?;
            if let Some(f) = std::env::var_os("PELICA_PICKS") {
                std::fs::write(f, bytes_u32(&picks))?;
            }
            let (mut all, mut host) = (0u64, 0u64);
            for x in picks.chunks_exact(3) {
                all += x[2] as u64;
                if eng.in_host(x[0] as usize, x[1] as usize) {
                    host += x[2] as u64;
                }
            }
            println!("generated {} tokens: {:.1} tok/s wall; expert uses {all}, from host RAM now {host} ({:.1}%); experts moved {}", st.generated, st.passes as f64 / st.decode_s.max(1e-9), host as f64 * 100.0 / all.max(1) as f64, eng.experts_moved());
            let steps = ids.len() - prompt_ids.len() - 1;
            print!("{}", eng.profile(prompt_ids.len(), 1, steps, Some(&ids))?);
            return Ok(());
        }
        print!("{}", eng.profile(ev("PELICA_PROF_POS", 100), ev("PELICA_PROF_BATCH", 1), ev("PELICA_PROF_STEPS", 10), None)?);
        return Ok(());
    }
    if let Ok(spec) = std::env::var("PELICA_DUMP") {
        // PELICA_DUMP=ops,file: activations after the first `ops` dispatches
        let (n_ops, file) = spec.split_once(',').context("PELICA_DUMP=ops,file")?;
        let (names, v) = eng.dump(&prompt_ids[..prompt_ids.len().min(k::PT)], n_ops.parse()?)?;
        for (i, w) in names.iter().enumerate() {
            println!("{i} {w}");
        }
        std::fs::write(file, bytes_f32(&v))?;
        println!("prompt_ids {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        return Ok(());
    }
    if let Some(n) = std::env::var("PELICA_ATTN_CHECK").ok().and_then(|v| v.parse::<usize>().ok()) {
        // after n generated tokens (the last one's pass), every layer
        eng.generate(&prompt_ids, n.max(1), 0, |_| true)?;
        for l in 0..eng.cfg.n_layer {
            println!("{}", eng.check_attn(prompt_ids.len() + n.max(1) - 1, l)?);
        }
        return Ok(());
    }
    if let Some(k0) = std::env::var("PELICA_FORCED").ok().and_then(|v| v.parse::<usize>().ok()) {
        // the argmax after each prefix of the ids from k0 on (teacher forcing),
        // one position per pass on top of the cache
        let mut forced = Vec::new();
        for i in k0.max(1)..=prompt_ids.len() {
            let mut t = 0;
            eng.generate(&prompt_ids[..i], 1, if i == k0.max(1) { 0 } else { i - 1 }, |x| {
                t = x;
                false
            })?;
            forced.push(t);
        }
        println!("forced {}", forced.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        return Ok(());
    }
    if teacher {
        println!("prompt_ids {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    let mut out = std::io::stdout();
    let mut gen = Vec::new();
    let mut printed = 0usize;
    let stats = eng.generate(&prompt_ids, n, 0, |t| {
        gen.push(t);
        if let Ok(s) = tok.decode(&gen, false) {
            if s.len() > printed && s.is_char_boundary(printed) {
                let _ = write!(out, "{}", &s[printed..]);
                let _ = out.flush();
                printed = s.len();
            }
        }
        ignore_eos || !eng.eos.contains(&t)
    })?;
    println!();
    if teacher {
        println!("greedy {}", gen.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    if let Ok(f) = std::env::var("PELICA_KV_DUMP") {
        // the cache of a few layers through the last generated token's position
        let n = prompt_ids.len() + gen.len() - 1;
        for l in [0, 1, eng.cfg.n_layer / 2, eng.cfg.n_layer - 1] {
            std::fs::write(format!("{f}.{l}"), eng.dump_kv(l, n)?)?;
        }
    }
    let _ = eng.save_usage();
    let dec = stats.generated.saturating_sub(1);
    eprintln!(
        "[pelica] prefill {} tokens in {:.3}s ({:.1} tok/s); decode {dec} tokens in {:.2}s: {:.1} tok/s",
        stats.prompt,
        stats.prefill_s,
        stats.prompt as f64 / stats.prefill_s.max(1e-9),
        stats.decode_s,
        dec as f64 / stats.decode_s.max(1e-9),
    );
    Ok(())
}
