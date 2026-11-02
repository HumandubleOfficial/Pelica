//! Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`), for models far larger
//! than VRAM + RAM, built like the DeepSeek V4 engine (`ds4`): the dense
//! weights sit in VRAM; routed experts sit in VRAM (the hottest), in a RAM
//! cache kept by heat (`ds4_io`), or on the drive, read when picked.
//!
//! The math follows llama.cpp's `qwen4exp` graph:
//! - hyper-connections: 4 copies of the hidden state; per block a grouped
//!   RMS norm, a low-rank sigmoid gate folds them into the block's input
//!   (their gated mean) and learned weights spread its output back;
//! - token mixers: Gated DeltaNet linear attention (36 of 48 layers: a
//!   causal conv, then a 128x128 state per value head updated by the delta
//!   rule, a gated RMS norm), and full attention every 4th layer (24 query
//!   heads, 2 KV heads of 256, gated output) that attends, past 2048
//!   positions, only to the blocks of 4 positions a small indexer scores
//!   highest (Qwen Sparse Attention), plus the newest positions;
//! - an n-gram embedding (layer 1): hashes of the last 1-3 tokens pick 16
//!   rows of a 51-billion-value table, mixed in through a gate and a
//!   dilated causal conv;
//! - experts: softmax top 10 of 512 (weights renormalized), plus a shared
//!   expert with a sigmoid gate.
//!
//! The GPU does the products and the hyper-connection mixes (`ds4_gpu`),
//! the CPU the rest: the DeltaNet recurrence, attention, routing, the
//! n-gram embedding, and the experts that sit in RAM when only a few tokens
//! use them. Prompts run layer by layer over chunks of up to `CHUNK` tokens.

#[path = "q4x_state.rs"]
mod state;
#[path = "q4x_vision.rs"]
pub mod vision;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::ds4::f16_to_f32;
use crate::ds4_gpu::{Batch, Gpu, Mat, NONE};
use crate::ds4_io::{Layout, Pin, Store};
use crate::gguf::{gpu_layout, Gguf, Tensor, Ty};
use crate::oss::Stats;
use crate::q4x_cpu::{self as cpu, Q8};

/// tokens per GPU sub-batch of a prompt
const TB: usize = 256;
/// tokens per layer-by-layer prompt pass
const CHUNK: usize = 4096;
/// expert rows (tokens) per GPU expert batch
const EROWS: usize = 1024;
/// VRAM slots that experts from RAM pass through for prompt passes
const STAGE: usize = 16;
/// experts used by at most this many tokens of a small pass run on the CPU (from RAM)
const CPU_MAX: usize = 4;
/// passes of at most this many tokens are "small"
const SMALL: usize = 8;
/// experts used by at most this many tokens of a prompt pass run on the CPU
const PF_CPU_ROWS: usize = 32;
/// bytes per weight arena
const ARENA: u64 = 1 << 30;
/// RAM left to the OS and other apps when sizing the expert cache
const RAM_MARGIN: u64 = 7 << 29;
/// expert moves into VRAM after a prompt, and every `REBAL_EVERY` tokens
const REBAL_PROMPT: usize = 256;
const REBAL_STEP: usize = 8;
const REBAL_EVERY: usize = 4;

/// time per stage (PELICA_PROF=1 prints it after a run)
static PROF: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];
/// experts run in small passes: in VRAM, on the CPU (from RAM), of those read from the drive
static EXP_COUNT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
const PROF_NAMES: [&str; 16] = ["n-gram embedding", "GPU mix+projections", "DeltaNet layers (GPU)", "attention (CPU)", "GPU out+ffn mix", "routing", "CPU experts (+drive)", "VRAM experts (+CPU's)", "staged experts (+drive)", "head", "  CPU waiting for drive", "  VRAM experts: building", "  VRAM experts: GPU wait", "  staged: drive wait", "  staged: upload", "  staged: GPU run"];

fn tick(i: usize, t: &mut Instant) {
    let now = Instant::now();
    PROF[i].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
    *t = now;
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f32 = cpu_dot(x, x);
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter_mut().zip(w).for_each(|(v, w)| *v *= inv * w);
}

#[inline]
fn cpu_dot(a: &[f32], b: &[f32]) -> f32 {
    crate::quant::dot_f32(a, b)
}

/// f32 to f16 bits, rounding to nearest even (as the hardware converts).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 112;
    if e2 >= 31 {
        return sign | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        let mf = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = mf >> shift;
        let rem = mf & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// f16 bits to f32, many at once
fn h2f(src: &[u16], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: the features were detected; lengths checked per chunk
            unsafe { h2f_f16c(src, dst) };
            return;
        }
    }
    for (o, &h) in dst.iter_mut().zip(src) {
        *o = f16_to_f32(h);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn h2f_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    while i < n {
        dst[i] = f16_to_f32(src[i]);
        i += 1;
    }
}

/// The files of a model split in several (`name-00001-of-00003.gguf`), or
/// the one file.
pub fn split_paths(path: &Path) -> Vec<PathBuf> {
    let s = path.to_string_lossy();
    let re = regex::Regex::new(r"-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    if let Some(c) = re.captures(&s) {
        let n: usize = c[2].parse().unwrap_or(1);
        let stem = &s[..c.get(0).unwrap().start()];
        return (1..=n).map(|i| PathBuf::from(format!("{stem}-{i:05}-of-{n:05}.gguf"))).collect();
    }
    vec![path.to_path_buf()]
}

/// The GGUF files of one model, searched by tensor name.
struct Shards {
    gs: Vec<Gguf>,
    paths: Vec<PathBuf>,
}

impl Shards {
    fn open(path: &Path) -> Result<Shards> {
        let paths = split_paths(path);
        let mut gs = Vec::new();
        for p in &paths {
            gs.push(Gguf::open(p).with_context(|| format!("opening {} (one part of the model)", p.display()))?);
        }
        Ok(Shards { gs, paths })
    }

    fn find(&self, name: &str) -> Result<(usize, &Tensor)> {
        for (i, g) in self.gs.iter().enumerate() {
            if let Some(t) = g.tensors.get(name) {
                return Ok((i, t));
            }
        }
        bail!("tensor {name} not in the model's files")
    }

    fn has(&self, name: &str) -> bool {
        self.gs.iter().any(|g| g.tensors.contains_key(name))
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let (i, t) = self.find(name)?;
        let b = self.gs[i].bytes(t);
        Ok(match t.ty {
            Ty::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Ty::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Ty::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            t => bail!("{name}: expected a float tensor, got {}", t.name()),
        })
    }

    /// A tensor's bytes, with large reads (faster than faulting in the map).
    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let (i, t) = self.find(name)?;
        let f = std::fs::File::open(&self.paths[i])?;
        let mut buf = vec![0u8; t.bytes()];
        let mut done = 0usize;
        while done < buf.len() {
            let end = (done + (64 << 20)).min(buf.len());
            #[cfg(windows)]
            let n = std::os::windows::fs::FileExt::seek_read(&f, &mut buf[done..end], (t.offset + done) as u64)?;
            #[cfg(unix)]
            let n = std::os::unix::fs::FileExt::read_at(&f, &mut buf[done..end], (t.offset + done) as u64)?;
            if n == 0 {
                bail!("unexpected end of file reading {name}");
            }
            done += n;
        }
        Ok(buf)
    }
}

fn wpr(ty: Ty, cols: usize) -> u32 {
    (cols / ty.gpu_values() * ty.gpu_words().expect("decodable")) as u32
}

pub fn handles(g: &Gguf) -> bool {
    g.arch() == "qwen4exp"
}

/// The model's shape, from the GGUF metadata.
struct Hp {
    n_layer: usize,
    d: usize,
    hc: usize,
    lr: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    rot: usize,
    rope_base: f32,
    /// M-RoPE: rotary pairs taking the time, row and column positions
    /// (interleaved: pair i takes component i % 3 while under 3x its count)
    rope_sections: [usize; 3],
    /// DeltaNet: key head size, key heads, value heads, conv taps
    dk: usize,
    nkh: usize,
    nvh: usize,
    conv: usize,
    /// sparse attention: indexer heads and size, positions kept, block size
    ih: usize,
    ihd: usize,
    topk: usize,
    kpool: usize,
    ne: usize,
    k: usize,
    ff: usize,
    eps: f32,
    vocab: usize,
    ple: Option<PleHp>,
}

struct PleHp {
    layer: usize,
    heads: usize,
    dim: usize,
    ngram: usize,
    per_gram: usize,
    taps: usize,
    eos: u64,
    mult: Vec<u64>,
    off: Vec<u64>,
    size: Vec<u64>,
}

/// A hyper-connection mix's weights: grouped norm (f32 words in its arena),
/// low-rank gate, and (blocks) the weights that spread the output back.
struct HcW {
    arena: usize,
    norm: u32,
    down: Mat,
    up: Mat,
    inj: Option<Mat>,
    /// down and inject as one dispatch (one token)
    mvs: Option<Mvs>,
}

enum Mixer {
    Lin { qkv: Mat, z: Mat, beta: Mat, alpha: Mat, out: Mat, conv_w: u32, dt_w: u32, a_w: u32, norm_w: u32, idx: usize },
    Att { q: Mat, k: Mat, v: Mat, o: Mat, qn: Vec<f32>, kn: Vec<f32>, iq: Mat, ik: Mat, iqn: Vec<f32>, ikn: Vec<f32>, idx: usize },
}

struct Ple {
    key: Mat,
    value: Mat,
    nk: Vec<f32>,
    nq: Vec<f32>,
    nc: Vec<f32>,
    conv: Vec<f32>,
}

struct Layer {
    hca: HcW,
    hcf: HcW,
    mix: Mixer,
    /// one-token product tables: the mixer's projections, the FFN's router,
    /// shared expert and gate (and the next layer's router when it shares
    /// the arena), each stream mix's down and inject
    mvs_mix: Option<Mvs>,
    mvs_ffn: Option<Mvs>,
    ffn_has_next: bool,
    ple: Option<Ple>,
    router: Mat,
    sgu: Mat,
    sd: Mat,
    sgate: Mat,
}

/// Word offsets of the activation regions (prompt sub-batch sized).
struct R {
    res: u32,
    xn: u32,
    lo: u32,
    gate: u32,
    mix: u32,
    inj: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    o: u32,
    y: u32,
    rl: u32,
    rl2: u32,
    cv: u32,
    /// DeltaNet states (per linear layer: heads x 128 x 128) and conv
    /// histories (3 rows of channels), and their prompt-end copies
    gst: u32,
    gcv: u32,
    sgst: u32,
    sgcv: u32,
    sgu: u32,
    sh: u32,
    sy: u32,
    sg: u32,
    emb: u32,
    pk: u32,
    pv: u32,
    xe: u32,
    egu: u32,
    eh: u32,
    ew: u32,
    ey: u32,
    logits: u32,
    arg: u32,
    /// the DeltaNet states before a drafted pass, to go back to when the
    /// model turns a draft down
    dgst: u32,
    dgcv: u32,
    /// a prompt pass's FFN inputs, the sum of its GPU experts' outputs, and
    /// a batch's rows' token indices
    fx: u32,
    fy: u32,
    eidx: u32,
}

/// An attention layer's cache (f16): keys, values, the indexer's raw keys
/// per position and its pooled keys per block of `kpool`.
#[derive(Clone, Default)]
struct Att {
    k: Vec<u16>,
    v: Vec<u16>,
    ik: Vec<u16>,
    pk: Vec<u16>,
}

/// The recurrent parts at the end of a prompt, to carry on from there.
struct Snap {
    pos: usize,
    ple: Vec<f32>,
}

struct State {
    att: Vec<Att>,
    /// the n-gram embedding's conv history (last (taps-1)*ngram inputs)
    ple: Vec<f32>,
    /// every token in the cache
    toks: Vec<u32>,
    pos: usize,
    snap: Option<Snap>,
    trace: Option<Vec<Vec<Vec<(u32, f32)>>>>,
    since_rebal: usize,
    /// the furthest position of this cache saved to disk (or loaded from it)
    saved_pos: usize,
    /// a sampled draft pass's logits, per position
    logits: Vec<f32>,
    /// with pictures in the cache: the rotary positions of this pass's
    /// positions from the first (and the blocks it completes)
    rp: Option<(usize, Vec<[u32; 3]>)>,
}

/// How expert bytes sit in one layer: piece types and words per row in the
/// GPU layout.
#[derive(Clone, Copy)]
struct Ex {
    ty: [Ty; 3],
    wpr_g: u32,
    wpr_d: u32,
}

struct Vram {
    slots: Vec<(usize, u32)>,
    /// each slot's size class, and each layer's: a layer's experts go only
    /// in slots of its class
    cls: Vec<u8>,
    lcls: Vec<u8>,
    key: Vec<u32>,
    slot_of: std::collections::HashMap<u32, usize>,
    moved: usize,
}

pub struct Engine {
    pub cfg: crate::moe::Cfg,
    pub name: String,
    pub max_ctx: usize,
    pub vram_experts: usize,
    pub host_experts: usize,
    pub host_bytes: usize,
    pub eos: Vec<u32>,
    hp: Hp,
    gpu: Gpu,
    sh: Shards,
    layers: Vec<Layer>,
    head: Mat,
    hc_head: HcW,
    embd: (usize, Tensor),
    ple_tab: Option<(usize, Tensor)>,
    store: Store,
    ex: Vec<Ex>,
    vram: Mutex<Vram>,
    stage: Vec<(usize, u32)>,
    st: Mutex<State>,
    r: R,
    n_lin: usize,
    usage: Vec<AtomicU32>,
    usage_base: Vec<u32>,
    usage_path: PathBuf,
    samp: Mutex<(f32, f32, u64)>,
    /// saved prompt states (`state`), and the chat template's
    /// `<|im_start|>` and `system` tokens
    state_dir: PathBuf,
    chat_marks: Option<(u32, u32)>,
    /// the attention cache's RAM per position, and the positions it has
    /// room for: set aside for KV_START at load, then taken from the expert
    /// cache as a chat grows past it
    kv_per_pos: usize,
    kv_room: AtomicUsize,
    /// pictures (`vision`): the encoder file, the pictures encoded so far by
    /// id, and the token the n-gram embedding reads at their positions
    mmproj: Option<PathBuf>,
    images: Mutex<Vec<(u32, std::sync::Arc<vision::Image>)>>,
    img_tok: u32,
    /// other computers lending their RAM and CPU (`pelica bridge`)
    remote: Option<std::sync::Arc<crate::bridge::Remote>>,
}

// --- GPU kernels of the hyper-connection mixes (see `Batch::kernel`) ---

/// Grouped RMS norm: row r (of n values) times weight row (r % groups).
/// args: 0 n, 1 in, 2 out, 3 groups, 4 weight word base, 5 eps, 6 rows
const HCNORM: &str = r#"
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x;
    if (row >= arg(6u)) { return; }
    let n = arg(0u);
    let xi = arg(1u) + row * n;
    let xo = arg(2u) + row * n;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = act[xi + i]; s += v * v; }
    s = subgroupAdd(s);
    if (li % 32u == 0u) { red[li / 32u] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red[j]; }
    let inv = 1.0 / sqrt(tot / f32(n) + argf(5u));
    let wb = arg(4u) + (row % arg(3u)) * n;
    for (var i = li; i < n; i += 256u) { act[xo + i] = act[xi + i] * inv * bitcast<f32>(w[wb + i]); }
}
"#;

/// y = silu(x * s). args: 0 n, 1 x, 2 y, 3 s
const SILUS: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arg(0u)) { return; }
    let x = act[arg(1u) + i] * argf(3u);
    act[arg(2u) + i] = x / (1.0 + exp(-x));
}
"#;

/// The gated mean of the streams: out[t][i] = mean_s xn[t][s][i] * sigmoid(gate[t][s][i]).
/// args: 0 d, 1 streams, 2 xn, 3 gate, 4 out, 5 T
const HCMIX: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let d = arg(0u);
    let h = arg(1u);
    if (i >= d || t >= arg(5u)) { return; }
    var s = 0.0;
    for (var k = 0u; k < h; k++) {
        let o = (t * h + k) * d + i;
        s += act[arg(2u) + o] / (1.0 + exp(-act[arg(3u) + o]));
    }
    act[arg(4u) + t * d + i] = s / f32(h);
}
"#;

/// The block output spread back: res[t][s][i] += y[t][i] * 2 sigmoid(inj[t][s] / streams).
/// args: 0 d, 1 streams, 2 res, 3 y, 4 inj, 5 T
const HCCOMB: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let ts = gid.y;
    let d = arg(0u);
    let h = arg(1u);
    if (i >= d || ts >= arg(5u) * h) { return; }
    let wt = 2.0 / (1.0 + exp(-act[arg(4u) + ts] / f32(h)));
    act[arg(2u) + ts * d + i] += act[arg(3u) + (ts / h) * d + i] * wt;
}
"#;

/// A one-token FFN's output added into the residuals on the GPU: the CPU's
/// part (shared expert and CPU experts) plus the VRAM experts' rows (already
/// weighted), times each stream's inject weight.
/// args: 0 d, 1 h, 2 res, 3 CPU part, 4 expert rows, 5 rows, 6 inject
const FFNCOMB: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let d = arg(0u);
    if (i >= d) { return; }
    var v = act[arg(3u) + i];
    for (var j = 0u; j < arg(5u); j++) { v += act[arg(4u) + j * d + i]; }
    let h = arg(1u);
    for (var s = 0u; s < h; s++) {
        let wt = 2.0 / (1.0 + exp(-act[arg(6u) + s] / f32(h)));
        act[arg(2u) + s * d + i] += v * wt;
    }
}
"#;

/// DeltaNet's causal conv (4 taps) over the last 3 inputs (state) and T new
/// rows, then silu; the state becomes the last 3 inputs. One thread per channel.
/// args: 0 channels, 1 T, 2 x, 3 out, 4 state, 5 weight word base ([channel][tap])
const CONV: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let ch = arg(0u);
    if (i >= ch) { return; }
    let wb = arg(5u) + i * 4u;
    let w0 = bitcast<f32>(w[wb]);
    let w1 = bitcast<f32>(w[wb + 1u]);
    let w2 = bitcast<f32>(w[wb + 2u]);
    let w3 = bitcast<f32>(w[wb + 3u]);
    let sb = arg(4u);
    var s0 = act[sb + i];
    var s1 = act[sb + ch + i];
    var s2 = act[sb + 2u * ch + i];
    for (var t = 0u; t < arg(1u); t++) {
        let xv = act[arg(2u) + t * ch + i];
        let a = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3;
        act[arg(3u) + t * ch + i] = a / (1.0 + exp(-a));
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    act[sb + i] = s0;
    act[sb + ch + i] = s1;
    act[sb + 2u * ch + i] = s2;
}
"#;

/// Per-head norms: row r (head r % heads of token r / heads) of n values,
/// L2 (mode 0: x / sqrt(sum x^2 + eps)) or RMS times a weight times
/// sigmoid(gate) (mode 1). One workgroup per row.
/// args: 0 n, 1 x, 2 x token stride, 3 heads, 4 rows, 5 eps, 6 mode,
///       7 weight word, 8 gate, 9 gate token stride, 10 out, 11 out token stride
const HEADNORM: &str = r#"
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x;
    if (row >= arg(4u)) { return; }
    let n = arg(0u);
    let t = row / arg(3u);
    let hh = row % arg(3u);
    let xi = arg(1u) + t * arg(2u) + hh * n;
    let xo = arg(10u) + t * arg(11u) + hh * n;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = act[xi + i]; s += v * v; }
    s = subgroupAdd(s);
    if (li % 32u == 0u) { red[li / 32u] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red[j]; }
    let rms = arg(6u) == 1u;
    var inv = 1.0 / sqrt(tot + argf(5u));
    if (rms) { inv = 1.0 / sqrt(tot / f32(n) + argf(5u)); }
    for (var i = li; i < n; i += 256u) {
        var v = act[xi + i] * inv;
        if (rms) {
            let g = act[arg(8u) + t * arg(9u) + hh * n + i];
            v = v * bitcast<f32>(w[arg(7u) + i]) / (1.0 + exp(-g));
        }
        act[xo + i] = v;
    }
}
"#;

/// The delta rule: each row j of a value head's state (column j of S) is
/// independent, so 4 lanes share a row (32 columns each) and a workgroup
/// takes 32 rows; step through the tokens: decay, delta from the key,
/// update, read with the query. Grid (value heads * 4).
/// args: 0 T, 1 conv out (q | k | v rows), 2 its row width, 3 alpha, 4 beta
///       (rows of value heads), 5 value heads, 6 key heads, 7 state base,
///       8 out (rows of heads * 128), 9 dt bias word, 10 A word, 11 scale
const GDN: &str = r#"
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x / 4u;
    let j = (wg.x % 4u) * 32u + li / 4u;
    let c0 = (li % 4u) * 32u;
    let T = arg(0u);
    let ch = arg(2u);
    let nvh = arg(5u);
    let nkh = arg(6u);
    let kh = h % nkh;
    let sb = arg(7u) + h * 16384u + j * 128u + c0;
    var m: array<f32, 32>;
    for (var i = 0u; i < 32u; i++) { m[i] = act[sb + i]; }
    let dt = bitcast<f32>(w[arg(9u) + h]);
    let a = bitcast<f32>(w[arg(10u) + h]);
    for (var t = 0u; t < T; t++) {
        let row = arg(1u) + t * ch;
        qs[li] = act[row + kh * 128u + li];
        ks[li] = act[row + (nkh + kh) * 128u + li];
        workgroupBarrier();
        let v = act[row + (2u * nkh + h) * 128u + j];
        let ar = act[arg(3u) + t * nvh + h] + dt;
        var sp = ar;
        if (ar <= 20.0) { sp = log(1.0 + exp(ar)); }
        let decay = exp(sp * a);
        let beta = 1.0 / (1.0 + exp(-act[arg(4u) + t * nvh + h]));
        var sk = 0.0;
        for (var i = 0u; i < 32u; i++) { m[i] = m[i] * decay; sk += m[i] * ks[c0 + i]; }
        sk += subgroupShuffleXor(sk, 1u);
        sk += subgroupShuffleXor(sk, 2u);
        let dl = (v - sk) * beta;
        var o = 0.0;
        for (var i = 0u; i < 32u; i++) { m[i] += dl * ks[c0 + i]; o += m[i] * qs[c0 + i]; }
        o += subgroupShuffleXor(o, 1u);
        o += subgroupShuffleXor(o, 2u);
        if (c0 == 0u) { act[arg(8u) + t * nvh * 128u + h * 128u + j] = o * argf(11u); }
        workgroupBarrier();
    }
    for (var i = 0u; i < 32u; i++) { act[sb + i] = m[i]; }
}
"#;

/// dst = src (mode 0) or 0 (mode 1) over n words. args: 0 n, 1 src, 2 dst, 3 mode
const FILL: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var i = gid.x; i < arg(0u); i += 1048576u) {
        if (arg(3u) == 1u) { act[arg(2u) + i] = 0.0; } else { act[arg(2u) + i] = act[arg(1u) + i]; }
    }
}
"#;

/// Rows by index: dst[j] = src[idx[j]] (d values each; idx as f32 bits).
/// args: 0 d, 1 src, 2 dst, 3 idx, 4 rows. Grid (d/256, rows).
const GATHER: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let d = arg(0u);
    if (i >= d || j >= arg(4u)) { return; }
    let t = bitcast<u32>(act[arg(3u) + j]);
    act[arg(2u) + j * d + i] = act[arg(1u) + t * d + i];
}
"#;

/// Rows added back by index, in order (rows of one token never race):
/// dst[idx[j]] += src[j]. args: 0 d, 1 dst, 2 src, 3 idx, 4 rows. Grid (d/256).
const SCATTER: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let d = arg(0u);
    if (i >= d) { return; }
    for (var j = 0u; j < arg(4u); j++) {
        let t = bitcast<u32>(act[arg(3u) + j]);
        act[arg(1u) + t * d + i] += act[arg(2u) + j * d + i];
    }
}
"#;

/// A picture's token: the flag, the picture's id, its row (`add_image`).
const IMG: u32 = 1 << 31;
const IMG_ROW_BITS: u32 = 11;

/// positions whose attention cache is set aside at load; a longer chat takes
/// its memory from the expert cache, KV_STEP positions at a time
const KV_START: usize = 16384;
const KV_STEP: usize = 4096;

/// threads copying a staged batch's experts to the GPU
const UPLOADERS: usize = 3;

/// Several one-token products in one dispatch (each with its own input
/// and output): a workgroup per 8 rows of one of them, picked from a table
/// in the weight arena (8 words per product: workgroup end, rows, cols,
/// output, weight base, words per row, type index, input). Fewer dispatches
/// in a decode step: each costs the GPU a barrier and wgpu a few us.
/// args: 0 table word, 1 products
fn mvs_src(tys: &[Ty]) -> String {
    let mut dq = String::new();
    let mut body = String::new();
    for (i, &ty) in tys.iter().enumerate() {
        dq += &crate::moe_kernels::deq4(ty, &format!("dq{i}"), "w");
        let (gpb, bw) = (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"));
        body += &format!("            if (ty == {i}u) {{ for (var j = lane; j < n; j += 32u) {{ let g = t0 + j; a += dot(dq{i}(rb + (g / {gpb}u) * {bw}u, g % {gpb}u), xt[j]); }} }}\n");
    }
    format!(
        r#"{dq}
var<workgroup> xt: array<vec4<f32>, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let tb = arg(0u);
    var s = 0u;
    var start = 0u;
    for (var i = 0u; i < arg(1u); i++) {{
        let e = w[tb + 8u * i];
        if (wg.x >= e) {{ s = i + 1u; start = e; }}
    }}
    let o = tb + 8u * s;
    let rows = w[o + 1u];
    let G = w[o + 2u] / 4u;
    let ty = w[o + 6u];
    let xb = w[o + 7u];
    let row0 = (wg.x - start) * 8u + li / 32u;
    let valid = row0 < rows;
    let row = min(row0, rows - 1u);
    let lane = li % 32u;
    let rb = w[o + 4u] + row * w[o + 5u];
    var a = 0.0;
    for (var t0 = 0u; t0 < G; t0 += 512u) {{
        let n = min(512u, G - t0);
        for (var i = li; i < n; i += 256u) {{ xt[i] = x4(xb + 4u * (t0 + i)); }}
        workgroupBarrier();
{body}        workgroupBarrier();
    }}
    let sum = subgroupAdd(a);
    if (lane == 0u && valid) {{ act[w[o + 3u] + row] = sum; }}
}}
"#
    )
}

/// A table of one-token products for \`mvs_src\`, built at load.
struct Mvs {
    buf: usize,
    tab: u32,
    n: u32,
    groups: u32,
    tys: Vec<Ty>,
    key: String,
}

impl Mvs {
    fn run(&self, b: &mut Batch) {
        b.kernel(&self.key, || mvs_src(&self.tys), self.buf, &[self.tab, self.n], [self.groups, 1, 1]);
    }
}

/// The FFN's inputs: its mix, router logits, the gated shared expert (where
/// the routed experts' outputs are added), the spreading weights, and the
/// next layer's router logits (read-ahead, small passes).
/// A one-token FFN whose sum is left for the next layer's batch to add in
/// on the GPU (one wait fewer per layer): the CPU's part and the VRAM
/// experts' rows in r.ey (their batch submitted, not waited on).
struct Pend {
    y: Vec<f32>,
    rows: u32,
}

struct FfnIn {
    f: Vec<f32>,
    lg: Vec<f32>,
    y: Vec<f32>,
    inj: Vec<f32>,
    ahead: Option<Vec<f32>>,
}

impl Engine {
    pub fn load(path: &Path, ctx: usize) -> Result<Engine> {
        let t0 = Instant::now();
        let progress = |p: usize| crate::moe::LOAD_PROGRESS.store(p.min(100), Ordering::Relaxed);
        progress(0);
        if crate::repack::unfinished(path) {
            bail!("speeding this model up (pelica repack) didn't finish: run it again to finish it, from Models or with `pelica repack`");
        }
        let sh = Shards::open(path)?;
        let g = &sh.gs[0];
        if !handles(g) {
            bail!("not a Qwen3.8 (qwen4exp) model");
        }
        let key = |s: &str| format!("qwen4exp.{s}");
        let ints = |s: &str| -> Result<Vec<u64>> { Ok(g.ints(&key(s))?.into_iter().map(|v| v as u64).collect()) };
        let n_layer = g.u(&key("block_count"))?;
        let ple = match g.ints(&key("ple.layers")) {
            Ok(v) if !v.is_empty() => {
                let ngram = g.u(&key("ple.ngram_size"))?;
                let per_gram = g.u(&key("ple.heads_per_ngram"))?;
                let heads = (ngram - 1) * per_gram;
                let p = PleHp {
                    layer: v[0] as usize,
                    heads,
                    dim: g.u(&key("embedding_length_per_layer_input"))?,
                    ngram,
                    per_gram,
                    taps: g.u(&key("ple.conv_kernel"))?,
                    eos: g.u(&key("ple.eos_token_id"))? as u64,
                    mult: ints("ple.layer_multipliers")?,
                    off: ints("ple.head_offsets")?,
                    size: ints("ple.head_vocab_sizes")?,
                };
                if v.len() != 1 || p.mult.len() < ngram || p.off.len() < heads || p.size.len() < heads || p.size.iter().any(|&s| s == 0) {
                    bail!("unsupported n-gram embedding settings");
                }
                Some(p)
            }
            _ => None,
        };
        let ratios = ints("attention.compress_ratios")?;
        let kpool = ratios.iter().copied().find(|&r| r != 0).unwrap_or(0) as usize;
        let hp = Hp {
            n_layer,
            d: g.u(&key("embedding_length"))?,
            hc: g.u(&key("hyper_connection.count"))?,
            lr: g.u(&key("hyper_connection.low_rank"))?,
            nh: g.u(&key("attention.head_count"))?,
            nkv: g.u(&key("attention.head_count_kv"))?,
            hd: g.u(&key("attention.key_length"))?,
            rot: g.u(&key("rope.dimension_count"))?,
            rope_base: g.f(&key("rope.freq_base"))?,
            rope_sections: match g.ints(&key("rope.dimension_sections")) {
                Ok(v) if v.len() >= 3 => [v[0] as usize, v[1] as usize, v[2] as usize],
                _ => [usize::MAX / 4, 0, 0],
            },
            dk: g.u(&key("ssm.state_size"))?,
            nkh: g.u(&key("ssm.group_count"))?,
            nvh: g.u(&key("ssm.time_step_rank"))?,
            conv: g.u(&key("ssm.conv_kernel"))?,
            ih: g.u(&key("attention.indexer.head_count"))?,
            ihd: g.u(&key("attention.indexer.key_length"))?,
            topk: g.u(&key("attention.indexer.top_k"))?,
            kpool,
            ne: g.u(&key("expert_count"))?,
            k: g.u(&key("expert_used_count"))?,
            ff: g.u(&key("expert_feed_forward_length"))?,
            eps: g.f(&key("attention.layer_norm_rms_epsilon"))?,
            vocab: sh.find("token_embd.weight")?.1.dims[1],
            ple,
        };
        let interval = g.u(&key("full_attention_interval")).unwrap_or(4);
        let is_att = |l: usize| (l + 1) % interval == 0;
        if hp.hc < 2 || hp.d % 256 != 0 || hp.k > 16 || hp.nvh % hp.nkh != 0 || hp.nh % hp.nkv != 0 || hp.rot > hp.hd || hp.rot > hp.ihd || kpool < 2 || hp.topk % kpool != 0 || g.u(&key("ssm.inner_size"))? != hp.nvh * hp.dk {
            bail!("unsupported Qwen3.8 shape");
        }
        if g.u(&key("expert_shared_feed_forward_length")).unwrap_or(hp.ff) != hp.ff {
            bail!("a shared expert of another size than the routed ones is not supported");
        }
        let max_ctx = g.u(&key("context_length")).unwrap_or(262_144);
        let ctx = ctx.clamp(1024, max_ctx);
        let (d, ff, hcd) = (hp.d, hp.ff, hp.hc * hp.d);
        let qkv_w = 2 * hp.nkh * hp.dk + hp.nvh * hp.dk;
        if hp.dk != 128 || hp.conv != 4 {
            bail!("unsupported DeltaNet shape (head size {}, conv {})", hp.dk, hp.conv);
        }
        let n_lin = (0..hp.n_layer).filter(|&l| !is_att(l)).count();

        // activations (words 0..512: the IQ3_S grid)
        let mut words = 512usize;
        let mut region = |n: usize| -> u32 {
            let at = words;
            words += n.div_ceil(64) * 64;
            at as u32
        };
        let r = R {
            res: region(TB * hcd),
            xn: region(TB * hcd),
            lo: region(TB * hp.lr),
            gate: region(TB * hcd),
            mix: region(TB * d),
            inj: region(TB * hp.hc),
            p1: region(TB * qkv_w.max(2 * hp.nh * hp.hd)),
            p2: region(TB * (hp.nvh * hp.dk).max(hp.nkv * hp.hd)),
            p3: region(TB * hp.nvh.max(hp.nkv * hp.hd)),
            p4: region(TB * hp.nvh.max(hp.ih * hp.ihd)),
            p5: region(TB * hp.ihd),
            o: region(TB * (hp.nvh * hp.dk).max(hp.nh * hp.hd)),
            y: region(TB * d),
            rl: region(TB * hp.ne),
            rl2: region(TB * hp.ne),
            cv: region(TB * qkv_w),
            gst: region(n_lin * hp.nvh * hp.dk * hp.dk),
            gcv: region(n_lin * (hp.conv - 1) * qkv_w),
            sgst: region(n_lin * hp.nvh * hp.dk * hp.dk),
            sgcv: region(n_lin * (hp.conv - 1) * qkv_w),
            sgu: region(TB * 2 * ff),
            sh: region(TB * ff),
            sy: region(TB * d),
            sg: region(TB),
            emb: region(TB * d),
            pk: region(TB * hcd),
            pv: region(TB * d),
            xe: region(EROWS * d),
            egu: region(EROWS * 2 * ff),
            eh: region(EROWS * ff),
            ew: region(EROWS),
            ey: region(EROWS * d),
            logits: region(SMALL * hp.vocab),
            arg: region(SMALL),
            dgst: region(n_lin * hp.nvh * hp.dk * hp.dk),
            dgcv: region(n_lin * (hp.conv - 1) * qkv_w),
            fx: region(CHUNK * d),
            fy: region(CHUNK * d),
            eidx: region(EROWS),
        };
        let mut gpu = Gpu::new(words)?;
        let name = gpu.name.clone();
        {
            let grid: Vec<f32> = crate::iq_tables::IQ3S_GRID.iter().map(|&v| f32::from_bits(v)).collect();
            let mut b = gpu.batch();
            b.upload(0, &grid);
            b.run()?;
        }

        // dense weights, packed into arenas a layer at a time
        let t = |l: usize, s: &str| format!("blk.{l}.{s}");
        let gpu_bytes = |n: &str| -> Result<usize> {
            let (_, tt) = sh.find(n)?;
            if tt.ty.gpu_words().is_none() {
                bail!("{n}: the GPU can't decode {}", tt.ty.name());
            }
            Ok(tt.dims.get(1).copied().unwrap_or(1) * wpr(tt.ty, tt.dims[0]) as usize * 4)
        };
        let dense = |l: usize| -> Vec<String> {
            let mut v: Vec<String> = Vec::new();
            for hcn in ["hc_attn", "hc_ffn"] {
                for s in ["down", "up", "inject"] {
                    v.push(t(l, &format!("{hcn}_{s}.weight")));
                }
            }
            let names: &[&str] = if is_att(l) {
                &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "indexer.q_proj.weight", "indexer.k_proj.weight"]
            } else {
                &["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight"]
            };
            v.extend(names.iter().map(|s| t(l, s)));
            v.extend(["ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"].iter().map(|s| t(l, s)));
            if hp.ple.as_ref().is_some_and(|p| p.layer == l) {
                v.extend(["ple_key.weight", "ple_value.weight"].iter().map(|s| t(l, s)));
            }
            v
        };
        let mut layer_bytes = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let mut b = 2 * hcd * 4 + 64 + 1024;
            if !is_att(l) {
                b += (hp.conv * qkv_w + 2 * hp.nvh + hp.dk) * 4 + 64;
            }
            for n in dense(l) {
                b += gpu_bytes(&n)?.div_ceil(16) * 16;
            }
            layer_bytes.push(b);
        }
        let mut arena_of = vec![0usize; n_layer];
        let mut at = 0usize;
        while at < n_layer {
            let mut end = at;
            let mut sum = 0u64;
            while end < n_layer && (end == at || sum + layer_bytes[end] as u64 <= ARENA) {
                sum += layer_bytes[end] as u64;
                end += 1;
            }
            let a = gpu.arena(sum)?;
            arena_of[at..end].fill(a);
            at = end;
        }
        let mut cursor = vec![0u64; gpu.bufs.len()];
        let put = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, bytes: &[u8]| -> u32 {
            let w = cursor[arena];
            gpu.write(arena, w, bytes);
            cursor[arena] += (bytes.len() as u64).div_ceil(16) * 4;
            w as u32
        };
        let mat = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, n: &str| -> Result<Mat> {
            let (_, tt) = sh.find(n)?;
            let (cols, rows) = (tt.dims[0], tt.dims.get(1).copied().unwrap_or(1));
            let ty = tt.ty;
            let base = put(gpu, cursor, arena, &gpu_layout(ty, &sh.read(n)?));
            Ok(Mat { buf: arena, base, ty, rows, cols, wpr: wpr(ty, cols) })
        };
        let floats_put = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, v: &[f32]| -> u32 { put(gpu, cursor, arena, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }) };
        let hcw = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, pre: &str, inject: bool| -> Result<HcW> {
            let nw = sh.floats(&format!("{pre}_norm.weight"))?;
            if nw.len() != hcd {
                bail!("{pre}_norm: {} values, expected {hcd}", nw.len());
            }
            let norm = floats_put(gpu, cursor, arena, &nw);
            let down = mat(gpu, cursor, arena, &format!("{pre}_down.weight"))?;
            let up = mat(gpu, cursor, arena, &format!("{pre}_up.weight"))?;
            let inj = if inject { Some(mat(gpu, cursor, arena, &format!("{pre}_inject.weight"))?) } else { None };
            if down.rows != hp.lr || down.cols != hcd || up.rows != hcd || up.cols != hp.lr || inj.is_some_and(|m| m.rows != hp.hc || m.cols != hcd) {
                bail!("{pre}: hyper-connection shapes don't match the metadata");
            }
            Ok(HcW { arena, norm, down, up, inj, mvs: None })
        };
        let mut layers = Vec::with_capacity(n_layer);
        let (mut n_lin, mut n_att) = (0usize, 0usize);
        for l in 0..n_layer {
            let a = arena_of[l];
            let hca = hcw(&gpu, &mut cursor, a, &t(l, "hc_attn"), true)?;
            let hcf = hcw(&gpu, &mut cursor, a, &t(l, "hc_ffn"), true)?;
            let mix = if is_att(l) {
                let m = Mixer::Att {
                    q: mat(&gpu, &mut cursor, a, &t(l, "attn_q.weight"))?,
                    k: mat(&gpu, &mut cursor, a, &t(l, "attn_k.weight"))?,
                    v: mat(&gpu, &mut cursor, a, &t(l, "attn_v.weight"))?,
                    o: mat(&gpu, &mut cursor, a, &t(l, "attn_output.weight"))?,
                    qn: sh.floats(&t(l, "attn_q_norm.weight"))?,
                    kn: sh.floats(&t(l, "attn_k_norm.weight"))?,
                    iq: mat(&gpu, &mut cursor, a, &t(l, "indexer.q_proj.weight"))?,
                    ik: mat(&gpu, &mut cursor, a, &t(l, "indexer.k_proj.weight"))?,
                    iqn: sh.floats(&t(l, "indexer.q_norm.weight"))?,
                    ikn: sh.floats(&t(l, "indexer.k_norm.weight"))?,
                    idx: n_att,
                };
                if let Mixer::Att { q, k, v, o, iq, ik, .. } = &m {
                    if q.rows != 2 * hp.nh * hp.hd || k.rows != hp.nkv * hp.hd || v.rows != hp.nkv * hp.hd || o.cols != hp.nh * hp.hd || iq.rows != hp.ih * hp.ihd || ik.rows != hp.ihd {
                        bail!("layer {l}: attention shapes don't match the metadata");
                    }
                }
                if ratios.get(l).copied().unwrap_or(0) as usize != kpool {
                    bail!("layer {l}: dense attention layers are not supported");
                }
                n_att += 1;
                m
            } else {
                let m = Mixer::Lin {
                    qkv: mat(&gpu, &mut cursor, a, &t(l, "attn_qkv.weight"))?,
                    z: mat(&gpu, &mut cursor, a, &t(l, "attn_gate.weight"))?,
                    beta: mat(&gpu, &mut cursor, a, &t(l, "ssm_beta.weight"))?,
                    alpha: mat(&gpu, &mut cursor, a, &t(l, "ssm_alpha.weight"))?,
                    out: mat(&gpu, &mut cursor, a, &t(l, "ssm_out.weight"))?,
                    conv_w: 0,
                    dt_w: 0,
                    a_w: 0,
                    norm_w: 0,
                    idx: n_lin,
                };
                let (conv, dt, sa, norm) = (sh.floats(&t(l, "ssm_conv1d.weight"))?, sh.floats(&t(l, "ssm_dt.bias"))?, sh.floats(&t(l, "ssm_a"))?, sh.floats(&t(l, "ssm_norm.weight"))?);
                let mut m = m;
                if let Mixer::Lin { qkv, z, beta, alpha, out, conv_w, dt_w, a_w, norm_w, .. } = &mut m {
                    if qkv.rows != qkv_w || z.rows != hp.nvh * hp.dk || beta.rows != hp.nvh || alpha.rows != hp.nvh || out.cols != hp.nvh * hp.dk || conv.len() != hp.conv * qkv_w || dt.len() != hp.nvh || sa.len() != hp.nvh || norm.len() != hp.dk {
                        bail!("layer {l}: linear attention shapes don't match the metadata");
                    }
                    *conv_w = floats_put(&gpu, &mut cursor, a, &conv);
                    *dt_w = floats_put(&gpu, &mut cursor, a, &dt);
                    *a_w = floats_put(&gpu, &mut cursor, a, &sa);
                    *norm_w = floats_put(&gpu, &mut cursor, a, &norm);
                }
                n_lin += 1;
                m
            };
            let ple = match &hp.ple {
                Some(p) if p.layer == l => {
                    let pl = Ple {
                        key: mat(&gpu, &mut cursor, a, &t(l, "ple_key.weight"))?,
                        value: mat(&gpu, &mut cursor, a, &t(l, "ple_value.weight"))?,
                        nk: sh.floats(&t(l, "ple_norm_key.weight"))?,
                        nq: sh.floats(&t(l, "ple_norm_query.weight"))?,
                        nc: sh.floats(&t(l, "ple_norm_conv.weight"))?,
                        conv: sh.floats(&t(l, "ple_conv1d.weight"))?,
                    };
                    if pl.key.cols != p.heads * p.dim || pl.key.rows != hcd || pl.value.rows != d || pl.conv.len() != p.taps * hcd {
                        bail!("layer {l}: n-gram embedding shapes don't match the metadata");
                    }
                    Some(pl)
                }
                _ => None,
            };
            let router = mat(&gpu, &mut cursor, a, &t(l, "ffn_gate_inp.weight"))?;
            let sg = mat(&gpu, &mut cursor, a, &t(l, "ffn_gate_shexp.weight"))?;
            let su = mat(&gpu, &mut cursor, a, &t(l, "ffn_up_shexp.weight"))?;
            let sd = mat(&gpu, &mut cursor, a, &t(l, "ffn_down_shexp.weight"))?;
            let sgate = mat(&gpu, &mut cursor, a, &t(l, "ffn_gate_inp_shexp.weight"))?;
            if su.base != sg.base + (sg.rows as u32) * sg.wpr || su.ty != sg.ty || router.rows != hp.ne {
                bail!("layer {l}: shared expert or router shapes don't line up");
            }
            layers.push(Layer { hca, hcf, mix, mvs_mix: None, mvs_ffn: None, ffn_has_next: false, ple, router, sgu: Mat { rows: 2 * sg.rows, ..sg }, sd, sgate });
            gpu.flush()?;
            progress((l + 1) * 40 / n_layer);
        }
        // one-token product tables, in each layer's arena
        let mk = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, segs: &[(Mat, u32, u32)]| -> Option<Mvs> {
            if segs.is_empty() || segs.iter().any(|(m, _, _)| m.buf != arena || m.cols % 4 != 0) {
                return None;
            }
            let mut tys: Vec<Ty> = Vec::new();
            for (m, _, _) in segs {
                if !tys.contains(&m.ty) {
                    tys.push(m.ty);
                }
            }
            let mut words: Vec<u32> = Vec::new();
            let mut end = 0u32;
            for &(m, x, y) in segs {
                end += (m.rows as u32).div_ceil(8);
                words.extend([end, m.rows as u32, m.cols as u32, y, m.base, m.wpr, tys.iter().position(|t| *t == m.ty).unwrap() as u32, x]);
            }
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let tab = put(gpu, cursor, arena, &bytes);
            let key = format!("q4x-mvs-{}", tys.iter().map(|t| t.name()).collect::<Vec<_>>().join("-"));
            Some(Mvs { buf: arena, tab, n: segs.len() as u32, groups: end, tys, key })
        };
        for l in 0..n_layer {
            let a = arena_of[l];
            let next = (l + 1 < n_layer && arena_of[l + 1] == a).then(|| layers[l + 1].router);
            let ly = &mut layers[l];
            for hc in [&mut ly.hca, &mut ly.hcf] {
                let mut segs = vec![(hc.down, r.xn, r.lo)];
                segs.extend(hc.inj.map(|m| (m, r.xn, r.inj)));
                hc.mvs = mk(&gpu, &mut cursor, a, &segs);
            }
            let segs: Vec<(Mat, u32, u32)> = match &ly.mix {
                Mixer::Lin { qkv, z, beta, alpha, .. } => vec![(*qkv, r.mix, r.p1), (*z, r.mix, r.p2), (*beta, r.mix, r.p3), (*alpha, r.mix, r.p4)],
                Mixer::Att { q, k, v, iq, ik, .. } => vec![(*q, r.mix, r.p1), (*k, r.mix, r.p2), (*v, r.mix, r.p3), (*iq, r.mix, r.p4), (*ik, r.mix, r.p5)],
            };
            ly.mvs_mix = mk(&gpu, &mut cursor, a, &segs);
            let mut segs = vec![(ly.router, r.mix, r.rl), (ly.sgu, r.mix, r.sgu), (ly.sgate, r.mix, r.sg)];
            segs.extend(next.map(|m| (m, r.mix, r.rl2)));
            ly.ffn_has_next = next.is_some();
            ly.mvs_ffn = mk(&gpu, &mut cursor, a, &segs);
        }
        gpu.flush()?;
        let head_arena = gpu.arena(gpu_bytes("output.weight")? as u64 + 3 * gpu_bytes("output_hc_down.weight")? as u64 + hcd as u64 * 4 + 64)?;
        cursor.push(0);
        let head = mat(&gpu, &mut cursor, head_arena, "output.weight")?;
        let hc_head = hcw(&gpu, &mut cursor, head_arena, "output_hc", false)?;
        gpu.flush()?;

        // experts: where they sit in the files, how they sit in VRAM
        let mut pieces = Vec::with_capacity(n_layer);
        let mut files = Vec::with_capacity(n_layer);
        let mut stride = Vec::with_capacity(n_layer);
        let mut ex = Vec::with_capacity(n_layer);
        let mut slot_words = 0u64;
        let mut expert_max = 0usize;
        let mut expert_sum = 0u64;
        let mut lwords: Vec<u64> = Vec::new();
        for l in 0..n_layer {
            let mut p = [(0u64, 0usize); 3];
            let mut fl = [0u8; 3];
            let mut ty = [Ty::F32; 3];
            // `pelica repack` renames a layer's expert tensors to *_expm once
            // each expert's three pieces sit together in their region
            let n_m = ["ffn_gate", "ffn_up", "ffn_down"].iter().filter(|s| sh.has(&t(l, &format!("{s}_expm.weight")))).count();
            if n_m != 0 && n_m != 3 {
                bail!("layer {l} of this model was left half repacked: run `pelica repack` on it again to finish");
            }
            let merged = n_m == 3;
            for (i, s) in ["ffn_gate", "ffn_up", "ffn_down"].iter().enumerate() {
                let name = t(l, &format!("{s}_{}.weight", if merged { "expm" } else { "exps" }));
                let (fi, tt) = sh.find(&name)?;
                if tt.dims.len() != 3 || tt.dims[2] != hp.ne || tt.ty.gpu_words().is_none() || !cpu::supports(tt.ty) {
                    bail!("{name}: unexpected expert tensor ({:?}, {})", tt.dims, tt.ty.name());
                }
                p[i] = (tt.offset as u64, tt.bytes() / hp.ne);
                fl[i] = fi as u8;
                ty[i] = tt.ty;

            }
            if ty[0] != ty[1] {
                bail!("layer {l}: expert gate and up of different types");
            }
            if merged {
                if fl[0] != fl[1] || fl[1] != fl[2] {
                    bail!("layer {l}: repacked expert pieces in different files");
                }
                let mut at = p.iter().map(|x| x.0).min().unwrap_or(0);
                for q in p.iter_mut() {
                    q.0 = at;
                    at += q.1 as u64;
                }
                stride.push(p.iter().map(|x| x.1).sum());
            } else {
                // apart: each piece steps by its own size
                stride.push(0);
            }
            let e = Ex { ty, wpr_g: wpr(ty[0], d), wpr_d: wpr(ty[2], ff) };
            lwords.push(2 * ff as u64 * e.wpr_g as u64 + d as u64 * e.wpr_d as u64);
            slot_words = slot_words.max(*lwords.last().unwrap());
            ex.push(e);
            expert_max = expert_max.max(p.iter().map(|x| (x.1 + 2 * 4096).div_ceil(4096) * 4096).sum());
            expert_sum += p.iter().map(|x| x.1 as u64).sum::<u64>() * hp.ne as u64;
            pieces.push(p);
            files.push(fl);
        }
        let slot_bytes = slot_words * 4;
        let total = n_layer * hp.ne;
        let mut stage = Vec::new();
        {
            let a = gpu.arena(STAGE as u64 * slot_bytes)?;
            for i in 0..STAGE {
                stage.push((a, (i as u64 * slot_words) as u32));
            }
        }
        let budget = match std::env::var("PELICA_EXPERTS_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(gb) => (gb * 1e9) as u64,
            None => crate::free_vram().map_or(0, |f| (f as u64).saturating_sub(crate::vram_margin() as u64 + (64 << 20))),
        };
        // slots sized for each layer's experts (they differ by layer), the
        // same number for every layer
        if std::env::var_os("PELICA_ONE_SLOT_SIZE").is_some() {
            lwords.iter_mut().for_each(|w| *w = slot_words);
        }
        let mut sizes = lwords.clone();
        sizes.sort_unstable();
        sizes.dedup();
        let lcls: Vec<u8> = lwords.iter().map(|w| sizes.iter().position(|s| s == w).unwrap() as u8).collect();
        let per_layer = ((budget / (lwords.iter().sum::<u64>() * 4).max(1)) as usize).min(hp.ne);
        let mut vslots = Vec::new();
        let mut vcls = Vec::new();
        'cls: for (c, &words) in sizes.iter().enumerate() {
            let want = lcls.iter().filter(|&&x| x as usize == c).count() * per_layer;
            let per_arena = (ARENA / (words * 4)).max(1) as usize;
            let mut got = 0;
            while got < want {
                let n = per_arena.min(want - got);
                let a = match gpu.arena(n as u64 * words * 4) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("[pelica] VRAM for experts ran out early: {e:#}");
                        break 'cls;
                    }
                };
                for i in 0..n {
                    vslots.push((a, (i as u64 * words) as u32));
                    vcls.push(c as u8);
                }
                got += n;
            }
        }
        let usage_path = PathBuf::from(format!("{}.pelica-usage", path.display()));
        let usage_base: Vec<u32> = std::fs::read(&usage_path)
            .ok()
            .filter(|b| b.len() == total * 4)
            .map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
            .unwrap_or_else(|| vec![0; total]);
        // (f16 keys, values and indexer keys per attention layer, and one
        // layer's keys and values in f32 while a pass reads them)
        let kv_per_pos = n_att * (4 * hp.nkv * hp.hd + 2 * hp.ihd) + 8 * hp.nkv * hp.hd + 64;
        let kv_room = ctx.min(std::env::var("PELICA_KV_START").ok().and_then(|v| v.parse().ok()).unwrap_or(KV_START));
        let ram = match std::env::var("PELICA_RAM_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(gb) => (gb * 1e9) as u64,
            None => crate::available_ram().saturating_sub(RAM_MARGIN + (kv_per_pos * kv_room) as u64),
        };
        let lay = Layout { pieces, stride, n_exp: hp.ne, file: files };
        let remote = {
            let size: u64 = sh.paths.iter().map(|p| std::fs::metadata(p).map_or(0, |m| m.len())).sum();
            let model = size ^ (total as u64) << 40 ^ (n_layer as u64) << 56 ^ 0x51a7_38c4;
            let tys: Vec<[u32; 3]> = ex.iter().map(|e| e.ty.map(|t| t.id())).collect();
            crate::bridge::Remote::new_q4x(sh.paths.clone(), model, d, ff, lay.clone(), tys)
        };
        let one_size = std::env::var_os("PELICA_ONE_SLOT_SIZE").is_some();
        let store = if one_size { Store::new_multi(&sh.paths, lay, ((ram / expert_max as u64) as usize).clamp(16, total), vec![0.0; total])? } else { Store::new_sized(&sh.paths, lay, ram, vec![0.0; total])? };
        let mut order: Vec<u32> = (0..total as u32).filter(|&k| usage_base[k as usize] > 0).collect();
        order.sort_by_key(|&k| (std::cmp::Reverse(usage_base[k as usize]), k));
        let mut vram = Vram { key: vec![u32::MAX; vslots.len()], slots: vslots, cls: vcls, lcls, slot_of: Default::default(), moved: 0 };
        let n_vram = vram.slots.len();
        // the hottest by use history: into VRAM while its layer's slots
        // last, then into RAM while those do
        let mut vfree: Vec<Vec<usize>> = vec![Vec::new(); sizes.len()];
        for s in (0..vram.slots.len()).rev() {
            vfree[vram.cls[s] as usize].push(s);
        }
        let mut rroom: std::collections::HashMap<usize, usize> = Default::default();
        let (mut warm_v, mut warm_s, mut warm_r) = (Vec::new(), Vec::new(), Vec::new());
        for &k in &order {
            let l = k as usize / hp.ne;
            if let Some(s) = vfree[vram.lcls[l] as usize].pop() {
                warm_v.push(k);
                warm_s.push(s);
            } else {
                let room = rroom.entry(store.class_of(l)).or_insert_with(|| store.class_slots(l).saturating_sub(16));
                if *room > 0 {
                    *room -= 1;
                    warm_r.push(k);
                }
            }
        }
        let todo = warm_v.len() + warm_r.len();
        for (i, chunk) in warm_v.chunks(16).enumerate() {
            let pins = store.request(chunk)?;
            for (j, (pin, &k)) in pins.iter().zip(chunk).enumerate() {
                store.wait(pin)?;
                let s = warm_s[i * 16 + j];
                Self::upload_expert(&gpu, &ex[k as usize / hp.ne], vram.slots[s], pin);
                vram.key[s] = k;
                vram.slot_of.insert(k, s);
            }
            gpu.flush()?;
            drop(pins);
            store.forget(chunk);
            progress(40 + (i * 16 + chunk.len()) * 60 / todo.max(1));
        }
        for (i, chunk) in warm_r.chunks(32).enumerate() {
            let pins = store.request(chunk)?;
            for pin in &pins {
                store.wait(pin)?;
            }
            progress(40 + (warm_v.len() + i * 32 + chunk.len()) * 60 / todo.max(1));
        }
        if let Some(r) = &remote {
            // the devices get the experts after those VRAM and RAM hold
            r.set_want(order.iter().copied().filter(|k| !vram.slot_of.contains_key(k)).skip(store.slots()).take(12_000).collect());
        }
        let state = State {
            att: (0..n_att).map(|_| Att::default()).collect(),
            ple: vec![0.0; hp.ple.as_ref().map_or(0, |p| (p.taps - 1) * p.ngram * hcd)],
            toks: Vec::new(),
            pos: 0,
            snap: None,
            trace: None,
            since_rebal: 0,
            saved_pos: 0,
            logits: Vec::new(),
            rp: None,
        };
        let cfg = crate::moe::Cfg {
            arch: "qwen4exp".into(),
            n_layer,
            d,
            n_head: hp.nh,
            n_kv: hp.nkv,
            hd: hp.hd,
            n_exp: hp.ne,
            top_k: hp.k,
            ff_exp: ff,
            vocab: hp.vocab,
            eps: hp.eps,
            ctx,
            rot: hp.rot,
            neox: true,
            norm_w: true,
            w_scale: 1.0,
            sigmoid: false,
        };
        let eos = crate::moe::stop_tokens(g);
        let chat_marks = g.strs("tokenizer.ggml.tokens").ok().and_then(|v| {
            let at = |s: &str| v.iter().position(|t| *t == s).map(|i| i as u32);
            Some((at("<|im_start|>")?, at("system")?))
        });
        let state_dir = PathBuf::from(format!("{}.pelica-state", path.display()));
        let host_experts = store.slots();
        let host_bytes = store.bytes();
        let embd = { let (i, tt) = sh.find("token_embd.weight")?; (i, tt.clone()) };
        if !cpu::supports(embd.1.ty) {
            bail!("token embeddings of type {} are not supported", embd.1.ty.name());
        }
        let ple_tab = match &hp.ple {
            Some(p) => {
                let (i, tt) = sh.find("per_layer_token_embd.weight")?;
                if tt.dims[0] != p.dim || !cpu::supports(tt.ty) || (0..p.heads).any(|h| p.off[h] + p.size[h] > tt.dims[1] as u64) {
                    bail!("the n-gram embedding table doesn't match its settings");
                }
                Some((i, tt.clone()))
            }
            None => None,
        };
        eprintln!(
            "[pelica] Qwen3.8: {n_layer} layers ({n_lin} DeltaNet, {n_att} attention), dense {:.2} GB in VRAM; experts {:.1} GB: {} in VRAM, {} cache slots in RAM ({:.1} GB), the rest read from the drive; warmed {} from use history; loaded in {:.1}s",
            layer_bytes.iter().sum::<usize>() as f64 / 1e9,
            expert_sum as f64 / 1e9,
            n_vram,
            host_experts,
            host_bytes as f64 / 1e9,
            todo,
            t0.elapsed().as_secs_f64()
        );
        progress(100);
        Ok(Engine {
            cfg,
            name,
            n_lin,
            max_ctx,
            vram_experts: n_vram,
            host_experts,
            host_bytes,
            eos,
            gpu,
            layers,
            head,
            hc_head,
            embd,
            ple_tab,
            store,
            ex,
            vram: Mutex::new(vram),
            stage,
            st: Mutex::new(state),
            r,
            usage: (0..total).map(|_| AtomicU32::new(0)).collect(),
            usage_base,
            usage_path,
            samp: Mutex::new((0.0, 1.0, 0)),
            state_dir,
            kv_per_pos,
            kv_room: AtomicUsize::new(kv_room),
            mmproj: vision::find_mmproj(path),
            images: Mutex::new(Vec::new()),
            img_tok: g.u(&key("ple.image_token_id")).map(|v| v as u32).unwrap_or(u32::MAX),
            chat_marks,
            remote,
            hp,
            sh,
        })
    }

    /// Copies an expert's three pieces into a VRAM slot (GPU layout).
    fn upload_expert(gpu: &Gpu, ex: &Ex, (arena, base): (usize, u32), pin: &Pin) {
        let mut w = base as u64;
        for p in 0..3 {
            let b = pin.piece(p);
            let ty = ex.ty[p];
            if ty.gpu_words().unwrap() * 4 == ty.block().1 {
                gpu.write(arena, w, b);
                w += (b.len() / 4) as u64;
            } else {
                let v = gpu_layout(ty, b);
                gpu.write(arena, w, &v);
                w += (v.len() / 4) as u64;
            }
        }
    }

    /// The hyper-connection mix of the `n` residuals at r.res: the block
    /// input at r.mix and (blocks) the spreading weights at r.inj.
    fn hc_ops(&self, b: &mut Batch, hc: &HcW, n: usize) {
        let (r, hp) = (&self.r, &self.hp);
        let (d, h, lr) = (hp.d as u32, hp.hc as u32, hp.lr as u32);
        let rows = n as u32 * h;
        b.kernel("q4x-hcnorm", || HCNORM.to_string(), hc.arena, &[d, r.res, r.xn, h, hc.norm, hp.eps.to_bits(), rows], [rows, 1, 1]);
        let one = n == 1 && hc.mvs.is_some();
        if one {
            hc.mvs.as_ref().unwrap().run(b);
        } else {
            b.mv(&hc.down, r.xn, d * h, r.lo, lr, n, false, None);
        }
        b.kernel("q4x-silus", || SILUS.to_string(), 0, &[n as u32 * lr, r.lo, r.lo, (1.0 / h as f32).to_bits()], [(n as u32 * lr).div_ceil(256), 1, 1]);
        b.mv(&hc.up, r.lo, lr, r.gate, d * h, n, false, None);
        b.kernel("q4x-hcmix", || HCMIX.to_string(), 0, &[d, h, r.xn, r.gate, r.mix, n as u32], [d.div_ceil(256), n as u32, 1]);
        if let (false, Some(m)) = (one, &hc.inj) {
            b.mv(m, r.xn, d * h, r.inj, h, n, false, None);
        }
    }

    /// The token's embedding, copied into the hc streams (a picture's row
    /// for its tokens).
    fn embed(&self, tok: u32, x: &mut [f32], imgs: &[(u32, std::sync::Arc<vision::Image>)]) {
        let d = self.hp.d;
        if tok & IMG != 0 {
            let (id, row) = ((tok & !IMG) >> IMG_ROW_BITS, (tok & ((1 << IMG_ROW_BITS) - 1)) as usize);
            if let Some((_, im)) = imgs.iter().find(|(i, _)| *i == id).filter(|(_, im)| (row + 1) * d <= im.rows.len()) {
                for c in x.chunks_mut(d) {
                    c.copy_from_slice(&im.rows[row * d..(row + 1) * d]);
                }
                return;
            }
        }
        let (i, t) = &self.embd;
        let tok = (tok as usize).min(self.hp.vocab - 1);
        let rb = t.ty.bytes(d);
        let b = self.sh.gs[*i].bytes(t);
        cpu::deq_row(t.ty, &b[tok * rb..(tok + 1) * rb], &mut x[..d]);
        let (a, rest) = x.split_at_mut(d);
        for c in rest.chunks_mut(d) {
            c.copy_from_slice(a);
        }
    }

    /// Rotary angles' cos and sin for position `pos` (as ggml builds them:
    /// the angle multiplied up step by step).
    fn rope_cs(&self, pos: usize) -> Vec<(f32, f32)> {
        let n = self.hp.rot;
        let scale = self.hp.rope_base.powf(-2.0 / n as f32);
        let mut th = pos as f32;
        (0..n / 2)
            .map(|_| {
                let v = (th.cos(), th.sin());
                th *= scale;
                v
            })
            .collect()
    }

    /// `rope_cs` for M-RoPE positions (time, row, column): pair i turns by
    /// the position its section names (as ggml's interleaved M-RoPE); text
    /// has all three equal, which is `rope_cs`.
    fn rope_cs3(&self, p: [u32; 3]) -> Vec<(f32, f32)> {
        if p[0] == p[1] && p[1] == p[2] {
            return self.rope_cs(p[0] as usize);
        }
        let n = self.hp.rot;
        let sec = self.hp.rope_sections;
        let scale = self.hp.rope_base.powf(-2.0 / n as f32);
        let mut th = p.map(|v| v as f32);
        (0..n / 2)
            .map(|i| {
                let c = match i % 3 {
                    1 if i < 3 * sec[1] => 1,
                    2 if i < 3 * sec[2] => 2,
                    _ => 0,
                };
                let v = (th[c].cos(), th[c].sin());
                th.iter_mut().for_each(|t| *t *= scale);
                v
            })
            .collect()
    }

    /// Rotary positions (time, row, column) of cache positions a..b: text
    /// counts on from where the last picture's positions end; a picture's
    /// rows all have the time of its start, and their row and column added
    /// (llama.cpp's M-RoPE layout: a picture takes max(rows, columns)
    /// positions).
    fn rope_pos(&self, toks: &[u32], a: usize, b: usize) -> Vec<[u32; 3]> {
        let imgs = self.images.lock().unwrap();
        let (mut next, mut base) = (0u32, 0u32);
        let mut out = Vec::with_capacity(b - a);
        for (p, &t) in toks[..b].iter().enumerate() {
            let r = if t & IMG != 0 {
                let (id, row) = ((t & !IMG) >> IMG_ROW_BITS, (t & ((1 << IMG_ROW_BITS) - 1)) as usize);
                let (nx, ny) = imgs.iter().find(|(i, _)| *i == id).map_or((1, 1), |(_, im)| (im.nx, im.ny));
                if row == 0 {
                    base = next;
                }
                if row + 1 >= nx * ny {
                    next = base + nx.max(ny) as u32;
                }
                [base, base + (row / nx) as u32, base + (row % nx) as u32]
            } else {
                next += 1;
                [next - 1; 3]
            };
            if p >= a {
                out.push(r);
            }
        }
        out
    }

    /// Whether pictures can go in (an encoder file beside the model).
    pub fn vision(&self) -> bool {
        self.mmproj.is_some()
    }

    /// Encodes a picture (RGB, w x h, multiples of 32) unless it was already;
    /// returns its id and size in rows (nx, ny). Its tokens are
    /// `image_token(id, row)`.
    pub fn add_image(&self, rgb: &[u8], w: usize, h: usize) -> Result<(u32, usize, usize)> {
        let Some(path) = &self.mmproj else { bail!("this model has no image encoder (mmproj) beside it") };
        let mut hsh = 0xcbf2_9ce4_8422_2325u64;
        for &b in rgb.iter().chain(&(w as u32).to_le_bytes()).chain(&(h as u32).to_le_bytes()) {
            hsh = (hsh ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
        let id = ((hsh ^ (hsh >> 32)) as u32) & ((1 << (31 - IMG_ROW_BITS)) - 1);
        if let Some((_, im)) = self.images.lock().unwrap().iter().find(|(i, _)| *i == id) {
            return Ok((id, im.nx, im.ny));
        }
        let t = Instant::now();
        let enc = vision::Encoder::open(path)?;
        if enc.out_dim() != self.hp.d {
            bail!("the image encoder's rows ({}) don't fit this model ({})", enc.out_dim(), self.hp.d);
        }
        let im = enc.encode(rgb, w, h)?;
        if im.nx * im.ny > 1 << IMG_ROW_BITS {
            bail!("the picture is too big");
        }
        eprintln!("[pelica] picture {w}x{h}: {} rows in {:.1}s", im.nx * im.ny, t.elapsed().as_secs_f64());
        let (nx, ny) = (im.nx, im.ny);
        let mut imgs = self.images.lock().unwrap();
        imgs.push((id, std::sync::Arc::new(im)));
        // the last few pictures stay (a chat sends them again every turn)
        if imgs.len() > 12 {
            imgs.remove(0);
        }
        Ok((id, nx, ny))
    }

    /// The token for row `row` of picture `id`.
    pub fn image_token(id: u32, row: usize) -> u32 {
        IMG | id << IMG_ROW_BITS | row as u32
    }

    /// The token `<|image_pad|>` the chat template marks a picture with.
    pub fn image_pad(&self) -> u32 {
        self.img_tok
    }

    /// NEOX rotation of the first `rot` dims of x (pairs i, i + rot/2).
    fn rope(&self, x: &mut [f32], cs: &[(f32, f32)]) {
        let h = self.hp.rot / 2;
        for (i, &(c, s)) in cs.iter().enumerate() {
            let (a, b) = (x[i], x[i + h]);
            x[i] = a * c - b * s;
            x[i + h] = a * s + b * c;
        }
    }

    /// Runs `toks` (positions pos0..) through the model; with `logits`, the
    /// next token after the last. `on_layer` hears of each finished layer;
    /// if it returns false the pass stops (the cache is then cleared).
    fn forward(&self, st: &mut State, toks: &[u32], pos0: usize, logits: bool, on_layer: &mut dyn FnMut(usize) -> bool) -> Result<Option<u32>> {
        Ok(self.forward_n(st, toks, pos0, logits as usize, on_layer)?.pop())
    }

    /// `forward`, with the next token after each of the last `heads`
    /// positions (greedy when more than one).
    fn forward_n(&self, st: &mut State, toks: &[u32], pos0: usize, heads: usize, on_layer: &mut dyn FnMut(usize) -> bool) -> Result<Vec<u32>> {
        let logits = heads > 0;
        let hp = &self.hp;
        let (d, c) = (hp.d, toks.len());
        let hcd = hp.hc * d;
        if st.pos != pos0 || st.toks.len() != pos0 {
            bail!("cache out of step ({} cached, pass at {pos0})", st.pos);
        }
        let mut x = vec![0f32; c * hcd];
        {
            let imgs = self.images.lock().unwrap();
            x.par_chunks_mut(hcd).zip(toks.par_iter()).for_each(|(xr, &t)| self.embed(t, xr, &imgs));
        }
        st.toks.extend_from_slice(toks);
        st.rp = if self.mmproj.is_some() && st.toks.iter().any(|&t| t & IMG != 0) {
            let from = pos0 / hp.kpool * hp.kpool;
            Some((from, self.rope_pos(&st.toks, from, pos0 + c)))
        } else {
            None
        };
        // the n-gram embedding's rows sit at random in a 29 GB table read
        // through the file map: touch them all at once, from a side thread,
        // while the first layers run, rather than one page fault at a time
        let rows: Vec<usize> = match (&hp.ple, &self.ple_tab) {
            (Some(p), Some((_, tab))) => {
                let rb = tab.ty.bytes(p.dim);
                (0..c).flat_map(|i| self.ple_rows(&st.toks, pos0 + i)).map(|r| r as usize * rb).collect()
            }
            _ => Vec::new(),
        };
        std::thread::scope(|sc| -> Result<()> {
            if let Some((ti, tab)) = self.ple_tab.as_ref().filter(|_| !rows.is_empty()) {
                let tb = self.sh.gs[*ti].bytes(tab);
                let rb = tab.ty.bytes(hp.ple.as_ref().unwrap().dim);
                sc.spawn(move || rows.par_iter().for_each(|&o| unsafe {
                    std::ptr::read_volatile(&tb[o]);
                    std::ptr::read_volatile(&tb[o + rb - 1]);
                }));
            }
            let mut pend = None;
            for l in 0..hp.n_layer {
                self.store.mark(l * 4);
                if self.layers[l].ple.is_some() {
                    self.ple(st, l, pos0, c, &mut x)?;
                }
                let fi = self.mixer(st, l, pos0, c, &mut x, pend.take())?;
                self.store.mark(l * 4 + 1);
                if c > SMALL {
                    self.store.sweep(Some(l));
                }
                pend = self.ffn(st, l, toks, &mut x, fi)?;
                if !on_layer(l) {
                    self.reset(st);
                    bail!(Cancelled);
                }
            }
            Ok(())
        })?;
        self.store.mark(255);
        self.store.sweep(None);
        st.pos = pos0 + c;
        self.store.advance(if c > SMALL { 64.min(c) } else { c });
        if !logits {
            return Ok(Vec::new());
        }
        let mut tm = Instant::now();
        let r = &self.r;
        let mut b = self.gpu.batch();
        if heads > 1 {
            let m = heads.min(c).min(SMALL);
            b.upload(r.res, &x[(c - m) * hcd..]);
            self.hc_ops(&mut b, &self.hc_head, m);
            b.mv(&self.head, r.mix, d as u32, r.logits, hp.vocab as u32, m, false, None);
            if self.samp.lock().unwrap().0 > 0.0 {
                // sampling: the caller checks the drafts against the logits
                b.read(r.logits, m * hp.vocab);
                st.logits = b.run()?.pop().unwrap_or_default();
                tick(9, &mut tm);
                return Ok(vec![0; m]);
            }
            for i in 0..m {
                b.argmax(hp.vocab, r.logits + (i * hp.vocab) as u32, r.arg + i as u32);
            }
            b.read(r.arg, m);
            let out = b.run()?;
            tick(9, &mut tm);
            return Ok(out[0].iter().map(|v| v.to_bits()).collect());
        }
        b.upload(r.res, &x[(c - 1) * hcd..]);
        self.hc_ops(&mut b, &self.hc_head, 1);
        b.mv(&self.head, r.mix, d as u32, r.logits, hp.vocab as u32, 1, false, None);
        let (temp, top_p, _) = *self.samp.lock().unwrap();
        if temp > 0.0 {
            b.read(r.logits, hp.vocab);
            let out = b.run()?;
            let t = {
                let mut sp = self.samp.lock().unwrap();
                crate::ds4::sample(&out[0], temp, top_p, &mut sp.2)
            };
            tick(9, &mut tm);
            return Ok(vec![t]);
        }
        if std::env::var_os("PELICA_LOGITS").is_some() {
            b.read(r.logits, hp.vocab);
            let out = b.run()?;
            let mut best: Vec<(usize, f32)> = out[0].iter().copied().enumerate().collect();
            best.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("[logits] {:?}", &best[..5]);
            tick(9, &mut tm);
            return Ok(vec![best[0].0 as u32]);
        }
        b.argmax(hp.vocab, r.logits, r.arg);
        b.read(r.arg, 1);
        let out = b.run()?;
        tick(9, &mut tm);
        Ok(vec![out[0][0].to_bits()])
    }

    fn gst_len(&self) -> u32 {
        let hp = &self.hp;
        (self.n_lin * hp.nvh * hp.dk * hp.dk) as u32
    }

    fn gcv_len(&self) -> u32 {
        let hp = &self.hp;
        (self.n_lin * (hp.conv - 1) * (2 * hp.nkh * hp.dk + hp.nvh * hp.dk)) as u32
    }

    /// GPU copies (src, dst, words, mode 0) or clears (mode 1) in one batch.
    fn fill(&self, jobs: &[(u32, u32, u32, u32)]) -> Result<()> {
        let mut b = self.gpu.batch();
        for &(src, dst, n, mode) in jobs {
            b.kernel("q4x-fill", || FILL.to_string(), 0, &[n, src, dst, mode], [n.div_ceil(256).min(4096), 1, 1]);
        }
        b.run()?;
        Ok(())
    }

    fn reset(&self, st: &mut State) {
        st.pos = 0;
        st.toks.clear();
        st.snap = None;
        st.saved_pos = 0;
        let r = &self.r;
        if let Err(e) = self.fill(&[(r.gst, r.gst, r.gcv + self.gcv_len() - r.gst, 1)]) {
            eprintln!("can't clear the DeltaNet state: {e}");
        }
        for a in st.att.iter_mut() {
            *a = Att::default();
        }
        st.ple.fill(0.0);
    }

    /// The n-gram embedding rows of the token at `i` of `toks`: indices into
    /// the table, head by head.
    fn ple_rows(&self, toks: &[u32], i: usize) -> Vec<u64> {
        let p = self.hp.ple.as_ref().unwrap();
        let tk = |t: u32| if t & IMG != 0 { self.img_tok as u64 } else { t as u64 };
        let mut ctx = vec![0u64; p.ngram];
        ctx[0] = tk(toks[i]);
        let mut cut = false;
        for s in 1..p.ngram {
            let t = if cut || i < s { None } else { Some(tk(toks[i - s])) };
            cut = cut || t.is_none() || t == Some(p.eos);
            ctx[s] = if cut { p.eos } else { t.unwrap() };
        }
        let mut out = vec![0u64; p.heads];
        for n in 2..=p.ngram {
            let mut mixed = ctx[0].wrapping_mul(p.mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(p.mult[j]);
            }
            for g in 0..p.per_gram {
                let h = (n - 2) * p.per_gram + g;
                out[h] = mixed % p.size[h] + p.off[h];
            }
        }
        out
    }

    /// Layer `l`'s n-gram embedding, added into the residuals.
    fn ple(&self, st: &mut State, l: usize, pos0: usize, c: usize, x: &mut [f32]) -> Result<()> {
        let mut tm = Instant::now();
        let hp = &self.hp;
        let p = hp.ple.as_ref().unwrap();
        let ly = self.layers[l].ple.as_ref().unwrap();
        let (d, h) = (hp.d, hp.hc);
        let hcd = h * d;
        let r = &self.r;
        let (ti, tab) = self.ple_tab.as_ref().unwrap();
        let tb = self.sh.gs[*ti].bytes(tab);
        let rb = tab.ty.bytes(p.dim);
        let ed = p.heads * p.dim;
        let toks = &st.toks;
        let mut emb = vec![0f32; c * ed];
        emb.par_chunks_mut(ed).enumerate().for_each(|(i, e)| {
            for (hh, &row) in self.ple_rows(toks, pos0 + i).iter().enumerate() {
                let row = row as usize;
                cpu::deq_row(tab.ty, &tb[row * rb..(row + 1) * rb], &mut e[hh * p.dim..(hh + 1) * p.dim]);
            }
        });
        let mut key = vec![0f32; c * hcd];
        let mut val = vec![0f32; c * d];
        for s in (0..c).step_by(TB) {
            let n = (c - s).min(TB);
            let mut b = self.gpu.batch();
            b.upload(r.emb, &emb[s * ed..(s + n) * ed]);
            b.mv(&ly.key, r.emb, ed as u32, r.pk, hcd as u32, n, false, None);
            b.mv(&ly.value, r.emb, ed as u32, r.pv, d as u32, n, false, None);
            b.read(r.pk, n * hcd);
            b.read(r.pv, n * d);
            let out = b.run()?;
            key[s * hcd..(s + n) * hcd].copy_from_slice(&out[0]);
            val[s * d..(s + n) * d].copy_from_slice(&out[1]);
        }
        // gate per stream, gated value and its norm (the conv's input)
        let mut gated = vec![0f32; c * hcd];
        let mut normed = vec![0f32; c * hcd];
        gated.par_chunks_mut(hcd).zip(normed.par_chunks_mut(hcd)).enumerate().for_each(|(i, (gr, nr))| {
            for s in 0..h {
                let mut k = key[i * hcd + s * d..i * hcd + (s + 1) * d].to_vec();
                rms(&mut k, &ly.nk[s * d..(s + 1) * d], hp.eps);
                let mut q = x[i * hcd + s * d..i * hcd + (s + 1) * d].to_vec();
                rms(&mut q, &ly.nq[s * d..(s + 1) * d], hp.eps);
                let sc = cpu_dot(&k, &q) / (d as f32).sqrt();
                let mag = sc.abs().max(1e-6).sqrt();
                let gt = sigmoid(if sc > 0.0 { mag } else if sc < 0.0 { -mag } else { 0.0 });
                let g = &mut gr[s * d..(s + 1) * d];
                for (o, v) in g.iter_mut().zip(&val[i * d..(i + 1) * d]) {
                    *o = v * gt;
                }
                let nn = &mut nr[s * d..(s + 1) * d];
                nn.copy_from_slice(g);
                rms(nn, &ly.nc[s * d..(s + 1) * d], hp.eps);
            }
        });
        // depthwise causal conv dilated by the n-gram size over the history
        let hist = (p.taps - 1) * p.ngram;
        let mut ext = st.ple.clone();
        ext.extend_from_slice(&normed);
        x.par_chunks_mut(hcd).enumerate().for_each(|(i, xr)| {
            for ch in 0..hcd {
                let mut s = 0f32;
                for k in 0..p.taps {
                    let row = hist + i - (p.taps - 1 - k) * p.ngram;
                    s += ly.conv[ch * p.taps + k] * ext[row * hcd + ch];
                }
                xr[ch] += gated[i * hcd + ch] + silu(s);
            }
        });
        let n_ext = ext.len() / hcd;
        st.ple.copy_from_slice(&ext[(n_ext - hist) * hcd..]);
        tick(0, &mut tm);
        Ok(())
    }

    /// Layer `l`'s token mixer (with its hyper-connection mix) added into
    /// the residuals, and the FFN's inputs. A DeltaNet layer runs whole on
    /// the GPU, one batch per sub-batch; an attention layer projects on the
    /// GPU, attends on the CPU, and goes on in a second batch.
    fn mixer(&self, st: &mut State, l: usize, pos0: usize, c: usize, x: &mut [f32], pend: Option<Pend>) -> Result<FfnIn> {
        let mut tm = Instant::now();
        let hp = &self.hp;
        let ly = &self.layers[l];
        let r = &self.r;
        let (d, h, ne) = (hp.d, hp.hc, hp.ne);
        let hcd = h * d;
        let ahead = c <= SMALL && l + 1 < hp.n_layer;
        let mut fi = FfnIn { f: vec![0f32; c * d], lg: vec![0f32; c * ne], y: vec![0f32; c * d], inj: vec![0f32; c * h], ahead: ahead.then(|| vec![0f32; c * ne]) };
        match &ly.mix {
            Mixer::Lin { qkv, z, beta, alpha, out, conv_w, dt_w, a_w, norm_w, idx } => {
                let (dk, nkh, nvh) = (hp.dk as u32, hp.nkh as u32, hp.nvh as u32);
                let ch = 2 * nkh * dk + nvh * dk;
                let vw = nvh * dk;
                let (gs, gc) = (r.gst + *idx as u32 * nvh * dk * dk, r.gcv + *idx as u32 * (hp.conv as u32 - 1) * ch);
                for s in (0..c).step_by(TB) {
                    let n = (c - s).min(TB);
                    let nn = n as u32;
                    let mut b = self.gpu.batch();
                    self.res_in(&mut b, x, s, n, &pend);
                    self.hc_ops(&mut b, &ly.hca, n);
                    if let (1, Some(m)) = (n, &ly.mvs_mix) {
                        m.run(&mut b);
                    } else {
                        b.mv(qkv, r.mix, d as u32, r.p1, ch, n, false, None);
                        b.mv(z, r.mix, d as u32, r.p2, vw, n, false, None);
                        b.mv(beta, r.mix, d as u32, r.p3, nvh, n, false, None);
                        b.mv(alpha, r.mix, d as u32, r.p4, nvh, n, false, None);
                    }
                    b.kernel("q4x-conv", || CONV.to_string(), qkv.buf, &[ch, nn, r.p1, r.cv, gc, *conv_w], [ch.div_ceil(256), 1, 1]);
                    b.kernel("q4x-headnorm", || HEADNORM.to_string(), 0, &[dk, r.cv, ch, 2 * nkh, nn * 2 * nkh, hp.eps.to_bits(), 0, NONE, NONE, 0, r.cv, ch], [nn * 2 * nkh, 1, 1]);
                    let ga = [nn, r.cv, ch, r.p4, r.p3, nvh, nkh, gs, r.o, *dt_w, *a_w, (1.0 / (dk as f32).sqrt()).to_bits()];
                    b.kernel("q4x-gdn", || GDN.to_string(), qkv.buf, &ga, [nvh * 4, 1, 1]);
                    b.kernel("q4x-headnorm", || HEADNORM.to_string(), qkv.buf, &[dk, r.o, vw, nvh, nn * nvh, hp.eps.to_bits(), 1, *norm_w, r.p2, vw, r.o, vw], [nn * nvh, 1, 1]);
                    b.mv(out, r.o, vw, r.y, d as u32, n, false, None);
                    self.ffn_ops(&mut b, l, n, ahead);
                    let o = b.run()?;
                    self.take_ffn_in(o, s, n, x, &mut fi);
                }
                tick(2, &mut tm);
            }
            Mixer::Att { q, k, v, o: om, iq, ik, idx, .. } => {
                let mats = [q, k, v, iq, ik];
                let widths: Vec<(u32, usize)> = vec![(r.p1, q.rows), (r.p2, k.rows), (r.p3, v.rows), (r.p4, iq.rows), (r.p5, ik.rows)];
                let mut outs: Vec<Vec<f32>> = widths.iter().map(|&(_, w)| vec![0f32; c * w]).collect();
                let mut inj = vec![0f32; c * h];
                for s in (0..c).step_by(TB) {
                    let n = (c - s).min(TB);
                    let mut b = self.gpu.batch();
                    self.res_in(&mut b, x, s, n, &pend);
                    self.hc_ops(&mut b, &ly.hca, n);
                    if let (1, Some(m)) = (n, &ly.mvs_mix) {
                        m.run(&mut b);
                    } else {
                        for (m, &(at, w)) in mats.iter().zip(&widths) {
                            b.mv(m, r.mix, d as u32, at, w as u32, n, false, None);
                        }
                    }
                    for &(at, w) in &widths {
                        b.read(at, n * w);
                    }
                    b.read(r.inj, n * h);
                    let mut out = b.run()?;
                    inj[s * h..(s + n) * h].copy_from_slice(&out.pop().unwrap());
                    for (o, (vv, &(_, w))) in outs.iter_mut().zip(out.into_iter().zip(&widths)) {
                        o[s * w..(s + n) * w].copy_from_slice(&vv);
                    }
                }
                tick(1, &mut tm);
                let o = self.attention(&mut st.att[*idx], ly, &outs, pos0, c, st.rp.as_ref());
                tick(3, &mut tm);
                let ow = om.cols;
                for s in (0..c).step_by(TB) {
                    let n = (c - s).min(TB);
                    let mut b = self.gpu.batch();
                    // the residuals on the GPU are newer than x when the last
                    // FFN was added in there (the first batch left them so)
                    if pend.is_none() {
                        b.upload(r.res, &x[s * hcd..(s + n) * hcd]);
                    }
                    b.upload(r.inj, &inj[s * h..(s + n) * h]);
                    b.upload(r.o, &o[s * ow..(s + n) * ow]);
                    b.mv(om, r.o, ow as u32, r.y, d as u32, n, false, None);
                    self.ffn_ops(&mut b, l, n, ahead);
                    let out = b.run()?;
                    self.take_ffn_in(out, s, n, x, &mut fi);
                }
                tick(4, &mut tm);
            }
        }
        Ok(fi)
    }

    /// The residuals for tokens s..s+n into r.res: from x, or (one token) by
    /// adding the last layer's pending FFN into those already there.
    fn res_in(&self, b: &mut Batch, x: &[f32], s: usize, n: usize, pend: &Option<Pend>) {
        let (r, hp) = (&self.r, &self.hp);
        let (d, h) = (hp.d as u32, hp.hc as u32);
        match pend {
            Some(p) => {
                b.upload(r.y, &p.y);
                b.kernel("q4x-ffncomb", || FFNCOMB.to_string(), 0, &[d, h, r.res, r.y, r.ey, p.rows, r.inj], [d.div_ceil(256), 1, 1]);
            }
            None => b.upload(r.res, &x[s * (h * d) as usize..(s + n) * (h * d) as usize]),
        }
    }

    /// Full attention over the chunk (keys stored first), restricted past
    /// `topk` positions to the indexer's best blocks plus the newest
    /// positions; returns [token][heads * size], gated.
    fn attention(&self, kv: &mut Att, ly: &Layer, p: &[Vec<f32>], pos0: usize, c: usize, rp: Option<&(usize, Vec<[u32; 3]>)>) -> Vec<f32> {
        let hp = &self.hp;
        let Mixer::Att { qn, kn, iqn, ikn, .. } = &ly.mix else { unreachable!() };
        let (nh, nkv, hd, ih, ihd, kp) = (hp.nh, hp.nkv, hp.hd, hp.ih, hp.ihd, hp.kpool);
        let (qf, kr, vr, iqr, ikr) = (&p[0], &p[1], &p[2], &p[3], &p[4]);
        // rotary angles: by position, or (pictures in the cache) M-RoPE's
        let cs_at = |q: usize| match rp {
            Some((from, v)) => self.rope_cs3(v[q - from]),
            None => self.rope_cs(q),
        };
        let css: Vec<Vec<(f32, f32)>> = (0..c).into_par_iter().map(|t| cs_at(pos0 + t)).collect();
        // keys, values, raw indexer keys into the cache
        let kvw = nkv * hd;
        if kv.k.len() != pos0 * kvw {
            kv.k.truncate(pos0 * kvw);
            kv.v.truncate(pos0 * kvw);
            kv.ik.truncate(pos0 * ihd);
            kv.pk.truncate(pos0 / kp * ihd);
        }
        for t in 0..c {
            let mut k = kr[t * kvw..(t + 1) * kvw].to_vec();
            for g in 0..nkv {
                let kg = &mut k[g * hd..(g + 1) * hd];
                rms(kg, kn, hp.eps);
                self.rope(kg, &css[t]);
            }
            kv.k.extend(k.iter().map(|&v| f32_to_f16(v)));
            kv.v.extend(vr[t * kvw..(t + 1) * kvw].iter().map(|&v| f32_to_f16(v)));
            kv.ik.extend(ikr[t * ihd..(t + 1) * ihd].iter().map(|&v| f32_to_f16(v)));
            let pos = pos0 + t;
            if (pos + 1) % kp == 0 {
                // a block is complete: its pooled key (mean of the raw keys,
                // normed, rotated to its first position)
                let b = pos / kp;
                let mut m = vec![0f32; ihd];
                let mut tmp = vec![0f32; ihd];
                for q in 0..kp {
                    h2f(&kv.ik[(b * kp + q) * ihd..(b * kp + q + 1) * ihd], &mut tmp);
                    m.iter_mut().zip(&tmp).for_each(|(a, v)| *a += v);
                }
                m.iter_mut().for_each(|v| *v /= kp as f32);
                rms(&mut m, ikn, hp.eps);
                self.rope(&mut m, &cs_at(b * kp));
                kv.pk.extend(m.iter().map(|&v| f32_to_f16(v)));
            }
        }
        let kv = &*kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let iscale = 1.0 / (ihd as f32).sqrt();
        let top_pool = hp.topk / kp;
        // the cache once in f32 for the whole pass
        let np = pos0 + c;
        let mut kf = vec![0f32; np * kvw];
        let mut vf = vec![0f32; np * kvw];
        kf.par_chunks_mut(kvw).zip(vf.par_chunks_mut(kvw)).enumerate().for_each(|(q, (kr, vr))| {
            h2f(&kv.k[q * kvw..(q + 1) * kvw], kr);
            h2f(&kv.v[q * kvw..(q + 1) * kvw], vr);
        });
        // the pooled keys once in f32 too
        let mut pkf = vec![0f32; kv.pk.len()];
        if pos0 + c > hp.topk {
            h2f(&kv.pk, &mut pkf);
        }
        // the positions each token sees
        let cells: Vec<Vec<usize>> = (0..c)
            .into_par_iter()
            .map(|t| {
                let pos = pos0 + t;
                let nv = (pos + 1) / kp;
                if nv <= top_pool {
                    return (0..=pos).collect();
                }
                let mut iq = iqr[t * ih * ihd..(t + 1) * ih * ihd].to_vec();
                for hh in 0..ih {
                    let q = &mut iq[hh * ihd..(hh + 1) * ihd];
                    rms(q, iqn, hp.eps);
                    self.rope(q, &css[t]);
                }
                let scores: Vec<f32> = (0..nv)
                    .map(|b| {
                        let kb = &pkf[b * ihd..(b + 1) * ihd];
                        let mut s = 0f32;
                        for hh in 0..ih {
                            let v = cpu_dot(&iq[hh * ihd..(hh + 1) * ihd], kb);
                            if v > 0.0 {
                                s += v * iscale;
                            }
                        }
                        s
                    })
                    .collect();
                let mut idx: Vec<usize> = (0..nv).collect();
                idx.select_nth_unstable_by(top_pool - 1, |&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                idx.truncate(top_pool);
                idx.sort_unstable();
                let mut cells: Vec<usize> = idx.iter().flat_map(|&b| b * kp..(b + 1) * kp).collect();
                cells.extend(nv * kp..=pos);
                cells
            })
            .collect();
        // one task per token and key group: the group's heads share each
        // key and value row, so a row is read once for all of them (each
        // head's sums still run in the same order)
        let hpg = nh / nkv;
        // few tokens (replies): smaller tasks, so every core has one
        let hn = if c * nkv >= 24 || hpg % 3 != 0 { hpg } else { 3 };
        let mut o = vec![0f32; c * nh * hd];
        o.par_chunks_mut(hn * hd).enumerate().for_each(|(i, out)| {
            let (t, h0) = (i / (nh / hn), i % (nh / hn) * hn);
            let g = h0 / hpg;
            let cl = &cells[t];
            let qo = |j: usize| t * 2 * nh * hd + (h0 + j) * 2 * hd;
            let mut q = vec![0f32; hn * hd];
            for (j, qj) in q.chunks_mut(hd).enumerate() {
                qj.copy_from_slice(&qf[qo(j)..qo(j) + hd]);
                rms(qj, qn, hp.eps);
                self.rope(qj, &css[t]);
            }
            let mut sc = vec![0f32; cl.len() * hn];
            let mut mx = vec![f32::NEG_INFINITY; hn];
            let q3 = hn % 3 == 0;
            for (sr, &cp) in sc.chunks_mut(hn).zip(cl) {
                let kr = &kf[cp * kvw + g * hd..cp * kvw + (g + 1) * hd];
                if q3 {
                    for j in (0..hn).step_by(3) {
                        let d = crate::quant::dot3_f32([&q[j * hd..(j + 1) * hd], &q[(j + 1) * hd..(j + 2) * hd], &q[(j + 2) * hd..(j + 3) * hd]], kr);
                        for u in 0..3 {
                            sr[j + u] = d[u] * scale;
                        }
                    }
                } else {
                    for (j, s) in sr.iter_mut().enumerate() {
                        *s = cpu_dot(&q[j * hd..(j + 1) * hd], kr) * scale;
                    }
                }
                for (m, s) in mx.iter_mut().zip(sr.iter()) {
                    *m = m.max(*s);
                }
            }
            let mut den = vec![0f32; hn];
            out.fill(0.0);
            let mut e = vec![0f32; hn];
            for (sr, &cp) in sc.chunks(hn).zip(cl) {
                let vr = &vf[cp * kvw + g * hd..cp * kvw + (g + 1) * hd];
                for (j, s) in sr.iter().enumerate() {
                    e[j] = (s - mx[j]).exp();
                    den[j] += e[j];
                }
                if q3 {
                    for j in (0..hn).step_by(3) {
                        crate::quant::axpy3_f32(&mut out[j * hd..(j + 3) * hd], hd, [e[j], e[j + 1], e[j + 2]], vr);
                    }
                } else {
                    for (j, &ej) in e.iter().enumerate() {
                        for (a, v) in out[j * hd..(j + 1) * hd].iter_mut().zip(vr) {
                            *a += ej * v;
                        }
                    }
                }
            }
            for (j, oj) in out.chunks_mut(hd).enumerate() {
                for (a, gv) in oj.iter_mut().zip(&qf[qo(j) + hd..qo(j) + 2 * hd]) {
                    *a = *a / den[j] * sigmoid(*gv);
                }
            }
        });
        o
    }

    /// Softmax top-k of the router logits, weights renormalized.
    fn route(&self, lg: &[f32]) -> ([u32; 16], [f32; 16]) {
        let (ne, k) = (self.hp.ne, self.hp.k);
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
        idx[..k].sort_unstable_by(|&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
        let mx = lg[idx[0]];
        let mut ids = [0u32; 16];
        let mut w = [0f32; 16];
        let mut sum = 0f32;
        for j in 0..k {
            ids[j] = idx[j] as u32;
            w[j] = (lg[idx[j]] - mx).exp();
            sum += w[j];
        }
        for v in w[..k].iter_mut() {
            *v /= sum;
        }
        (ids, w)
    }

    /// The batch tail every layer shares: the token mixer's output (r.y)
    /// spread into the residuals (r.res, weights r.inj), the FFN's mix, its
    /// router logits (and the next layer's, for read-ahead) and the shared
    /// expert; reads back what `take_ffn_in` takes.
    fn ffn_ops(&self, b: &mut Batch, l: usize, n: usize, ahead: bool) {
        let (hp, r, ly) = (&self.hp, &self.r, &self.layers[l]);
        let (d, ff, ne, h) = (hp.d, hp.ff, hp.ne, hp.hc);
        b.kernel("q4x-hccomb", || HCCOMB.to_string(), 0, &[d as u32, h as u32, r.res, r.y, r.inj, n as u32], [(d as u32).div_ceil(256), (n * h) as u32, 1]);
        self.hc_ops(b, &ly.hcf, n);
        let one = n == 1 && ly.mvs_ffn.is_some();
        if one {
            // router, shared expert's gate+up, its gate (and the next router)
            ly.mvs_ffn.as_ref().unwrap().run(b);
        } else {
            b.mv(&ly.router, r.mix, d as u32, r.rl, ne as u32, n, false, None);
            b.mv(&ly.sgu, r.mix, d as u32, r.sgu, (2 * ff) as u32, n, false, None);
        }
        b.swiglu(ff, r.sgu, r.sgu + ff as u32, r.sh, (2 * ff) as u32, ff as u32, None, n, 0.0);
        b.mv(&ly.sd, r.sh, ff as u32, r.sy, d as u32, n, false, None);
        if !one {
            b.mv(&ly.sgate, r.mix, d as u32, r.sg, 1, n, false, None);
        }
        b.read(r.res, n * h * d);
        b.read(r.mix, n * d);
        b.read(r.rl, n * ne);
        b.read(r.sy, n * d);
        b.read(r.sg, n);
        b.read(r.inj, n * h);
        if ahead {
            if !(one && ly.ffn_has_next) {
                b.mv(&self.layers[l + 1].router, r.mix, d as u32, r.rl2, ne as u32, n, false, None);
            }
            b.read(r.rl2, n * ne);
        }
    }

    /// Takes `ffn_ops`' readbacks for tokens s..s+n into the residuals and `fi`.
    fn take_ffn_in(&self, out: Vec<Vec<f32>>, s: usize, n: usize, x: &mut [f32], fi: &mut FfnIn) {
        let (d, ne, h) = (self.hp.d, self.hp.ne, self.hp.hc);
        let mut out = out.into_iter();
        let mut next = || out.next().unwrap();
        x[s * h * d..(s + n) * h * d].copy_from_slice(&next());
        fi.f[s * d..(s + n) * d].copy_from_slice(&next());
        fi.lg[s * ne..(s + n) * ne].copy_from_slice(&next());
        let sy = next();
        let sg = next();
        for i in 0..n {
            let g = sigmoid(sg[i]);
            for (a, v) in fi.y[(s + i) * d..(s + i + 1) * d].iter_mut().zip(&sy[i * d..(i + 1) * d]) {
                *a = v * g;
            }
        }
        fi.inj[s * h..(s + n) * h].copy_from_slice(&next());
        if let Some(a) = fi.ahead.as_mut() {
            a[s * ne..(s + n) * ne].copy_from_slice(&next());
        }
    }

    fn ffn(&self, st: &mut State, l: usize, toks: &[u32], x: &mut [f32], fi: FfnIn) -> Result<Option<Pend>> {
        let hp = &self.hp;
        let r = &self.r;
        let (d, ff, c, k, ne, h) = (hp.d, hp.ff, toks.len(), hp.k, hp.ne, hp.hc);
        let ex = self.ex[l];
        let mut tm = Instant::now();
        let FfnIn { f, lg, mut y, inj, ahead } = fi;
        let routes: Vec<([u32; 16], [f32; 16])> = lg.par_chunks(ne).map(|row| self.route(row)).collect();
        if let Some(tr) = st.trace.as_mut() {
            for (i, (ids, w)) in routes.iter().enumerate() {
                if tr.len() <= i {
                    tr.resize(i + 1, Vec::new());
                }
                tr[i].push((0..k).map(|j| (ids[j], w[j])).collect());
            }
        }
        let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        let mut keys = Vec::new();
        for (i, (ids, w)) in routes.iter().enumerate() {
            for j in 0..k {
                let e = ids[j] as usize;
                if rows[e].is_empty() {
                    keys.push((l * ne + e) as u32);
                }
                rows[e].push((i, w[j]));
                self.usage[l * ne + e].fetch_add(1, Ordering::Relaxed);
            }
        }
        let touched: Vec<u32> = routes.iter().flat_map(|(ids, _)| ids[..k].iter().map(|&e| (l * ne + e as usize) as u32)).collect();
        self.store.touch(&touched, if c > SMALL { (64.0 / c as f32).min(1.0) } else { 1.0 });
        let small = c <= SMALL;
        let (mut on_vram, mut on_cpu, mut on_stage) = (Vec::new(), Vec::new(), Vec::new());
        {
            let v = self.vram.lock().unwrap();
            for &key in &keys {
                let e = key as usize % ne;
                if let Some(&s) = v.slot_of.get(&key) {
                    on_vram.push((key, v.slots[s]));
                } else if rows[e].len() <= if small { CPU_MAX } else { PF_CPU_ROWS } {
                    on_cpu.push(key);
                } else {
                    on_stage.push(key);
                }
            }
        }
        self.store.cancel_guesses(l, &keys);
        if small {
            EXP_COUNT[0].fetch_add(on_vram.len() as u64, Ordering::Relaxed);
            EXP_COUNT[1].fetch_add(on_cpu.len() as u64, Ordering::Relaxed);
            EXP_COUNT[2].fetch_add(on_cpu.iter().filter(|&&k| !self.store.has(k)).count() as u64, Ordering::Relaxed);
        }
        // a layer's experts have only their slot size's share of the RAM
        // slots: the CPU's (held for the whole layer) and the staging window
        // stay within it, the CPU's beyond a quarter going through staging
        let room = self.store.class_slots(l);
        if on_cpu.len() > room / 4 {
            on_stage.extend(on_cpu.split_off(room / 4));
        }
        // experts a device holds (small passes, not in RAM here): sent there
        // now, collected after this PC's own; any that fail run here instead
        let mut far: Vec<(std::sync::Arc<crate::bridge::Link>, Vec<u32>, Instant)> = Vec::new();
        if let (true, Some(r)) = (small, &self.remote) {
            let mut by: Vec<(std::sync::Arc<crate::bridge::Link>, Vec<u32>)> = Vec::new();
            on_cpu.retain(|&key| {
                if self.store.has(key) {
                    return true;
                }
                let Some(link) = r.holder(key) else { return true };
                match by.iter_mut().find(|(lk, _)| std::sync::Arc::ptr_eq(lk, &link)) {
                    Some((_, ks)) => ks.push(key),
                    None => by.push((link, vec![key])),
                }
                false
            });
            for (link, ks) in by {
                let jobs: Vec<(u32, f32, Vec<(u16, f32)>)> = ks.iter().map(|&k| (k, 0.0, rows[k as usize % ne].iter().map(|&(i, w)| (i as u16, w)).collect())).collect();
                let t = Instant::now();
                match r.start(&link, c, &f, &jobs) {
                    Ok(()) => far.push((link, ks, t)),
                    Err(e) => {
                        r.fail(&link, &format!("{e:#}"));
                        on_cpu.extend(ks);
                    }
                }
            }
        }
        let cpu_pins = self.store.request(&on_cpu)?;
        let win = (room / 2).clamp(8, 128);
        let mut stage_pins: std::collections::VecDeque<(u32, Pin)> = std::collections::VecDeque::new();
        let mut next_stage = 0usize;
        let refill = |stage_pins: &mut std::collections::VecDeque<(u32, Pin)>, next_stage: &mut usize| -> Result<()> {
            if *next_stage < on_stage.len() && stage_pins.len() < win {
                let take = (win - stage_pins.len()).min(on_stage.len() - *next_stage);
                let ks = &on_stage[*next_stage..*next_stage + take];
                for (kk, p) in ks.iter().zip(self.store.request(ks)?) {
                    stage_pins.push_back((*kk, p));
                }
                *next_stage += take;
            }
            Ok(())
        };
        refill(&mut stage_pins, &mut next_stage)?;
        // read-ahead: the next layer's likeliest experts from this FFN input
        if let (true, Some(a)) = (small, &ahead) {
            self.guess(l + 1, a, c);
        }
        self.store.mark(l * 4 + 2);
        tick(5, &mut tm);

        let mat_of = |(arena, base): (usize, u32)| -> (Mat, Mat) {
            let gu = Mat { buf: arena, base, ty: ex.ty[0], rows: 2 * ff, cols: d, wpr: ex.wpr_g };
            let dn = Mat { buf: arena, base: base + (2 * ff) as u32 * ex.wpr_g, ty: ex.ty[2], rows: d, cols: ff, wpr: ex.wpr_d };
            (gu, dn)
        };
        let gather = |b: &mut Batch, list: &[(u32, (usize, u32), usize, usize)], read: bool| -> Vec<(usize, u32, usize, usize)> {
            let mut xe = Vec::new();
            let mut we = Vec::new();
            let mut at = Vec::new();
            for &(key, slot, lo, nr) in list {
                let e = key as usize % ne;
                let (gu, dn) = mat_of(slot);
                let r0 = xe.len() / d;
                for &(i, w) in &rows[e][lo..lo + nr] {
                    xe.extend_from_slice(&f[i * d..(i + 1) * d]);
                    we.push(w);
                }
                b.mv(&gu, r.xe + (r0 * d) as u32, d as u32, r.egu + (r0 * 2 * ff) as u32, (2 * ff) as u32, nr, false, None);
                b.swiglu(ff, r.egu + (r0 * 2 * ff) as u32, r.egu + (r0 * 2 * ff + ff) as u32, r.eh + (r0 * ff) as u32, (2 * ff) as u32, ff as u32, Some(r.ew + r0 as u32), nr, 0.0);
                b.mv(&dn, r.eh + (r0 * ff) as u32, ff as u32, r.ey + (r0 * d) as u32, d as u32, nr, false, None);
                at.push((r0, key, lo, nr));
            }
            b.upload(r.xe, &xe);
            b.upload(r.ew, &we);
            if read && !xe.is_empty() {
                b.read(r.ey, xe.len());
            }
            at
        };
        // prompt passes: the tokens' FFN inputs go to the GPU once (r.fx)
        // and the experts' outputs add up there (r.fy); a batch gathers its
        // rows by index, so nothing goes back and forth per batch and no
        // batch is waited on
        let gather_big = |b: &mut Batch, list: &[(u32, (usize, u32), usize, usize)]| {
            let (mut idx, mut we) = (Vec::new(), Vec::new());
            for &(key, _, lo, nr) in list {
                for &(i, w) in &rows[key as usize % ne][lo..lo + nr] {
                    idx.push(f32::from_bits(i as u32));
                    we.push(w);
                }
            }
            let n = idx.len() as u32;
            if n == 0 {
                return;
            }
            b.upload(r.eidx, &idx);
            b.upload(r.ew, &we);
            b.kernel("q4x-gather", || GATHER.to_string(), 0, &[d as u32, r.fx, r.xe, r.eidx, n], [(d as u32).div_ceil(256), n, 1]);
            let mut r0 = 0usize;
            for &(_, slot, _, nr) in list {
                let (gu, dn) = mat_of(slot);
                b.mv(&gu, r.xe + (r0 * d) as u32, d as u32, r.egu + (r0 * 2 * ff) as u32, (2 * ff) as u32, nr, false, None);
                b.swiglu(ff, r.egu + (r0 * 2 * ff) as u32, r.egu + (r0 * 2 * ff + ff) as u32, r.eh + (r0 * ff) as u32, (2 * ff) as u32, ff as u32, Some(r.ew + r0 as u32), nr, 0.0);
                b.mv(&dn, r.eh + (r0 * ff) as u32, ff as u32, r.ey + (r0 * d) as u32, d as u32, nr, false, None);
                r0 += nr;
            }
            b.kernel("q4x-scatter", || SCATTER.to_string(), 0, &[d as u32, r.fy, r.ey, r.eidx, n], [(d as u32).div_ceil(256), 1, 1]);
        };
        let scatter = |y: &mut [f32], ey: &[f32], at: &[(usize, u32, usize, usize)]| {
            for &(r0, key, lo, nr) in at {
                for (j, &(i, _)) in rows[key as usize % ne][lo..lo + nr].iter().enumerate() {
                    let src = &ey[(r0 + j) * d..(r0 + j + 1) * d];
                    y[i * d..(i + 1) * d].iter_mut().zip(src).for_each(|(a, b)| *a += b);
                }
            }
        };
        let pack = |list: &[(u32, (usize, u32))]| -> Vec<Vec<(u32, (usize, u32), usize, usize)>> {
            let mut groups = Vec::new();
            let mut cur = Vec::new();
            let mut nrow = 0;
            for &(key, slot) in list {
                let n = rows[key as usize % ne].len();
                let mut lo = 0;
                while lo < n {
                    if nrow == EROWS || (nrow + (n - lo).min(EROWS) > EROWS && n - lo <= EROWS) {
                        groups.push(std::mem::take(&mut cur));
                        nrow = 0;
                    }
                    let take = (n - lo).min(EROWS - nrow);
                    cur.push((key, slot, lo, take));
                    nrow += take;
                    lo += take;
                }
            }
            if !cur.is_empty() {
                groups.push(cur);
            }
            groups
        };
        // one token: the VRAM experts' batch goes off unwaited, the CPU's
        // experts run meanwhile, and the next layer's batch adds the lot in
        let defer = c == 1
            && on_stage.is_empty()
            && on_vram.len() <= EROWS
            && l + 1 < hp.n_layer
            && self.layers[l + 1].ple.is_none()
            && std::env::var_os("PELICA_NO_DEFER").is_none();
        if defer {
            let mut nrows = 0;
            if !on_vram.is_empty() {
                let mut tb = Instant::now();
                let list: Vec<(u32, (usize, u32), usize, usize)> = on_vram.iter().map(|&(key, slot)| (key, slot, 0, 1)).collect();
                let mut b = self.gpu.batch();
                nrows = gather(&mut b, &list, false).len() as u32;
                drop(b.submit());
                tick(11, &mut tb);
            }
            if !cpu_pins.is_empty() {
                let mut t2 = Instant::now();
                self.cpu_experts(l, &cpu_pins, &on_cpu, &rows, &f, &mut y)?;
                tick(6, &mut t2);
            }
            self.far_in(l, &far, &rows, &f, &mut y)?;
            self.store.mark(l * 4 + 3);
            tick(7, &mut tm);
            return Ok(Some(Pend { y, rows: nrows }));
        }
        let mut y_cpu = vec![0f32; c * d];
        std::thread::scope(|sc| -> Result<()> {
            let cpu_job = if !small && !cpu_pins.is_empty() { Some(sc.spawn(|| self.cpu_experts(l, &cpu_pins, &on_cpu, &rows, &f, &mut y_cpu))) } else { None };
            let groups = pack(&on_vram);
            if !small {
                let mut b = self.gpu.batch();
                b.upload(r.fx, &f);
                b.kernel("q4x-fill", || FILL.to_string(), 0, &[(c * d) as u32, r.fy, r.fy, 1], [((c * d) as u32).div_ceil(256).min(4096), 1, 1]);
                for gr in &groups {
                    gather_big(&mut b, gr);
                    drop(std::mem::replace(&mut b, self.gpu.batch()).submit());
                }
                drop(b.submit());
            }
            let mut first = true;
            for gr in groups.iter().filter(|_| small) {
                let mut tb = Instant::now();
                let mut b = self.gpu.batch();
                let at = gather(&mut b, gr, true);
                let pending = b.submit();
                tick(11, &mut tb);
                if first && small && !cpu_pins.is_empty() {
                    let mut t2 = Instant::now();
                    self.cpu_experts(l, &cpu_pins, &on_cpu, &rows, &f, &mut y)?;
                    tick(6, &mut t2);
                }
                first = false;
                let mut tw = Instant::now();
                let got = pending.wait()?.pop();
                tick(12, &mut tw);
                if let Some(ey) = got {
                    scatter(&mut y, &ey, &at);
                }
            }
            if groups.is_empty() && small && !cpu_pins.is_empty() {
                let mut t2 = Instant::now();
                self.cpu_experts(l, &cpu_pins, &on_cpu, &rows, &f, &mut y)?;
                tick(6, &mut t2);
            }
            self.store.mark(l * 4 + 3);
            tick(7, &mut tm);
            while !stage_pins.is_empty() {
                let mut list = Vec::new();
                let mut pins = Vec::new();
                let mut nrow = 0;
                while let Some((key, _)) = stage_pins.front() {
                    let n = rows[*key as usize % ne].len();
                    if list.len() == self.stage.len() || (nrow + n > EROWS && !list.is_empty()) {
                        break;
                    }
                    let (key, pin) = stage_pins.pop_front().unwrap();
                    let slot = self.stage[list.len()];
                    list.push((key, slot));
                    pins.push(pin);
                    nrow += n;
                }
                refill(&mut stage_pins, &mut next_stage)?;
                // the batch's experts as they arrive, a few copies at once
                // (each to its own slot, all before the batch is submitted)
                let mut ts = Instant::now();
                let next = std::sync::atomic::AtomicUsize::new(0);
                let (list, pins, next) = (&list, &pins, &next);
                std::thread::scope(|us| -> Result<()> {
                    let hs: Vec<_> = (0..if small { 0 } else { UPLOADERS.min(pins.len()) })
                        .map(|_| {
                            us.spawn(move || -> Result<()> {
                                loop {
                                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let Some(pin) = pins.get(i) else { return Ok(()) };
                                    self.store.wait(pin)?;
                                    Self::upload_expert(&self.gpu, &ex, list[i].1, pin);
                                }
                            })
                        })
                        .collect();
                    for h in hs {
                        h.join().map_err(|_| anyhow::anyhow!("an upload thread failed"))??;
                    }
                    if small {
                        for (pin, &(_, slot)) in pins.iter().zip(list) {
                            self.store.wait(pin)?;
                            Self::upload_expert(&self.gpu, &ex, slot, pin);
                        }
                    }
                    Ok(())
                })?;
                tick(14, &mut ts);
                for gr in pack(&list) {
                    let mut ts = Instant::now();
                    let mut b = self.gpu.batch();
                    if small {
                        let at = gather(&mut b, &gr, true);
                        let ey = b.run()?.pop().unwrap_or_default();
                        scatter(&mut y, &ey, &at);
                    } else {
                        gather_big(&mut b, &gr);
                        drop(b.submit());
                    }
                    tick(15, &mut ts);
                }
            }
            if !small {
                // the GPU experts' sum, once they've all run
                let mut ts = Instant::now();
                let mut b = self.gpu.batch();
                b.read(r.fy, c * d);
                let fy = b.run()?.pop().unwrap_or_default();
                y.iter_mut().zip(&fy).for_each(|(a, b)| *a += b);
                tick(15, &mut ts);
            }
            if let Some(j) = cpu_job {
                j.join().map_err(|_| anyhow::anyhow!("the CPU expert thread failed"))??;
            }
            Ok(())
        })?;
        y.iter_mut().zip(&y_cpu).for_each(|(a, b)| *a += b);
        self.far_in(l, &far, &rows, &f, &mut y)?;
        tick(8, &mut tm);
        combine(x, &y, &inj, d, h);
        Ok(None)
    }

    /// The devices' outputs (they worked while this PC did), added into y;
    /// experts a device no longer held, or all of a device that failed, are
    /// worked here.
    fn far_in(&self, l: usize, far: &[(std::sync::Arc<crate::bridge::Link>, Vec<u32>, Instant)], rows: &[Vec<(usize, f32)>], f: &[f32], y: &mut [f32]) -> Result<()> {
        let Some(r) = &self.remote else { return Ok(()) };
        for (link, ks, t) in far {
            let here = match r.finish(link, *t, ks.len()) {
                Ok((yr, missing)) => {
                    y.iter_mut().zip(&yr).for_each(|(a, b)| *a += b);
                    missing
                }
                Err(e) => {
                    r.fail(link, &format!("{e:#}"));
                    ks.clone()
                }
            };
            if !here.is_empty() {
                let pins = self.store.request(&here)?;
                self.cpu_experts(l, &pins, &here, rows, f, y)?;
            }
            FAR.fetch_add(ks.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Tells the devices which experts to hold (by heat, after those VRAM and
    /// RAM hold), and frees the RAM copies of experts a device now holds.
    fn remote_want(&self) {
        let Some(r) = &self.remote else { return };
        let held = r.held();
        if !held.is_empty() {
            self.store.forget(&held);
        }
        if r.capacity() == 0 {
            return;
        }
        let heat = self.store.heat();
        let v = self.vram.lock().unwrap();
        let mut order: Vec<u32> = (0..heat.len() as u32).filter(|k| heat[*k as usize] > 0.0 && !v.slot_of.contains_key(k)).collect();
        drop(v);
        order.sort_by(|a, b| heat[*b as usize].total_cmp(&heat[*a as usize]).then(a.cmp(b)));
        r.set_want(order.into_iter().skip(self.store.slots()).take(12_000).collect());
    }

    /// Experts in RAM for a few tokens each, on the CPU, added into y.
    fn cpu_experts(&self, l: usize, pins: &[Pin], keys: &[u32], rows: &[Vec<(usize, f32)>], f: &[f32], y: &mut [f32]) -> Result<()> {
        let (d, ff, ne) = (self.hp.d, self.hp.ff, self.hp.ne);
        let tys = self.ex[l].ty;
        let mut qx: std::collections::HashMap<usize, Q8> = std::collections::HashMap::new();
        for &key in keys {
            for &(i, _) in &rows[key as usize % ne] {
                qx.entry(i).or_insert_with(|| Q8::new(&f[i * d..(i + 1) * d]));
            }
        }
        let mut done = vec![false; pins.len()];
        let mut outs: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); pins.len()];
        let mut left = pins.len();
        while left > 0 {
            let mut tw = Instant::now();
            let n0 = self.store.wait_any(pins, &done)?;
            tick(10, &mut tw);
            let mut batch = vec![n0];
            batch.extend((0..pins.len()).filter(|&i| i != n0 && !done[i] && self.store.is_ready(&pins[i])));
            for &n in &batch {
                done[n] = true;
            }
            left -= batch.len();
            let one = |n: usize| -> Vec<(usize, Vec<f32>)> {
                let (pin, key) = (&pins[n], keys[n]);
                let toks = &rows[key as usize % ne];
                let xs: Vec<&Q8> = toks.iter().map(|(i, _)| &qx[i]).collect();
                let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
                let o = cpu::expert(tys, pin.piece(0), pin.piece(1), pin.piece(2), d, ff, &xs, &wts);
                toks.iter().enumerate().map(|(t, &(i, _))| (i, o[t * d..(t + 1) * d].to_vec())).collect()
            };
            let res: Vec<Vec<(usize, Vec<f32>)>> = if batch.len() == 1 { vec![one(n0)] } else { batch.par_iter().map(|&n| one(n)).collect() };
            for (&n, v) in batch.iter().zip(res) {
                outs[n] = v;
            }
        }
        for (i, o) in outs.iter().flatten() {
            y[i * d..(i + 1) * d].iter_mut().zip(o).for_each(|(a, b)| *a += b);
        }
        Ok(())
    }

    /// Starts reading the experts layer `l` will likely want, from its
    /// router applied to the layer before's FFN input (logits `lg`), unless
    /// in VRAM.
    fn guess(&self, l: usize, lg: &[f32], c: usize) {
        if std::env::var("PELICA_LOOKAHEAD").is_ok_and(|v| v == "0") {
            return;
        }
        let ne = self.hp.ne;
        let kk = std::env::var("PELICA_EARLY_K").ok().and_then(|v| v.parse().ok()).unwrap_or(6usize).min(self.hp.k);
        let mut keys = Vec::new();
        for i in 0..c {
            let (ids, _) = self.route(&lg[i * ne..(i + 1) * ne]);
            for &e in &ids[..kk] {
                let key = (l * ne + e as usize) as u32;
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        let v = self.vram.lock().unwrap();
        let far = |key: &u32| self.remote.as_ref().is_some_and(|r| r.has(*key));
        keys.retain(|key| !v.slot_of.contains_key(key) && !far(key));
        drop(v);
        self.store.guess(&keys);
    }

    /// Moves up to `budget` experts into VRAM: the hottest by heat that are
    /// not there, in place of the coldest that are.
    fn rebalance(&self, budget: usize) -> Result<usize> {
        let heat = self.store.heat();
        let mut v = self.vram.lock().unwrap();
        let ne = self.hp.ne;
        // per slot size: the hottest of those layers' experts that are not
        // in VRAM, each in place of the coldest there
        let mut add_all: Vec<(u32, usize)> = Vec::new();
        for c in 0..=v.cls.iter().copied().max().map_or(0, |m| m as usize) {
            let slots: Vec<usize> = (0..v.slots.len()).filter(|&s| v.cls[s] as usize == c).collect();
            let mut order: Vec<u32> = (0..heat.len() as u32).filter(|&k| v.lcls[k as usize / ne] as usize == c).collect();
            let n = slots.len().min(order.len());
            if n == 0 {
                continue;
            }
            order.select_nth_unstable_by(n - 1, |&a, &b| heat[b as usize].total_cmp(&heat[a as usize]));
            let want: std::collections::HashSet<u32> = order[..n].iter().copied().collect();
            let mut add: Vec<u32> = order[..n].iter().copied().filter(|k| !v.slot_of.contains_key(k) && heat[*k as usize] > 0.0).collect();
            add.sort_by(|&a, &b| heat[b as usize].total_cmp(&heat[a as usize]));
            let mut free: Vec<usize> = slots.into_iter().filter(|&s| v.key[s] == u32::MAX || !want.contains(&v.key[s])).collect();
            free.sort_by(|&a, &b| {
                let ha = if v.key[a] == u32::MAX { f32::NEG_INFINITY } else { heat[v.key[a] as usize] };
                let hb = if v.key[b] == u32::MAX { f32::NEG_INFINITY } else { heat[v.key[b] as usize] };
                ha.total_cmp(&hb)
            });
            add_all.extend(add.into_iter().zip(free));
        }
        add_all.sort_by(|a, b| heat[b.0 as usize].total_cmp(&heat[a.0 as usize]));
        add_all.truncate(budget);
        let add: Vec<u32> = add_all.iter().map(|a| a.0).collect();
        let mut moved = 0;
        for chunk in add.chunks(16) {
            let pins = self.store.request(chunk)?;
            for (pin, &key) in pins.iter().zip(chunk) {
                self.store.wait(pin)?;
                let s = add_all[moved].1;
                let old = v.key[s];
                if old != u32::MAX {
                    v.slot_of.remove(&old);
                }
                Self::upload_expert(&self.gpu, &self.ex[key as usize / self.hp.ne], v.slots[s], pin);
                v.key[s] = key;
                v.slot_of.insert(key, s);
                moved += 1;
            }
            self.gpu.flush()?;
            drop(pins);
            self.store.forget(chunk);
        }
        v.moved += moved;
        Ok(moved)
    }

    /// Brings the cache back to its first `c` positions if it can: the
    /// recurrent state exists only at the cache's end and where the last
    /// prompt ended. Returns the positions kept.
    fn rollback(&self, st: &mut State, c: usize) -> usize {
        if c >= st.pos {
            return st.pos;
        }
        if let Some(sn) = &st.snap {
            if sn.pos <= c && sn.pos <= st.pos {
                let p = sn.pos;
                let r = &self.r;
                if self.fill(&[(r.sgst, r.gst, self.gst_len(), 0), (r.sgcv, r.gcv, self.gcv_len(), 0)]).is_err() {
                    self.reset(st);
                    return 0;
                }
                st.ple.copy_from_slice(&sn.ple);
                self.truncate(st, p);
                return p;
            }
        }
        self.reset(st);
        0
    }

    fn truncate(&self, st: &mut State, p: usize) {
        let (kvw, ihd, kp) = (self.hp.nkv * self.hp.hd, self.hp.ihd, self.hp.kpool);
        for a in st.att.iter_mut() {
            a.k.truncate(p * kvw);
            a.v.truncate(p * kvw);
            a.ik.truncate(p * ihd);
            a.pk.truncate(p / kp * ihd);
        }
        st.toks.truncate(p);
        st.pos = p;
    }

    pub fn generate(&self, prompt: &[u32], n: usize, reuse: usize, on_token: impl FnMut(u32) -> bool) -> Result<Stats> {
        self.generate_with(prompt, n, reuse, on_token, |_, _| true)
    }

    /// Decoding (greedy, or sampled: set_sampling) of up to `n` tokens after
    /// `prompt`, whose first `reuse` positions may still be in the cache.
    /// Prompt passes report progress to `on_prefill` (done, total); if it
    /// returns false, stops with nothing generated.
    /// Room in RAM for the attention cache of `pos` positions: past what's
    /// set aside, the expert cache gives back memory, a step at a time.
    fn kv_make_room(&self, pos: usize) {
        let room = self.kv_room.load(Ordering::Relaxed);
        if pos <= room {
            return;
        }
        let want = pos.next_multiple_of(KV_STEP).min(self.cfg.ctx.max(pos));
        let freed = self.store.retire(((want - room) * self.kv_per_pos) as u64);
        let got = room + freed as usize / self.kv_per_pos;
        self.kv_room.store(got.max(pos), Ordering::Relaxed);
        {
            eprintln!("[pelica] chat memory for {} positions: {:.2} GB from the expert cache", want, freed as f64 / 1e9);
        }
    }

    pub fn generate_with(&self, prompt: &[u32], n: usize, reuse: usize, mut on_token: impl FnMut(u32) -> bool, mut on_prefill: impl FnMut(usize, usize) -> bool) -> Result<Stats> {
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        if prompt.len() + n > self.cfg.ctx {
            bail!("prompt ({}) + tokens ({n}) exceed the context ({})", prompt.len(), self.cfg.ctx);
        }
        let mut st = self.st.lock().unwrap();
        st.trace = None;
        let t0 = Instant::now();
        self.kv_make_room(prompt.len() + n);
        // what's cached must match the prompt as far as it's kept
        let same = st.toks.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let mut c = self.rollback(&mut st, reuse.min(same).min(prompt.len() - 1));
        // a state saved on disk that goes further than the cache
        if let Some((p, f)) = self.best_saved(prompt, c) {
            match self.load_state(&mut st, &f, prompt) {
                Ok(p) => c = p,
                Err(e) => eprintln!("[pelica] the prompt state saved at {p} won't do ({e:#}); reading the prompt"),
            }
        }
        let total = prompt.len();
        // where the system prompt ends, if that isn't saved yet: a pass ends
        // there and the state is saved, for new chats with the same start
        let mark = self.sys_end(prompt).filter(|&b| b > c && b >= state::MIN_SAVE && b < total && !self.has_saved(&prompt[..b]));
        let nl = self.hp.n_layer;
        let mut next = None;
        while c < total {
            let mut t = (total - c).min(CHUNK);
            if let Some(b) = mark.filter(|&b| c < b && b < c + t) {
                t = b - c;
            }
            let last = c + t == total;
            let mut on_layer = |l: usize| t < SMALL || on_prefill(c + t * (l + 1) / nl, total);
            match self.forward(&mut st, &prompt[c..c + t], c, last, &mut on_layer) {
                Ok(v) => next = v,
                Err(e) if e.is::<Cancelled>() => {
                    return Ok(Stats { prompt: 0, generated: 0, prefill_s: t0.elapsed().as_secs_f64(), decode_s: 0.0, passes: 0, log: Vec::new() });
                }
                Err(e) => {
                    self.reset(&mut st);
                    return Err(e);
                }
            }
            c += t;
            if Some(c) == mark {
                self.save_state(&mut st)?;
            }
        }
        // where this prompt ends: a later prompt that extends it carries on from here
        let r = &self.r;
        self.fill(&[(r.gst, r.sgst, self.gst_len(), 0), (r.gcv, r.sgcv, self.gcv_len(), 0)])?;
        st.snap = Some(Snap { pos: st.pos, ple: st.ple.clone() });
        // and on disk, once enough is new since the last saved state
        if st.pos >= st.saved_pos + state::MIN_SAVE && !self.has_saved(prompt) {
            self.save_state(&mut st)?;
        }
        if total - reuse.min(total) > SMALL {
            self.rebalance(REBAL_PROMPT)?;
            st.since_rebal = 0;
            self.remote_want();
        }
        let t_first = Instant::now();
        let mut tok = next.context("no token")?;
        let mut produced = 0usize;
        let mut passes = 0usize;
        let mut pos = total;
        let drafting = std::env::var_os("PELICA_NO_DRAFT").is_none();
        let mut dr = Drafter::new();
        loop {
            produced += 1;
            if !on_token(tok) || produced >= n {
                break;
            }
            let drafts: Vec<u32> = if drafting { dr.propose(&st.toks, tok).into_iter().take_while(|t| t & IMG == 0).collect() } else { Vec::new() };
            passes += 1;
            if drafts.is_empty() {
                let r = self.forward(&mut st, &[tok], pos, true, &mut |_| true);
                tok = match r {
                    Ok(Some(t)) => t,
                    Ok(None) => unreachable!(),
                    Err(e) => {
                        self.reset(&mut st);
                        return Err(e);
                    }
                };
                pos += 1;
                st.since_rebal += 1;
            } else {
                // the token and its drafts in one pass; the model's own pick
                // after each says how many of the drafts it would have written
                match self.draft_pass(&mut st, tok, &drafts, pos, &mut dr, &mut |t| {
                    produced += 1;
                    on_token(t) && produced < n
                }) {
                    Ok((kept, next)) => {
                        pos += kept;
                        st.since_rebal += kept;
                        match next {
                            Some(t) => tok = t,
                            None => break,
                        }
                    }
                    Err(e) => {
                        self.reset(&mut st);
                        return Err(e);
                    }
                }
            }
            if st.since_rebal >= REBAL_EVERY {
                self.rebalance(REBAL_STEP)?;
                st.since_rebal = 0;
            }
            if self.remote.is_some() && passes % 64 == 0 {
                self.remote_want();
            }
        }
        let end = Instant::now();
        if drafting && std::env::var_os("PELICA_PROF").is_some() {
            eprintln!("[pelica] drafts: {} passes for {produced} tokens; {} drafted, {} taken", passes, dr.drafted, dr.taken);
        }
        Ok(Stats { prompt: total, generated: produced, prefill_s: (t_first - t0).as_secs_f64(), decode_s: (end - t_first).as_secs_f64(), passes, log: Vec::new() })
    }

    /// One pass over `tok` (at `pos`) and its `drafts`. The drafts the
    /// model agrees with go to `emit` in order (it returns false to stop);
    /// the cache then holds `tok` and those, going back over the pass
    /// (states from before it, then the kept tokens again) when it turned
    /// some down. Returns the positions kept and the next token (none if
    /// `emit` stopped).
    fn draft_pass(&self, st: &mut State, tok: u32, drafts: &[u32], pos: usize, dr: &mut Drafter, emit: &mut dyn FnMut(u32) -> bool) -> Result<(usize, Option<u32>)> {
        let r = &self.r;
        self.fill(&[(r.gst, r.dgst, self.gst_len(), 0), (r.gcv, r.dgcv, self.gcv_len(), 0)])?;
        let ple0 = st.ple.clone();
        let mut seq = vec![tok];
        seq.extend_from_slice(drafts);
        let mut picks = self.forward_n(st, &seq, pos, seq.len(), &mut |_| true)?;
        let (temp, top_p, _) = *self.samp.lock().unwrap();
        if temp > 0.0 {
            // sampling: each draft kept as likely as the model would write
            // it; the first turned down gets the model's own draw instead
            let v = self.hp.vocab;
            let mut sp = self.samp.lock().unwrap();
            for (i, &dt) in drafts.iter().enumerate() {
                let (ok, t) = crate::ds4::sample_draft(&st.logits[i * v..(i + 1) * v], temp, top_p, dt, &mut sp.2);
                picks[i] = t;
                if !ok {
                    break;
                }
            }
            let i = drafts.len();
            if picks[..i] == *drafts {
                picks[i] = crate::ds4::sample(&st.logits[i * v..(i + 1) * v], temp, top_p, &mut sp.2);
            }
        }
        let agree = drafts.iter().zip(&picks).take_while(|(a, b)| a == b).count();
        dr.done(drafts.len(), agree);
        let mut kept = 1;
        let mut go = true;
        for &t in &drafts[..agree] {
            kept += 1;
            if !emit(t) {
                go = false;
                break;
            }
        }
        if kept < seq.len() {
            self.fill(&[(r.dgst, r.gst, self.gst_len(), 0), (r.dgcv, r.gcv, self.gcv_len(), 0)])?;
            st.ple = ple0;
            self.truncate(st, pos);
            self.forward(st, &seq[..kept], pos, false, &mut |_| true)?;
        }
        Ok((kept, go.then(|| picks[agree])))
    }

    /// Forced decoding: runs `toks` one pass per chunk and returns the
    /// greedy pick after each position from `from` on (teacher forcing).
    pub fn forced(&self, toks: &[u32], from: usize) -> Result<Vec<u32>> {
        self.kv_make_room(toks.len());
        let mut st = self.st.lock().unwrap();
        self.reset(&mut st);
        let mut out = Vec::new();
        let mut c = 0;
        while c < from {
            let t = (from - c).min(CHUNK);
            if let Some(v) = self.forward(&mut st, &toks[c..c + t], c, c + t == from, &mut |_| true)? {
                out.push(v);
            }
            c += t;
        }
        while c < toks.len() {
            if let Some(v) = self.forward(&mut st, &toks[c..c + 1], c, true, &mut |_| true)? {
                out.push(v);
            }
            c += 1;
        }
        Ok(out)
    }

    /// The experts each token of `toks` goes to in every layer, with their
    /// weights: [token][layer][(expert, weight)]. Runs from position 0.
    pub fn trace(&self, toks: &[u32]) -> Result<Vec<Vec<Vec<(u32, f32)>>>> {
        if toks.is_empty() {
            bail!("nothing to trace");
        }
        let mut st = self.st.lock().unwrap();
        self.reset(&mut st);
        let mut out: Vec<Vec<Vec<(u32, f32)>>> = Vec::new();
        let mut c = 0;
        while c < toks.len() {
            let t = (toks.len() - c).min(CHUNK);
            st.trace = Some(Vec::new());
            let r = self.forward(&mut st, &toks[c..c + t], c, false, &mut |_| true);
            let tr = st.trace.take().unwrap_or_default();
            if let Err(e) = r {
                self.reset(&mut st);
                return Err(e);
            }
            out.extend(tr);
            c += t;
        }
        Ok(out)
    }

    pub fn set_sampling(&self, temp: f32, top_p: f32, seed: Option<u64>) {
        let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64));
        *self.samp.lock().unwrap() = (temp.max(0.0), top_p.clamp(0.01, 1.0), seed | 1);
    }

    /// Positions of `prompt` the cache can keep: the cache's end, or where
    /// the last prompt ended, if the prompt goes on from there.
    pub fn reusable(&self, cached: &[u32], prompt: &[u32]) -> usize {
        let st = self.st.lock().unwrap();
        let _ = cached;
        let same = st.toks.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1));
        let here = if same >= st.pos {
            st.pos
        } else {
            match &st.snap {
                Some(s) if s.pos <= same => s.pos,
                _ => 0,
            }
        };
        // or a state saved on disk
        self.best_saved(prompt, here).map_or(here, |(p, _)| p)
    }

    pub fn save_usage(&self) -> Result<()> {
        let sum: Vec<u32> = self.usage_base.iter().zip(&self.usage).map(|(a, b)| a.saturating_add(b.load(Ordering::Relaxed))).collect();
        let bytes: Vec<u8> = sum.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&self.usage_path, bytes).with_context(|| format!("writing {}", self.usage_path.display()))
    }

    pub fn activity(&self, prev: &mut Vec<u32>) -> Result<Vec<u32>> {
        let now: Vec<u32> = self.usage.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let ne = self.hp.ne;
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

    pub fn expert_grid(&self) -> (usize, usize) {
        (self.hp.n_layer, self.hp.ne)
    }

    /// (drive reads, RAM hits, read-ahead guesses, moves into VRAM)
    /// Running totals for the speed panel (`speed_report` of two).
    pub fn speed_counters(&self) -> Vec<u64> {
        let (reads, hits, _) = self.store.counts();
        let mut v: Vec<u64> = PROF.iter().chain(&EXP_COUNT).map(|c| c.load(Ordering::Relaxed)).collect();
        v.extend([reads, hits, self.store.io_prof().1]);
        v
    }

    pub fn io_counts(&self) -> (u64, u64, u64, usize) {
        let (a, b, c) = self.store.counts();
        (a, b, c, self.vram.lock().unwrap().moved)
    }
}

/// Where a reply's time went, from `speed_counters` before and after it:
/// seconds on the GPU's layers, the attention (CPU), experts in VRAM, on the
/// CPU, waiting for the drive, and the rest; and the experts' counts.
pub fn speed_report(a: &[u64], b: &[u64]) -> serde_json::Value {
    let d = |i: usize| b.get(i).zip(a.get(i)).map_or(0, |(x, y)| x.saturating_sub(*y));
    let s = |i: usize| d(i) as f64 / 1e9;
    let drive = s(10) + s(13);
    serde_json::json!({
        "gpu_s": s(1) + s(2) + s(4) + s(9),
        "attention_s": s(3),
        "experts_vram_s": (s(7) - s(6)).max(0.0) + (s(8) - s(13)).max(0.0),
        "experts_cpu_s": (s(6) - s(10)).max(0.0),
        "drive_s": drive,
        "other_s": s(0) + s(5),
        "in_vram": d(16), "on_cpu": d(17), "from_drive": d(18),
        "drive_reads": d(19), "ram_hits": d(20), "drive_gb": d(21) as f64 / 1e9,
    })
}

/// res[t][s] += y[t] * 2 sigmoid(inj[t][s] / streams)
fn combine(x: &mut [f32], y: &[f32], inj: &[f32], d: usize, h: usize) {
    x.par_chunks_mut(h * d).enumerate().for_each(|(t, xr)| {
        for s in 0..h {
            let w = 2.0 * sigmoid(inj[t * h + s] / h as f32);
            for (a, v) in xr[s * d..(s + 1) * d].iter_mut().zip(&y[t * d..(t + 1) * d]) {
                *a += v * w;
            }
        }
    });
}

/// Drafts for decoding (prompt lookup): where the last few tokens came
/// before in the cache, the tokens that followed then. Code edits, quotes
/// and lists repeat a lot of what's already there; one pass then checks
/// several tokens. The draft grows while the model keeps agreeing with it
/// and shrinks when it doesn't (a turned-down draft costs a pass over the
/// tokens it kept).
struct Drafter {
    /// tokens to draft next time
    k: usize,
    drafted: usize,
    taken: usize,
    /// passes to skip drafting after drafts went unused, and drafts turned
    /// down in a row (each doubles the rest: prose rarely repeats itself)
    rest: usize,
    misses: u32,
}

impl Drafter {
    /// at least this many tokens must match before the place counts
    const MATCH: usize = 3;
    const LONGEST: usize = 8;

    fn new() -> Drafter {
        Drafter { k: 3, drafted: 0, taken: 0, rest: 0, misses: 0 }
    }

    fn propose(&mut self, hist: &[u32], tok: u32) -> Vec<u32> {
        if self.rest > 0 {
            self.rest -= 1;
            return Vec::new();
        }
        let mut h = Vec::with_capacity(Self::LONGEST);
        h.extend_from_slice(&hist[hist.len().saturating_sub(Self::LONGEST - 1)..]);
        h.push(tok);
        // the longest suffix seen before, its latest place
        for n in (Self::MATCH..=h.len()).rev() {
            let suf = &h[h.len() - n..];
            let end = hist.len() + 1 - n;
            for j in (0..end.saturating_sub(1)).rev() {
                if j + n <= hist.len() && hist[j..j + n] == *suf {
                    let from = j + n;
                    let to = (from + self.k).min(hist.len());
                    if from < to {
                        return hist[from..to].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }

    fn done(&mut self, drafted: usize, agreed: usize) {
        self.drafted += drafted;
        self.taken += agreed;
        if agreed == drafted {
            self.k = (self.k + 2).min(SMALL - 1);
        } else {
            self.k = (agreed + 1).max(2);
        }
        if agreed * 2 >= drafted {
            self.misses = 0;
        } else {
            self.misses = (self.misses + 1).min(5);
            self.rest = 1 << self.misses;
        }
    }
}

/// experts worked on other computers (`pelica bridge`)
static FAR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A pass stopped by its caller.
#[derive(Debug)]
struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stopped")
    }
}

impl std::error::Error for Cancelled {}

fn print_prof(eng: &Engine) {
if std::env::var_os("PELICA_PROF").is_some() {
    let (busy, bytes, wasted) = eng.store.io_prof();
    eprintln!("  drive busy {busy:.1}s, read {:.1} GB ({:.2} GB/s while busy), guesses never used {wasted}", bytes as f64 / 1e9, bytes as f64 / 1e9 / busy.max(1e-9));
    if let Some(r) = eng.remote.as_ref().filter(|r| !r.held().is_empty() || FAR.load(Ordering::Relaxed) > 0) {
        eprintln!("  devices: {} experts worked there, {} held now; {}", FAR.load(Ordering::Relaxed), r.held().len(), r.status());
    }
    let bp: Vec<u64> = crate::ds4_gpu::BATCH_PROF.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    eprintln!("  GPU batches {} on {}: building+submitting {:.2}s, waiting {:.2}s", bp[0], eng.gpu.name, bp[1] as f64 / 1e9, bp[2] as f64 / 1e9);
    for (i, n) in PROF_NAMES.iter().enumerate() {
        eprintln!("  {n:<26} {:8.2}s", PROF[i].load(Ordering::Relaxed) as f64 / 1e9);
    }
    let ec: Vec<u64> = EXP_COUNT.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    eprintln!("  small-pass experts: {} in VRAM, {} on the CPU ({} of them read from the drive)", ec[0], ec[1], ec[2]);
}
}

/// `pelica -m model.gguf ...` for this architecture: load, generate, report.
/// PELICA_FORCED=<file of token ids, prompt length first> runs teacher
/// forcing and prints the picks.
pub fn run(path: &Path, prompt: &str, ids: Option<&[u32]>, raw: bool, n: usize, ctx: usize, teacher: bool, ignore_eos: bool) -> Result<()> {
    use std::io::Write;
    if std::env::var_os("PELICA_RAWBENCH").is_some() {
        // random reads over the largest part, at this model's piece sizes
        let big = split_paths(path).into_iter().max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).unwrap();
        let span = std::fs::metadata(&big)?.len();
        for chunk in [256usize << 10, 1 << 20, 3 << 20] {
            for th in [2usize, 4, 8, 16] {
                eprintln!("chunk {:>4} KB, {th:>2} threads: {:.2} GB/s", chunk >> 10, crate::ds4_io::bench(&big, chunk, th, span, 2.0)?);
            }
        }
        return Ok(());
    }
    let g = Gguf::open(path)?;
    let tok = crate::gguf::tokenizer(&g)?;
    let template = crate::chat_template::ChatTemplate::from_gguf(&g);
    drop(g);
    let eng = Engine::load(path, ctx)?;
    if let Some(t) = std::env::var("PELICA_TEMP").ok().and_then(|v| v.parse::<f32>().ok()) {
        eng.set_sampling(t, 0.95, std::env::var("PELICA_SEED").ok().and_then(|v| v.parse().ok()));
    }
    if std::env::var_os("PELICA_GPUBENCH").is_some() {
        // a GPU round trip's fixed cost: one tiny kernel and a read
        let r = &eng.r;
        for _ in 0..2 {
            let t = Instant::now();
            for _ in 0..500 {
                let mut b = eng.gpu.batch();
                b.kernel("q4x-fill", || FILL.to_string(), 0, &[64, r.y, r.sy, 0], [1, 1, 1]);
                b.read(r.sy, 1);
                b.run()?;
            }
            eprintln!("round trip: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 500.0);
        }
        // single-token products on layer 0's matrices: GB/s
        let ly = &eng.layers[0];
        let mut mats: Vec<(&str, &Mat)> = vec![("router F32", &ly.router), ("shared gate+up", &ly.sgu), ("shared down", &ly.sd), ("hc down", &ly.hca.down), ("hc up", &ly.hca.up)];
        if let Mixer::Lin { qkv, z, out, .. } = &ly.mix {
            mats.extend([("qkv", qkv), ("z", z), ("out", out)]);
        }
        for (name, m) in mats {
            let bytes = m.rows as f64 * m.wpr as f64 * 4.0;
            let t = Instant::now();
            for _ in 0..100 {
                let mut b = eng.gpu.batch();
                for _ in 0..10 {
                    b.mv(m, r.emb, m.cols as u32, r.p1, m.rows as u32, 1, false, None);
                }
                b.read(r.p1, 1);
                b.run()?;
            }
            let el = t.elapsed().as_secs_f64() / 1000.0;
            eprintln!("{name:<16} {} x {} {}: {:.1} us, {:.0} GB/s", m.rows, m.cols, m.ty.name(), el * 1e6, bytes / el / 1e9);
        }
        return Ok(());
    }
    if let Some(f) = std::env::var_os("PELICA_FORCED") {
        let s = std::fs::read_to_string(f)?;
        let v: Vec<u32> = s.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
        let (from, toks) = (v[0] as usize, &v[1..]);
        let t = Instant::now();
        let picks = eng.forced(toks, from)?;
        println!("forced {}", picks.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        let (reads, hits, guesses, _) = eng.io_counts();
        eprintln!("[pelica] forced {} tokens in {:.1}s; expert reads {reads} (read-ahead {guesses}), RAM hits {hits}", toks.len(), t.elapsed().as_secs_f64());
        print_prof(&eng);
        return Ok(());
    }
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
    if teacher {
        println!("prompt_ids {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    let mut out = std::io::stdout();
    let mut gen = Vec::new();
    let mut printed = 0usize;
    let prof = std::env::var_os("PELICA_PROF").is_some();
    let stats = eng.generate(&prompt_ids, n, 0, |t| {
        if prof && gen.is_empty() {
            // (the profile covers decoding only)
            eng.store.io_prof_reset();
            PROF.iter().chain(crate::ds4_gpu::BATCH_PROF.iter()).chain(EXP_COUNT.iter()).for_each(|c| c.store(0, Ordering::Relaxed));
        }
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
    let _ = eng.save_usage();
    print_prof(&eng);
    let dec = stats.generated.saturating_sub(1);
    let (reads, hits, guesses, moved) = eng.io_counts();
    eprintln!(
        "[pelica] prefill {} tokens in {:.1}s ({:.2} tok/s); decode {dec} tokens in {:.1}s: {:.2} tok/s; expert reads {reads} (read-ahead {guesses}), RAM hits {hits}, moved to VRAM {moved}",
        stats.prompt,
        stats.prefill_s,
        stats.prompt as f64 / stats.prefill_s.max(1e-9),
        stats.decode_s,
        dec as f64 / stats.decode_s.max(1e-9),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for &v in &[0.0f32, 1.0, -2.5, 65504.0, 1e-5, -3.14159, 1e-8, 0.333333] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {back}");
        }
        // ties go to even
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn split_names() {
        let p = split_paths(Path::new("C:/m/Q-UD-IQ4_XS-00001-of-00003.gguf"));
        assert_eq!(p.len(), 3);
        assert!(p[2].to_string_lossy().ends_with("-00003-of-00003.gguf"));
        assert_eq!(split_paths(Path::new("C:/m/one.gguf")).len(), 1);
    }
}
