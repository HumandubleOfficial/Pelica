//! DeepSeek V4 (GGUF architecture `deepseek4`), for models far larger than
//! VRAM + RAM: the dense weights (attention, shared experts, compressors,
//! indexers, the head) sit in VRAM; routed experts sit in VRAM (the hottest),
//! in a RAM cache kept by heat (`ds4_io`), or on the drive, read when picked.
//!
//! The math follows DeepSeek's reference (`inference/model.py`) and
//! llama.cpp's `deepseek4` graph:
//! - hyper-connections: 4 copies of the hidden state; per block, a learned
//!   mix (Sinkhorn-normalized) folds them into the block's input and spreads
//!   its output back;
//! - attention: 64 query heads against one 512-wide key that is also the
//!   value (rotary on its last 64 dims, undone on the output), over a sliding
//!   window of 128 raw positions plus compressed entries: groups of 4
//!   positions (overlapping, picked top-512 by a learned indexer) or of 128
//!   positions (all of them), pooled by learned softmax weights; per-head
//!   sinks; a grouped low-rank output projection;
//! - experts: sqrt(softplus) scores, top 6 of 256 (by token id in the first
//!   3 layers), clamped SwiGLU, plus a shared expert.
//!
//! The GPU does the products (`ds4_gpu`), the CPU the rest: hyper-connection
//! mixes, norms, rotary, compression, the indexer, attention, and the
//! experts that sit in RAM when only a few tokens use them. Prompts run
//! layer by layer over chunks of up to `CHUNK` tokens, so each expert is
//! read once per chunk; decoding runs one token through the same path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::ds4_cpu::{self as cpu, Q8x};
use crate::ds4_gpu::{Batch, Gpu, Mat};
use crate::ds4_io::{Layout, Pin, Store};
use crate::gguf::{gpu_layout, Gguf, Tensor, Ty};
use crate::oss::Stats;

/// raw positions (and compressor inputs) kept per layer
const RING: usize = 1024;
/// tokens per GPU sub-batch of a prompt
const TB: usize = 256;
/// tokens per layer-by-layer prompt pass
const CHUNK: usize = 4096;
/// expert rows (tokens) per GPU expert batch
const EROWS: usize = 1024;
/// VRAM slots that experts from RAM pass through for prompt passes
const STAGE: usize = 12;
/// experts used by at most this many tokens of a pass run on the CPU (from RAM)
const CPU_MAX: usize = 4;
/// passes of at most this many tokens are "small" (CPU experts allowed)
const SMALL: usize = 8;
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
/// experts worked on other computers (`pelica bridge`)
static FAR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const PROF_NAMES: [&str; 16] = ["hc+norm (attn)", "GPU projections", "cache+compress", "CPU attention", "GPU output proj", "hc+route (ffn)", "shared+VRAM experts", "CPU experts (+drive)", "VRAM expert groups", "staged experts (+drive)", "hc_post", "head", "  staged: drive wait", "  staged: upload", "  staged: GPU", "devices (waiting)"];

/// decode picks in order (PELICA_PICKS=<file>), u32::MAX after each token
static PICKS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// layers read ahead (PELICA_AHEAD)
fn ahead() -> usize {
    std::env::var("PELICA_AHEAD").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// free an expert's RAM copy once it's in VRAM (PELICA_DEDUP=0: keep both)
fn dedup() -> bool {
    std::env::var("PELICA_DEDUP").map_or(true, |v| v != "0")
}

/// how many decode tokens' worth of heat a prompt pass counts as
/// (PELICA_PF_WEIGHT, 0: every prompt token counts as one). Counting each
/// token wiped the decode history and filled the RAM cache by prompt
/// counts: decode after a 1.5k-token prompt went 2.67 -> 2.89 tok/s at 64.
fn pf_weight() -> Option<f32> {
    let w = std::env::var("PELICA_PF_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(64.0);
    (w > 0.0).then_some(w)
}

/// unsure read-ahead (lower ranks, two layers on) while the drive would
/// otherwise sit idle (PELICA_IDLE_GUESS=1)
fn idle_guess() -> bool {
    std::env::var("PELICA_IDLE_GUESS").is_ok_and(|v| v != "0")
}

fn pf_guess() -> bool {
    std::env::var("PELICA_PF_GUESS").map_or(true, |v| v != "0")
}

/// a prompt pass works experts with at most this many tokens on the CPU, the
/// rest on the GPU (PELICA_PF_CPU_ROWS; 0 = all on the GPU)
fn pf_cpu_rows() -> usize {
    std::env::var("PELICA_PF_CPU_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(64)
}

/// ranks past the late / early guesses read only on an idle drive
/// (PELICA_IDLE_LATE, PELICA_IDLE_EARLY)
fn idle_ranks(kind: usize) -> usize {
    let v = if kind == 0 { "PELICA_IDLE_EARLY" } else { "PELICA_IDLE_LATE" };
    std::env::var(v).ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// CPU experts that are ready together run side by side (PELICA_CPU_BATCH=0:
/// one at a time, as before)
fn cpu_batch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PELICA_CPU_BATCH").map(|v| v != "0").unwrap_or(true))
}

/// experts read ahead per token from this layer's input before attention
/// (PELICA_LATE_K). 3 since the repack: 4 read 7% more for 3.6% less speed.
/// A single token reads by router margin instead (guess_margin)
fn late_k() -> usize {
    std::env::var("PELICA_LATE_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}

/// experts read ahead per token from an earlier layer (PELICA_EARLY_K). 4 with
/// two readers fills the drive while attention runs: 4.33 tok/s vs 4.25 at 3
/// (5: 4.29, 6: 4.10, as the extra reads go unused). A single token reads by
/// router margin instead (guess_margin)
fn early_k() -> usize {
    std::env::var("PELICA_EARLY_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}

/// a single token's early / late guess reads the picks whose router score
/// beats the best expert left out by at least this much (always the top
/// one), in place of a fixed top-k (PELICA_EARLY_M / PELICA_LATE_M; 0 = top-k).
/// The margin says more than the rank: early 0.12 reads 11% less than top 4
/// for as many hits, late 0.2 hits 37% more than top 3 for 10% more reads
fn guess_margin(kind: usize) -> f32 {
    let v = if kind == 0 { "PELICA_EARLY_M" } else { "PELICA_LATE_M" };
    std::env::var(v).ok().and_then(|v| v.parse().ok()).unwrap_or(if kind == 0 { 0.12 } else { 0.2 })
}

/// experts of the next layer read ahead per token from this layer's input
/// before attention (PELICA_MID_K)
fn mid_k() -> usize {
    std::env::var("PELICA_MID_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// a read-ahead prediction: expert, margin (route_m), in VRAM or RAM when guessed
type Pred = (u32, f32, bool);

/// predictions and outcomes (PELICA_MARGINS=<file>): kind, rank, margin, hit, cached
static MARGINS: Mutex<Vec<[f32; 5]>> = Mutex::new(Vec::new());

/// prediction hits: early by rank (0..6), late hits, predictions counted,
/// late by rank (8..14), mid by rank (16..22)
static PRED: [std::sync::atomic::AtomicU64; 24] = [const { std::sync::atomic::AtomicU64::new(0) }; 24];

fn tick(i: usize, t: &mut Instant) {
    let now = Instant::now();
    PROF[i].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
    *t = now;
}

pub fn f16_to_f32(h: u16) -> f32 {
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rms(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ss: f32 = cpu::dot(x, x);
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    match w {
        Some(w) => x.iter_mut().zip(w).for_each(|(v, w)| *v *= inv * w),
        None => x.iter_mut().for_each(|v| *v *= inv),
    }
}

/// The model's shape, from the GGUF metadata.
struct Hp {
    n_layer: usize,
    d: usize,
    nh: usize,
    hd: usize,
    rot: usize,
    ql: usize,
    olr: usize,
    og: usize,
    win: usize,
    ratios: Vec<usize>,
    ih: usize,
    ihd: usize,
    topk: usize,
    ne: usize,
    k: usize,
    ff: usize,
    w_scale: f32,
    hash: usize,
    clamp: Vec<f32>,
    hc: usize,
    iters: usize,
    hc_eps: f32,
    eps: f32,
    vocab: usize,
}

/// Rotary angles for the last `rot` dims of a vector (pairs 2i, 2i+1), with
/// YaRN's frequency blend when the layer uses it (as the reference computes
/// them).
struct Rope {
    freqs: Vec<f32>,
}

impl Rope {
    fn new(rot: usize, base: f32, yarn: Option<(f32, f32, f32, f32)>) -> Rope {
        let dim = rot as f32;
        let mut freqs: Vec<f32> = (0..rot / 2).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim)).collect();
        if let Some((factor, orig, fast, slow)) = yarn {
            let corr = |n: f32| dim * (orig / (n * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
            let low = corr(fast).floor().max(0.0);
            let mut high = corr(slow).ceil().min(dim - 1.0);
            if low == high {
                high += 0.001;
            }
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / factor * (1.0 - smooth) + *f * smooth;
            }
        }
        Rope { freqs }
    }

    fn apply(&self, x: &mut [f32], pos: usize, inverse: bool) {
        for (i, f) in self.freqs.iter().enumerate() {
            let ang = pos as f32 * f;
            let (s, c) = ang.sin_cos();
            let s = if inverse { -s } else { s };
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = a * s + b * c;
        }
    }
}

/// A hyper-connection mix: `rows` x (hc*d) weights, scales, biases.
struct Hc {
    f: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
}

struct Layer {
    ratio: usize,
    q_a: Mat,
    q_a_norm: u32,
    q_b: Mat,
    kv: Mat,
    o_a: Mat,
    o_b: Mat,
    /// shared expert: gate and up rows together, then down
    sgu: Mat,
    sd: Mat,
    comp: Option<(Mat, Mat)>,
    icomp: Option<(Mat, Mat)>,
    idx_q: Option<Mat>,
    idx_w: Option<Mat>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    sinks: Vec<f32>,
    hc_attn: Hc,
    hc_ffn: Hc,
    router: Vec<f32>,
    bias: Option<Vec<f32>>,
    hash: Option<Vec<i32>>,
    ape: Vec<f32>,
    cnorm: Vec<f32>,
    iape: Vec<f32>,
    icnorm: Vec<f32>,
}

/// Word offsets of the activation regions (prompt sub-batch sized).
struct R {
    a: u32,
    qa: u32,
    qr: u32,
    q: u32,
    kv: u32,
    ckv: u32,
    cg: u32,
    ickv: u32,
    icg: u32,
    iq: u32,
    iw: u32,
    oa: u32,
    ob: u32,
    f: u32,
    sgu: u32,
    sh: u32,
    sy: u32,
    xe: u32,
    egu: u32,
    eh: u32,
    ew: u32,
    ey: u32,
    hx: u32,
    logits: u32,
    arg: u32,
}

/// Per layer: raw positions (ring), compressed entries, the compressors'
/// inputs (ring), and the indexer's.
struct Kv {
    raw: Vec<f32>,
    comp: Vec<f32>,
    hkv: Vec<f32>,
    hsc: Vec<f32>,
    icomp: Vec<f32>,
    ihkv: Vec<f32>,
    ihsc: Vec<f32>,
}

struct State {
    kv: Vec<Kv>,
    /// positions in the cache
    pos: usize,
    trace: Option<Vec<Vec<Vec<(u32, f32)>>>>,
    since_rebal: usize,
}

/// How expert bytes sit: piece types and words per row in the GPU layout.
struct Ex {
    ty: [Ty; 3],
    wpr_g: u32,
    wpr_d: u32,
    /// words of one expert in a VRAM slot
    words: u64,
    cpu_ok: bool,
}

struct Vram {
    /// (arena, word) of each slot, and the expert in it
    slots: Vec<(usize, u32)>,
    key: Vec<u32>,
    slot_of: std::collections::HashMap<u32, usize>,
    moved: usize,
}

pub struct Engine {
    pub cfg: crate::moe::Cfg,
    /// the GPU and API, for display
    pub name: String,
    pub max_ctx: usize,
    pub vram_experts: usize,
    pub host_experts: usize,
    pub host_bytes: usize,
    pub eos: Vec<u32>,
    hp: Hp,
    gpu: Gpu,
    g: Gguf,
    layers: Vec<Layer>,
    head: Mat,
    out_norm: Vec<f32>,
    hc_head: Hc,
    embd: Tensor,
    store: Store,
    ex: Ex,
    vram: Mutex<Vram>,
    stage: Vec<(usize, u32)>,
    st: Mutex<State>,
    r: R,
    rope_base: Rope,
    rope_comp: Rope,
    usage: Vec<AtomicU32>,
    usage_base: Vec<u32>,
    usage_path: PathBuf,
    zeros: Vec<f32>,
    /// per layer: the latest early and late read-ahead predictions (ranked)
    preds: Mutex<Vec<[Vec<Pred>; 3]>>,
    /// sampling: (temperature, top-p, random state); temperature 0 = greedy
    samp: Mutex<(f32, f32, u64)>,
    /// other computers lending RAM (`pelica bridge`): a tier between RAM and the drive
    remote: Option<Arc<crate::bridge::Remote>>,
}

fn floats(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    let b = g.bytes(t);
    Ok(match t.ty {
        Ty::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Ty::F16 => b.par_chunks(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        Ty::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        t => bail!("{name}: expected a float tensor, got {}", t.name()),
    })
}

/// A tensor's bytes, with large reads (faster than faulting in the map).
fn read_tensor(f: &std::fs::File, t: &Tensor) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; t.bytes()];
    let mut done = 0usize;
    while done < buf.len() {
        let end = (done + (64 << 20)).min(buf.len());
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..end], (t.offset + done) as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..end], (t.offset + done) as u64)?;
        if n == 0 {
            bail!("unexpected end of file reading a tensor");
        }
        done += n;
    }
    Ok(buf)
}

fn wpr(ty: Ty, cols: usize) -> u32 {
    (cols / ty.gpu_values() * ty.gpu_words().expect("decodable")) as u32
}

pub fn handles(g: &Gguf) -> bool {
    g.arch() == "deepseek4"
}

impl Engine {
    pub fn load(path: &Path, ctx: usize) -> Result<Engine> {
        let t0 = Instant::now();
        let progress = |p: usize| crate::moe::LOAD_PROGRESS.store(p.min(100), Ordering::Relaxed);
        progress(0);
        if crate::repack::unfinished(path) {
            bail!("speeding this model up (pelica repack) didn't finish: run it again to finish it, from Models or with `pelica repack`");
        }
        let g = Gguf::open(path)?;
        if !handles(&g) {
            bail!("not a DeepSeek V4 model");
        }
        let key = |s: &str| format!("deepseek4.{s}");
        let ints = |s: &str| -> Result<Vec<usize>> { Ok(g.ints(&key(s))?.into_iter().map(|v| v as usize).collect()) };
        let n_layer = g.u(&key("block_count"))?;
        let clamp: Vec<f32> = match g.meta.get(&key("swiglu_clamp_exp")) {
            Some(crate::gguf::Val::Arr(v)) => v.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect(),
            Some(v) => vec![v.as_f64().unwrap_or(0.0) as f32; n_layer],
            None => vec![0.0; n_layer],
        };
        let hp = Hp {
            n_layer,
            d: g.u(&key("embedding_length"))?,
            nh: g.u(&key("attention.head_count"))?,
            hd: g.u(&key("attention.key_length"))?,
            rot: g.u(&key("rope.dimension_count"))?,
            ql: g.u(&key("attention.q_lora_rank"))?,
            olr: g.u(&key("attention.output_lora_rank"))?,
            og: g.u(&key("attention.output_group_count"))?,
            win: g.u(&key("attention.sliding_window"))?,
            ratios: ints("attention.compress_ratios")?,
            ih: g.u(&key("attention.indexer.head_count"))?,
            ihd: g.u(&key("attention.indexer.key_length"))?,
            topk: g.u(&key("attention.indexer.top_k"))?,
            ne: g.u(&key("expert_count"))?,
            k: g.u(&key("expert_used_count"))?,
            ff: g.u(&key("expert_feed_forward_length"))?,
            w_scale: g.f(&key("expert_weights_scale")).unwrap_or(1.0),
            hash: g.u(&key("hash_layer_count")).unwrap_or(0),
            clamp,
            hc: g.u(&key("hyper_connection.count"))?,
            iters: g.u(&key("hyper_connection.sinkhorn_iterations"))?,
            hc_eps: g.f(&key("hyper_connection.epsilon"))?,
            eps: g.f(&key("attention.layer_norm_rms_epsilon"))?,
            vocab: g.tensor("token_embd.weight")?.dims[1],
        };
        if hp.ratios.len() < n_layer || hp.hc != 4 || hp.hd != g.u(&key("attention.value_length"))? || hp.nh % hp.og != 0 || hp.d % 256 != 0 || hp.ff % 256 != 0 || hp.win + TB > RING {
            bail!("unsupported DeepSeek V4 shape");
        }
        if g.u(&key("expert_gating_func")).unwrap_or(0) != 4 {
            bail!("DeepSeek V4 with a scoring function other than sqrt(softplus) is not supported");
        }
        if let Some(r) = hp.ratios[..n_layer].iter().find(|&&r| r != 0 && r != 4 && r != 128) {
            bail!("DeepSeek V4 compression ratio {r} is not supported");
        }
        let max_ctx = g.u(&key("context_length")).unwrap_or(65536);
        let ctx = ctx.clamp(1024, max_ctx);
        let rope_base = Rope::new(hp.rot, g.f(&key("rope.freq_base"))?, None);
        let yarn = (
            g.f(&key("rope.scaling.factor")).unwrap_or(1.0),
            g.u(&key("rope.scaling.original_context_length")).unwrap_or(0) as f32,
            g.f(&key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
            g.f(&key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
        );
        let rope_comp = Rope::new(hp.rot, g.f(&key("attention.compress_rope_freq_base"))?, (yarn.1 > 0.0).then_some(yarn));

        // activations
        let (d, ff) = (hp.d, hp.ff);
        let mut words = 0usize;
        let mut region = |n: usize| -> u32 {
            let at = words;
            words += n.div_ceil(64) * 64;
            at as u32
        };
        let r = R {
            a: region(TB * d),
            qa: region(TB * hp.ql),
            qr: region(TB * hp.ql),
            q: region(TB * hp.nh * hp.hd),
            kv: region(TB * hp.hd),
            ckv: region(TB * 2 * hp.hd),
            cg: region(TB * 2 * hp.hd),
            ickv: region(TB * 2 * hp.ihd),
            icg: region(TB * 2 * hp.ihd),
            iq: region(TB * hp.ih * hp.ihd),
            iw: region(TB * hp.ih),
            oa: region(TB * hp.og * hp.olr),
            ob: region(TB * d),
            f: region(TB * d),
            sgu: region(TB * 2 * ff),
            sh: region(TB * ff),
            sy: region(TB * d),
            xe: region(EROWS * d),
            egu: region(EROWS * 2 * ff),
            eh: region(EROWS * ff),
            ew: region(EROWS),
            ey: region(EROWS * d),
            hx: region(d),
            logits: region(hp.vocab),
            arg: region(4),
        };
        let mut gpu = Gpu::new(words)?;
        let name = gpu.name.clone();

        // dense weights, packed into arenas a layer at a time
        let file = std::fs::File::open(path)?;
        let t = |l: usize, s: &str| format!("blk.{l}.{s}");
        let has = |n: &str| g.tensors.contains_key(n);
        let gpu_bytes = |n: &str| -> Result<usize> {
            let t = g.tensor(n)?;
            if t.ty.gpu_words().is_none() {
                bail!("{n}: the GPU can't decode {}", t.ty.name());
            }
            Ok(t.dims[1] * wpr(t.ty, t.dims[0]) as usize * 4)
        };
        let dense = |l: usize| -> Vec<String> {
            let mut v: Vec<String> = ["attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"].iter().map(|s| t(l, s)).collect();
            for s in ["attn_compressor_kv.weight", "attn_compressor_gate.weight", "indexer_compressor_kv.weight", "indexer_compressor_gate.weight", "indexer.attn_q_b.weight", "indexer.proj.weight"] {
                if has(&t(l, s)) {
                    v.push(t(l, s));
                }
            }
            v
        };
        let mut layer_bytes = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let mut b = hp.ql * 4;
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
            let tt = g.tensor(n)?;
            let (cols, rows) = (tt.dims[0], tt.dims[1]);
            let base = put(gpu, cursor, arena, &gpu_layout(tt.ty, &read_tensor(&file, tt)?));
            Ok(Mat { buf: arena, base, ty: tt.ty, rows, cols, wpr: wpr(tt.ty, cols) })
        };
        let hcm = |l: usize, s: &str| -> Result<Hc> { Ok(Hc { f: floats(&g, &t(l, &format!("hc_{s}_fn.weight")))?, scale: floats(&g, &t(l, &format!("hc_{s}_scale.weight")))?, base: floats(&g, &t(l, &format!("hc_{s}_base.weight")))? }) };
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let a = arena_of[l];
            let ratio = hp.ratios[l];
            let q_a_norm = floats(&g, &t(l, "attn_q_a_norm.weight"))?;
            let qn = put(&gpu, &mut cursor, a, unsafe { std::slice::from_raw_parts(q_a_norm.as_ptr() as *const u8, q_a_norm.len() * 4) });
            let q_a = mat(&gpu, &mut cursor, a, &t(l, "attn_q_a.weight"))?;
            let q_b = mat(&gpu, &mut cursor, a, &t(l, "attn_q_b.weight"))?;
            let kv = mat(&gpu, &mut cursor, a, &t(l, "attn_kv.weight"))?;
            let o_a = mat(&gpu, &mut cursor, a, &t(l, "attn_output_a.weight"))?;
            let o_b = mat(&gpu, &mut cursor, a, &t(l, "attn_output_b.weight"))?;
            let sg = mat(&gpu, &mut cursor, a, &t(l, "ffn_gate_shexp.weight"))?;
            let su = mat(&gpu, &mut cursor, a, &t(l, "ffn_up_shexp.weight"))?;
            let sd = mat(&gpu, &mut cursor, a, &t(l, "ffn_down_shexp.weight"))?;
            if su.base != sg.base + (sg.rows as u32) * sg.wpr || su.ty != sg.ty {
                bail!("layer {l}: shared expert gate and up don't line up");
            }
            if q_b.rows != hp.nh * hp.hd || kv.rows != hp.hd || o_a.rows != hp.og * hp.olr || o_a.cols != hp.nh * hp.hd / hp.og || o_b.cols != hp.og * hp.olr {
                bail!("layer {l}: attention shapes don't match the metadata");
            }
            let comp = if ratio != 0 { Some((mat(&gpu, &mut cursor, a, &t(l, "attn_compressor_kv.weight"))?, mat(&gpu, &mut cursor, a, &t(l, "attn_compressor_gate.weight"))?)) } else { None };
            let (icomp, idx_q, idx_w) = if ratio == 4 {
                (
                    Some((mat(&gpu, &mut cursor, a, &t(l, "indexer_compressor_kv.weight"))?, mat(&gpu, &mut cursor, a, &t(l, "indexer_compressor_gate.weight"))?)),
                    Some(mat(&gpu, &mut cursor, a, &t(l, "indexer.attn_q_b.weight"))?),
                    Some(mat(&gpu, &mut cursor, a, &t(l, "indexer.proj.weight"))?),
                )
            } else {
                (None, None, None)
            };
            let hash = if l < hp.hash {
                let tt = g.tensor(&t(l, "ffn_gate_tid2eid.weight"))?;
                if tt.ty != Ty::Other(26) && tt.ty.name() != "I32" {
                    bail!("layer {l}: token-to-expert table of type {}", tt.ty.name());
                }
                Some(g.bytes(tt).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
            } else {
                None
            };
            let opt = |s: &str| -> Result<Vec<f32>> { if has(&t(l, s)) { floats(&g, &t(l, s)) } else { Ok(Vec::new()) } };
            layers.push(Layer {
                ratio,
                q_a,
                q_a_norm: qn,
                q_b,
                kv,
                o_a,
                o_b,
                sgu: Mat { rows: 2 * sg.rows, ..sg },
                sd,
                comp,
                icomp,
                idx_q,
                idx_w,
                attn_norm: floats(&g, &t(l, "attn_norm.weight"))?,
                ffn_norm: floats(&g, &t(l, "ffn_norm.weight"))?,
                kv_norm: floats(&g, &t(l, "attn_kv_a_norm.weight"))?,
                sinks: floats(&g, &t(l, "attn_sinks.weight"))?,
                hc_attn: hcm(l, "attn")?,
                hc_ffn: hcm(l, "ffn")?,
                router: floats(&g, &t(l, "ffn_gate_inp.weight"))?,
                bias: if has(&t(l, "exp_probs_b.bias")) { Some(floats(&g, &t(l, "exp_probs_b.bias"))?) } else { None },
                hash,
                ape: opt("attn_compressor_ape.weight")?,
                cnorm: opt("attn_compressor_norm.weight")?,
                iape: opt("indexer_compressor_ape.weight")?,
                icnorm: opt("indexer_compressor_norm.weight")?,
            });
            gpu.flush()?;
            progress((l + 1) * 40 / n_layer);
        }
        let head_arena = gpu.arena(gpu_bytes("output.weight")? as u64)?;
        cursor.push(0);
        let head = mat(&gpu, &mut cursor, head_arena, "output.weight")?;
        let out_norm = floats(&g, "output_norm.weight")?;
        let hc_head = Hc { f: floats(&g, "output_hc_fn.weight")?, scale: floats(&g, "output_hc_scale.weight")?, base: floats(&g, "output_hc_base.weight")? };
        gpu.flush()?;
        drop(file);

        // experts: where they sit in the file, how they sit in VRAM
        let mut pieces = Vec::with_capacity(n_layer);
        let mut stride = Vec::with_capacity(n_layer);
        let mut tys = None;
        for l in 0..n_layer {
            let mut p = [(0u64, 0usize); 3];
            let mut ty = [Ty::F32; 3];
            // `pelica repack` renames a layer's expert tensors to *_expm once
            // each expert's three pieces sit together in their region
            let merged = crate::repack::layer_state(&g, l)?;
            for (i, s) in ["ffn_gate", "ffn_up", "ffn_down"].iter().enumerate() {
                let name = t(l, &format!("{s}_{}.weight", if merged { "expm" } else { "exps" }));
                let tt = g.tensor(&name)?;
                if tt.dims.len() != 3 || tt.dims[2] != hp.ne || tt.ty.gpu_words().is_none() {
                    bail!("{name}: unexpected expert tensor ({:?}, {})", tt.dims, tt.ty.name());
                }
                p[i] = (tt.offset as u64, tt.bytes() / hp.ne);
                ty[i] = tt.ty;
            }
            if merged {
                let base = p.iter().map(|x| x.0).min().unwrap_or(0);
                let mut at = base;
                for q in p.iter_mut() {
                    q.0 = at;
                    at += q.1 as u64;
                }
                stride.push(p.iter().map(|x| x.1).sum());
            } else {
                // apart: each piece steps by its own size
                stride.push(0);
            }
            if tys.is_some_and(|x| x != ty) || ty[0] != ty[1] {
                bail!("experts of mixed types are not supported");
            }
            tys = Some(ty);
            pieces.push(p);
        }
        let ty = tys.context("no layers")?;
        let ex = Ex { ty, wpr_g: wpr(ty[0], d), wpr_d: wpr(ty[2], ff), words: 2 * ff as u64 * wpr(ty[0], d) as u64 + d as u64 * wpr(ty[2], ff) as u64, cpu_ok: ty.iter().all(|&x| x == Ty::Q4K) };
        let slot_bytes = ex.words * 4;
        let total = n_layer * hp.ne;
        let per_arena = (ARENA / slot_bytes).max(1) as usize;
        // staging slots (prompt passes)
        let mut stage = Vec::new();
        {
            let a = gpu.arena(STAGE as u64 * slot_bytes)?;
            for i in 0..STAGE {
                stage.push((a, (i as u64 * ex.words) as u32));
            }
        }
        // VRAM experts: what's free after the rest, less the margin
        let budget = match std::env::var("PELICA_EXPERTS_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(gb) => (gb * 1e9) as u64,
            None => crate::free_vram().map_or(0, |f| (f as u64).saturating_sub(crate::vram_margin() as u64 + (64 << 20))),
        };
        let n_vram = ((budget / slot_bytes) as usize).min(total);
        let mut vslots = Vec::with_capacity(n_vram);
        while vslots.len() < n_vram {
            let n = per_arena.min(n_vram - vslots.len());
            let a = match gpu.arena(n as u64 * slot_bytes) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("[pelica] VRAM for experts ran out early: {e:#}");
                    break;
                }
            };
            for i in 0..n {
                vslots.push((a, (i as u64 * ex.words) as u32));
            }
        }
        // use history: seeds heat; the hottest go to VRAM, the next to RAM
        let usage_path = PathBuf::from(format!("{}.pelica-usage", path.display()));
        let usage_base: Vec<u32> = std::fs::read(&usage_path)
            .ok()
            .filter(|b| b.len() == total * 4)
            .map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
            .unwrap_or_else(|| vec![0; total]);
        let ram = match std::env::var("PELICA_RAM_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(gb) => (gb * 1e9) as u64,
            // (the compressed attention cache, reserved below at its exact size:
            // ~14 KB per position, where 64 KB used to be held back)
            None => crate::available_ram().saturating_sub(RAM_MARGIN + kv_bytes(&hp, n_layer, ctx)),
        };
        let remote = if ex.cpu_ok {
            let size = std::fs::metadata(path).map_or(0, |m| m.len());
            let model = size ^ (total as u64) << 40 ^ (n_layer as u64) << 56 ^ 0x9e37_79b9;
            crate::bridge::Remote::new(path.to_path_buf(), model, d, ff, Layout { pieces: pieces.clone(), stride: stride.clone(), n_exp: hp.ne, file: Vec::new() })
        } else {
            None
        };
        let lay = Layout { pieces, stride, n_exp: hp.ne, file: Vec::new() };
        let cap_guess = (slot_bytes as usize + 3 * 8192) as u64;
        let n_ram = ((ram / cap_guess) as usize).clamp(16, total);
        let store = Store::new(path, lay, n_ram, vec![0.0; total])?;
        let mut order: Vec<u32> = (0..total as u32).filter(|&k| usage_base[k as usize] > 0).collect();
        order.sort_by_key(|&k| (std::cmp::Reverse(usage_base[k as usize]), k));
        let mut vram = Vram { key: vec![u32::MAX; vslots.len()], slots: vslots, slot_of: Default::default(), moved: 0 };
        let n_vram = vram.slots.len();
        let warm_v: Vec<u32> = order.iter().copied().take(n_vram).collect();
        let warm_r: Vec<u32> = order.iter().copied().skip(n_vram).take(store.slots().saturating_sub(64)).collect();
        let todo = warm_v.len() + warm_r.len();
        for (i, chunk) in warm_v.chunks(16).enumerate() {
            let pins = store.request(chunk)?;
            for (j, (pin, &k)) in pins.iter().zip(chunk).enumerate() {
                store.wait(pin)?;
                let s = i * 16 + j;
                Self::upload_expert(&gpu, &ex, vram.slots[s], pin);
                vram.key[s] = k;
                vram.slot_of.insert(k, s);
            }
            gpu.flush()?;
            drop(pins);
            if dedup() {
                store.forget(chunk);
            }
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
        let ratios = &hp.ratios;
        let kv = (0..n_layer)
            .map(|l| {
                let r = ratios[l];
                let coff = if r == 4 { 2 } else { 1 };
                Kv {
                    raw: vec![0.0; RING * hp.hd],
                    comp: if r != 0 { Vec::with_capacity(ctx / r * hp.hd) } else { Vec::new() },
                    hkv: if r != 0 { vec![0.0; RING * coff * hp.hd] } else { Vec::new() },
                    hsc: if r != 0 { vec![0.0; RING * coff * hp.hd] } else { Vec::new() },
                    icomp: if r == 4 { Vec::with_capacity(ctx / 4 * hp.ihd) } else { Vec::new() },
                    ihkv: if r == 4 { vec![0.0; RING * 2 * hp.ihd] } else { Vec::new() },
                    ihsc: if r == 4 { vec![0.0; RING * 2 * hp.ihd] } else { Vec::new() },
                }
            })
            .collect();
        let cfg = crate::moe::Cfg {
            arch: "deepseek4".into(),
            n_layer,
            d,
            n_head: hp.nh,
            n_kv: 1,
            hd: hp.hd,
            n_exp: hp.ne,
            top_k: hp.k,
            ff_exp: ff,
            vocab: hp.vocab,
            eps: hp.eps,
            ctx,
            rot: hp.rot,
            neox: false,
            norm_w: true,
            w_scale: hp.w_scale,
            sigmoid: false,
        };
        let eos = crate::moe::stop_tokens(&g);
        let host_experts = store.slots();
        let host_bytes = store.slots() * store.slot_bytes();
        let embd = g.tensor("token_embd.weight")?.clone();
        if embd.ty != Ty::F16 && embd.ty != Ty::F32 {
            bail!("token embeddings of type {} are not supported", embd.ty.name());
        }
        eprintln!(
            "[pelica] DeepSeek V4: {n_layer} layers, dense {:.2} GB in VRAM; experts {:.1} GB: {} in VRAM, {} cache slots in RAM ({:.1} GB), the rest read from the drive; warmed {} from use history; loaded in {:.1}s",
            layer_bytes.iter().sum::<usize>() as f64 / 1e9,
            (total as u64 * slot_bytes) as f64 / 1e9,
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
            max_ctx,
            vram_experts: n_vram,
            host_experts,
            host_bytes,
            eos,
            gpu,
            layers,
            head,
            out_norm,
            hc_head,
            embd,
            store,
            ex,
            vram: Mutex::new(vram),
            stage,
            st: Mutex::new(State { kv, pos: 0, trace: None, since_rebal: 0 }),
            remote,
            r,
            rope_base,
            rope_comp,
            usage: (0..total).map(|_| AtomicU32::new(0)).collect(),
            usage_base,
            usage_path,
            zeros: vec![0.0; d],
            preds: Mutex::new(vec![[Vec::new(), Vec::new(), Vec::new()]; n_layer]),
            samp: Mutex::new((0.0, 1.0, 0)),
            g,
            hp,
        })
    }

    /// Copies an expert's three pieces into a VRAM slot (GPU layout).
    fn upload_expert(gpu: &Gpu, ex: &Ex, (arena, base): (usize, u32), pin: &Pin) {
        let mut w = base as u64;
        for p in 0..3 {
            let b = pin.piece(p);
            let ty = ex.ty[p];
            if ty.gpu_words().unwrap() * 4 == ty.block().1 || matches!(ty, Ty::F32 | Ty::F16 | Ty::BF16) {
                gpu.write(arena, w, b);
                w += (b.len() / 4) as u64;
            } else {
                let v = gpu_layout(ty, b);
                gpu.write(arena, w, &v);
                w += (v.len() / 4) as u64;
            }
        }
    }

    fn rope(&self, l: usize) -> &Rope {
        if self.layers[l].ratio != 0 { &self.rope_comp } else { &self.rope_base }
    }

    /// The hyper-connection mix of `x` (hc copies): the block input `y`, and
    /// the post weights and combination matrix for `hc_post`.
    fn hc_pre(&self, x: &[f32], hc: &Hc, y: &mut [f32], post: &mut [f32], comb: &mut [f32]) {
        let (n, d, e) = (x.len(), self.hp.d, self.hp.hc_eps);
        let rs = 1.0 / (cpu::dot(x, x) / n as f32 + self.hp.eps).sqrt();
        let mut mix = [0f32; 24];
        let dots = |(ci, ch): (usize, &mut [f32])| {
            for (j, m) in ch.iter_mut().enumerate() {
                let k = ci * 3 + j;
                *m = cpu::dot(&hc.f[k * n..(k + 1) * n], x) * rs;
            }
        };
        if rayon::current_thread_index().is_some() {
            mix.chunks_mut(3).enumerate().for_each(dots);
        } else {
            mix.par_chunks_mut(3).enumerate().for_each(dots);
        }
        let mut pre = [0f32; 4];
        for j in 0..4 {
            pre[j] = sigmoid(mix[j] * hc.scale[0] + hc.base[j]) + e;
            post[j] = 2.0 * sigmoid(mix[4 + j] * hc.scale[1] + hc.base[4 + j]);
        }
        let mut c = [[0f32; 4]; 4];
        for j in 0..4 {
            for k in 0..4 {
                c[j][k] = mix[8 + 4 * j + k] * hc.scale[2] + hc.base[8 + 4 * j + k];
            }
            let m = c[j].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut s = 0.0;
            for k in 0..4 {
                c[j][k] = (c[j][k] - m).exp();
                s += c[j][k];
            }
            for k in 0..4 {
                c[j][k] = c[j][k] / s + e;
            }
        }
        let cols = |c: &mut [[f32; 4]; 4]| {
            for k in 0..4 {
                let s = c[0][k] + c[1][k] + c[2][k] + c[3][k] + e;
                for row in c.iter_mut() {
                    row[k] /= s;
                }
            }
        };
        cols(&mut c);
        for _ in 1..self.hp.iters {
            for row in c.iter_mut() {
                let s = row[0] + row[1] + row[2] + row[3] + e;
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            cols(&mut c);
        }
        for j in 0..4 {
            comb[4 * j..4 * j + 4].copy_from_slice(&c[j]);
        }
        for i in 0..d {
            y[i] = pre[0] * x[i] + pre[1] * x[d + i] + pre[2] * x[2 * d + i] + pre[3] * x[3 * d + i];
        }
    }

    /// x (the residual copies) = post * f + comb^T x
    fn hc_post(&self, x: &mut [f32], f: &[f32], post: &[f32], comb: &[f32]) {
        let d = self.hp.d;
        let res = x.to_vec();
        for k in 0..4 {
            let o = &mut x[k * d..(k + 1) * d];
            for i in 0..d {
                o[i] = post[k] * f[i] + comb[k] * res[i] + comb[4 + k] * res[d + i] + comb[8 + k] * res[2 * d + i] + comb[12 + k] * res[3 * d + i];
            }
        }
    }

    /// Pools block `b` of a compressor from its inputs in the ring (per dim,
    /// softmax over the block's positions; the overlapping kind also over
    /// the previous block's, from the first half of the dims), then norm and
    /// rotary at the block's first position.
    fn compress(&self, l: usize, hkv: &[f32], hsc: &[f32], dd: usize, ratio: usize, b: usize, norm: &[f32], out: &mut [f32]) {
        let overlap = ratio == 4;
        let w = if overlap { 2 * dd } else { dd };
        let mut rows: Vec<(usize, usize)> = Vec::with_capacity(2 * ratio);
        if overlap && b > 0 {
            for p in (b - 1) * ratio..b * ratio {
                rows.push((p % RING, 0));
            }
        }
        for p in b * ratio..(b + 1) * ratio {
            rows.push((p % RING, if overlap { dd } else { 0 }));
        }
        for i in 0..dd {
            let mut m = f32::NEG_INFINITY;
            for &(ri, o) in &rows {
                m = m.max(hsc[ri * w + o + i]);
            }
            let (mut s, mut v) = (0f32, 0f32);
            for &(ri, o) in &rows {
                let e = (hsc[ri * w + o + i] - m).exp();
                s += e;
                v += e * hkv[ri * w + o + i];
            }
            out[i] = v / s;
        }
        rms(out, Some(norm), self.hp.eps);
        let rot = self.hp.rot;
        self.rope(l).apply(&mut out[dd - rot..], b * ratio, false);
    }

    /// Attention of one token (heads already normed and rotated in `q`) over
    /// `keys`, with per-head sinks; output rotated back. `par`: heads in
    /// parallel.
    fn attend(&self, l: usize, q: &[f32], keys: &[&[f32]], pos: usize, out: &mut [f32], par: bool) {
        let (hd, rot) = (self.hp.hd, self.hp.rot);
        let scale = 1.0 / (hd as f32).sqrt();
        let sinks = &self.layers[l].sinks;
        let one = |h: usize, o: &mut [f32]| {
            let qh = &q[h * hd..(h + 1) * hd];
            let s: Vec<f32> = keys.iter().map(|k| cpu::dot(qh, k) * scale).collect();
            let m = s.iter().fold(sinks[h], |a, &b| a.max(b));
            let mut den = (sinks[h] - m).exp();
            o.fill(0.0);
            for (k, &sv) in keys.iter().zip(&s) {
                let p = (sv - m).exp();
                den += p;
                cpu::axpy(o, p, k);
            }
            let inv = 1.0 / den;
            o.iter_mut().for_each(|v| *v *= inv);
            self.rope(l).apply(&mut o[hd - rot..], pos, true);
        };
        if par {
            out.par_chunks_mut(hd).enumerate().for_each(|(h, o)| one(h, o));
        } else {
            out.chunks_mut(hd).enumerate().for_each(|(h, o)| one(h, o));
        }
    }

    /// The compressed entries a token at `pos` attends to in a top-k layer:
    /// all visible ones, or the indexer's top k.
    fn select(&self, kv: &Kv, iq: &[f32], iw: &[f32], nv: usize, par: bool) -> Vec<usize> {
        let k = self.hp.topk;
        if nv <= k {
            return (0..nv).collect();
        }
        let (ih, ihd) = (self.hp.ih, self.hp.ihd);
        let wscale = 1.0 / ((ih * ihd) as f32).sqrt();
        let score = |b: usize| -> f32 {
            let key = &kv.icomp[b * ihd..(b + 1) * ihd];
            let mut s = 0f32;
            for h in 0..ih {
                let v = cpu::dot(&iq[h * ihd..(h + 1) * ihd], key);
                if v > 0.0 {
                    s += v * iw[h] * wscale;
                }
            }
            s
        };
        let scores: Vec<f32> = if par { (0..nv).into_par_iter().map(score).collect() } else { (0..nv).map(score).collect() };
        let mut idx: Vec<usize> = (0..nv).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }

    /// The token's embedding, copied into the hc streams.
    fn embed(&self, tok: u32, x: &mut [f32]) {
        let d = self.hp.d;
        let b = self.g.bytes(&self.embd);
        let tok = (tok as usize).min(self.hp.vocab - 1);
        if self.embd.ty == Ty::F16 {
            let row = &b[tok * d * 2..(tok + 1) * d * 2];
            for i in 0..d {
                x[i] = f16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            }
        } else {
            let row = &b[tok * d * 4..(tok + 1) * d * 4];
            for i in 0..d {
                x[i] = f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        let (a, rest) = x.split_at_mut(d);
        for c in rest.chunks_mut(d) {
            c.copy_from_slice(a);
        }
    }

    /// Runs `toks` (positions pos0..) through the model; with `logits`, the
    /// greedy next token after the last. `on_layer` hears of each finished
    /// layer; if it returns false the pass stops (the cache is then cleared).
    fn forward(&self, st: &mut State, toks: &[u32], pos0: usize, logits: bool, on_layer: &mut dyn FnMut(usize) -> bool) -> Result<Option<u32>> {
        let hp = &self.hp;
        let (d, c) = (hp.d, toks.len());
        let hcd = hp.hc * d;
        let mut x = vec![0f32; c * hcd];
        x.par_chunks_mut(hcd).zip(toks.par_iter()).for_each(|(xr, &t)| self.embed(t, xr));
        if c <= SMALL {
            // the token-routed layers' experts are known now
            for l in 0..hp.hash.min(hp.n_layer) {
                self.guess(l, &[], toks, 0);
            }
        }
        for l in 0..hp.n_layer {
            self.store.mark(l * 4);
            self.attention(st, l, toks, pos0, &mut x)?;
            self.store.mark(l * 4 + 1);
            self.ffn(st, l, toks, &mut x)?;
            if !on_layer(l) {
                self.reset(st);
                bail!(Cancelled);
            }
        }
        self.store.mark(255);
        st.pos = pos0 + c;
        self.store.advance(if c > SMALL { pf_weight().map_or(c, |n| (n as usize).clamp(1, c)) } else { c });
        if c == 1 && std::env::var_os("PELICA_PICKS").is_some() {
            PICKS.lock().unwrap().push(u32::MAX);
        }
        if !logits {
            return Ok(None);
        }
        let mut tm = Instant::now();
        let xl = &x[(c - 1) * hcd..];
        let n = xl.len();
        let rs = 1.0 / (cpu::dot(xl, xl) / n as f32 + hp.eps).sqrt();
        let mut y = vec![0f32; d];
        let h = &self.hc_head;
        for j in 0..4 {
            let pre = sigmoid(cpu::dot(&h.f[j * n..(j + 1) * n], xl) * rs * h.scale[0] + h.base[j]) + hp.hc_eps;
            cpu::axpy(&mut y, pre, &xl[j * d..(j + 1) * d]);
        }
        rms(&mut y, Some(&self.out_norm), hp.eps);
        let r = &self.r;
        let mut b = self.gpu.batch();
        b.upload(r.hx, &y);
        b.mv(&self.head, r.hx, d as u32, r.logits, hp.vocab as u32, 1, false, None);
        let (temp, top_p, _) = *self.samp.lock().unwrap();
        if temp > 0.0 {
            b.read(r.logits, hp.vocab);
            let out = b.run()?;
            let t = {
                let mut sp = self.samp.lock().unwrap();
                sample(&out[0], temp, top_p, &mut sp.2)
            };
            tick(11, &mut tm);
            return Ok(Some(t));
        }
        b.argmax(hp.vocab, r.logits, r.arg);
        b.read(r.arg, 1);
        let out = b.run()?;
        tick(11, &mut tm);
        Ok(Some(out[0][0].to_bits()))
    }

    fn reset(&self, st: &mut State) {
        st.pos = 0;
        for kv in st.kv.iter_mut() {
            kv.comp.clear();
            kv.icomp.clear();
        }
    }

    fn attention(&self, st: &mut State, l: usize, toks: &[u32], pos0: usize, x: &mut [f32]) -> Result<()> {
        let hp = &self.hp;
        let ly = &self.layers[l];
        let r = &self.r;
        let (d, hd, nh, ql, rot) = (hp.d, hp.hd, hp.nh, hp.ql, hp.rot);
        let hcd = hp.hc * d;
        let ratio = ly.ratio;
        let coff = if ratio == 4 { 2 } else { 1 };
        let (ih, ihd) = (hp.ih, hp.ihd);
        // late read-ahead (small passes, one sub-batch): this layer's FFN
        // input as it stands before attention, worked out while the GPU projects
        let late = toks.len() <= SMALL && l >= hp.hash && std::env::var("PELICA_LATE").map_or(true, |v| v != "0");
        let late_first = std::env::var_os("PELICA_LATE_FIRST").is_some();
        let late_guess = |x: &[f32]| {
            let mut f = vec![0f32; toks.len() * d];
            for (fr, xr) in f.chunks_mut(d).zip(x.chunks(hcd)) {
                let (mut post, mut comb) = ([0f32; 4], [0f32; 16]);
                self.hc_pre(xr, &ly.hc_ffn, fr, &mut post, &mut comb);
                rms(fr, Some(&ly.ffn_norm), hp.eps);
            }
            self.guess(l, &f, toks, 1);
            // and the next layer's likeliest experts from the same state: their
            // reads fill the drive's idle time during this attention
            let mk = mid_k().min(hp.k);
            if mk > 0 && l + 1 < hp.n_layer {
                let nx = &self.layers[l + 1];
                for (fr, xr) in f.chunks_mut(d).zip(x.chunks(hcd)) {
                    let (mut post, mut comb) = ([0f32; 4], [0f32; 16]);
                    self.hc_pre(xr, &nx.hc_ffn, fr, &mut post, &mut comb);
                    rms(fr, Some(&nx.ffn_norm), hp.eps);
                }
                self.guess_ranks(l + 1, &f, toks, 3, mk, mk);
            }
        };
        if late && late_first {
            late_guess(x);
        }
        for s in (0..toks.len()).step_by(TB) {
            let mut tm = Instant::now();
            let n = (toks.len() - s).min(TB);
            let xs = &mut x[s * hcd..(s + n) * hcd];
            let mut a = vec![0f32; n * d];
            let mut pc = vec![0f32; n * 20];
            let pre = |((a, pc), xr): ((&mut [f32], &mut [f32]), &[f32])| {
                let (post, comb) = pc.split_at_mut(4);
                self.hc_pre(xr, &ly.hc_attn, a, post, comb);
                rms(a, Some(&ly.attn_norm), hp.eps);
            };
            // one token: on this thread, so hc_pre spreads its own work
            if n == 1 {
                a.chunks_mut(d).zip(pc.chunks_mut(20)).zip(xs.chunks(hcd)).for_each(pre);
            } else {
                a.par_chunks_mut(d).zip(pc.par_chunks_mut(20)).zip(xs.par_chunks(hcd)).for_each(pre);
            }
            tick(0, &mut tm);
            // projections
            let mut b = self.gpu.batch();
            b.upload(r.a, &a);
            b.mv(&ly.q_a, r.a, d as u32, r.qa, ql as u32, n, false, None);
            b.rmsnorm(ql, r.qa, r.qr, ql as u32, Some((ly.q_a.buf, ly.q_a_norm)), hp.eps, n);
            b.mv(&ly.q_b, r.qr, ql as u32, r.q, (nh * hd) as u32, n, false, None);
            b.mv(&ly.kv, r.a, d as u32, r.kv, hd as u32, n, false, None);
            b.read(r.q, n * nh * hd);
            b.read(r.kv, n * hd);
            if let Some((ckv, cg)) = &ly.comp {
                b.mv(ckv, r.a, d as u32, r.ckv, (coff * hd) as u32, n, false, None);
                b.mv(cg, r.a, d as u32, r.cg, (coff * hd) as u32, n, false, None);
                b.read(r.ckv, n * coff * hd);
                b.read(r.cg, n * coff * hd);
            }
            if let (Some((ickv, icg)), Some(iqm), Some(iwm)) = (&ly.icomp, &ly.idx_q, &ly.idx_w) {
                b.mv(ickv, r.a, d as u32, r.ickv, (2 * ihd) as u32, n, false, None);
                b.mv(icg, r.a, d as u32, r.icg, (2 * ihd) as u32, n, false, None);
                b.mv(iqm, r.qr, ql as u32, r.iq, (ih * ihd) as u32, n, false, None);
                b.mv(iwm, r.a, d as u32, r.iw, ih as u32, n, false, None);
                b.read(r.ickv, n * 2 * ihd);
                b.read(r.icg, n * 2 * ihd);
                b.read(r.iq, n * ih * ihd);
                b.read(r.iw, n * ih);
            }
            let pending = b.submit();
            if late && !late_first && s == 0 {
                late_guess(xs);
            }
            let mut rb = pending.wait()?.into_iter();
            let mut q = rb.next().unwrap();
            let mut kvr = rb.next().unwrap();
            let comp = ly.comp.as_ref().map(|_| (rb.next().unwrap(), rb.next().unwrap()));
            let idx = ly.icomp.as_ref().map(|_| (rb.next().unwrap(), rb.next().unwrap(), rb.next().unwrap(), rb.next().unwrap()));
            tick(1, &mut tm);
            // the cache, in position order
            let kv = &mut st.kv[l];
            for i in 0..n {
                let pos = pos0 + s + i;
                let row = &mut kvr[i * hd..(i + 1) * hd];
                rms(row, Some(&ly.kv_norm), hp.eps);
                self.rope(l).apply(&mut row[hd - rot..], pos, false);
                kv.raw[(pos % RING) * hd..(pos % RING + 1) * hd].copy_from_slice(row);
                if let Some((ckv, cg)) = &comp {
                    let w = coff * hd;
                    let ri = pos % RING;
                    kv.hkv[ri * w..(ri + 1) * w].copy_from_slice(&ckv[i * w..(i + 1) * w]);
                    let ape = &ly.ape[(pos % ratio) * w..(pos % ratio + 1) * w];
                    for j in 0..w {
                        kv.hsc[ri * w + j] = cg[i * w + j] + ape[j];
                    }
                    if (pos + 1) % ratio == 0 {
                        let bidx = pos / ratio;
                        let mut out = vec![0f32; hd];
                        self.compress(l, &kv.hkv, &kv.hsc, hd, ratio, bidx, &ly.cnorm, &mut out);
                        kv.comp.truncate(bidx * hd);
                        kv.comp.extend_from_slice(&out);
                    }
                }
                if let Some((ickv, icg, _, _)) = &idx {
                    let w = 2 * ihd;
                    let ri = pos % RING;
                    kv.ihkv[ri * w..(ri + 1) * w].copy_from_slice(&ickv[i * w..(i + 1) * w]);
                    let ape = &ly.iape[(pos % 4) * w..(pos % 4 + 1) * w];
                    for j in 0..w {
                        kv.ihsc[ri * w + j] = icg[i * w + j] + ape[j];
                    }
                    if (pos + 1) % 4 == 0 {
                        let bidx = pos / 4;
                        let mut out = vec![0f32; ihd];
                        self.compress(l, &kv.ihkv, &kv.ihsc, ihd, 4, bidx, &ly.icnorm, &mut out);
                        kv.icomp.truncate(bidx * ihd);
                        kv.icomp.extend_from_slice(&out);
                    }
                }
            }
            tick(2, &mut tm);
            // attention per token
            let kv = &st.kv[l];
            let mut o = vec![0f32; n * nh * hd];
            let par_heads = n < 4;
            let tok = |i: usize, qi: &mut [f32], oi: &mut [f32]| {
                let pos = pos0 + s + i;
                for h in 0..nh {
                    let qh = &mut qi[h * hd..(h + 1) * hd];
                    rms(qh, None, hp.eps);
                    self.rope(l).apply(&mut qh[hd - rot..], pos, false);
                }
                let lo = pos.saturating_sub(hp.win - 1);
                let mut keys: Vec<&[f32]> = (lo..=pos).map(|p| &kv.raw[(p % RING) * hd..(p % RING + 1) * hd]).collect();
                if ratio != 0 {
                    let nv = (pos + 1) / ratio;
                    let sel: Vec<usize> = match &idx {
                        Some((_, _, iq, iw)) => {
                            let mut iqi = iq[i * ih * ihd..(i + 1) * ih * ihd].to_vec();
                            if nv > hp.topk {
                                for h in 0..ih {
                                    self.rope(l).apply(&mut iqi[h * ihd + ihd - rot..(h + 1) * ihd], pos, false);
                                }
                            }
                            self.select(kv, &iqi, &iw[i * ih..(i + 1) * ih], nv, par_heads)
                        }
                        None => (0..nv).collect(),
                    };
                    keys.extend(sel.iter().map(|&bk| &kv.comp[bk * hd..(bk + 1) * hd]));
                }
                self.attend(l, qi, &keys, pos, oi, par_heads);
            };
            if par_heads {
                for (i, (qi, oi)) in q.chunks_mut(nh * hd).zip(o.chunks_mut(nh * hd)).enumerate() {
                    tok(i, qi, oi);
                }
            } else {
                q.par_chunks_mut(nh * hd).zip(o.par_chunks_mut(nh * hd)).enumerate().for_each(|(i, (qi, oi))| tok(i, qi, oi));
            }
            tick(3, &mut tm);
            // output projection
            let mut b = self.gpu.batch();
            b.upload(r.q, &o);
            let ogd = (nh / hp.og * hd) as u32;
            b.mv(&ly.o_a, r.q, (nh * hd) as u32, r.oa, (hp.og * hp.olr) as u32, n, false, Some((hp.olr as u32, ogd)));
            b.mv(&ly.o_b, r.oa, (hp.og * hp.olr) as u32, r.ob, d as u32, n, false, None);
            b.read(r.ob, n * d);
            let ob = b.run()?.pop().unwrap();
            tick(4, &mut tm);
            xs.par_chunks_mut(hcd).zip(ob.par_chunks(d)).zip(pc.par_chunks(20)).for_each(|((xr, f), pc)| self.hc_post(xr, f, &pc[..4], &pc[4..]));
        }
        Ok(())
    }

    /// The experts each token goes to, and their weights.
    fn route(&self, l: usize, f: &[f32], tok: u32) -> ([u32; 8], [f32; 8]) {
        let (ids, w, _) = self.route_m(l, f, tok);
        (ids, w)
    }

    /// route, plus how far each pick's biased score sits above the best
    /// expert left out (how sure a guess from it is; infinite when the
    /// layer routes by token)
    fn route_m(&self, l: usize, f: &[f32], tok: u32) -> ([u32; 8], [f32; 8], [f32; 8]) {
        let hp = &self.hp;
        let ly = &self.layers[l];
        let (d, ne, k) = (hp.d, hp.ne, hp.k);
        let mut sc = vec![0f32; ne];
        let score = |(ci, ch): (usize, &mut [f32])| {
            for (j, s) in ch.iter_mut().enumerate() {
                let e = ci * 32 + j;
                let v = cpu::dot(&ly.router[e * d..(e + 1) * d], f);
                let sp = if v > 20.0 { v } else { v.exp().ln_1p() };
                *s = sp.sqrt();
            }
        };
        // across threads unless already on one (a pass over many tokens)
        if rayon::current_thread_index().is_some() {
            sc.chunks_mut(32).enumerate().for_each(score);
        } else {
            sc.par_chunks_mut(32).enumerate().for_each(score);
        }
        let mut ids = [0u32; 8];
        let mut m = [f32::INFINITY; 8];
        match &ly.hash {
            Some(tab) => {
                let t = (tok as usize).min(hp.vocab - 1);
                for j in 0..k {
                    ids[j] = tab[t * k + j].clamp(0, ne as i32 - 1) as u32;
                }
            }
            None => {
                let biased: Vec<f32> = match &ly.bias {
                    Some(b) => sc.iter().zip(b).map(|(s, b)| s + b).collect(),
                    None => sc.clone(),
                };
                let mut idx: Vec<usize> = (0..ne).collect();
                let kk = (k + 1).min(ne);
                idx.select_nth_unstable_by(kk - 1, |&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
                idx[..kk].sort_unstable_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
                for j in 0..k {
                    ids[j] = idx[j] as u32;
                    if kk > k {
                        m[j] = biased[idx[j]] - biased[idx[k]];
                    }
                }
            }
        }
        let mut w = [0f32; 8];
        let mut sum = 0f32;
        for j in 0..k {
            w[j] = sc[ids[j] as usize];
            sum += w[j];
        }
        for v in w[..k].iter_mut() {
            *v = *v / sum * hp.w_scale;
        }
        (ids, w, m)
    }

    fn ffn(&self, st: &mut State, l: usize, toks: &[u32], x: &mut [f32]) -> Result<()> {
        let hp = &self.hp;
        let ly = &self.layers[l];
        let r = &self.r;
        let (d, ff, c, k) = (hp.d, hp.ff, toks.len(), hp.k);
        let hcd = hp.hc * d;
        let lim = hp.clamp[l];
        let mut tm = Instant::now();
        let mut f = vec![0f32; c * d];
        let mut pc = vec![0f32; c * 20];
        let pre = |((fr, pc), xr): ((&mut [f32], &mut [f32]), &[f32])| {
            let (post, comb) = pc.split_at_mut(4);
            self.hc_pre(xr, &ly.hc_ffn, fr, post, comb);
            rms(fr, Some(&ly.ffn_norm), hp.eps);
        };
        // one token: on this thread, so hc_pre and route spread their own work
        let routes: Vec<([u32; 8], [f32; 8])> = if c == 1 {
            f.chunks_mut(d).zip(pc.chunks_mut(20)).zip(x.chunks(hcd)).for_each(pre);
            vec![self.route(l, &f, toks[0])]
        } else {
            f.par_chunks_mut(d).zip(pc.par_chunks_mut(20)).zip(x.par_chunks(hcd)).for_each(pre);
            f.par_chunks(d).zip(toks.par_iter()).map(|(fr, &t)| self.route(l, fr, t)).collect()
        };
        if let Some(tr) = st.trace.as_mut() {
            for (i, (ids, w)) in routes.iter().enumerate() {
                if tr.len() <= i {
                    tr.resize(i + 1, Vec::new());
                }
                tr[i].push((0..k).map(|j| (ids[j], w[j])).collect());
            }
        }
        // per expert: its tokens and their weights
        let ne = hp.ne;
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
        if c == 1 && std::env::var_os("PELICA_PICKS").is_some() {
            PICKS.lock().unwrap().extend_from_slice(&touched);
        }
        if c == 1 && ly.hash.is_none() {
            let mut p = self.preds.lock().unwrap();
            let act = &routes[0].0[..k];
            if !p[l][0].is_empty() && !p[l][1].is_empty() {
                if std::env::var_os("PELICA_MARGINS").is_some() {
                    let mut mg = MARGINS.lock().unwrap();
                    for kind in 0..2 {
                        for (r, &(e, m, c)) in p[l][kind].iter().enumerate() {
                            mg.push([kind as f32, r as f32, m, act.contains(&e) as u8 as f32, c as u8 as f32]);
                        }
                    }
                }
                for (r, (e, _, _)) in p[l][0].iter().enumerate().take(6) {
                    if act.contains(e) {
                        PRED[r].fetch_add(1, Ordering::Relaxed);
                    }
                }
                PRED[6].fetch_add(p[l][1].iter().filter(|e| act.contains(&e.0)).count() as u64, Ordering::Relaxed);
                for (r, (e, _, _)) in p[l][1].iter().enumerate().take(6) {
                    if act.contains(e) {
                        PRED[8 + r].fetch_add(1, Ordering::Relaxed);
                    }
                }
                for (r, (e, _, _)) in p[l][2].iter().enumerate().take(6) {
                    if act.contains(e) {
                        PRED[16 + r].fetch_add(1, Ordering::Relaxed);
                    }
                }
                PRED[7].fetch_add(1, Ordering::Relaxed);
            }
            p[l] = [Vec::new(), Vec::new(), Vec::new()];
        }
        self.store.touch(&touched, if c > SMALL { pf_weight().map_or(1.0, |n| (n / c as f32).min(1.0)) } else { 1.0 });
        let small = c <= SMALL;
        let (mut on_vram, mut on_cpu, mut on_stage) = (Vec::new(), Vec::new(), Vec::new());
        {
            let v = self.vram.lock().unwrap();
            for &key in &keys {
                let e = key as usize % ne;
                if let Some(&s) = v.slot_of.get(&key) {
                    on_vram.push((key, v.slots[s]));
                } else if self.ex.cpu_ok && rows[e].len() <= if small { CPU_MAX } else { pf_cpu_rows() } {
                    on_cpu.push(key);
                } else {
                    on_stage.push(key);
                }
            }
        }
        // experts a device holds (small passes, not in RAM here): sent there
        // now, collected after this PC's own; any that fail run here instead
        let mut far: Vec<(Arc<crate::bridge::Link>, Vec<u32>, Instant)> = Vec::new();
        if let (true, Some(r)) = (small, &self.remote) {
            let mut by: Vec<(Arc<crate::bridge::Link>, Vec<u32>)> = Vec::new();
            on_cpu.retain(|&key| {
                if self.store.has(key) {
                    return true;
                }
                let Some(link) = r.holder(key) else { return true };
                match by.iter_mut().find(|(l, _)| Arc::ptr_eq(l, &link)) {
                    Some((_, ks)) => ks.push(key),
                    None => by.push((link, vec![key])),
                }
                false
            });
            for (link, ks) in by {
                let jobs: Vec<(u32, f32, Vec<(u16, f32)>)> = ks.iter().map(|&k| (k, lim, rows[k as usize % ne].iter().map(|&(i, w)| (i as u16, w)).collect())).collect();
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
        // drive reads first (CPU experts, then staged ones in windows)
        self.store.cancel_guesses(l, &keys);
        let cpu_pins = self.store.request(&on_cpu)?;
        let win = (self.store.slots() / 2).clamp(8, 128);
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
        // read-ahead for the next layer (a prompt pass too: its reads start
        // while this layer's experts and the next attention run)
        if small || pf_guess() {
            for a in 1..=ahead() {
                if l + a >= hp.hash {
                    self.guess(l + a, &f, toks, 0);
                }
            }
            // two layers on, on an idle drive only
            if idle_guess() && c == 1 && l + ahead() + 1 >= hp.hash {
                self.guess_ranks(l + ahead() + 1, &f, toks, 2, 0, early_k().min(hp.k));
            }
        }
        self.store.mark(l * 4 + 2);
        tick(5, &mut tm);

        let mut y = vec![0f32; c * d];
        let mat_of = |(arena, base): (usize, u32)| -> (Mat, Mat) {
            let gu = Mat { buf: arena, base, ty: self.ex.ty[0], rows: 2 * ff, cols: d, wpr: self.ex.wpr_g };
            let dn = Mat { buf: arena, base: base + (2 * ff) as u32 * self.ex.wpr_g, ty: self.ex.ty[2], rows: d, cols: ff, wpr: self.ex.wpr_d };
            (gu, dn)
        };
        // a GPU batch of expert row ranges: rows gathered at r.xe..; returns
        // the (row offset, expert, first row, row count) to scatter afterwards
        let gather = |b: &mut Batch, list: &[(u32, (usize, u32), usize, usize)]| -> Vec<(usize, u32, usize, usize)> {
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
                b.swiglu(ff, r.egu + (r0 * 2 * ff) as u32, r.egu + (r0 * 2 * ff + ff) as u32, r.eh + (r0 * ff) as u32, (2 * ff) as u32, ff as u32, Some(r.ew + r0 as u32), nr, lim);
                b.mv(&dn, r.eh + (r0 * ff) as u32, ff as u32, r.ey + (r0 * d) as u32, d as u32, nr, false, None);
                at.push((r0, key, lo, nr));
            }
            b.upload(r.xe, &xe);
            b.upload(r.ew, &we);
            if !xe.is_empty() {
                b.read(r.ey, xe.len());
            }
            at
        };
        let scatter = |y: &mut [f32], ey: &[f32], at: &[(usize, u32, usize, usize)]| {
            for &(r0, key, lo, nr) in at {
                for (j, &(i, _)) in rows[key as usize % ne][lo..lo + nr].iter().enumerate() {
                    let src = &ey[(r0 + j) * d..(r0 + j + 1) * d];
                    y[i * d..(i + 1) * d].iter_mut().zip(src).for_each(|(a, b)| *a += b);
                }
            }
        };
        // experts packed into batches of at most EROWS rows; an expert with
        // more rows than that (long prompts) is split across batches
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
        // a prompt pass works the CPU's experts on another thread, beside the
        // GPU's (whose experts have to be uploaded first)
        let mut y_cpu = vec![0f32; if small { 0 } else { c * d }];
        std::thread::scope(|sc| -> Result<()> {
        let cpu_job = if !small && !cpu_pins.is_empty() { Some(sc.spawn(|| self.cpu_experts(&cpu_pins, &on_cpu, &rows, &f, &mut y_cpu, lim))) } else { None };
        let groups = pack(&on_vram);
        // the shared expert, sub-batch by sub-batch; a small pass carries the
        // VRAM experts along in the same submission
        let mut first_group = if small { groups.first().cloned() } else { None };
        for s in (0..c).step_by(TB) {
            let n = (c - s).min(TB);
            let mut b = self.gpu.batch();
            b.upload(r.f, &f[s * d..(s + n) * d]);
            b.mv(&ly.sgu, r.f, d as u32, r.sgu, (2 * ff) as u32, n, false, None);
            b.swiglu(ff, r.sgu, r.sgu + ff as u32, r.sh, (2 * ff) as u32, ff as u32, None, n, lim);
            b.mv(&ly.sd, r.sh, ff as u32, r.sy, d as u32, n, false, None);
            b.read(r.sy, n * d);
            let at = match first_group.take() {
                Some(gr) => gather(&mut b, &gr),
                None => Vec::new(),
            };
            let pending = b.submit();
            tick(6, &mut tm);
            // CPU experts while the GPU works (small passes only)
            if s == 0 && small && !cpu_pins.is_empty() {
                self.cpu_experts(&cpu_pins, &on_cpu, &rows, &f, &mut y, lim)?;
                tick(7, &mut tm);
            }
            let mut out = pending.wait()?.into_iter();
            let sy = out.next().unwrap();
            y[s * d..(s + n) * d].iter_mut().zip(&sy).for_each(|(a, b)| *a += b);
            if let Some(ey) = out.next() {
                scatter(&mut y, &ey, &at);
            }
            tick(6, &mut tm);
        }
        for gr in groups.iter().skip(if small { 1 } else { 0 }) {
            let mut b = self.gpu.batch();
            let at = gather(&mut b, gr);
            let ey = b.run()?.pop().unwrap_or_default();
            scatter(&mut y, &ey, &at);
        }
        // the devices' outputs (they worked while this PC did)
        for (link, ks, t) in &far {
            let r = self.remote.as_ref().unwrap();
            match r.finish(link, *t, ks.len()) {
                Ok((yr, missing)) => {
                    y.iter_mut().zip(&yr).for_each(|(a, b)| *a += b);
                    // swapped out while the RUN was on its way: worked here
                    if !missing.is_empty() {
                        let pins = self.store.request(&missing)?;
                        self.cpu_experts(&pins, &missing, &rows, &f, &mut y, lim)?;
                    }
                }
                Err(e) => {
                    r.fail(link, &format!("{e:#}"));
                    let pins = self.store.request(ks)?;
                    self.cpu_experts(&pins, ks, &rows, &f, &mut y, lim)?;
                }
            }
            FAR.fetch_add(ks.len() as u64, Ordering::Relaxed);
        }
        if !far.is_empty() {
            tick(15, &mut tm);
        }
        self.store.mark(l * 4 + 3);
        tick(8, &mut tm);
        // staged experts: through the VRAM staging slots, in read order
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
                let mut ts = Instant::now();
                self.store.wait(&pin)?;
                tick(12, &mut ts);
                let slot = self.stage[list.len()];
                Self::upload_expert(&self.gpu, &self.ex, slot, &pin);
                tick(13, &mut ts);
                list.push((key, slot));
                pins.push(pin);
                nrow += n;
            }
            refill(&mut stage_pins, &mut next_stage)?;
            let mut ts = Instant::now();
            for gr in pack(&list) {
                let mut b = self.gpu.batch();
                let at = gather(&mut b, &gr);
                let ey = b.run()?.pop().unwrap_or_default();
                scatter(&mut y, &ey, &at);
            }
            tick(14, &mut ts);
            drop(pins);
        }
        if let Some(j) = cpu_job {
            j.join().map_err(|_| anyhow::anyhow!("the CPU expert thread failed"))??;
        }
        Ok(())
        })?;
        if !small {
            y.iter_mut().zip(&y_cpu).for_each(|(a, b)| *a += b);
        }
        tick(9, &mut tm);
        x.par_chunks_mut(hcd).zip(y.par_chunks(d)).zip(pc.par_chunks(20)).for_each(|((xr, yr), pc)| self.hc_post(xr, yr, &pc[..4], &pc[4..]));
        tick(10, &mut tm);
        Ok(())
    }

    /// Experts in RAM for a few tokens each, on the CPU, added into y.
    fn cpu_experts(&self, pins: &[Pin], keys: &[u32], rows: &[Vec<(usize, f32)>], f: &[f32], y: &mut [f32], lim: f32) -> Result<()> {
        use rayon::prelude::*;
        let (d, ff, ne) = (self.hp.d, self.hp.ff, self.hp.ne);
        let mut qx: std::collections::HashMap<usize, Q8x> = std::collections::HashMap::new();
        for &key in keys {
            for &(i, _) in &rows[key as usize % ne] {
                qx.entry(i).or_insert_with(|| Q8x::new(&f[i * d..(i + 1) * d]));
            }
        }
        let inorder = std::env::var_os("PELICA_INORDER").is_some();
        // experts in RAM first, those still coming from the drive as they
        // arrive (in order, the CPU sat idle behind each read); every expert
        // ready at the time is worked side by side: one at a time across all
        // threads reached 47 GB/s of weights, three or more at once 57-60
        // (test mem_speed), as each expert's two passes wait on their slowest
        // thread. Each row's sum is the same either way, and the outputs are
        // added in the original order, so the sums don't change
        let mut done = vec![false; pins.len()];
        let mut outs: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); pins.len()];
        let mut left = pins.len();
        while left > 0 {
            let n0 = if inorder {
                let n = done.iter().position(|x| !x).unwrap_or(0);
                self.store.wait(&pins[n])?;
                n
            } else {
                self.store.wait_any(pins, &done)?
            };
            let mut batch = vec![n0];
            if !inorder && cpu_batch() {
                batch.extend((0..pins.len()).filter(|&i| i != n0 && !done[i] && self.store.is_ready(&pins[i])));
            }
            for &n in &batch {
                done[n] = true;
            }
            left -= batch.len();
            let one = |n: usize| -> Vec<(usize, Vec<f32>)> {
                let (pin, key) = (&pins[n], keys[n]);
                let (gw, uw, dw) = (pin.piece(0), pin.piece(1), pin.piece(2));
                // all of this expert's tokens at once: each weight row is read
                // once for every token (same sums as one token at a time)
                let toks = &rows[key as usize % ne];
                let xs: Vec<&Q8x> = toks.iter().map(|(i, _)| &qx[i]).collect();
                let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
                let o = cpu::expert(gw, uw, dw, d, ff, &xs, &wts, lim);
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

    /// Moves up to `budget` experts into VRAM: the hottest by heat that are
    /// not there, in place of the coldest that are.
    fn rebalance(&self, budget: usize) -> Result<usize> {
        let heat = self.store.heat();
        let mut v = self.vram.lock().unwrap();
        let n = v.slots.len();
        if n == 0 {
            return Ok(0);
        }
        let mut order: Vec<u32> = (0..heat.len() as u32).collect();
        order.select_nth_unstable_by(n - 1, |&a, &b| heat[b as usize].total_cmp(&heat[a as usize]));
        let want: std::collections::HashSet<u32> = order[..n].iter().copied().collect();
        let mut add: Vec<u32> = order[..n].iter().copied().filter(|k| !v.slot_of.contains_key(k)).collect();
        add.sort_by(|&a, &b| heat[b as usize].total_cmp(&heat[a as usize]));
        add.truncate(budget);
        if add.is_empty() {
            return Ok(0);
        }
        // the slots to give up: empty ones first, then the coldest unwanted
        let mut free: Vec<usize> = (0..n).filter(|&s| v.key[s] == u32::MAX || !want.contains(&v.key[s])).collect();
        free.sort_by(|&a, &b| {
            let ha = if v.key[a] == u32::MAX { f32::NEG_INFINITY } else { heat[v.key[a] as usize] };
            let hb = if v.key[b] == u32::MAX { f32::NEG_INFINITY } else { heat[v.key[b] as usize] };
            ha.total_cmp(&hb)
        });
        let add: Vec<u32> = add.into_iter().take(free.len()).collect();
        let mut moved = 0;
        for chunk in add.chunks(16) {
            let pins = self.store.request(chunk)?;
            for (pin, &key) in pins.iter().zip(chunk) {
                self.store.wait(pin)?;
                let s = free[moved];
                let old = v.key[s];
                if old != u32::MAX {
                    v.slot_of.remove(&old);
                }
                Self::upload_expert(&self.gpu, &self.ex, v.slots[s], pin);
                v.key[s] = key;
                v.slot_of.insert(key, s);
                moved += 1;
            }
            self.gpu.flush()?;
            drop(pins);
            if dedup() {
                self.store.forget(chunk);
            }
        }
        v.moved += moved;
        Ok(moved)
    }

    /// Restores the cache to its first `c` positions if it can (the ring
    /// still holds what continuing needs); returns the positions kept.
    fn rollback(&self, st: &mut State, c: usize) -> usize {
        if c >= st.pos {
            return st.pos.min(c);
        }
        if c + RING - self.hp.win - 8 < st.pos {
            self.reset(st);
            return 0;
        }
        for (l, kv) in st.kv.iter_mut().enumerate() {
            let r = self.hp.ratios[l];
            if r != 0 {
                kv.comp.truncate(c / r * self.hp.hd);
            }
            if r == 4 {
                kv.icomp.truncate(c / 4 * self.hp.ihd);
            }
        }
        st.pos = c;
        c
    }

    pub fn generate(&self, prompt: &[u32], n: usize, reuse: usize, on_token: impl FnMut(u32) -> bool) -> Result<Stats> {
        self.generate_with(prompt, n, reuse, on_token, |_, _| true)
    }

    /// Decoding (greedy, or sampled: set_sampling) of up to `n` tokens after `prompt`, whose first
    /// `reuse` positions are still in the cache. Prompt passes report
    /// progress to `on_prefill` (done, total); if it returns false, stops
    /// with nothing generated and `Stats::prompt` the tokens now cached.
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
        if std::env::var_os("PELICA_PROF").is_some() {
            self.store.io_prof_reset();
        }
        let mut c = self.rollback(&mut st, reuse.min(prompt.len() - 1));
        let total = prompt.len();
        let nl = self.hp.n_layer;
        let mut next = None;
        while c < total {
            let t = (total - c).min(CHUNK);
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
        }
        if total - reuse.min(total) > SMALL {
            self.rebalance(REBAL_PROMPT)?;
            st.since_rebal = 0;
            self.remote_want();
        }
        let t_first = Instant::now();
        if std::env::var_os("PELICA_PROF").is_some() {
            // the prompt pass, before the counters start over for decode
            let (busy, bytes, _) = self.store.io_prof();
            let idle = self.store.idle_by_mark();
            let phase = |p: usize| (0..nl).map(|l| idle[l * 4 + p]).sum::<f64>();
            eprintln!("  prompt pass: {:.1}s, drive busy {busy:.1}s, read {:.1} GB; drive idle: attention {:.1}s, routing {:.1}s, experts {:.1}s, after experts {:.1}s, other {:.1}s", t0.elapsed().as_secs_f64(), bytes as f64 / 1e9, phase(0), phase(1), phase(2), phase(3), idle[255]);
            self.store.io_prof_reset();
            crate::ds4_gpu::BATCH_PROF.iter().for_each(|c| c.store(0, Ordering::Relaxed));
        }
        let mut tok = next.context("no token")?;
        let mut produced = 0usize;
        let mut pos = total;
        loop {
            produced += 1;
            if !on_token(tok) || produced >= n {
                break;
            }
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
            if st.since_rebal >= REBAL_EVERY {
                self.rebalance(REBAL_STEP)?;
                st.since_rebal = 0;
            }
            if produced % 64 == 0 {
                self.remote_want();
            }
        }
        let end = Instant::now();
        Ok(Stats { prompt: total, generated: produced, prefill_s: (t_first - t0).as_secs_f64(), decode_s: (end - t_first).as_secs_f64(), passes: produced.saturating_sub(1), log: Vec::new() })
    }

    /// The experts each token of `toks` goes to in every layer, with their
    /// weights: [token][layer][(expert, weight)]. Runs from position 0, so
    /// the cache no longer holds what it held before.
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

    /// How later tokens are picked: temperature (0: always the likeliest)
    /// and top-p; `seed` fixes the random draws.
    pub fn set_sampling(&self, temp: f32, top_p: f32, seed: Option<u64>) {
        let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64));
        *self.samp.lock().unwrap() = (temp.max(0.0), top_p.clamp(0.01, 1.0), seed | 1);
    }

    pub fn reusable(&self, cached: &[u32], prompt: &[u32]) -> usize {
        cached.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1))
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
    pub fn io_counts(&self) -> (u64, u64, u64, usize) {
        let (a, b, c) = self.store.counts();
        (a, b, c, self.vram.lock().unwrap().moved)
    }

    /// Starts reading the experts layer `l` will likely want, from the FFN
    /// inputs `f` of an earlier layer (exact for token-routed layers), unless
    /// they're in VRAM.
    fn guess(&self, l: usize, f: &[f32], toks: &[u32], kind: usize) {
        let k = if kind == 0 { early_k().min(self.hp.k) } else { late_k().min(self.hp.k) };
        // the next ranks are unsure: read them only on an idle drive
        let extra = if toks.len() == 1 { (k + idle_ranks(kind)).min(self.hp.k) } else { k };
        let m = if toks.len() == 1 { guess_margin(kind) } else { 0.0 };
        self.guess_ranks_m(l, f, toks, kind, k, extra, m);
    }

    /// Guesses ranks [0, k) as reads, ranks [k, extra) as idle-drive reads.
    fn guess_ranks(&self, l: usize, f: &[f32], toks: &[u32], kind: usize, k: usize, extra: usize) {
        self.guess_ranks_m(l, f, toks, kind, k, extra, 0.0);
    }

    /// guess_ranks; with `min_m` > 0 a token's reads are the picks at least
    /// that far above the best expert left out (at least one) instead of k
    #[allow(clippy::too_many_arguments)]
    fn guess_ranks_m(&self, l: usize, f: &[f32], toks: &[u32], kind: usize, k: usize, extra: usize, min_m: f32) {
        if l >= self.hp.n_layer || std::env::var("PELICA_LOOKAHEAD").is_ok_and(|v| v == "0") {
            return;
        }
        let (d, ne) = (self.hp.d, self.hp.ne);
        let mut idle = Vec::new();
        let mut keys = Vec::new();
        for (i, &t) in toks.iter().enumerate() {
            let fr = if f.is_empty() { &[][..] } else { &f[i * d..(i + 1) * d] };
            let (ids, _, mg) = if self.layers[l].hash.is_some() || !fr.is_empty() { self.route_m(l, if fr.is_empty() { &self.zeros } else { fr }, t) } else { continue };
            if toks.len() == 1 && self.layers[l].hash.is_none() && kind != 2 {
                let v = self.vram.lock().unwrap();
                let pr: Vec<Pred> = (0..self.hp.k).map(|j| {
                    let key = (l * ne + ids[j] as usize) as u32;
                    (ids[j], mg[j], v.slot_of.contains_key(&key) || self.store.has(key))
                }).collect();
                drop(v);
                self.preds.lock().unwrap()[l][kind.min(2)] = pr;
            }
            let k = if min_m > 0.0 && self.layers[l].hash.is_none() { mg[..self.hp.k].iter().filter(|&&m| m >= min_m).count().max(1) } else { k };
            for (j, &e) in ids[..extra.max(k)].iter().enumerate() {
                let key = (l * ne + e as usize) as u32;
                let to = if j < k { &mut keys } else { &mut idle };
                if !to.contains(&key) {
                    to.push(key);
                }
            }
        }
        let v = self.vram.lock().unwrap();
        let far = |key: &u32| self.remote.as_ref().is_some_and(|r| r.has(*key));
        keys.retain(|key| !v.slot_of.contains_key(key) && !far(key));
        idle.retain(|key| !v.slot_of.contains_key(key) && !keys.contains(key) && !far(key));
        drop(v);
        self.store.guess(&keys);
        if !idle.is_empty() {
            self.store.guess_idle(&idle);
        }
    }
}

/// RAM the compressed attention cache takes at `ctx` positions: compressed
/// keys (one per `ratio` positions) and, on ratio-4 layers, the indexer's.
fn kv_bytes(hp: &Hp, n_layer: usize, ctx: usize) -> u64 {
    let per: usize = hp.ratios[..n_layer].iter().map(|&r| if r == 0 { 0 } else { ctx / r * hp.hd + if r == 4 { ctx / 4 * hp.ihd } else { 0 } }).sum();
    per as u64 * 4
}

/// A token drawn from the logits at temperature `temp` from the smallest set
/// of likeliest tokens holding `top_p` of the probability. Tokens more than
/// 30 nats (at temperature 1) below the top are left out: together they hold
/// under 1e-8.
pub(crate) fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> u32 {
    let c = dist(logits, temp, top_p);
    let mass: f64 = c.iter().map(|x| x.1).sum();
    let mut r = uniform(rng) * mass;
    for x in &c {
        r -= x.1;
        if r < 0.0 {
            return x.0;
        }
    }
    c[c.len() - 1].0
}

/// One draft checked against the model (exact speculative sampling): kept
/// with the probability `sample` gives it, else a token drawn as `sample`
/// would with the draft left out. Either way the token comes out exactly as
/// likely as from `sample`. Returns (draft kept, token).
pub(crate) fn sample_draft(logits: &[f32], temp: f32, top_p: f32, draft: u32, rng: &mut u64) -> (bool, u32) {
    let c = dist(logits, temp, top_p);
    let mass: f64 = c.iter().map(|x| x.1).sum();
    let pd = c.iter().find(|x| x.0 == draft).map_or(0.0, |x| x.1);
    if uniform(rng) * mass < pd {
        return (true, draft);
    }
    let rest = mass - pd;
    let mut r = uniform(rng) * rest;
    let mut last = None;
    for x in c.iter().filter(|x| x.0 != draft) {
        last = Some(x.0);
        r -= x.1;
        if r < 0.0 {
            return (false, x.0);
        }
    }
    (false, last.unwrap_or(draft))
}

/// xorshift64*, as a number in [0, 1)
fn uniform(rng: &mut u64) -> f64 {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
}

/// The tokens `sample` draws from, likeliest first, with their weights
/// (not normalized).
fn dist(logits: &[f32], temp: f32, top_p: f32) -> Vec<(u32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![(logits.iter().position(|v| *v == max).unwrap_or(0) as u32, 1.0)];
    }
    let cut = max - 30.0 * temp;
    let mut c: Vec<(u32, f64)> = logits.iter().enumerate().filter(|(_, &v)| v >= cut).map(|(i, &v)| (i as u32, (((v - max) / temp) as f64).exp())).collect();
    c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let total: f64 = c.iter().map(|x| x.1).sum();
    let mut keep = c.len();
    if top_p < 1.0 {
        let mut acc = 0.0;
        for (i, x) in c.iter().enumerate() {
            acc += x.1;
            if acc >= top_p as f64 * total {
                keep = i + 1;
                break;
            }
        }
    }
    c.truncate(keep);
    c
}

/// A pass stopped by its caller.
#[derive(Debug)]
struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stopped")
    }
}

impl std::error::Error for Cancelled {}

pub fn run(path: &Path, prompt: &str, ids: Option<&[u32]>, raw: bool, n: usize, ctx: usize, teacher: bool, ignore_eos: bool) -> Result<()> {
    use std::io::Write;
    if std::env::var_os("PELICA_RAWBENCH").is_some() {
        let span = std::fs::metadata(path)?.len().min(150 << 30);
        for chunk in [1usize << 20, 4 << 20, 8 << 20, 16 << 20] {
            for th in [1usize, 2, 4, 8] {
                eprintln!("chunk {:>2} MB, {th} threads: {:.2} GB/s", chunk >> 20, crate::ds4_io::bench(path, chunk, th, span, 2.0)?);
            }
        }
        return Ok(());
    }
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
    if teacher {
        println!("prompt_ids {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    if std::env::var_os("PELICA_BENCH").is_some() {
        // drive read speed for experts not in RAM (one in every 37)
        let keys: Vec<u32> = (0..eng.hp.n_layer * eng.hp.ne).step_by(37).map(|k| k as u32).collect();
        let t = Instant::now();
        for ch in keys.chunks(64) {
            let pins = eng.store.request(ch)?;
            for p in &pins {
                eng.store.wait(p)?;
            }
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("[pelica] read {} experts in {s:.2}s: {:.2} GB/s", keys.len(), keys.len() as f64 * eng.store.slot_bytes() as f64 / 3.0 * 3.0 / s / 1e9);
        return Ok(());
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
    if let Some(f) = std::env::var_os("PELICA_MARGINS") {
        let s: String = MARGINS.lock().unwrap().iter().map(|r| format!("{},{},{},{},{}
", r[0], r[1], r[2], r[3], r[4])).collect();
        std::fs::write(f, s)?;
    }
    if let Some(f) = std::env::var_os("PELICA_PICKS") {
        let v: Vec<u8> = PICKS.lock().unwrap().iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(f, v)?;
    }
    let _ = eng.save_usage();
    if std::env::var_os("PELICA_PROF").is_some() {
        let np = PRED[7].load(Ordering::Relaxed).max(1) as f64;
        eprintln!("  early guess hit rate by rank: {}; late guess hits {:.2}/6 ({})", (0..6).map(|r| format!("{:.0}%", PRED[r].load(Ordering::Relaxed) as f64 * 100.0 / np)).collect::<Vec<_>>().join(" "), PRED[6].load(Ordering::Relaxed) as f64 / np, (8..14).map(|r| format!("{:.0}%", PRED[r].load(Ordering::Relaxed) as f64 * 100.0 / np)).collect::<Vec<_>>().join(" "));
        if mid_k() > 0 {
            eprintln!("  next-layer guess at attention, hit rate by rank: {}", (16..22).map(|r| format!("{:.0}%", PRED[r].load(Ordering::Relaxed) as f64 * 100.0 / np)).collect::<Vec<_>>().join(" "));
        }
        let (busy, bytes, wasted) = eng.store.io_prof();
        let dup = { let v = eng.vram.lock().unwrap(); v.key.iter().filter(|&&k| k != u32::MAX && eng.store.has(k)).count() };
        eprintln!("  VRAM experts also held in RAM: {dup} of {} RAM slots", eng.store.slots());
        let qd = eng.store.io_qd();
        eprintln!("  drive time by reads in flight: 0: {:.1}s, 1: {:.1}s, 2: {:.1}s, 3: {:.1}s, 4+: {:.1}s", qd[0], qd[1], qd[2], qd[3], qd[4]);
        let idle = eng.store.idle_by_mark();
        let nl = eng.hp.n_layer;
        let phase = |p: usize| (0..nl).map(|l| idle[l * 4 + p]).sum::<f64>();
        eprintln!("  drive idle: attention {:.1}s, routing {:.1}s, experts {:.1}s, after experts {:.1}s, between tokens {:.1}s", phase(0), phase(1), phase(2), phase(3), idle[255]);
        eprintln!("  drive busy {busy:.1}s, read {:.1} GB ({:.2} GB/s while busy), guesses never used {wasted}", bytes as f64 / 1e9, bytes as f64 / 1e9 / busy.max(1e-9));
        if let Some(r) = &eng.remote {
            eprintln!("  devices: {} experts worked there (all passes), {} held now; {}", FAR.load(Ordering::Relaxed), r.held().len(), r.status());
        }
        let bp: Vec<u64> = crate::ds4_gpu::BATCH_PROF.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        eprintln!("  GPU batches {} on {}: building+submitting {:.2}s, waiting {:.2}s", bp[0], eng.gpu.name, bp[1] as f64 / 1e9, bp[2] as f64 / 1e9);
        for (i, n) in PROF_NAMES.iter().enumerate() {
            eprintln!("  {n:<26} {:8.2}s", PROF[i].load(Ordering::Relaxed) as f64 / 1e9);
        }
    }
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
    use super::sample;

    #[test]
    fn sampling() {
        let lg = [0.0f32, 5.0, 4.0, -1.0];
        let mut rng = 12345u64;
        // a tiny top-p keeps only the likeliest token
        assert!((0..200).all(|_| sample(&lg, 1.0, 0.01, &mut rng) == 1));
        // a low temperature is nearly greedy
        assert!((0..200).all(|_| sample(&lg, 0.05, 1.0, &mut rng) == 1));
        // temperature 1: token 2 comes up about e^-1 / (1 + e^-1 + ...) of the time
        let n = 20000;
        let twos = (0..n).filter(|_| sample(&lg, 1.0, 1.0, &mut rng) == 2).count() as f64 / n as f64;
        let want = (-1f64).exp() / (1.0 + (-1f64).exp() + (-5f64).exp() + (-6f64).exp());
        assert!((twos - want).abs() < 0.015, "{twos} vs {want}");
    }
}
