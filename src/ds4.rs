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
use std::sync::Mutex;
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
const RAM_MARGIN: u64 = 5 << 30;
/// expert moves into VRAM after a prompt, and every `REBAL_EVERY` tokens
const REBAL_PROMPT: usize = 256;
const REBAL_STEP: usize = 6;
const REBAL_EVERY: usize = 16;

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
        let mut put = |gpu: &Gpu, cursor: &mut Vec<u64>, arena: usize, bytes: &[u8]| -> u32 {
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
        let mut tys = None;
        for l in 0..n_layer {
            let mut p = [(0u64, 0usize); 3];
            let mut ty = [Ty::F32; 3];
            for (i, s) in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"].iter().enumerate() {
                let tt = g.tensor(&t(l, s))?;
                if tt.dims.len() != 3 || tt.dims[2] != hp.ne || tt.ty.gpu_words().is_none() {
                    bail!("{}: unexpected expert tensor ({:?}, {})", t(l, s), tt.dims, tt.ty.name());
                }
                p[i] = (tt.offset as u64, tt.bytes() / hp.ne);
                ty[i] = tt.ty;
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
            None => crate::available_ram().saturating_sub(RAM_MARGIN + (ctx as u64 * 64 << 10)),
        };
        let lay = Layout { pieces, n_exp: hp.ne };
        let cap_guess = (slot_bytes as usize + 3 * 8192) as u64;
        let n_ram = ((ram / cap_guess) as usize).clamp(16, total);
        let store = Store::new(path, lay, n_ram, usage_base.iter().map(|&u| u as f32).collect())?;
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
            progress(40 + (i * 16 + chunk.len()) * 60 / todo.max(1));
        }
        for (i, chunk) in warm_r.chunks(32).enumerate() {
            let pins = store.request(chunk)?;
            for pin in &pins {
                store.wait(pin)?;
            }
            progress(40 + (warm_v.len() + i * 32 + chunk.len()) * 60 / todo.max(1));
        }
        let ratios = &hp.ratios;
        let kv = (0..n_layer)
            .map(|l| {
                let r = ratios[l];
                let coff = if r == 4 { 2 } else { 1 };
                Kv {
                    raw: vec![0.0; RING * hp.hd],
                    comp: Vec::new(),
                    hkv: if r != 0 { vec![0.0; RING * coff * hp.hd] } else { Vec::new() },
                    hsc: if r != 0 { vec![0.0; RING * coff * hp.hd] } else { Vec::new() },
                    icomp: Vec::new(),
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
            r,
            rope_base,
            rope_comp,
            usage: (0..total).map(|_| AtomicU32::new(0)).collect(),
            usage_base,
            usage_path,
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
            let words = if p < 2 { ex.wpr_g as u64 * (b.len() / ty.bytes(ty.block().0) * ty.block().0) as u64 / (ex.wpr_g as u64 * ty.gpu_values() as u64 / ty.gpu_words().unwrap() as u64) } else { 0 };
            let _ = words;
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
        for (k, m) in mix.iter_mut().enumerate() {
            *m = cpu::dot(&hc.f[k * n..(k + 1) * n], x) * rs;
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
        for l in 0..hp.n_layer {
            self.attention(st, l, toks, pos0, &mut x)?;
            self.ffn(st, l, toks, &mut x)?;
            if !on_layer(l) {
                self.reset(st);
                bail!(Cancelled);
            }
        }
        st.pos = pos0 + c;
        if !logits {
            return Ok(None);
        }
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
        b.argmax(hp.vocab, r.logits, r.arg);
        b.read(r.arg, 1);
        let out = b.run()?;
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
        for s in (0..toks.len()).step_by(TB) {
            let n = (toks.len() - s).min(TB);
            let xs = &mut x[s * hcd..(s + n) * hcd];
            let mut a = vec![0f32; n * d];
            let mut pc = vec![0f32; n * 20];
            a.par_chunks_mut(d).zip(pc.par_chunks_mut(20)).zip(xs.par_chunks(hcd)).for_each(|((a, pc), xr)| {
                let (post, comb) = pc.split_at_mut(4);
                self.hc_pre(xr, &ly.hc_attn, a, post, comb);
                rms(a, Some(&ly.attn_norm), hp.eps);
            });
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
            let mut rb = b.run()?.into_iter();
            let mut q = rb.next().unwrap();
            let mut kvr = rb.next().unwrap();
            let comp = ly.comp.as_ref().map(|_| (rb.next().unwrap(), rb.next().unwrap()));
            let idx = ly.icomp.as_ref().map(|_| (rb.next().unwrap(), rb.next().unwrap(), rb.next().unwrap(), rb.next().unwrap()));
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
            // output projection
            let mut b = self.gpu.batch();
            b.upload(r.q, &o);
            let ogd = (nh / hp.og * hd) as u32;
            b.mv(&ly.o_a, r.q, (nh * hd) as u32, r.oa, (hp.og * hp.olr) as u32, n, false, Some((hp.olr as u32, ogd)));
            b.mv(&ly.o_b, r.oa, (hp.og * hp.olr) as u32, r.ob, d as u32, n, false, None);
            b.read(r.ob, n * d);
            let ob = b.run()?.pop().unwrap();
            xs.par_chunks_mut(hcd).zip(ob.par_chunks(d)).zip(pc.par_chunks(20)).for_each(|((xr, f), pc)| self.hc_post(xr, f, &pc[..4], &pc[4..]));
        }
        Ok(())
    }

    /// The experts each token goes to, and their weights.
    fn route(&self, l: usize, f: &[f32], tok: u32) -> ([u32; 8], [f32; 8]) {
        let hp = &self.hp;
        let ly = &self.layers[l];
        let (d, ne, k) = (hp.d, hp.ne, hp.k);
        let mut sc = vec![0f32; ne];
        for (e, s) in sc.iter_mut().enumerate() {
            let v = cpu::dot(&ly.router[e * d..(e + 1) * d], f);
            let sp = if v > 20.0 { v } else { v.exp().ln_1p() };
            *s = sp.sqrt();
        }
        let mut ids = [0u32; 8];
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
                idx.select_nth_unstable_by(k - 1, |&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
                idx[..k].sort_unstable_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
                for j in 0..k {
                    ids[j] = idx[j] as u32;
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
        (ids, w)
    }

    fn ffn(&self, st: &mut State, l: usize, toks: &[u32], x: &mut [f32]) -> Result<()> {
        let hp = &self.hp;
        let ly = &self.layers[l];
        let r = &self.r;
        let (d, ff, c, k) = (hp.d, hp.ff, toks.len(), hp.k);
        let hcd = hp.hc * d;
        let lim = hp.clamp[l];
        let mut f = vec![0f32; c * d];
        let mut pc = vec![0f32; c * 20];
        f.par_chunks_mut(d).zip(pc.par_chunks_mut(20)).zip(x.par_chunks(hcd)).for_each(|((fr, pc), xr)| {
            let (post, comb) = pc.split_at_mut(4);
            self.hc_pre(xr, &ly.hc_ffn, fr, post, comb);
            rms(fr, Some(&ly.ffn_norm), hp.eps);
        });
        let routes: Vec<([u32; 8], [f32; 8])> = f.par_chunks(d).zip(toks.par_iter()).map(|(fr, &t)| self.route(l, fr, t)).collect();
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
        self.store.touch(&touched);
        let small = c <= SMALL;
        let (mut on_vram, mut on_cpu, mut on_stage) = (Vec::new(), Vec::new(), Vec::new());
        {
            let v = self.vram.lock().unwrap();
            for &key in &keys {
                let e = key as usize % ne;
                if let Some(&s) = v.slot_of.get(&key) {
                    on_vram.push((key, v.slots[s]));
                } else if small && self.ex.cpu_ok && rows[e].len() <= CPU_MAX {
                    on_cpu.push(key);
                } else {
                    on_stage.push(key);
                }
            }
        }
        // drive reads first (CPU experts, then staged ones in windows)
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

        let mut y = vec![0f32; c * d];
        let mat_of = |(arena, base): (usize, u32)| -> (Mat, Mat) {
            let gu = Mat { buf: arena, base, ty: self.ex.ty[0], rows: 2 * ff, cols: d, wpr: self.ex.wpr_g };
            let dn = Mat { buf: arena, base: base + (2 * ff) as u32 * self.ex.wpr_g, ty: self.ex.ty[2], rows: d, cols: ff, wpr: self.ex.wpr_d };
            (gu, dn)
        };
        // a GPU batch of experts: rows gathered at r.xe..; returns the batch
        // and the (row offset, expert rows) to scatter afterwards
        let gather = |b: &mut Batch, list: &[(u32, (usize, u32))]| -> Vec<(usize, u32)> {
            let mut xe = Vec::new();
            let mut we = Vec::new();
            let mut at = Vec::new();
            for &(key, slot) in list {
                let e = key as usize % ne;
                let (gu, dn) = mat_of(slot);
                let r0 = xe.len() / d;
                for &(i, w) in &rows[e] {
                    xe.extend_from_slice(&f[i * d..(i + 1) * d]);
                    we.push(w);
                }
                let nr = rows[e].len();
                b.mv(&gu, r.xe + (r0 * d) as u32, d as u32, r.egu + (r0 * 2 * ff) as u32, (2 * ff) as u32, nr, false, None);
                b.swiglu(ff, r.egu + (r0 * 2 * ff) as u32, r.egu + (r0 * 2 * ff + ff) as u32, r.eh + (r0 * ff) as u32, (2 * ff) as u32, ff as u32, Some(r.ew + r0 as u32), nr, lim);
                b.mv(&dn, r.eh + (r0 * ff) as u32, ff as u32, r.ey + (r0 * d) as u32, d as u32, nr, false, None);
                at.push((r0, key));
            }
            b.upload(r.xe, &xe);
            b.upload(r.ew, &we);
            if !xe.is_empty() {
                b.read(r.ey, xe.len());
            }
            at
        };
        let scatter = |y: &mut [f32], ey: &[f32], at: &[(usize, u32)]| {
            for &(r0, key) in at {
                for (j, &(i, _)) in rows[key as usize % ne].iter().enumerate() {
                    let src = &ey[(r0 + j) * d..(r0 + j + 1) * d];
                    y[i * d..(i + 1) * d].iter_mut().zip(src).for_each(|(a, b)| *a += b);
                }
            }
        };
        // groups of VRAM experts within the row budget
        let mut groups: Vec<Vec<(u32, (usize, u32))>> = Vec::new();
        {
            let mut cur = Vec::new();
            let mut nrow = 0;
            for &(key, slot) in &on_vram {
                let n = rows[key as usize % ne].len();
                if nrow + n > EROWS && !cur.is_empty() {
                    groups.push(std::mem::take(&mut cur));
                    nrow = 0;
                }
                cur.push((key, slot));
                nrow += n;
            }
            if !cur.is_empty() {
                groups.push(cur);
            }
        }
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
            // CPU experts while the GPU works (small passes only)
            if s == 0 && !cpu_pins.is_empty() {
                self.cpu_experts(&cpu_pins, &on_cpu, &rows, &f, &mut y, lim)?;
            }
            let mut out = pending.wait()?.into_iter();
            let sy = out.next().unwrap();
            y[s * d..(s + n) * d].iter_mut().zip(&sy).for_each(|(a, b)| *a += b);
            if let Some(ey) = out.next() {
                scatter(&mut y, &ey, &at);
            }
        }
        for gr in groups.iter().skip(if small { 1 } else { 0 }) {
            let mut b = self.gpu.batch();
            let at = gather(&mut b, gr);
            let ey = b.run()?.pop().unwrap_or_default();
            scatter(&mut y, &ey, &at);
        }
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
                self.store.wait(&pin)?;
                let slot = self.stage[list.len()];
                Self::upload_expert(&self.gpu, &self.ex, slot, &pin);
                list.push((key, slot));
                pins.push(pin);
                nrow += n;
            }
            refill(&mut stage_pins, &mut next_stage)?;
            let mut b = self.gpu.batch();
            let at = gather(&mut b, &list);
            let ey = b.run()?.pop().unwrap_or_default();
            drop(pins);
            scatter(&mut y, &ey, &at);
        }
        if c > 0 && cpu_pins.is_empty() && !on_cpu.is_empty() {
            unreachable!();
        }
        x.par_chunks_mut(hcd).zip(y.par_chunks(d)).zip(pc.par_chunks(20)).for_each(|((xr, yr), pc)| self.hc_post(xr, yr, &pc[..4], &pc[4..]));
        Ok(())
    }

    /// Experts in RAM for a few tokens each, on the CPU, added into y.
    fn cpu_experts(&self, pins: &[Pin], keys: &[u32], rows: &[Vec<(usize, f32)>], f: &[f32], y: &mut [f32], lim: f32) -> Result<()> {
        let (d, ff, ne) = (self.hp.d, self.hp.ff, self.hp.ne);
        let bpr_g = d / 256 * 144;
        let bpr_d = ff / 256 * 144;
        let mut qx: std::collections::HashMap<usize, Q8x> = std::collections::HashMap::new();
        for (pin, &key) in pins.iter().zip(keys) {
            self.store.wait(pin)?;
            let (gw, uw, dw) = (pin.piece(0), pin.piece(1), pin.piece(2));
            for &(i, wt) in &rows[key as usize % ne] {
                let xq = qx.entry(i).or_insert_with(|| Q8x::new(&f[i * d..(i + 1) * d]));
                let mut g = vec![0f32; ff];
                let mut u = vec![0f32; ff];
                g.par_chunks_mut(64).enumerate().for_each(|(ci, o)| cpu::q4k_rows(&gw[ci * 64 * bpr_g..], xq, o));
                u.par_chunks_mut(64).enumerate().for_each(|(ci, o)| cpu::q4k_rows(&uw[ci * 64 * bpr_g..], xq, o));
                let h: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(&gv, &uv)| {
                        let (gv, uv) = if lim > 0.0 { (gv.min(lim), uv.clamp(-lim, lim)) } else { (gv, uv) };
                        gv / (1.0 + (-gv).exp()) * uv * wt
                    })
                    .collect();
                let hq = Q8x::new(&h);
                let mut o = vec![0f32; d];
                o.par_chunks_mut(64).enumerate().for_each(|(ci, oc)| cpu::q4k_rows(&dw[ci * 64 * bpr_d..], &hq, oc));
                y[i * d..(i + 1) * d].iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            }
        }
        Ok(())
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

    /// Greedy decoding of up to `n` tokens after `prompt`, whose first
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
        }
        let t_first = Instant::now();
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

    /// (drive reads, RAM hits, moves into VRAM)
    pub fn io_counts(&self) -> (u64, u64, usize) {
        let (a, b) = self.store.counts();
        (a, b, self.vram.lock().unwrap().moved)
    }
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
    let _ = eng.save_usage();
    let dec = stats.generated.saturating_sub(1);
    let (reads, hits, moved) = eng.io_counts();
    eprintln!(
        "[pelica] prefill {} tokens in {:.1}s ({:.2} tok/s); decode {dec} tokens in {:.1}s: {:.2} tok/s; expert reads {reads}, RAM hits {hits}, moved to VRAM {moved}",
        stats.prompt,
        stats.prefill_s,
        stats.prompt as f64 / stats.prefill_s.max(1e-9),
        stats.decode_s,
        dec as f64 / stats.decode_s.max(1e-9),
    );
    Ok(())
}
