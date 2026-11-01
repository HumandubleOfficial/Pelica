//! Pelica's GPU-resident engine for models that fit in VRAM (gpt-oss for
//! now), read from a GGUF file.
//!
//! The whole token runs on the GPU: the model's weights stay in their stored
//! formats (MXFP4 experts, Q8_0 attention and output; only re-laid out so
//! each block is word-aligned, values untouched), activations are f32, and
//! the routing, top-k, attention and argmax happen in kernels too.
//!
//! A pass runs one or more tokens through the model. The prompt but its last
//! token goes in prompt passes of up to PMAX tokens (`oss_prefill`: tiled
//! products, on the tensor cores where offered, that only fill the cache);
//! its last token takes the exact path below. After it, each pass checks the current token
//! plus a draft of the tokens that may follow it (prompt lookup: what came
//! after the last place the text ended the same way), keeps the drafted
//! tokens the model agrees with and adds the model's own next one. Drafting
//! and checking happen on the GPU, so the CPU never waits on a pass: it
//! records pass after pass and reads tokens back a couple of passes behind.
//!
//! The kernels (`oss_batch`) are built for passes of 1, 2, 4 and 8 tokens;
//! every build does each token's arithmetic the same way, so drafting never
//! changes the output: it is exactly greedy decoding one token at a time.

use crate::gguf::{Gguf, Tensor, Ty};
use crate::oss_batch as kb;
use crate::oss_kernels as k;
use crate::oss_prefill as kp;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;

/// Percent of layers uploaded by the `Engine::load` in progress.
pub static LOAD_PROGRESS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Model shape, from the GGUF metadata.
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub n_exp: usize,
    pub top_k: usize,
    pub ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub swa: usize,
    pub ctx: usize,
}

/// state buffer: the token history starts here (the words below: oss_batch)
const PROMPT: u32 = 256;
/// argmax partials (workgroups in the first pass)
const ARG_GROUPS: u32 = 256;
/// passes whose tokens are not yet read back when the next one is recorded
const LAG: usize = 2;
/// prompt passes reaching past this position are submitted a layer at a
/// time and waited for (their attention takes long)
const LONG_PASS: usize = 4096;
/// The kernels' builds: register arrays for this many tokens.
const BUILDS: [usize; 4] = [1, 2, 4, 8];

/// The build a pass of `t` tokens uses.
fn build(t: usize) -> usize {
    BUILDS.iter().position(|&b| t <= b).unwrap_or(BUILDS.len() - 1)
}

/// How a dispatch changes with the batched build in use.
#[derive(Clone, Copy, PartialEq)]
enum Vary {
    No,
    /// one pipeline (and bind group) per build
    Build,
    /// and a grid row per expert the pass can need
    Experts,
}

struct Disp {
    pipe: usize,
    binds: Vec<wgpu::BindGroup>,
    groups: (u32, u32),
    vary: Vary,
}

/// Where a dispatch goes.
#[derive(Clone, Copy, PartialEq)]
enum Grp {
    Body,
    Head,
    Spec,
    /// a prompt pass (oss_prefill)
    Pre,
}

/// A byte range of one of the weight buffers.
#[derive(Clone, Copy)]
struct Reg {
    buf: usize,
    off: u64,
    size: u64,
}

/// One Q8_0 matrix in VRAM: int8 values (row-major) and f16 block scales.
struct GQ8 {
    q: Reg,
    d: Reg,
    rows: usize,
    cols: usize,
}

/// One MXFP4 expert tensor: 16-byte value blocks and e8m0 block scales.
struct GMx {
    q: Reg,
    e: Reg,
}

/// Host-side staging of several pieces that become one VRAM buffer (the
/// allocator wastes space on many medium-sized buffers).
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

pub struct Engine {
    pub cfg: Cfg,
    /// the GPU and API, for display
    pub name: String,
    /// the model's own context length (the most `load` accepts)
    pub max_ctx: usize,
    /// positions of the full-attention layers' cache held in VRAM; the rest
    /// (up to ctx) live in host RAM that the GPU reads in place
    pub resident: usize,
    /// bytes of host RAM holding cache positions past `resident`
    pub host_bytes: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Vec<wgpu::ComputePipeline>,
    /// a pass: the layers, then each token's logits and argmax
    body: Vec<Disp>,
    head: Vec<Disp>,
    /// after a pass's argmaxes: accept, draft
    spec: Vec<Disp>,
    /// a prompt pass of up to PMAX tokens
    pre: Vec<Disp>,
    st: wgpu::Buffer,
    reads: Vec<wgpu::Buffer>,
    /// activations, kept for `check`
    actb: wgpu::Buffer,
    xin: wgpu::Buffer,
    xa: wgpu::Buffer,
    blog: u32,
    /// batched regions: x, q, router logits, down outputs; attention output (xin)
    breg: [u32; 4],
    batt: u32,
    /// the prompt passes' expert plan (kept for `profile`)
    pl: wgpu::Buffer,
    /// per layer: a full layer's host-RAM keys and values, if any; and the
    /// VRAM staging copy prompt passes read them from
    far: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    stage: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

/// Pipelines, by index.
const P_EMBED_T: usize = 0;
const P_NORM_T: usize = 1;
const P_ROPE_T: usize = 2;
const P_ATTN_T: usize = 3;
const P_ROUTE_T: usize = 4;
const P_PLAN_T: usize = 5;
const P_ARG1_T: usize = 6;
const P_ARG2_T: usize = 7;
const P_ACCEPT: usize = 8;
const P_DRAFT: usize = 9;
/// one pipeline per build from here on (matvec: for 1, 2 and 4 rows per subgroup)
const P_MV_T: usize = 10;
const P_GATEUP_T: usize = 22;
const P_DOWN_T: usize = 26;
const MV_ROWS: [usize; 3] = [1, 2, 4];
/// prompt passes
const P_EMBED_P: usize = 30;
const P_NORM_P: usize = 31;
const P_ROUTE_P: usize = 32;
const P_PLAN_P: usize = 33;
const P_MM_Q8: usize = 34;
const P_MM_GATEUP: usize = 35;
const P_MM_DOWN: usize = 36;
const P_ATTN_P: usize = 37;
const P_TC_Q8: usize = 38;
const P_TC_GATEUP: usize = 39;
const P_TC_DOWN: usize = 40;
/// full-layer attention in passes: split over keys, then merged
const P_ATTN_S: usize = 41;
const P_ATTN_M: usize = 42;
const P_NAMES: [&str; 43] = [
    "embed", "combine+norm", "rope+kv", "attention", "norm+router", "expert plan", "argmax 1", "argmax 2", "accept", "draft",
    "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8", "matvec q8",
    "moe gate/up", "moe gate/up", "moe gate/up", "moe gate/up", "moe down", "moe down", "moe down", "moe down",
    "embed", "combine+norm", "norm+router", "expert plan", "matmul q8", "moe gate/up mm", "moe down mm", "attention",
    "matmul q8 tc", "moe gate/up tc", "moe down tc", "attention split", "attention merge",
];

/// Timings of one generation.
pub struct Stats {
    pub prompt: usize,
    pub generated: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    /// decode passes (each checks a token plus any draft)
    pub passes: usize,
    /// per decode pass: tokens, experts used (summed over layers), tokens kept
    pub log: Vec<(u32, u32, u32)>,
}

/// Split Q8_0 blocks (f16 scale + 32 x i8) into a value array and a scale array.
fn split_q8(raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let nb = raw.len() / 34;
    let mut q = vec![0u8; nb * 32];
    let mut d = vec![0u8; nb * 2];
    q.par_chunks_mut(32 * 1024).zip(d.par_chunks_mut(2 * 1024)).enumerate().for_each(|(c, (q, d))| {
        for b in 0..q.len() / 32 {
            let src = &raw[(c * 1024 + b) * 34..][..34];
            d[b * 2..b * 2 + 2].copy_from_slice(&src[..2]);
            q[b * 32..b * 32 + 32].copy_from_slice(&src[2..]);
        }
    });
    (q, d)
}

/// Split MXFP4 blocks (e8m0 scale + 16 bytes) into values and scales.
fn split_mx(raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let nb = raw.len() / 17;
    let mut q = vec![0u8; nb * 16];
    let mut e = vec![0u8; nb.div_ceil(4) * 4];
    q.par_chunks_mut(16 * 4096).zip(e.par_chunks_mut(4096)).enumerate().for_each(|(c, (q, e))| {
        for b in 0..q.len() / 16 {
            let src = &raw[(c * 4096 + b) * 17..][..17];
            e[b] = src[0];
            q[b * 16..b * 16 + 16].copy_from_slice(&src[1..]);
        }
    });
    (q, e)
}

fn f32s(g: &Gguf, t: &Tensor) -> Result<Vec<f32>> {
    if t.ty != Ty::F32 {
        bail!("expected an f32 tensor, got {:?}", t.ty);
    }
    Ok(g.bytes(t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

impl Engine {
    pub fn load(path: &std::path::Path, ctx: usize) -> Result<Engine> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        if g.s("general.architecture") != Some("gpt-oss") {
            bail!("the GPU-resident engine supports gpt-oss GGUF files (got {:?})", g.s("general.architecture"));
        }
        let a = "gpt-oss";
        let cfg = Cfg {
            n_layer: g.u(&format!("{a}.block_count"))?,
            d: g.u(&format!("{a}.embedding_length"))?,
            n_head: g.u(&format!("{a}.attention.head_count"))?,
            n_kv: g.u(&format!("{a}.attention.head_count_kv"))?,
            hd: g.u(&format!("{a}.attention.key_length"))?,
            n_exp: g.u(&format!("{a}.expert_count"))?,
            top_k: g.u(&format!("{a}.expert_used_count"))?,
            ff: g.u(&format!("{a}.expert_feed_forward_length"))?,
            vocab: g.tensor("output.weight")?.dims[1],
            eps: g.f(&format!("{a}.attention.layer_norm_rms_epsilon"))?,
            swa: g.u(&format!("{a}.attention.sliding_window"))?,
            ctx,
        };
        let max_ctx = g.u(&format!("{a}.context_length")).unwrap_or(131072);
        if ctx > max_ctx {
            bail!("--ctx {ctx} is past the model's context length ({max_ctx})");
        }
        if 64 / (cfg.n_head / cfg.n_kv) < kb::TMAX {
            bail!("unsupported gpt-oss shape (more than 8 query heads per kv head): {cfg:?}");
        }
        if cfg.hd != 64 || 64 % (cfg.n_head / cfg.n_kv) != 0 || cfg.top_k != 4 || cfg.n_exp > 32 || cfg.d > 4096 || cfg.ff > 4096 || cfg.n_head * cfg.hd > 8192 || cfg.d % 32 != 0 || cfg.ff % 32 != 0 {
            bail!("unsupported gpt-oss shape: {cfg:?}");
        }

        // device with subgroups (one 32-lane subgroup per matrix row)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if !adapter.features().contains(wgpu::Features::SUBGROUP) || info.subgroup_min_size > 32 || info.subgroup_max_size < 32 {
            bail!("{} lacks 32-wide subgroups, which the GPU-resident engine needs", info.name);
        }
        let limits = adapter.limits();
        // (host buffers the GPU reads in place, for cache past VRAM)
        let mappable = adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        // tensor cores for the prompt passes' products, if offered
        // (PELICA_TC: which of q8,gu,down; 0: none)
        let tcf = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        let tc_set = std::env::var("PELICA_TC").unwrap_or_else(|_| "q8,gu,down".into());
        let tc = adapter.features().contains(tcf)
            && adapter.cooperative_matrix_properties().iter().any(|c| c.m_size == 16 && c.n_size == 16 && c.k_size == 16 && c.ab_type == wgpu::CooperativeScalarType::F16 && c.cr_type == wgpu::CooperativeScalarType::F32)
            && tc_set != "0";
        let use_tc = |k: &str| tc && tc_set.split(',').any(|x| x == k);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pelica-oss"),
            required_features: wgpu::Features::SUBGROUP | mappable | (adapter.features() & prof_features()) | if tc { tcf } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            // SAFETY: cooperative matrices are only used as wgpu documents
            // them (16x16 f16 x f16 + f32, a configuration the device lists)
            experimental_features: if tc { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;
        let name = format!("{} ({:?})", info.name, info.backend);

        // expert rows each subgroup sums at once, per build
        let nrows: Vec<usize> = std::env::var("PELICA_NROWS").unwrap_or_else(|_| "2,2,2,2".into()).split(',').map(|v| v.parse().unwrap_or(2)).collect();
        let src = |body: &str, _tb: usize| {
            // enable directives first
            let (en, body): (Vec<&str>, Vec<&str>) = body.lines().partition(|l| l.starts_with("enable "));
            let body = body.join("\n");
            format!(
                "{}{}{}{}",
                en.iter().map(|l| format!("{l}\n")).collect::<String>(),
                k::COMMON,
                if body.contains("lo4(") { k::MX } else { "" },
                body.replace("TOPK", k::TOPK)
                    .replace("BCOMMON", kb::BCOMMON)
                    .replace("ROUTED", kb::ROUTED)
                    .replace("PCOMMON", &kp::pcommon())
                    .replace("MAXN", &cfg.swa.to_string())

            )
        };
        let mut sources: Vec<String> = [kb::EMBED_T, kb::NORM_T, kb::ROPE_KV_T, kb::ATTN_T, kb::ROUTE_T, kb::PLAN_T, kb::ARGMAX1_T, kb::ARGMAX2_T, kb::ACCEPT, kb::DRAFT]
        .iter()
        .map(|s| src(s, 1))
        .collect();
        for nr in MV_ROWS {
            for tb in BUILDS {
                sources.push(src(&kb::mv_q8(tb, nr), tb));
            }
        }
        let nr = |tb: usize| nrows.get(build(tb)).copied().unwrap_or(2).clamp(1, 4);
        for tb in BUILDS {
            sources.push(src(&kb::gateup(tb, nr(tb)), tb));
        }
        for tb in BUILDS {
            sources.push(src(&kb::down(tb, nr(tb)), tb));
        }
        for s in [kp::EMBED_P, kp::NORM_P, kp::ROUTE_P, kp::PLAN_P] {
            sources.push(src(s, 1));
        }
        // tiles: rows, rows per thread, tokens per thread (PELICA_MM_*: tuning)
        let tile = |name: &str, def: [usize; 3]| -> [usize; 3] {
            let v: Vec<usize> = std::env::var(name).ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
            if v.len() == 3 { [v[0], v[1], v[2]] } else { def }
        };
        let (fq, fg, fd) = (tile("PELICA_MM_Q8", [64, 4, 4]), tile("PELICA_MM_GATEUP", [64, 4, 4]), tile("PELICA_MM_DOWN", [128, 8, 4]));
        // (tensor cores: subgroup columns, 16x16 tiles per subgroup down and across)
        let (cq, cg, cd) = (tile("PELICA_TC_Q8", [2, 2, 2]), tile("PELICA_TC_GATEUP", [2, 1, 2]), tile("PELICA_TC_DOWN", [2, 2, 2]));
        // each product's kernel: (pipeline, tile rows, tile tokens)
        let pick = |name: &str, f: [usize; 3], c: [usize; 3], pf: usize, pc: usize| -> (usize, u32, u32) {
            let k = if use_tc(name) { (pc, 8 / c[0] * 16 * c[1], c[0] * 16 * c[2]) } else { (pf, f[0], 256 / (f[0] / f[1]) * f[2]) };
            (k.0, k.1 as u32, k.2 as u32)
        };
        let (kq, kg, kd) = (pick("q8", fq, cq, P_MM_Q8, P_TC_Q8), pick("gu", fg, cg, P_MM_GATEUP, P_TC_GATEUP), pick("down", fd, cd, P_MM_DOWN, P_TC_DOWN));
        assert!(kg.2 as usize == kp::MOE_TT && kd.2 as usize == kp::MOE_TT && kp::PMAX % kq.2 as usize == 0, "tile shapes");
        let tt = |t: [usize; 3]| 256 / (t[0] / t[1]) * t[2];
        sources.push(src(&kp::mm(kp::Mm::Q8, fq[0], tt(fq), fq[1], fq[2]), 1));
        sources.push(src(&kp::mm(kp::Mm::GateUp, fg[0], tt(fg), fg[1], fg[2]), 1));
        sources.push(src(&kp::mm(kp::Mm::Down, fd[0], tt(fd), fd[1], fd[2]), 1));
        sources.push(src(&kp::attn_p(), 1));
        // (without tensor cores: placeholders)
        for (kind, c, n) in [(kp::Mm::Q8, cq, "q8"), (kp::Mm::GateUp, cg, "gu"), (kp::Mm::Down, cd, "down")] {
            sources.push(if use_tc(n) { src(&kp::mm_tc(kind, c[0], c[1], c[2]), 1) } else { sources[P_MM_Q8].clone() });
        }
        sources.push(src(&kb::attn_s(), 1));
        sources.push(src(kb::ATTN_M, 1));
        assert_eq!(sources.len(), P_NAMES.len());
        let pipes: Vec<wgpu::ComputePipeline> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("oss kernel {i}")), source: wgpu::ShaderSource::Wgsl(s.as_str().into()) });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("oss kernel {i}")),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        let vram = std::cell::Cell::new(0usize);
        // one reusable staging buffer (mapped-at-creation buffers each keep a
        // staging copy alive until the driver gets round to freeing it)
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 64 << 20,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let upload = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| -> wgpu::Buffer {
            let size = (bytes.len().max(16) as u64).div_ceil(16) * 16;
            vram.set(vram.get() + size as usize);
            let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let b = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            if let Some(e) = pollster::block_on(scope.pop()) {
                panic!("allocating {label} ({size} bytes, {:.2} GB so far): {e}", vram.get() as f64 / 1e9);
            }
            let n4 = bytes.len() / 4 * 4;
            crate::gpu::write_through(&device, &queue, &staging, &[(&b, 0, &bytes[..n4])]);
            if n4 < bytes.len() {
                let mut tail = [0u8; 4];
                tail[..bytes.len() - n4].copy_from_slice(&bytes[n4..]);
                crate::gpu::write_through(&device, &queue, &staging, &[(&b, n4 as u64, &tail)]);
            }
            b
        };
        let ro = wgpu::BufferUsages::STORAGE;
        let mut wbufs: Vec<wgpu::Buffer> = Vec::new();
        let q8 = |name: &str, blob: &mut Blob| -> Result<GQ8> {
            let t = g.tensor(name)?;
            if t.ty != Ty::Q8_0 {
                bail!("{name}: expected Q8_0, got {:?}", t.ty);
            }
            let (q, d) = split_q8(g.bytes(t));
            Ok(GQ8 { q: blob.add(&q), d: blob.add(&d), rows: t.dims[1], cols: t.dims[0] })
        };
        let mx = |name: &str, blob: &mut Blob| -> Result<GMx> {
            let t = g.tensor(name)?;
            if t.ty != Ty::Mxfp4 {
                bail!("{name}: expected MXFP4, got {:?}", t.ty);
            }
            let (q, e) = split_mx(g.bytes(t));
            Ok(GMx { q: blob.add(&q), e: blob.add(&e) })
        };
        let finish = |mut blob: Blob, wbufs: &mut Vec<wgpu::Buffer>| {
            assert_eq!(blob.buf, wbufs.len());
            let n = blob.data.len().div_ceil(16) * 16;
            blob.data.resize(n, 0);
            wbufs.push(upload("weights", &blob.data, ro));
        };

        // small f32 weights, all in one buffer
        let mut fw: Vec<f32> = Vec::new();
        let mut put = |v: Vec<f32>| -> u32 {
            let at = fw.len() as u32;
            fw.extend(v);
            while fw.len() % 4 != 0 {
                fw.push(0.0);
            }
            at
        };

        struct L {
            attn_norm: u32,
            post_norm: u32,
            qkv: GQ8,
            qkv_b: u32,
            o: GQ8,
            o_b: u32,
            sinks: u32,
            router: u32,
            router_b: u32,
            gate: GMx,
            up: GMx,
            down: GMx,
            gate_b: u32,
            up_b: u32,
            down_b: u32,
        }
        let mut blob = Blob { data: Vec::new(), buf: 0 };
        let embd = q8("token_embd.weight", &mut blob)?;
        finish(blob, &mut wbufs);
        let mut blob = Blob { data: Vec::new(), buf: 1 };
        let out = q8("output.weight", &mut blob)?;
        finish(blob, &mut wbufs);
        let out_norm = put(f32s(&g, g.tensor("output_norm.weight")?)?);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let t = |s: &str| format!("blk.{l}.{s}");
            let mut blob = Blob { data: Vec::new(), buf: wbufs.len() };
            // q, k and v share their input: stack them into one matrix
            let (mut qq, mut qd) = (Vec::new(), Vec::new());
            let mut rows = 0;
            for n in ["attn_q", "attn_k", "attn_v"] {
                let tt = g.tensor(&t(&format!("{n}.weight")))?;
                if tt.ty != Ty::Q8_0 || tt.dims[0] != cfg.d {
                    bail!("{}: unexpected layout", t(n));
                }
                let (q, d) = split_q8(g.bytes(tt));
                qq.extend(q);
                qd.extend(d);
                rows += tt.dims[1];
            }
            let qkv = GQ8 { q: blob.add(&qq), d: blob.add(&qd), rows, cols: cfg.d };
            let mut qkv_bias = f32s(&g, g.tensor(&t("attn_q.bias"))?)?;
            qkv_bias.extend(f32s(&g, g.tensor(&t("attn_k.bias"))?)?);
            qkv_bias.extend(f32s(&g, g.tensor(&t("attn_v.bias"))?)?);
            layers.push(L {
                attn_norm: put(f32s(&g, g.tensor(&t("attn_norm.weight"))?)?),
                post_norm: put(f32s(&g, g.tensor(&t("post_attention_norm.weight"))?)?),
                qkv,
                qkv_b: put(qkv_bias),
                o: q8(&t("attn_output.weight"), &mut blob)?,
                o_b: put(f32s(&g, g.tensor(&t("attn_output.bias"))?)?),
                sinks: put(f32s(&g, g.tensor(&t("attn_sinks.weight"))?)?),
                router: put(f32s(&g, g.tensor(&t("ffn_gate_inp.weight"))?)?),
                router_b: put(f32s(&g, g.tensor(&t("ffn_gate_inp.bias"))?)?),
                gate: mx(&t("ffn_gate_exps.weight"), &mut blob)?,
                up: mx(&t("ffn_up_exps.weight"), &mut blob)?,
                down: mx(&t("ffn_down_exps.weight"), &mut blob)?,
                gate_b: put(f32s(&g, g.tensor(&t("ffn_gate_exps.bias"))?)?),
                up_b: put(f32s(&g, g.tensor(&t("ffn_up_exps.bias"))?)?),
                down_b: put(f32s(&g, g.tensor(&t("ffn_down_exps.bias"))?)?),
            });
            finish(blob, &mut wbufs);
            eprint!("\r[pelica] uploading layer {}/{} ({:.2} GB)   ", l + 1, cfg.n_layer, vram.get() as f64 / 1e9);
            LOAD_PROGRESS.store((l + 1) * 100 / cfg.n_layer, std::sync::atomic::Ordering::Relaxed);
        }
        let fw_buf = upload("fw", bytemuck_f32(&fw), ro);

        // RoPE table (YaRN, NeoX pairs), computed in f64
        let rope = rope_table(&g, &cfg)?;
        let rope_buf = upload("rope", bytemuck_f32(&rope), ro);

        // VRAM left once the weights are in (sizes the cache below)
        let free_now = crate::free_vram();
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage: rw, mapped_at_creation: false });
        let st = mk("state", (PROMPT as u64 + ctx as u64 + kb::TMAX as u64 + 8) * 4);
        // activations: each region holds a prompt pass's tokens' worth (the
        // logits and argmax partials: a batched pass's)
        let (tm, pm) = (kb::TMAX, kp::PMAX);
        let qkv_rows = (cfg.n_head + 2 * cfg.n_kv) * cfg.hd;
        let mut bsize = 0u32;
        let mut region = |n: usize, toks: usize| -> u32 {
            let at = bsize;
            bsize += (toks * n).div_ceil(64) as u32 * 64;
            at
        };
        let (bx, bq, blg, bwt, bp4, bpart, blog) = (region(cfg.d, pm), region(qkv_rows, pm), region(32, pm), region(4, pm), region(4 * cfg.d, pm), region(512, tm), region(cfg.vocab, tm));
        let actb = mk("act batched", bsize as u64 * 4);
        // matrix inputs, in buffers of their own so kernels read them as
        // read-only vec4s: normed x and attention output; the expert hidden
        let (bxn, batt) = (0u32, (pm * cfg.d) as u32);
        let xin = mk("batched inputs", ((pm * cfg.d + pm * cfg.n_head * cfg.hd) * 4) as u64);
        let ba = 0u32;
        let xa = mk("batched expert hidden", (pm * 4 * cfg.ff * 4) as u64);
        let pl = mk("expert plan", kp::PL_WORDS as u64 * 4);

        // KV cache. Sliding-window layers keep a ring of `swa` (plus a prompt
        // pass's tokens, which are written before the first of them reads the
        // oldest of its window). Full layers keep every position: the first
        // `res` in VRAM, as many as fit, the rest in host RAM, which the GPU
        // reads in place when decoding and through a VRAM staging copy in
        // prompt passes (which read every key 64 times).
        let kv_row = cfg.n_kv * cfg.hd;
        let n_full = cfg.n_layer / 2;
        let swa_cap = (cfg.swa + kp::PMAX).min(ctx);
        // passes' full-layer attention: key splits of a fixed size
        let split = (ctx / 512).next_power_of_two().clamp(32, 256);
        let n_split = ctx.div_ceil(split);
        let part = mk("attention partials", (kb::TMAX * cfg.n_head * n_split * kb::PART * 4) as u64);
        let others = bsize as usize * 4 + xin.size() as usize + xa.size() as usize + pl.size() as usize + part.size() as usize + st.size() as usize;
        let ring_floats = (cfg.n_layer - n_full) * swa_cap * kv_row;
        // a position in VRAM (keys and values of every full layer), and one
        // in host RAM (its staging copy)
        let (pos_bytes, stage_bytes) = (n_full * kv_row * 8, kv_row * 8);
        let bind_max = (limits.max_storage_buffer_binding_size as usize / 4).saturating_sub(ring_floats) / (n_full * kv_row);
        let res = match std::env::var("PELICA_KV_RESIDENT").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(r) => r,
            None => match free_now {
                None => ctx,
                Some(f) => {
                    let budget = f.saturating_sub(crate::vram_margin() + others + ring_floats * 8);
                    if ctx * pos_bytes <= budget { ctx } else { budget.saturating_sub(ctx * stage_bytes) / (pos_bytes - stage_bytes) / 256 * 256 }
                }
            },
        }
        .min(ctx)
        .min(bind_max);
        let far_rows = ctx - res;
        let host_bytes = n_full * far_rows * kv_row * 8;
        if far_rows > 0 {
            if mappable.is_empty() {
                bail!("--ctx {ctx} needs more VRAM than is free (room for about {res} positions), and this GPU cannot read host memory");
            }
            let ram = crate::available_ram() as usize;
            let keep = 4usize << 30;
            if host_bytes + keep > ram {
                let fits = res + ram.saturating_sub(keep) / (n_full * kv_row * 8);
                bail!("--ctx {ctx} needs {:.1} GB of RAM for the cache past VRAM ({:.1} GB free); about {fits} positions fit", host_bytes as f64 / 1e9, ram as f64 / 1e9);
            }
        }
        let mut kv_base = Vec::new();
        let mut kv_floats = 0usize;
        for l in 0..cfg.n_layer {
            let cap = if l % 2 == 0 { swa_cap } else { res };
            kv_base.push((kv_floats as u32, cap as u32));
            kv_floats += cap * kv_row;
        }
        let kc = mk("k cache", kv_floats as u64 * 4);
        let vc = mk("v cache", kv_floats as u64 * 4);
        let host = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (far_rows * kv_row * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let far: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> = (0..cfg.n_layer).map(|l| (l % 2 == 1 && far_rows > 0).then(|| (host("k cache (RAM)"), host("v cache (RAM)")))).collect();
        let stage = (far_rows > 0).then(|| (mk("k staging", (far_rows * kv_row * 4) as u64), mk("v staging", (far_rows * kv_row * 4) as u64)));
        let dummy = mk("no far cache", 16);
        vram.set(vram.get() + kv_floats * 8 + others + stage.as_ref().map_or(0, |s| s.0.size() as usize * 2));
        let reads = (0..LAG + 2)
            .map(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("token readback"), size: 256, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
            .collect();

        // every dispatch, with its parameters, recorded once
        let groups = |n: usize| -> (u32, u32) {
            let x = n.min(65535) as u32;
            (x, (n as u32).div_ceil(x))
        };
        let eps = cfg.eps.to_bits();
        let d = cfg.d as u32;
        let none = 0xffff_ffffu32;
        // (pipe, extra bindings, groups, where, args, vary)
        let mut ops: Vec<(usize, Vec<(u32, wgpu::BufferBinding)>, (u32, u32), Grp, Vec<u32>, Vary)> = Vec::new();
        let w = |r: Reg| wgpu::BufferBinding { buffer: &wbufs[r.buf], offset: r.off, size: wgpu::BufferSize::new(r.size) };
        fn all(b: &wgpu::Buffer) -> wgpu::BufferBinding<'_> {
            b.as_entire_buffer_binding()
        }
        let rope_threads = (cfg.n_head + cfg.n_kv) * 32 + cfg.n_kv * cfg.hd;
        let ne = cfg.n_exp as u32;
        let scale = (1.0f32 / (cfg.hd as f32).sqrt()).to_bits();
        let hpk = (cfg.n_head / cfg.n_kv) as u32;
        // rows each subgroup computes: more amortises loading the input
        let (rq, ro, rh) = (env_usize("PELICA_RPS_QKV", 2), env_usize("PELICA_RPS_O", 2), env_usize("PELICA_RPS_HEAD", 4));
        let mvp = |r: usize| P_MV_T + BUILDS.len() * MV_ROWS.iter().position(|&x| x == r).expect("rows per subgroup: 1, 2 or 4");
        let chunk = (cfg.vocab as u32).div_ceil(ARG_GROUPS);
        use Vary::No;

        // a pass of up to TMAX tokens
        let t = tm as u32;
        let attw = (cfg.n_head * cfg.hd) as u32;
        ops.push((P_EMBED_T, vec![(1, w(embd.q)), (2, w(embd.d)), (3, all(&actb)), (4, all(&st))], (d.div_ceil(256), t), Grp::Body, vec![d, bx], No));
        // a layer's far (host RAM) cache, or a placeholder, and the residency
        // its kernels test (sliding-window layers: !0, the ring)
        let far_of = |l: usize| far[l].as_ref().map_or((all(&dummy), all(&dummy)), |(k, v)| (all(k), all(v)));
        let res_of = |l: usize| if l % 2 == 0 { none } else { res as u32 };
        for (l, ly) in layers.iter().enumerate() {
            let (kvb, cap) = kv_base[l];
            let win = if l % 2 == 0 { cfg.swa as u32 } else { 0 };
            let fwb = all(&fw_buf);
            let (fk, fv) = far_of(l);
            // the attention norm, after adding the previous layer's experts
            let comb = if l == 0 { vec![none, 0, 0] } else { vec![bp4, layers[l - 1].down_b, bwt] };
            ops.push((P_NORM_T, vec![(3, all(&actb)), (4, all(&st)), (5, all(&fw_buf)), (10, all(&xin))], (t, 1), Grp::Body, [vec![d, bx, bxn, ly.attn_norm, eps], comb].concat(), No));
            ops.push((mvp(rq), vec![(1, w(ly.qkv.q)), (2, w(ly.qkv.d)), (3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (9, all(&xin))], groups(qkv_rows.div_ceil(8 * rq)), Grp::Body, vec![qkv_rows as u32, d, bxn / 4, bq, qkv_rows as u32, ly.qkv_b, 0], Vary::Build));
            ops.push((P_ROPE_T, vec![(3, all(&actb)), (4, all(&st)), (6, all(&rope_buf)), (7, all(&kc)), (8, all(&vc)), (11, fk.clone()), (12, fv.clone())], (rope_threads.div_ceil(256) as u32, t), Grp::Body, vec![bq, cfg.n_head as u32, cfg.n_kv as u32, kvb, cap, qkv_rows as u32, res_of(l)], No));
            if win > 0 {
                ops.push((
                    P_ATTN_T,
                    vec![(3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (7, all(&kc)), (8, all(&vc)), (10, all(&xin))],
                    (cfg.n_head as u32, t),
                    Grp::Body,
                    vec![bq, batt, kvb, cap, win, ly.sinks, scale, cfg.n_kv as u32, hpk, qkv_rows as u32, attw],
                    No,
                ));
            } else {
                ops.push((
                    P_ATTN_S,
                    vec![(3, all(&actb)), (4, all(&st)), (7, all(&kc)), (8, all(&vc)), (11, fk.clone()), (12, fv.clone()), (13, all(&part))],
                    (cfg.n_kv as u32, n_split as u32),
                    Grp::Body,
                    vec![bq, n_split as u32, kvb, res as u32, split as u32, cfg.n_head as u32, scale, cfg.n_kv as u32, hpk, qkv_rows as u32, far_rows as u32],
                    No,
                ));
                ops.push((P_ATTN_M, vec![(4, all(&st)), (5, fwb.clone()), (10, all(&xin)), (13, all(&part))], (cfg.n_head as u32, t), Grp::Body, vec![batt, n_split as u32, split as u32, ly.sinks, attw, cfg.n_head as u32], No));
            }
            ops.push((mvp(ro), vec![(1, w(ly.o.q)), (2, w(ly.o.d)), (3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (9, all(&xin))], groups(ly.o.rows.div_ceil(8 * ro)), Grp::Body, vec![ly.o.rows as u32, ly.o.cols as u32, batt / 4, bx, d, ly.o_b, 1], Vary::Build));
            ops.push((P_ROUTE_T, vec![(3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (10, all(&xin))], (ne, 1), Grp::Body, vec![d, bx, bxn, ly.post_norm, eps, ly.router, ly.router_b, blg], No));
            ops.push((P_PLAN_T, vec![(3, all(&actb)), (4, all(&st))], (1, 1), Grp::Body, vec![blg, ne, bwt], No));
            ops.push((
                P_GATEUP_T,
                vec![(1, w(ly.gate.q)), (2, w(ly.gate.e)), (4, all(&st)), (5, fwb.clone()), (6, w(ly.up.q)), (7, w(ly.up.e)), (9, all(&xin)), (10, all(&xa))],
                (cfg.ff.div_ceil(32) as u32, ne),
                Grp::Body,
                vec![cfg.ff as u32, d, bxn / 4, ba, ly.gate_b, ly.up_b],
                Vary::Experts,
            ));
            ops.push((P_DOWN_T, vec![(1, w(ly.down.q)), (2, w(ly.down.e)), (3, all(&actb)), (4, all(&st)), (9, all(&xa))], (d.div_ceil(32), ne), Grp::Body, vec![d, cfg.ff as u32, ba / 4, bp4], Vary::Experts));
        }
        let last = layers.last().unwrap().down_b;
        ops.push((P_NORM_T, vec![(3, all(&actb)), (4, all(&st)), (5, all(&fw_buf)), (10, all(&xin))], (t, 1), Grp::Head, vec![d, bx, bxn, out_norm, eps, bp4, last, bwt], No));
        ops.push((mvp(rh), vec![(1, w(out.q)), (2, w(out.d)), (3, all(&actb)), (4, all(&st)), (5, all(&fw_buf)), (9, all(&xin))], groups(out.rows.div_ceil(8 * rh)), Grp::Head, vec![out.rows as u32, out.cols as u32, bxn / 4, blog, cfg.vocab as u32, none, 0], Vary::Build));
        ops.push((P_ARG1_T, vec![(3, all(&actb)), (4, all(&st))], (ARG_GROUPS, t), Grp::Head, vec![cfg.vocab as u32, blog, bpart, chunk], No));
        ops.push((P_ARG2_T, vec![(3, all(&actb)), (4, all(&st))], (t, 1), Grp::Head, vec![bpart, ARG_GROUPS], No));

        // after either: accept, draft
        ops.push((P_ACCEPT, vec![(4, all(&st))], (1, 1), Grp::Spec, vec![PROMPT], No));
        // a prompt pass of up to PMAX tokens: only the cache matters, so the
        // last layer stops once its keys and values are stored
        let p = pm as u32;
        let mt = kp::max_tiles(cfg.n_exp) as u32;
        ops.push((P_EMBED_P, vec![(1, w(embd.q)), (2, w(embd.d)), (3, all(&actb)), (4, all(&st))], (d.div_ceil(256), p), Grp::Pre, vec![d, bx, PROMPT], No));
        let (sk, sv) = stage.as_ref().map_or((all(&dummy), all(&dummy)), |(k, v)| (all(k), all(v)));
        for (l, ly) in layers.iter().enumerate() {
            let (kvb, cap) = kv_base[l];
            let win = if l % 2 == 0 { cfg.swa as u32 } else { 0 };
            let fwb = all(&fw_buf);
            let (fk, fv) = far_of(l);
            let comb = if l == 0 { none } else { bp4 };
            ops.push((P_NORM_P, vec![(3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (10, all(&xin))], (p, 1), Grp::Pre, vec![d, bx, bxn, ly.attn_norm, eps, comb], No));
            ops.push((kq.0, vec![(1, w(ly.qkv.q)), (2, w(ly.qkv.d)), (3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (9, all(&xin))], ((qkv_rows as u32).div_ceil(kq.1), p / kq.2), Grp::Pre, vec![qkv_rows as u32, d, bxn / 4, bq, qkv_rows as u32, ly.qkv_b, 0], No));
            ops.push((P_ROPE_T, vec![(3, all(&actb)), (4, all(&st)), (6, all(&rope_buf)), (7, all(&kc)), (8, all(&vc)), (11, fk), (12, fv)], (rope_threads.div_ceil(256) as u32, p), Grp::Pre, vec![bq, cfg.n_head as u32, cfg.n_kv as u32, kvb, cap, qkv_rows as u32, res_of(l)], No));
            if l + 1 == layers.len() {
                break;
            }
            ops.push((
                P_ATTN_P,
                vec![(3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (7, all(&kc)), (8, all(&vc)), (10, all(&xin)), (11, sk.clone()), (12, sv.clone())],
                (cfg.n_kv as u32, p.div_ceil(64 / hpk)),
                Grp::Pre,
                vec![bq, batt, kvb, cap, win, ly.sinks, scale, cfg.n_kv as u32, hpk, qkv_rows as u32, attw, res_of(l), far_rows as u32],
                No,
            ));
            ops.push((kq.0, vec![(1, w(ly.o.q)), (2, w(ly.o.d)), (3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (9, all(&xin))], ((ly.o.rows as u32).div_ceil(kq.1), p / kq.2), Grp::Pre, vec![ly.o.rows as u32, ly.o.cols as u32, batt / 4, bx, d, ly.o_b, 1], No));
            ops.push((P_ROUTE_P, vec![(3, all(&actb)), (4, all(&st)), (5, fwb.clone()), (10, all(&xin)), (11, all(&pl))], (p, 1), Grp::Pre, vec![d, bx, bxn, ly.post_norm, eps, ly.router, ly.router_b, blg, ne, bwt], No));
            ops.push((P_PLAN_P, vec![(4, all(&st)), (11, all(&pl))], (1, 1), Grp::Pre, vec![ne], No));
            ops.push((
                kg.0,
                vec![(1, w(ly.gate.q)), (2, w(ly.gate.e)), (5, fwb.clone()), (6, w(ly.up.q)), (7, w(ly.up.e)), (9, all(&xin)), (10, all(&xa)), (11, all(&pl))],
                ((cfg.ff as u32).div_ceil(kg.1), mt),
                Grp::Pre,
                vec![cfg.ff as u32, d, bxn / 4, ba, ly.gate_b, ly.up_b],
                No,
            ));
            ops.push((kd.0, vec![(1, w(ly.down.q)), (2, w(ly.down.e)), (3, all(&actb)), (5, fwb.clone()), (9, all(&xa)), (11, all(&pl))], (d.div_ceil(kd.1), mt), Grp::Pre, vec![d, cfg.ff as u32, ba / 4, bp4, ly.down_b, bwt], No));
        }

        let nmax = env_usize("PELICA_DRAFT_NMAX", 6).clamp(1, 8) as u32;
        let nmin = env_usize("PELICA_DRAFT_NMIN", 1).clamp(nmax.saturating_sub(7) as usize, nmax as usize) as u32;
        ops.push((P_DRAFT, vec![(4, all(&st))], (1, 1), Grp::Spec, vec![PROMPT, nmax, nmin, ctx as u32], No));

        let stride = 256usize;
        let mut pbytes = vec![0u8; ops.len() * stride];
        for (i, op) in ops.iter().enumerate() {
            for (j, v) in op.4.iter().enumerate() {
                pbytes[i * stride + j * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let pbuf = upload("params", &pbytes, wgpu::BufferUsages::UNIFORM);
        let (mut body, mut head, mut spec, mut pre) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, (pipe, binds, grp, at, _, vary)) in ops.into_iter().enumerate() {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &pbuf, offset: (i * stride) as u64, size: wgpu::BufferSize::new(64) }),
            }];
            for (b, buf) in binds {
                entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::Buffer(buf) });
            }
            let builds = if vary == No { 1 } else { BUILDS.len() };
            let binds = (0..builds).map(|v| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipes[pipe + v].get_bind_group_layout(0), entries: &entries })).collect();
            let dsp = Disp { pipe, binds, groups: grp, vary };
            match at {
                Grp::Head => head.push(dsp),
                Grp::Body => body.push(dsp),
                Grp::Spec => spec.push(dsp),
                Grp::Pre => pre.push(dsp),
            }
        }
        // the bind groups hold references to every buffer they use
        drop((layers, embd, out, fw_buf, rope_buf, kc, vc, pbuf, wbufs, part, dummy));
        let vram = vram.get();
        eprintln!("\r[pelica] {} on {name}: {:.2} GB in VRAM, loaded in {:.1}s   ", path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(), vram as f64 / 1e9, t0.elapsed().as_secs_f64());
        if far_rows > 0 {
            eprintln!("[pelica] cache: {res} positions in VRAM, {far_rows} in RAM ({:.2} GB; VRAM free before the cache: {:.2} GB)", host_bytes as f64 / 1e9, free_now.unwrap_or(0) as f64 / 1e9);
        }
        Ok(Engine { cfg, name, max_ctx, resident: res, host_bytes, device, queue, pipes, body, head, spec, pre, st, reads, actb, xin, xa, blog, breg: [bx, bq, blg, bp4], batt, pl, far, stage })
    }

    /// The dispatches of a pass with build `b`: the layers, then (if
    /// `logits`) the argmaxes, then (if `spec`) accepting and drafting.
    fn dispatches(&self, b: usize, logits: bool, spec: bool) -> Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32), usize)> {
        let heads: &[Disp] = if logits { &self.head } else { &[] };
        let specs: &[Disp] = if spec { &self.spec } else { &[] };
        self.body
            .iter()
            .chain(heads)
            .chain(specs)
            .map(|d| match d.vary {
                Vary::No => (&self.pipes[d.pipe], &d.binds[0], d.groups, d.pipe),
                Vary::Build => (&self.pipes[d.pipe + b], &d.binds[b], d.groups, d.pipe + b),
                // a union of at most 4 experts per token
                Vary::Experts => (&self.pipes[d.pipe + b], &d.binds[b], (d.groups.0, d.groups.1.min(4 * BUILDS[b] as u32)), d.pipe + b),
            })
            .collect()
    }

    fn record(&self, enc: &mut wgpu::CommandEncoder, b: usize, logits: bool, spec: bool) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for (p, bind, g, _) in self.dispatches(b, logits, spec) {
            pass.set_pipeline(p);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(g.0, g.1, 1);
        }
    }

    /// A prompt pass of the `n` prompt tokens at `pos`.. (already in the
    /// history): it fills their cache entries. Before each full layer's
    /// attention, the part of its cache in host RAM (where rope just stored
    /// the pass's own keys and values) is copied to the VRAM staging buffers.
    /// Deep in a long context the pass goes to the GPU a layer at a time:
    /// one submission running for seconds gets the device reset by Windows.
    fn run_pre(&self, pos: usize, n: usize) -> wgpu::SubmissionIndex {
        self.queue.write_buffer(&self.st, 0, bytemuck_u32(&[0, pos as u32, 0, n as u32]));
        let far_bytes = ((pos + n).saturating_sub(self.resident) * self.cfg.n_kv * self.cfg.hd * 4) as u64;
        let split = pos + n > LONG_PASS;
        let new_enc = || self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prompt pass") });
        let mut enc = new_enc();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime();
        let mut layer = 0;
        for d in &self.pre {
            if d.pipe == P_ATTN_P {
                if let (Some((fk, fv)), Some((sk, sv))) = (&self.far[layer], &self.stage) {
                    if far_bytes > 0 {
                        drop(pass);
                        enc.copy_buffer_to_buffer(fk, 0, sk, 0, far_bytes);
                        enc.copy_buffer_to_buffer(fv, 0, sv, 0, far_bytes);
                        pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime();
                    }
                }
            }
            pass.set_pipeline(&self.pipes[d.pipe]);
            pass.set_bind_group(0, &d.binds[0], &[]);
            pass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
            if d.pipe == P_ATTN_P {
                if split && layer % 2 == 1 {
                    drop(pass);
                    self.queue.submit([std::mem::replace(&mut enc, new_enc()).finish()]);
                    pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime();
                }
                layer += 1;
            }
        }
        drop(pass);
        self.queue.submit([enc.finish()])
    }

    /// Prompt passes for all of `prompt` from `start` but its last token;
    /// returns how many tokens are in the cache after them. Deep in a long
    /// context each pass is waited for and reported to `progress` (done,
    /// total), which can stop them by returning false.
    fn prefill(&self, prompt: &[u32], start: usize, progress: &mut dyn FnMut(usize, usize) -> bool) -> usize {
        let mut c = start;
        while c + 1 < prompt.len() {
            let n = (prompt.len() - 1 - c).min(kp::PMAX);
            let idx = self.run_pre(c, n);
            c += n;
            if c > LONG_PASS {
                let _ = self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
                if !progress(c, prompt.len()) {
                    break;
                }
            }
        }
        c
    }

    /// The state words a pass reads: its first position and tokens.
    fn pass_state(&self, pos: usize, toks: &[u32], mode: u32, draft: u32) {
        let mut w = [0u32; 40];
        w[1] = pos as u32;
        w[3] = toks.len() as u32;
        w[5] = mode;
        w[6] = draft;
        w[32..32 + toks.len()].copy_from_slice(toks);
        self.queue.write_buffer(&self.st, 0, bytemuck_u32(&w));
    }

    fn read_buf(&self, buf: &wgpu::Buffer, off: u64, n: usize) -> Result<Vec<u32>> {
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, off, &rb, 0, n as u64 * 4);
        let idx = self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
        let v = rb.slice(..).get_mapped_range().context("mapping")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(v)
    }

    /// Debug: the last position's logits after `prompt`, from one-token
    /// passes and from passes of `chunk` tokens, compared bit for bit.
    pub fn check(&self, prompt: &[u32], chunk: usize) -> Result<String> {
        let v = self.cfg.vocab;
        let mut hist = vec![0u32; PROMPT as usize + prompt.len()];
        hist[PROMPT as usize..].copy_from_slice(prompt);
        self.queue.write_buffer(&self.st, 0, bytemuck_u32(&hist));
        for (p, &tok) in prompt.iter().enumerate() {
            self.pass_state(p, &[tok], 0, 0);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.record(&mut enc, 0, p + 1 == prompt.len(), false);
            self.queue.submit([enc.finish()]);
        }
        let one = self.read_buf(&self.actb, self.blog as u64 * 4, v)?;
        let mut c = if chunk == 0 { self.prefill(prompt, 0, &mut |_, _| true) } else { 0 };
        let how = if c > 0 { format!("prompt passes of up to {} tokens", kp::PMAX) } else { format!("batched passes of {}", chunk.max(1)) };
        let chunk = chunk.max(1);
        let mut last = 0;
        while c < prompt.len() {
            let t = (prompt.len() - c).min(chunk).min(kb::TMAX);
            self.pass_state(c, &prompt[c..c + t], 0, 0);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.record(&mut enc, build(t), c + t == prompt.len(), false);
            self.queue.submit([enc.finish()]);
            last = t;
            c += t;
        }
        let bat = self.read_buf(&self.actb, (self.blog as u64 + ((last - 1) * v) as u64) * 4, v)?;
        let diff = one.iter().zip(&bat).filter(|(a, b)| a != b).count();
        let maxd = one.iter().zip(&bat).map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs()).fold(0f32, f32::max);
        let am = |x: &[u32]| x.iter().enumerate().max_by(|a, b| f32::from_bits(*a.1).total_cmp(&f32::from_bits(*b.1))).map(|(i, _)| i).unwrap_or(0);
        Ok(format!("{} tokens, {how}: {diff} of {v} logits differ (max |diff| {maxd:e}); argmax {} vs {}", prompt.len(), am(&one), am(&bat)))
    }

    /// Debug: layer 0's stages for `toks` at position 0.., one-token
    /// passes against one pass of them all with build `b`, bit for bit.
    pub fn check_ops(&self, toks: &[u32], b: usize) -> Result<String> {
        let c = &self.cfg;
        let (d, qn, ff) = (c.d, (c.n_head + 2 * c.n_kv) * c.hd, c.ff);
        let att = c.n_head * c.hd;
        let [bx, bq, blg, bp4] = self.breg;
        // (stage, dispatches done, (buf, off), floats per token)
        let stages: Vec<(&str, usize, (&wgpu::Buffer, u32), usize)> = vec![
            ("embed", 1, (&self.actb, bx), d),
            ("qkv", 3, (&self.actb, bq), qn),
            ("rope", 4, (&self.actb, bq), qn),
            ("attn", 5, (&self.xin, self.batt), att),
            ("o", 6, (&self.actb, bx), d),
            ("router", 7, (&self.actb, blg), 32),
            ("gateup", 9, (&self.xa, 0), 4 * ff),
            ("down", 10, (&self.actb, bp4), 4 * d),
            ("combine", 11, (&self.actb, bx), d),
        ];
        let mut out = String::new();
        let mut hist = vec![0u32; PROMPT as usize + toks.len()];
        hist[PROMPT as usize..].copy_from_slice(toks);
        for (name, nd, (buf, off), per) in stages {
            let mut one = Vec::new();
            for (p, &tok) in toks.iter().enumerate() {
                self.queue.write_buffer(&self.st, 0, bytemuck_u32(&hist));
                self.pass_state(p, &[tok], 0, 0);
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    for (pp, bind, g, _) in self.dispatches(0, false, false).into_iter().take(nd) {
                        pass.set_pipeline(pp);
                        pass.set_bind_group(0, bind, &[]);
                        pass.dispatch_workgroups(g.0, g.1, 1);
                    }
                }
                self.queue.submit([enc.finish()]);
                one.extend(self.read_buf(buf, off as u64 * 4, per)?);
            }
            self.pass_state(0, toks, 0, 0);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                for (pp, bind, g, _) in self.dispatches(b, false, false).into_iter().take(nd) {
                    pass.set_pipeline(pp);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(g.0, g.1, 1);
                }
            }
            self.queue.submit([enc.finish()]);
            let bat = self.read_buf(buf, off as u64 * 4, per * toks.len())?;
            let mut line = format!("  {name:8}");
            for t in 0..toks.len() {
                let (x, y) = (&one[t * per..(t + 1) * per], &bat[t * per..(t + 1) * per]);
                let diff = x.iter().zip(y).filter(|(a, b)| a != b).count();
                let maxd = x.iter().zip(y).map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs()).fold(0f32, f32::max);
                line += &format!("  t{t}: {diff}/{per} differ (max {maxd:.2e})");
            }
            out += &line;
            out.push('\n');
        }
        Ok(out)
    }

    /// Greedy generation: the prompt TMAX tokens per pass, then passes that
    /// each check the current token plus a draft of up to `draft` tokens.
    /// Calls `on_token` with each new token (return false to stop); with
    /// `every`, first with the argmax after every prompt position. The cache
    /// must already hold `prompt[..reuse]` (see `reusable`); those positions
    /// are not run again.
    pub fn generate(&self, prompt: &[u32], n: usize, draft: usize, every: bool, reuse: usize, on_token: impl FnMut(u32) -> bool) -> Result<Stats> {
        self.generate_with(prompt, n, draft, every, reuse, on_token, |_, _| true)
    }

    /// `generate`, reporting long prompts' progress to `on_prefill` (tokens
    /// done, total); if it returns false, stops with nothing generated and
    /// `Stats::prompt` the tokens now in the cache.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with(&self, prompt: &[u32], n: usize, draft: usize, every: bool, reuse: usize, mut on_token: impl FnMut(u32) -> bool, mut on_prefill: impl FnMut(usize, usize) -> bool) -> Result<Stats> {
        let tm = kb::TMAX;
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        if prompt.len() + n > self.cfg.ctx {
            bail!("prompt ({}) + tokens ({n}) exceed --ctx {}", prompt.len(), self.cfg.ctx);
        }
        let kmax = draft.min(tm - 1);
        let mut hist = vec![0u32; PROMPT as usize + prompt.len()];
        hist[PROMPT as usize..].copy_from_slice(prompt);
        self.queue.write_buffer(&self.st, 0, bytemuck_u32(&hist));

        let t0 = Instant::now();
        let mut t_first = None;
        let mut produced = 0usize;
        let mut passes = 0usize;
        let mut stop = false;
        let mut pol = Drafts::new(kmax);
        let mut log = Vec::new();
        let trace = std::env::var_os("PELICA_TRACE").is_some();
        // (slot, submission, prompt chunk length if a teacher readback)
        let mut pending: VecDeque<(usize, wgpu::SubmissionIndex, usize)> = VecDeque::new();
        let mut slot = 0usize;
        let mut take = |slot: usize, idx: wgpu::SubmissionIndex, chunk: usize, produced: &mut usize, passes: &mut usize, pol: &mut Drafts, stop: &mut bool, t_first: &mut Option<Instant>| -> Result<Vec<u32>> {
            let buf = &self.reads[slot];
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
            let words: Vec<u32> = {
                let v = buf.slice(..).get_mapped_range().context("mapping token readback")?;
                v[..256].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
            };
            buf.unmap();
            if chunk == 0 {
                let decode = t_first.is_some();
                if decode && !*stop {
                    *passes += 1;
                    log.push((words[9], words[8], words[4]));
                    if trace {
                        eprintln!("[trace] pass of {} kept {} offer {} (match of {}) cap {} pos {}", words[9], words[4], words[7], words[10], words[6], words[1]);
                    }
                }
                pol.observe(&words, decode);
            }
            let toks = if chunk > 0 { &words[40..40 + chunk] } else { &words[40..40 + (words[4] as usize).min(tm)] };
            for &tok in toks {
                if *stop {
                    break;
                }
                if chunk == 0 {
                    if t_first.is_none() {
                        *t_first = Some(Instant::now());
                    }
                    *produced += 1;
                }
                if !on_token(tok) || (chunk == 0 && *produced >= n) {
                    *stop = true;
                }
            }
            Ok(words)
        };
        let cap = |b: usize| BUILDS[b] as u32 - 1;
        let mut next = 0usize;

        // the prompt: prompt passes, then its last token (or with `every`,
        // all of it, for each position's argmax) TMAX tokens per pass
        let reuse = if every { 0 } else { reuse.min(prompt.len() - 1) };
        let prefill = !every && !std::env::var("PELICA_PREFILL").is_ok_and(|v| v == "0");
        let mut c = if prefill { self.prefill(prompt, reuse, &mut on_prefill) } else { reuse };
        if prefill && c + 1 < prompt.len() {
            // stopped while reading the prompt
            return Ok(Stats { prompt: c, generated: 0, prefill_s: t0.elapsed().as_secs_f64(), decode_s: 0.0, passes: 0, log: Vec::new() });
        }
        let mut words = Vec::new();
        while c < prompt.len() {
            let t = (prompt.len() - c).min(tm);
            let last = c + t == prompt.len();
            self.pass_state(c, &prompt[c..c + t], (last && !every) as u32, cap(next));
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefill") });
            self.record(&mut enc, build(t), last || every, last && !every);
            if last || every {
                enc.copy_buffer_to_buffer(&self.st, 0, &self.reads[slot], 0, 256);
            }
            let idx = self.queue.submit([enc.finish()]);
            if last || every {
                // a teacher readback reports every position; the last chunk's
                // (non-teacher) readback is the first generated token
                pending.push_back((slot, idx, if every { t } else { 0 }));
                slot = (slot + 1) % self.reads.len();
            }
            while pending.len() > LAG || (every && last && !pending.is_empty()) {
                let (sl, ix, ch) = pending.pop_front().unwrap();
                words = take(sl, ix, ch, &mut produced, &mut passes, &mut pol, &mut stop, &mut t_first)?;
            }
            c += t;
        }
        if every {
            // the last position's argmax is the first generated token: start
            // decoding from it
            let tok = words[40 + (prompt.len() - 1) % tm];
            t_first = Some(Instant::now());
            produced = 1;
            let p = prompt.len();
            self.pass_state(p, &[tok], 0, 0);
            self.queue.write_buffer(&self.st, (PROMPT as u64 + p as u64) * 4, bytemuck_u32(&[tok]));
        }

        // decode passes; the draft limit written now applies to the draft
        // this pass makes, so to the pass after it
        while !stop && produced < n {
            let b = next;
            next = pol.choose();
            self.queue.write_buffer(&self.st, 6 * 4, bytemuck_u32(&[cap(next)]));
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pass") });
            self.record(&mut enc, b, true, true);
            enc.copy_buffer_to_buffer(&self.st, 0, &self.reads[slot], 0, 256);
            let idx = self.queue.submit([enc.finish()]);
            pending.push_back((slot, idx, 0));
            slot = (slot + 1) % self.reads.len();
            while pending.len() > LAG {
                let (sl, ix, ch) = pending.pop_front().unwrap();
                take(sl, ix, ch, &mut produced, &mut passes, &mut pol, &mut stop, &mut t_first)?;
            }
        }
        while let Some((sl, ix, ch)) = pending.pop_front() {
            take(sl, ix, ch, &mut produced, &mut passes, &mut pol, &mut stop, &mut t_first)?;
        }
        let end = Instant::now();
        let first = t_first.unwrap_or(end);
        drop(take);
        Ok(Stats { prompt: prompt.len(), generated: produced, prefill_s: (first - t0).as_secs_f64(), decode_s: (end - first).as_secs_f64(), passes, log })
    }

    /// How much of `prompt` the cache still holds after a generation that
    /// ran `cached` through the model and wrote positions below `written`:
    /// their common start, unless later positions may have overwritten the
    /// sliding-window ring entries it needs (ring of swa + PMAX).
    pub fn reusable(&self, cached: &[u32], written: usize, prompt: &[u32]) -> usize {
        let r = cached.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1));
        if written < r + kp::PMAX { r } else { 0 }
    }

    /// GPU time per kernel over `steps` passes of `batch` tokens at position
    /// `pos` (the cache is whatever is in it; only timing matters). Needs
    /// timestamp queries.
    pub fn profile(&self, pos: usize, steps: usize, batch: usize) -> Result<String> {
        if !self.device.features().contains(prof_features()) {
            bail!("this GPU/driver does not offer timestamp queries inside passes");
        }
        let batch = batch.clamp(1, kp::PMAX);
        let b = std::env::var("PELICA_PROF_BUILD").ok().and_then(|v| v.parse().ok()).unwrap_or(build(batch)).max(build(batch)).min(BUILDS.len() - 1);
        // past TMAX tokens: a prompt pass (of tokens written into the history)
        let big = batch > kb::TMAX;
        let list = if big { self.pre.iter().map(|d| (&self.pipes[d.pipe], &d.binds[0], d.groups, d.pipe)).collect() } else { self.dispatches(b, true, false) };
        if big {
            let toks: Vec<u32> = (0..(pos + steps + batch) as u32).map(|t| (t * 7919 + 13) % 50000 + 100).collect();
            self.queue.write_buffer(&self.st, PROMPT as u64 * 4, bytemuck_u32(&toks));
        }
        let n = list.len() + 1;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: n as u32 });
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let mb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n as u64 * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let period = self.queue.get_timestamp_period() as f64;
        let mut per = vec![0f64; P_NAMES.len()];
        let mut total = 0f64;
        let mut cpu = 0f64;
        let toks: Vec<u32> = (0..batch.min(kb::TMAX) as u32).map(|t| 1000 + 37 * t).collect();
        for s in 0..steps {
            if big {
                self.queue.write_buffer(&self.st, 0, bytemuck_u32(&[0, (pos + s) as u32, 0, batch as u32]));
            } else {
                self.pass_state(pos + s, &toks, 0, 0);
            }
            let t = Instant::now();
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.write_timestamp(&qs, 0);
                for (q, (p, bind, g, _)) in list.iter().enumerate() {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, *bind, &[]);
                    pass.dispatch_workgroups(g.0, g.1, 1);
                    pass.write_timestamp(&qs, q as u32 + 1);
                }
            }
            enc.resolve_query_set(&qs, 0..n as u32, &rb, 0);
            enc.copy_buffer_to_buffer(&rb, 0, &mb, 0, n as u64 * 8);
            let idx = self.queue.submit([enc.finish()]);
            cpu += t.elapsed().as_secs_f64();
            mb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
            {
                let v = mb.slice(..).get_mapped_range().context("mapping timestamps")?;
                let ts: Vec<u64> = v.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                for (i, l) in list.iter().enumerate() {
                    let dt = ts[i + 1].saturating_sub(ts[i]) as f64 * period / 1e6;
                    per[l.3] += dt;
                    total += dt;
                }
            }
            mb.unmap();
        }
        let mut out = format!("GPU time per pass of {batch} token(s) at position {pos} (avg of {steps}): {:.3} ms; CPU recording {:.3} ms\n", total / steps as f64, cpu * 1e3 / steps as f64);
        if big {
            // the last layer's expert tiles: how full they are
            let n = self.read_buf(&self.pl, 0, 1)?[0];
            out += &format!("  expert tiles (layer {}): {n} for {} pairs ({:.0}% full)\n", self.cfg.n_layer - 2, 4 * batch, 400.0 * batch as f64 / (n as f64 * kp::MOE_TT as f64));
        }
        // by kernel (a kernel's builds share its name)
        let mut by: Vec<(&str, f64)> = Vec::new();
        for (i, t) in per.iter().enumerate() {
            match by.iter_mut().find(|(n, _)| *n == P_NAMES[i]) {
                Some(e) => e.1 += t,
                None => by.push((P_NAMES[i], *t)),
            }
        }
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (name, t) in by {
            if t > 0.0 {
                out += &format!("  {:12} {:7.3} ms  {:4.1}%\n", name, t / steps as f64, t / total * 100.0);
            }
        }
        Ok(out)
    }
}

/// Chooses each pass's draft length: the chance that a drafted token is
/// kept, learned from the passes read back (per length of the match that
/// drafted it, longer matches being surer), against what a pass of each
/// size costs. The choice is made two passes ahead of the pass it is for.
struct Drafts {
    kmax: usize,
    /// decayed counts, per match length: drafted tokens kept, drafts cut short
    kept: [f64; 9],
    cut: [f64; 9],
    /// the latest read-back draft on offer, and the length of its match
    offer: usize,
    n: usize,
    /// a pass's cost per build, relative to one token (PELICA_DRAFT_COST)
    cost: Vec<f64>,
}

impl Drafts {
    fn new(kmax: usize) -> Drafts {
        let cost = std::env::var("PELICA_DRAFT_COST").unwrap_or_else(|_| "1,1.2,1.5,2.7".into()).split(',').filter_map(|v| v.parse().ok()).collect();
        Drafts { kmax, kept: [1.0; 9], cut: [1.0; 9], offer: 0, n: 0, cost }
    }

    /// A pass's state words read back (`decode`: a pass that checked a draft
    /// or a lone token, not the prompt's).
    fn observe(&mut self, w: &[u32], decode: bool) {
        let (t, kept) = (w[9] as usize, w[4] as usize);
        if decode && t > 1 {
            // its draft came from the match seen in the previous read-back
            let n = self.n.min(8);
            self.kept[n] = self.kept[n] * 0.9 + kept.saturating_sub(1) as f64;
            self.cut[n] = self.cut[n] * 0.9 + (kept < t) as u32 as f64;
        }
        self.offer = w[7] as usize;
        self.n = w[10] as usize;
    }

    /// The build for the pass after next.
    fn choose(&self) -> usize {
        if self.kmax == 0 || self.offer == 0 {
            return 0;
        }
        let n = self.n.min(8);
        let p = (self.kept[n] / (self.kept[n] + self.cut[n])).min(0.999);
        let mut best = (1.0, 0);
        for b in 1..BUILDS.len() {
            let k = (BUILDS[b] - 1).min(self.kmax).min(self.offer) as i32;
            // tokens a pass keeps when each drafted one is kept with chance p
            let gain = (1.0 - p.powi(k + 1)) / (1.0 - p);
            let v = gain / self.cost.get(b).copied().unwrap_or(BUILDS[b] as f64);
            if v > best.0 {
                best = (v, b);
            }
        }
        best.1
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn prof_features() -> wgpu::Features {
    wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// cos/sin per position and rotary pair, as ggml's YaRN RoPE computes them
/// (NeoX layout: pair j rotates elements j and j + head_dim/2).
fn rope_table(g: &Gguf, c: &Cfg) -> Result<Vec<f32>> {
    let a = "gpt-oss";
    let base = g.f(&format!("{a}.rope.freq_base"))? as f64;
    let factor = g.meta.get(&format!("{a}.rope.scaling.factor")).and_then(|v| v.as_f64()).unwrap_or(1.0);
    let orig = g.meta.get(&format!("{a}.rope.scaling.original_context_length")).and_then(|v| v.as_u64()).unwrap_or(4096) as f64;
    let beta_fast = g.meta.get(&format!("{a}.rope.scaling.yarn_beta_fast")).and_then(|v| v.as_f64()).unwrap_or(32.0);
    let beta_slow = g.meta.get(&format!("{a}.rope.scaling.yarn_beta_slow")).and_then(|v| v.as_f64()).unwrap_or(1.0);
    let n_dims = c.hd as f64;
    let freq_scale = 1.0 / factor;
    let corr = |rot: f64| n_dims * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let (lo, hi) = ((corr(beta_fast).floor()).max(0.0), (corr(beta_slow).ceil()).min(n_dims - 1.0));
    let ext = if factor != 1.0 { 1.0 } else { 0.0 };
    // ggml: mscale = attn_factor * (1 + 0.1 ln(1/freq_scale)), and llama.cpp
    // sets attn_factor so that the two YaRN magnitude terms cancel to this one
    let mscale = if ext != 0.0 { 1.0 + 0.1 * (1.0 / freq_scale).ln() } else { 1.0 };
    let half = c.hd / 2;
    let mut t = vec![0f32; c.ctx * c.hd];
    for pos in 0..c.ctx {
        for j in 0..half {
            let i0 = 2 * j;
            let theta_extrap = pos as f64 * base.powf(-(i0 as f64) / n_dims);
            let theta_interp = freq_scale * theta_extrap;
            let y = (j as f64 - lo) / (hi - lo).max(0.001);
            let ramp = (1.0 - y.clamp(0.0, 1.0)) * ext;
            let theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
            t[pos * c.hd + 2 * j] = (theta.cos() * mscale) as f32;
            t[pos * c.hd + 2 * j + 1] = (theta.sin() * mscale) as f32;
        }
    }
    Ok(t)
}

/// `pelica -m model.gguf ...`: load, generate, report.
pub fn run(path: &std::path::Path, prompt: &str, ids: Option<&[u32]>, raw: bool, n: usize, ctx: usize, teacher: bool, ignore_eos: bool) -> Result<()> {
    use std::io::Write;
    let g = Gguf::open(path)?;
    let tok = crate::gguf::tokenizer(&g)?;
    drop(g);
    let eng = Engine::load(path, ctx)?;
    if std::env::var_os("PELICA_PROF").is_some() {
        print!("{}", eng.profile(env_usize("PELICA_PROF_POS", 100), env_usize("PELICA_PROF_STEPS", 20), env_usize("PELICA_PROF_BATCH", 1))?);
        return Ok(());
    }
    let prompt_ids: Vec<u32> = match ids {
        Some(v) => v.to_vec(),
        None => {
            let text = if raw {
                prompt.to_string()
            } else {
                // harmony chat format, no system prompt, low reasoning effort
                format!("<|start|>system<|message|>Reasoning: low<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant")
            };
            tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec()
        }
    };
    if let Ok(b) = std::env::var("PELICA_CHECK_OPS") {
        let b: usize = b.parse().unwrap_or(1);
        let t = BUILDS[b].min(prompt_ids.len());
        print!("{}", eng.check_ops(&prompt_ids[..t], b)?);
        return Ok(());
    }
    if let Ok(c) = std::env::var("PELICA_CHECK") {
        for ch in c.split(',').filter_map(|s| s.parse::<usize>().ok()) {
            println!("{}", eng.check(&prompt_ids, ch)?);
        }
        return Ok(());
    }
    if teacher {
        println!("prompt_ids {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    // <|return|>, <|call|> end a turn
    let stops: Vec<u32> = ["<|return|>", "<|call|>", "<|endoftext|>"].iter().filter_map(|s| tok.token_to_id(s)).collect();
    let mut out = std::io::stdout();
    let mut gen = Vec::new();
    let mut printed = 0usize;
    let mut forced = Vec::new();
    let stats = eng.generate(&prompt_ids, n, env_usize("PELICA_DRAFT", 7), teacher, 0, |t| {
        if teacher && forced.len() + 1 < prompt_ids.len() {
            forced.push(t);
            return true;
        }
        gen.push(t);
        if !teacher {
            // print the newly completed text (decoding the whole tail keeps
            // multi-byte characters intact)
            if let Ok(s) = tok.decode(&gen, false) {
                if s.len() > printed && s.is_char_boundary(printed) {
                    let _ = write!(out, "{}", &s[printed..]);
                    let _ = out.flush();
                    printed = s.len();
                }
            }
        }
        ignore_eos || !stops.contains(&t)
    })?;
    println!();
    if teacher {
        println!("teacher {}", forced.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        println!("greedy {}", gen.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    let dec = stats.generated.saturating_sub(1);
    if std::env::var_os("PELICA_STATS").is_some() {
        // by pass size: count, experts used per layer, tokens kept
        let layers = eng.cfg.n_layer as f64;
        for t in 1..=kb::TMAX as u32 {
            let v: Vec<_> = stats.log.iter().filter(|e| e.0 == t).collect();
            if !v.is_empty() {
                let n = v.len() as f64;
                eprintln!("[pelica] passes of {t}: {}, experts per layer {:.1}, tokens kept {:.2}", v.len(), v.iter().map(|e| e.1 as f64).sum::<f64>() / n / layers, v.iter().map(|e| e.2 as f64).sum::<f64>() / n);
            }
        }
    }
    eprintln!(
        "[pelica] prefill {} tokens in {:.3}s ({:.1} tok/s); decode {dec} tokens in {:.2}s: {:.1} tok/s ({} passes, {:.2} tokens each)",
        stats.prompt,
        stats.prefill_s,
        stats.prompt as f64 / stats.prefill_s.max(1e-9),
        stats.decode_s,
        dec as f64 / stats.decode_s.max(1e-9),
        stats.passes,
        dec as f64 / stats.passes.max(1) as f64
    );
    Ok(())
}

/// Debug: how much VRAM this device hands out in 132 MB buffers, with and
/// without writing them.
pub fn alloc_test(write: bool) -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| anyhow::anyhow!("{e}"))?;
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None, required_features: wgpu::Features::SUBGROUP, required_limits: limits, experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance, trace: wgpu::Trace::Off })).context("device")?;
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64 << 20, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let data = vec![1u8; 132_710_400];
    let mut keep = Vec::new();
    for i in 0..200 {
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: data.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if let Some(e) = pollster::block_on(scope.pop()) {
            println!("failed after {i} buffers ({:.2} GB): {e}", i as f64 * data.len() as f64 / 1e9);
            return Ok(());
        }
        if write {
            crate::gpu::write_through(&device, &queue, &staging, &[(&b, 0, &data)]);
        }
        keep.push(b);
    }
    println!("200 buffers ok");
    Ok(())
}

/// Debug: where host-side cache buffers land and how fast the GPU copies
/// from them into VRAM.
pub fn host_test() -> Result<()> {
    let vram_used = || {
        std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("mappable primary buffers: {}", adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS));
    let limits = adapter.limits();
    println!("max buffer {:.2} GB, max storage binding {:.2} GB", limits.max_buffer_size as f64 / 1e9, limits.max_storage_buffer_binding_size as f64 / 1e9);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None, required_features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS, required_limits: limits, experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance, trace: wgpu::Trace::Off })).context("device")?;
    println!("VRAM used before: {}", vram_used());
    let sz = 256u64 << 20;
    let t0 = std::time::Instant::now();
    let host: Vec<wgpu::Buffer> = (0..12)
        .map(|i| {
            let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: true });
            b.slice(..).get_mapped_range_mut().expect("mapped").slice(..).copy_from_slice(&vec![i as u8 + 1; sz as usize]);
            b.unmap();
            b
        })
        .collect();
    println!("12 x 256 MB host buffers filled in {:.2}s; VRAM used: {}", t0.elapsed().as_secs_f64(), vram_used());
    let dst = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    for round in 0..3 {
        let t0 = std::time::Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        for h in &host {
            enc.copy_buffer_to_buffer(h, 0, &dst, 0, sz);
        }
        let idx = queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("{e}"))?;
        let s = t0.elapsed().as_secs_f64();
        println!("round {round}: host -> VRAM {:.1} GB/s", 12.0 * sz as f64 / s / 1e9);
    }
    let t0 = std::time::Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    for h in &host {
        enc.copy_buffer_to_buffer(&dst, 0, h, 0, sz);
    }
    let idx = queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("VRAM -> host {:.1} GB/s", 12.0 * sz as f64 / t0.elapsed().as_secs_f64() / 1e9);
    // a shader reading a host buffer directly, vs the same from VRAM
    let hs = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 1 << 20, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    var s = vec4<f32>(0.0);
    let n = arrayLength(&src);
    for (var i = g.x; i < n; i += nw.x * 256u) { s += src[i]; }
    dst[g.x % 262144u] = s.x + s.y + s.z + s.w;
}"
            .into(),
        ),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &sm, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    for (name, b) in [("host", &hs), ("VRAM", &dst)] {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() }],
        });
        for round in 0..3 {
            let t0 = std::time::Instant::now();
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&pipe);
                p.set_bind_group(0, &bg, &[]);
                for _ in 0..4 {
                    p.dispatch_workgroups(1024, 1, 1);
                }
            }
            let idx = queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("{e}"))?;
            if round == 2 {
                println!("shader reading {name}: {:.1} GB/s", 4.0 * sz as f64 / t0.elapsed().as_secs_f64() / 1e9);
            }
        }
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let big = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 2 << 30, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    println!("one 2 GB host buffer: {}; VRAM used: {}", pollster::block_on(scope.pop()).map(|e| e.to_string()).unwrap_or_else(|| "ok".into()), vram_used());
    drop(big);
    Ok(())
}
