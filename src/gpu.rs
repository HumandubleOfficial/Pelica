//! Tier 1: the GPU. Holds the weights every token needs and the hottest
//! experts in VRAM, and runs their matrix-vector products there.
//!
//! Built on wgpu (Vulkan / DX12 / Metal), so it needs no vendor SDK. Work is
//! recorded into a `Batch`: inputs are uploaded into one scratch buffer, a
//! chain of kernels runs (outputs of one can feed the next without leaving
//! the GPU), and the requested results come back in a single readback.

use crate::quant::{Mat, Q4_GROUP};
use anyhow::{anyhow, Context, Result};
use std::sync::Mutex;


const SHADER: &str = r#"
// r: rows per head; token t, row in head h reads x[x_off + t*xs + h*xh ..]
// and writes act[o_off + t*os + row] (a plain matvec: r = rows, xs = cols, xh = 0, os = rows)
struct P { rows: u32, cols: u32, s: u32, x_off: u32, o_off: u32, w_off: u32, s_off: u32, f: f32,
           r: u32, xs: u32, xh: u32, os: u32, p0: u32, p1: u32, p2: u32, p3: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;

var<workgroup> red: array<vec4<f32>, 64>;

fn x4(c: u32) -> vec4<f32> {
    return vec4<f32>(act[c], act[c + 1u], act[c + 2u], act[c + 3u]);
}

// out[t, row] = scale[row] * sum_i w8[row, i] * x[t, i]; one workgroup per row,
// four tokens per pass over the row.
@compute @workgroup_size(64)
fn q8(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
      @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * nw.x;
    if (row >= p.rows) { return; }
    let wpr = p.cols / 4u;
    let base = p.w_off + row * wpr;
    let xb = p.x_off + (row / p.r) * p.xh;
    for (var t0 = 0u; t0 < p.s; t0 += 4u) {
        var acc = vec4<f32>(0.0);
        for (var i = li; i < wpr; i += 64u) {
            let v = i32(w[base + i]);
            let wv = vec4<f32>(f32(extractBits(v, 0u, 8u)), f32(extractBits(v, 8u, 8u)),
                               f32(extractBits(v, 16u, 8u)), f32(extractBits(v, 24u, 8u)));
            for (var j = 0u; j < 4u; j++) {
                if (t0 + j < p.s) {
                    acc[j] += dot(wv, x4(xb + (t0 + j) * p.xs + i * 4u));
                }
            }
        }
        red[li] = acc;
        workgroupBarrier();
        for (var st = 32u; st > 0u; st >>= 1u) {
            if (li < st) { red[li] += red[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) {
            let s = sc[p.s_off + row];
            for (var j = 0u; j < 4u; j++) {
                if (t0 + j < p.s) { act[p.o_off + (t0 + j) * p.os + row] = red[0][j] * s; }
            }
        }
        workgroupBarrier();
    }
}

// int4 in groups of 64: 8 words per group, low nibble first, level + 8.
@compute @workgroup_size(64)
fn q4(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
      @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * nw.x;
    if (row >= p.rows) { return; }
    // one word (8 values) per thread per step, so neighbouring threads read
    // neighbouring words; the group scale is applied per word
    let wpr = p.cols / 8u;
    let base = p.w_off + row * wpr;
    let sb = p.s_off + row * (p.cols / 64u);
    let xb = p.x_off + (row / p.r) * p.xh;
    for (var t0 = 0u; t0 < p.s; t0 += 4u) {
        var acc = vec4<f32>(0.0);
        for (var i = li; i < wpr; i += 64u) {
            let v = w[base + i];
            var scale: f32;
            if (p.p0 == 1u) {
                // packed: s_off is a block of [256-entry table | one index byte per group]
                let g = row * (p.cols / 64u) + i / 8u;
                let word = bitcast<u32>(sc[p.s_off + 256u + g / 4u]);
                scale = sc[p.s_off + ((word >> ((g % 4u) * 8u)) & 255u)];
            } else {
                scale = sc[sb + i / 8u];
            }
            let lo = (vec4<f32>(f32(v & 15u), f32((v >> 4u) & 15u), f32((v >> 8u) & 15u),
                                f32((v >> 12u) & 15u)) - vec4<f32>(8.0)) * scale;
            let hi = (vec4<f32>(f32((v >> 16u) & 15u), f32((v >> 20u) & 15u),
                                f32((v >> 24u) & 15u), f32(v >> 28u)) - vec4<f32>(8.0)) * scale;
            for (var j = 0u; j < 4u; j++) {
                if (t0 + j < p.s) {
                    let c = xb + (t0 + j) * p.xs + i * 8u;
                    acc[j] += dot(lo, x4(c)) + dot(hi, x4(c + 4u));
                }
            }
        }
        red[li] = acc;
        workgroupBarrier();
        for (var st = 32u; st > 0u; st >>= 1u) {
            if (li < st) { red[li] += red[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) {
            for (var j = 0u; j < 4u; j++) {
                if (t0 + j < p.s) { act[p.o_off + (t0 + j) * p.os + row] = red[0][j]; }
            }
        }
        workgroupBarrier();
    }
}

// gate = silu(min(gate, f)) * clamp(up, -f, f), in place over `rows` values
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p.rows) { return; }
    let g = min(act[p.x_off + i], p.f);
    let u = clamp(act[p.o_off + i], -p.f, p.f);
    act[p.x_off + i] = g / (1.0 + exp(-g)) * u;
}
"#;

const ATTN_SHADER: &str = r#"
// Attention kernels. Every entry point touches all four bindings so the
// automatic layouts agree.
struct A { a: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> pa: A;
@group(0) @binding(1) var<storage, read> cst: array<f32>;
@group(0) @binding(2) var<storage, read_write> st: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;

fn arg(i: u32) -> u32 { return pa.a[i / 4u][i % 4u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn touch() { _ = cst[0]; _ = st[0]; }

// KDA short causal conv + SiLU. One thread per channel of q|k|v (3p channels),
// looping over the tokens in order. args: ch, p, s, q, k, v, out([t][3p]),
// taps (cst, [ch][kk]), window (st, [ch][kk]), kk
@compute @workgroup_size(256)
fn kconv(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    touch();
    let ch = gid.x + gid.y * nw.x * 256u;
    let n = arg(0u); let p = arg(1u); let s = arg(2u); let kk = arg(9u);
    if (ch >= n) { return; }
    let which = ch / p; let c = ch % p;
    var src = arg(3u);
    if (which == 1u) { src = arg(4u); }
    if (which == 2u) { src = arg(5u); }
    let w0 = arg(8u) + ch * kk;
    let t0 = arg(7u) + ch * kk;
    for (var t = 0u; t < s; t++) {
        for (var j = 0u; j + 1u < kk; j++) { st[w0 + j] = st[w0 + j + 1u]; }
        st[w0 + kk - 1u] = act[src + t * p + c];
        var acc = 0.0;
        for (var j = 0u; j < kk; j++) { acc += cst[t0 + j] * st[w0 + j]; }
        act[arg(6u) + t * n + ch] = acc / (1.0 + exp(-acc));
    }
}

var<workgroup> sk: array<f32, 128>;
var<workgroup> sq: array<f32, 128>;
var<workgroup> al: array<f32, 128>;
var<workgroup> rd: array<f32, 128>;

fn reduce128(li: u32, v: f32) -> f32 {
    rd[li] = v;
    workgroupBarrier();
    for (var w = 64u; w > 0u; w >>= 1u) {
        if (li < w) { rd[li] += rd[li + w]; }
        workgroupBarrier();
    }
    let r = rd[0];
    workgroupBarrier();
    return r;
}

// KDA gated delta rule, per-head RMSNorm x o_norm x sigmoid(gate). One
// workgroup per head, one thread per value column of the [k][v] state.
// args: heads, hd, s, mixed, fb, beta, gate, out, dt, a, onorm, gate_lb, eps
@compute @workgroup_size(128)
fn kdelta(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    touch();
    let h = wg.x; let nh = arg(0u); let hd = arg(1u); let s = arg(2u);
    let p = nh * hd; let base = h * hd;
    let qscale = 1.0 / sqrt(f32(hd));
    let a = cst[arg(9u) + h];
    for (var t = 0u; t < s; t++) {
        let m0 = arg(3u) + t * 3u * p;
        let qv = act[m0 + base + li];
        let kv = act[m0 + p + base + li];
        let vv = act[m0 + 2u * p + base + li];
        let sumq = reduce128(li, qv * qv);
        let sumk = reduce128(li, kv * kv);
        let qn = qscale / sqrt(1e-6 + sumq);
        let kn = 1.0 / sqrt(1e-6 + sumk);
        let dec = argf(11u) * sigm(a * (act[arg(4u) + t * p + base + li] + cst[arg(8u) + base + li]));
        al[li] = exp(dec);
        sk[li] = kv * kn;
        sq[li] = qv * qn;
        let b = sigm(act[arg(5u) + t * nh + h]);
        workgroupBarrier();
        var mem = 0.0;
        for (var k = 0u; k < hd; k++) {
            let i = (base + k) * hd + li;
            let r = st[i] * al[k];
            st[i] = r;
            mem += sk[k] * r;
        }
        var o = 0.0;
        for (var k = 0u; k < hd; k++) {
            let i = (base + k) * hd + li;
            let r = st[i] + sk[k] * b * (vv - mem);
            st[i] = r;
            o += sq[k] * r;
        }
        let ss = reduce128(li, o * o);
        let inv = 1.0 / sqrt(ss / f32(hd) + argf(12u));
        act[arg(7u) + t * p + base + li] = o * inv * cst[arg(10u) + li] * sigm(act[arg(6u) + t * p + base + li]);
        workgroupBarrier();
    }
}

var<workgroup> r256: array<f32, 256>;

// In-place RMSNorm of `s` rows of `len`: x * w / rms. args: len, s, x, w, eps
@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    touch();
    let len = arg(0u); let row = wg.x;
    if (row >= arg(1u)) { return; }
    let x0 = arg(2u) + row * len;
    var acc = 0.0;
    for (var i = li; i < len; i += 256u) { acc += act[x0 + i] * act[x0 + i]; }
    r256[li] = acc;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w >>= 1u) {
        if (li < w) { r256[li] += r256[li + w]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(r256[0] / f32(len) + argf(4u));
    for (var i = li; i < len; i += 256u) { act[x0 + i] = act[x0 + i] * inv * cst[arg(3u) + i]; }
}

// Copy `n` floats from act[src] to st[dst] (the MLA latent cache).
@compute @workgroup_size(256)
fn put(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    touch();
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= arg(0u)) { return; }
    st[arg(2u) + i] = act[arg(1u) + i];
}

var<workgroup> score: array<f32, 2056>;
var<workgroup> r64: array<f32, 64>;

// Causal attention in latent space, one workgroup per (token, head).
// args: heads, lr, q ([t][h][lr]), out ([t][h][lr]), base, scale
@compute @workgroup_size(64)
fn mla(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
       @builtin(local_invocation_index) li: u32) {
    touch();
    let th = wg.x + wg.y * nw.x;
    let nh = arg(0u); let lr = arg(1u);
    let t = th / nh;
    let seen = arg(4u) + t + 1u;
    let q0 = arg(2u) + th * lr;
    var top = -3.4e38;
    for (var j = li; j < seen; j += 64u) {
        var acc = 0.0;
        for (var i = 0u; i < lr; i++) { acc += act[q0 + i] * st[j * lr + i]; }
        let v = acc * argf(5u);
        score[j] = v;
        top = max(top, v);
    }
    r64[li] = top;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if (li < w) { r64[li] = max(r64[li], r64[li + w]); }
        workgroupBarrier();
    }
    let m = r64[0];
    workgroupBarrier();
    var tot = 0.0;
    for (var j = li; j < seen; j += 64u) {
        let e = exp(score[j] - m);
        score[j] = e;
        tot += e;
    }
    r64[li] = tot;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if (li < w) { r64[li] += r64[li + w]; }
        workgroupBarrier();
    }
    let total = r64[0];
    let o0 = arg(3u) + th * lr;
    for (var d = li; d < lr; d += 64u) {
        var acc = 0.0;
        for (var j = 0u; j < seen; j++) { acc += score[j] / total * st[j * lr + d]; }
        act[o0 + d] = acc;
    }
}
"#;

/// Writes through a staging buffer, packed into as few round trips as fit;
/// returns once the GPU has the data.
pub(crate) fn write_through(device: &wgpu::Device, queue: &wgpu::Queue, staging: &wgpu::Buffer, items: &[(&wgpu::Buffer, u64, &[u8])]) {
    let cap = staging.size() as usize;
        let mut pieces = Vec::new();
        for &(dst, off, data) in items {
            for (i, c) in data.chunks(cap).enumerate() {
                pieces.push((dst, off + (i * cap) as u64, c));
            }
        }
        let mut next = 0;
        while next < pieces.len() {
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Write, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let mut copies = Vec::new();
            {
                let mut view = slice.get_mapped_range_mut().expect("mapping the staging buffer");
                let mut used = 0;
                while next < pieces.len() && used + pieces[next].2.len() <= cap {
                    let (dst, off, data) = pieces[next];
                    view.slice(used..used + data.len()).copy_from_slice(data);
                    copies.push((dst, off, used as u64, data.len() as u64));
                    used += data.len();
                    next += 1;
                }
            }
            staging.unmap();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("upload") });
            for (dst, off, at, len) in copies {
                enc.copy_buffer_to_buffer(&staging, at, dst, off, len);
            }
            queue.submit([enc.finish()]);
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

/// Uploads from another thread: see `Gpu::uploader`.
pub struct Uploader {
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging: wgpu::Buffer,
}

impl Uploader {
    pub fn write_all(&self, items: &[(&wgpu::Buffer, u64, &[u8])]) {
        write_through(&self.device, &self.queue, &self.staging, items);
    }
}

/// Kernels of the attention shader.
#[derive(Clone, Copy, Debug)]
pub enum AK {
    Conv,
    Delta,
    Rms,
    Put,
    Mla,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GKind {
    Q8,
    Q4,
}

/// A matrix living in VRAM. `w_off` (u32 words) and `s_off` (floats) locate it
/// inside its buffers, so several matrices can share one buffer.
#[derive(Clone)]
pub struct GMat {
    pub kind: GKind,
    pub w: wgpu::Buffer,
    pub s: wgpu::Buffer,
    pub w_off: u32,
    pub s_off: u32,
    pub rows: usize,
    pub cols: usize,
    /// int4 with packed scales: `s_off` locates a [table | indices] block
    pub packed: bool,
}

impl GMat {
    pub fn bytes(&self) -> usize {
        match self.kind {
            GKind::Q8 => self.rows * self.cols + self.rows * 4,
            GKind::Q4 => self.rows * self.cols / 2 + self.rows * self.cols / Q4_GROUP * 4,
        }
    }
}

/// A resident weight on whichever device holds it.
pub enum W {
    Cpu(Mat),
    Gpu(GMat),
}

impl W {
    pub fn gpu(&self) -> Option<&GMat> {
        match self {
            W::Gpu(g) => Some(g),
            W::Cpu(_) => None,
        }
    }
}

/// Starting sizes of the scratch activation and readback buffers.
const SCRATCH_ACT: u64 = 64 << 20;
const SCRATCH_READ: u64 = 32 << 20;

struct Scratch {
    act: wgpu::Buffer,
    params: wgpu::Buffer,
    read: wgpu::Buffer,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    q8: wgpu::ComputePipeline,
    q4: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    attn: [wgpu::ComputePipeline; 5],
    scratch: Mutex<Scratch>,
    staging: Mutex<wgpu::Buffer>,
    pub name: String,
    pub max_binding: u64,
}

const PARAM_STRIDE: u64 = 256;

fn storage(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(256), usage, mapped_at_creation: false })
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        match std::env::var("PELICA_BACKEND").as_deref() {
            Ok("dx12") => desc.backends = wgpu::Backends::DX12,
            Ok("vulkan") => desc.backends = wgpu::Backends::VULKAN,
            _ => {}
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(anyhow!("only a software adapter ({}) is available", info.name));
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pelica"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pelica kernels"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (q8, q4, swiglu) = (pipe("q8"), pipe("q4"), pipe("swiglu"));
        let amod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pelica attention"),
            source: wgpu::ShaderSource::Wgsl(ATTN_SHADER.into()),
        });
        let apipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &amod,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let attn = [apipe("kconv"), apipe("kdelta"), apipe("rmsnorm"), apipe("put"), apipe("mla")];
        let scratch = Scratch {
            act: storage(&device, "act", SCRATCH_ACT, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            params: storage(&device, "params", 1 << 20, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            read: storage(&device, "read", SCRATCH_READ, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        };
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 64 << 20,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Gpu {
            device,
            queue,
            q8,
            q4,
            swiglu,
            attn,
            scratch: Mutex::new(scratch),
            staging: Mutex::new(staging),
            name: format!("{} ({:?})", info.name, info.backend),
            max_binding: limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
        })
    }

    /// A storage buffer holding `data`.
    pub fn buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let b = storage(&self.device, label, data.len() as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        self.write(&b, 0, data);
        b
    }

    /// Copy `data` into `dst` at `offset` through one reusable staging buffer.
    /// `queue.write_buffer` would allocate staging memory per call, and the
    /// allocator keeps those blocks: gigabytes of RAM after loading a model.
    pub fn write(&self, dst: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.write_all(&[(dst, offset, data)]);
    }

    /// Several writes, packed into as few staging round trips as fit.
    /// Every length and offset must be a multiple of 4.
    pub fn write_all(&self, items: &[(&wgpu::Buffer, u64, &[u8])]) {
        let staging = self.staging.lock().unwrap();
        write_through(&self.device, &self.queue, &staging, items);
    }

    /// A writer with its own staging buffer, for uploads off the main thread.
    pub fn uploader(&self, staging_bytes: u64) -> Uploader {
        Uploader {
            device: self.device.clone(),
            queue: self.queue.clone(),
            staging: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upload staging"),
                size: staging_bytes,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        }
    }

    /// Submit pending writes and wait for the GPU to finish them.
    pub fn flush(&self) {
        self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }


    /// A zeroed storage buffer, or None if the device is out of memory. The
    /// driver may grant less VRAM than is nominally free, so callers that fill
    /// the card use this to find the real limit.
    /// Shrink the scratch buffers back to their starting size (they grow to
    /// fit prefill batches) and let the driver release the memory.
    pub fn trim_scratch(&self) {
        {
            let mut st = self.scratch.lock().unwrap();
            let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
            if st.act.size() > SCRATCH_ACT {
                st.act = storage(&self.device, "act", SCRATCH_ACT, rw);
            }
            if st.read.size() > SCRATCH_READ {
                st.read = storage(&self.device, "read", SCRATCH_READ, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            }
        }
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    pub fn try_empty(&self, label: &str, size: u64) -> Option<wgpu::Buffer> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let b = storage(&self.device, label, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        match pollster::block_on(scope.pop()) {
            None => Some(b),
            Some(_) => None,
        }
    }

    /// Copy a quantized CPU matrix into VRAM (int8 and int4 only).
    pub fn upload(&self, m: &Mat) -> Option<GMat> {
        let (kind, q, s, rows, cols): (GKind, &[u8], &[f32], usize, usize) = match m {
            Mat::Q8 { q, s, rows, cols } => {
                let q = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len()) };
                (GKind::Q8, q, s, *rows, *cols)
            }
            Mat::Q4 { q, s, rows, cols } => (GKind::Q4, q, s, *rows, *cols),
            Mat::F32 { .. } => return None,
        };
        if q.len() % 4 != 0 || q.len() as u64 > self.max_binding {
            return None;
        }
        let sb = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
        Some(GMat { kind, w: self.buffer("w", q), s: self.buffer("s", sb), w_off: 0, s_off: 0, rows, cols, packed: false })
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch { gpu: self, ops: Vec::new(), uploads: Vec::new(), reads: Vec::new(), cursor: 0 }
    }
}

/// Check the kernels against the CPU ones on random matrices; returns a report.
pub fn self_test(g: &Gpu) -> Result<String> {
    use crate::quant::matmul;
    let mut seed = 7u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    };
    let mut report = String::new();
    for (bits, rows, cols, s) in [(8, 300, 4096, 1), (8, 70000, 512, 3), (4, 2048, 4096, 1), (4, 4096, 2048, 5)] {
        let m = Mat::from_f32(rnd(rows * cols), rows, cols, bits);
        let x = rnd(s * cols);
        let mut want = vec![0f32; s * rows];
        matmul(&mut want, m.r(), &x, s);
        let gm = g.upload(&m).context("upload failed")?;
        let mut b = g.batch();
        let xr = b.input(&x);
        let o = b.matvec(&gm, xr, s);
        b.read(o);
        let got = b.run().pop().unwrap();
        let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        report += &format!("int{bits} {rows}x{cols} x{s}: max |gpu - cpu| = {err:.2e}\n");
        if err > 1e-2 {
            return Err(anyhow!("GPU kernel mismatch:\n{report}"));
        }
    }
    // swiglu chain
    let (gv, uv) = (rnd(1000).iter().map(|v| v * 40.0).collect::<Vec<_>>(), rnd(1000).iter().map(|v| v * 40.0).collect::<Vec<_>>());
    let mut b = g.batch();
    let (gr, ur) = (b.input(&gv), b.input(&uv));
    b.swiglu(gr, ur, 10.0);
    b.read(gr);
    let got = b.run().pop().unwrap();
    let err = gv
        .iter()
        .zip(&uv)
        .zip(&got)
        .map(|((g, u), o)| {
            let gc = g.min(10.0);
            (gc / (1.0 + (-gc).exp()) * u.clamp(-10.0, 10.0) - o).abs()
        })
        .fold(0f32, f32::max);
    report += &format!("swiglu: max err {err:.2e}\n");

    // round-trip overhead and bandwidth
    let small = g.upload(&Mat::from_f32(rnd(64 * 512), 64, 512, 8)).unwrap();
    let big = g.upload(&Mat::from_f32(rnd(8192 * 4096), 8192, 4096, 8)).unwrap();
    let big4 = g.upload(&Mat::from_f32(rnd(8192 * 4096), 8192, 4096, 4)).unwrap();
    let xs = rnd(4096);
    for (name, m, reps) in [("tiny matvec round trip", &small, 200), ("8192x4096 int8", &big, 50), ("8192x4096 int4", &big4, 50)] {
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut b = g.batch();
            let xr = b.input(&xs[..m.cols]);
            let o = b.matvec(m, xr, 1);
            b.read(o);
            b.run();
        }
        let us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
        report += &format!("{name}: {us:.0} us/batch ({:.0} GB/s incl. overhead)\n", m.bytes() as f64 / us / 1e3);
    }
    // ten big matvecs in one batch: kernel time without per-batch overhead
    let t0 = std::time::Instant::now();
    let mut b = g.batch();
    let xr = b.input(&xs);
    for _ in 0..10 {
        let o = b.matvec(&big, xr, 1);
        b.read(o);
    }
    b.run();
    let us = t0.elapsed().as_secs_f64() * 1e6 / 10.0;
    report += &format!("8192x4096 int8 batched x10: {us:.0} us each ({:.0} GB/s)\n", big.bytes() as f64 / us / 1e3);
    Ok(report)
}

/// A region of the scratch activation buffer, in floats.
#[derive(Clone, Copy, Debug)]
pub struct Reg {
    off: u32,
    len: u32,
}

impl Reg {
    pub fn off(&self) -> u32 {
        self.off
    }
}

enum Op {
    /// heads: (rows per head, x token stride, x head stride, out token stride)
    Mat { m: GMat, x: u32, o: u32, s: u32, heads: [u32; 4] },
    Swiglu { g: u32, u: u32, n: u32, limit: f32 },
    /// an attention kernel over (constants, state, act), `groups` workgroups
    Attn { k: AK, cst: wgpu::Buffer, st: wgpu::Buffer, args: [u32; 16], groups: u32 },
}

pub struct Batch<'g> {
    gpu: &'g Gpu,
    ops: Vec<Op>,
    uploads: Vec<(u32, Vec<f32>)>,
    reads: Vec<Reg>,
    cursor: u32,
}

impl Batch<'_> {
    fn alloc(&mut self, n: usize) -> Reg {
        let r = Reg { off: self.cursor, len: n as u32 };
        self.cursor += (n as u32).div_ceil(64) * 64; // keep regions 256-byte aligned
        r
    }

    /// A scratch region of `n` floats for a kernel to write.
    pub fn space(&mut self, n: usize) -> Reg {
        self.alloc(n)
    }

    pub fn input(&mut self, data: &[f32]) -> Reg {
        let r = self.alloc(data.len());
        self.uploads.push((r.off, data.to_vec()));
        r
    }

    /// out[t, :] = m * x[t, :] for `s` tokens; returns the output region.
    pub fn matvec(&mut self, m: &GMat, x: Reg, s: usize) -> Reg {
        debug_assert!(x.len as usize >= s * m.cols);
        let o = self.alloc(s * m.rows);
        let heads = [m.rows as u32, m.cols as u32, 0, m.rows as u32];
        self.ops.push(Op::Mat { m: m.clone(), x: x.off, o: o.off, s: s as u32, heads });
        o
    }

    /// Per-head products: rows come in heads of `r`; head h of token t reads
    /// x[t * xs + h * xh ..] and row j of token t lands at out[t * os + j].
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_heads(&mut self, m: &GMat, x: Reg, s: usize, r: usize, xs: usize, xh: usize, os: usize) -> Reg {
        let o = self.alloc(s * os);
        let heads = [r as u32, xs as u32, xh as u32, os as u32];
        self.ops.push(Op::Mat { m: m.clone(), x: x.off, o: o.off, s: s as u32, heads });
        o
    }

    /// Run an attention kernel; `args` are its u32 (or f32-bit) arguments.
    pub fn attn(&mut self, k: AK, cst: &wgpu::Buffer, st: &wgpu::Buffer, args: &[u32], groups: usize) {
        let mut a = [0u32; 16];
        a[..args.len()].copy_from_slice(args);
        self.ops.push(Op::Attn { k, cst: cst.clone(), st: st.clone(), args: a, groups: groups as u32 });
    }

    /// gate = silu(min(gate, limit)) * clamp(up, -limit, limit), into `gate`.
    pub fn swiglu(&mut self, gate: Reg, up: Reg, limit: f32) {
        self.ops.push(Op::Swiglu { g: gate.off, u: up.off, n: gate.len, limit });
    }

    /// Ask for a region back; results come out of `run` in request order.
    pub fn read(&mut self, r: Reg) -> usize {
        self.reads.push(r);
        self.reads.len() - 1
    }

    pub fn run(self) -> Vec<Vec<f32>> {
        let g = self.gpu;
        if self.ops.is_empty() && self.reads.is_empty() {
            return Vec::new();
        }
        let mut st = g.scratch.lock().unwrap();
        let act_bytes = self.cursor as u64 * 4;
        if st.act.size() < act_bytes {
            st.act = storage(&g.device, "act", act_bytes.next_power_of_two(), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        }
        let param_bytes = self.ops.len() as u64 * PARAM_STRIDE;
        if st.params.size() < param_bytes {
            st.params = storage(&g.device, "params", param_bytes.next_power_of_two(), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        }
        let read_floats: u64 = self.reads.iter().map(|r| r.len as u64).sum();
        if st.read.size() < read_floats * 4 {
            st.read = storage(&g.device, "read", (read_floats * 4).next_power_of_two(), wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        }

        for (off, data) in &self.uploads {
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            g.queue.write_buffer(&st.act, *off as u64 * 4, bytes);
        }
        let mut params = vec![0u8; param_bytes as usize];
        for (i, op) in self.ops.iter().enumerate() {
            let p: [u32; 16] = match op {
                Op::Mat { m, x, o, s, heads } => {
                    let h = heads;
                    [m.rows as u32, m.cols as u32, *s, *x, *o, m.w_off, m.s_off, 0, h[0], h[1], h[2], h[3], m.packed as u32, 0, 0, 0]
                }
                Op::Swiglu { g, u, n, limit } => [*n, 0, 0, *g, *u, 0, 0, limit.to_bits(), 1, 0, 0, 0, 0, 0, 0, 0],
                Op::Attn { args, .. } => *args,
            };
            for (k, v) in p.iter().enumerate() {
                params[i * PARAM_STRIDE as usize + k * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
        }
        if !params.is_empty() {
            g.queue.write_buffer(&st.params, 0, &params);
        }

        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            for (i, op) in self.ops.iter().enumerate() {
                let pb = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &st.params,
                    offset: i as u64 * PARAM_STRIDE,
                    size: wgpu::BufferSize::new(64),
                });
                let (pipe, n_groups, bind) = match op {
                    Op::Mat { m, .. } => {
                        let pipe = if m.kind == GKind::Q8 { &g.q8 } else { &g.q4 };
                        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &pipe.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: pb },
                                wgpu::BindGroupEntry { binding: 1, resource: m.w.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: m.s.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: st.act.as_entire_binding() },
                            ],
                        });
                        (pipe, m.rows as u32, bind)
                    }
                    Op::Swiglu { n, .. } => {
                        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &g.swiglu.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: pb },
                                wgpu::BindGroupEntry { binding: 3, resource: st.act.as_entire_binding() },
                            ],
                        });
                        (&g.swiglu, n.div_ceil(256), bind)
                    }
                    Op::Attn { k, cst, st: state, groups, .. } => {
                        let pipe = &g.attn[*k as usize];
                        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &pipe.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: pb },
                                wgpu::BindGroupEntry { binding: 1, resource: cst.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: state.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: st.act.as_entire_binding() },
                            ],
                        });
                        (pipe, *groups, bind)
                    }
                };
                let x = n_groups.min(65535);
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(x, n_groups.div_ceil(x), 1);
            }
        }
        let mut at = 0u64;
        for r in &self.reads {
            enc.copy_buffer_to_buffer(&st.act, r.off as u64 * 4, &st.read, at * 4, r.len as u64 * 4);
            at += r.len as u64;
        }
        g.queue.submit([enc.finish()]);

        let mut out = Vec::with_capacity(self.reads.len());
        if read_floats > 0 {
            let slice = st.read.slice(..read_floats * 4);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            {
                let view = slice.get_mapped_range().expect("mapping the readback buffer");
                let all: &[f32] = unsafe { std::slice::from_raw_parts(view.as_ptr() as *const f32, read_floats as usize) };
                let mut at = 0usize;
                for r in &self.reads {
                    out.push(all[at..at + r.len as usize].to_vec());
                    at += r.len as usize;
                }
            }
            st.read.unmap();
        }
        out
    }
}
