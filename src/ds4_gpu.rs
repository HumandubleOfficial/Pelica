//! GPU side of the DeepSeek V4 engine (`ds4`): weights in VRAM arenas, one
//! activation buffer, and four kernels (matrix products for any GGUF type
//! the GPU decodes, RMS norm, clamped SwiGLU, argmax). Every dispatch reads
//! its arguments from its own slot of one uniform buffer (dynamic offset);
//! a `Batch` records dispatches, uploads and readbacks and runs them as one
//! submission.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};

use crate::gguf::Ty;
use crate::moe_kernels::{deq4, DEQ_HELPERS};

/// "no buffer offset" in kernel arguments
pub const NONE: u32 = u32::MAX;
/// dispatches one batch holds at most
const SLOTS: usize = 8192;
/// bytes one batch reads back at most
const READ_CAP: u64 = 128 << 20;

const COMMON: &str = r#"
struct P { a: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
fn arg(i: u32) -> u32 { return p.a[i / 4u][i % 4u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
fn x4(o: u32) -> vec4<f32> { return vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u]); }
"#;

/// y[t][r] (+)= W[r] . x[t]: 8 rows per workgroup (a subgroup each), up to
/// 64 tokens per workgroup (grid y), 8 tokens per reading of the weights.
/// Rows may take their input from groups of x (`rows per group` rows read x
/// shifted by `x step` per group). Grid (rows/8, T/64).
/// args: 0 rows, 1 cols, 2 x, 3 x stride, 4 y, 5 y stride, 6 weight word
///       base, 7 words per row, 8 tokens, 9 add into y, 10 rows per group
///       (0: none), 11 x step per group
fn mv(ty: Ty) -> String {
    let (gpb, bw) = (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"));
    let acc: String = (0..8)
        .map(|j| format!("            if (n > {j}u) {{ a{j} += dot(v, x4(xo + {j}u * xs + o)); }}\n"))
        .collect();
    let fin: String = (0..8)
        .map(|j| {
            format!(
                "        {{ let s = subgroupAdd(a{j}); if (lane == 0u && valid && n > {j}u) {{ let yo = arg(4u) + (c + {j}u) * arg(5u) + row; if (arg(9u) == 1u) {{ act[yo] += s; }} else {{ act[yo] = s; }} }} }}\n"
            )
        })
        .collect();
    format!(
        r#"{COMMON}
{DEQ_HELPERS}
{dq}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let rows = arg(0u);
    let row0 = wg.x * 8u + li / 32u;
    let valid = row0 < rows;
    let row = min(row0, rows - 1u);
    let lane = li % 32u;
    let G = arg(1u) / 4u;
    let rb = arg(6u) + row * arg(7u);
    var xb = arg(2u);
    if (arg(10u) != 0u) {{ xb += (row / arg(10u)) * arg(11u); }}
    let xs = arg(3u);
    let t1 = min(arg(8u), (wg.y + 1u) * 64u);
    for (var c = wg.y * 64u; c < t1; c += 8u) {{
        let n = min(8u, t1 - c);
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0; var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        let xo = xb + c * xs;
        for (var g = lane; g < G; g += 32u) {{
            let v = dq(rb + (g / {gpb}u) * {bw}u, g % {gpb}u);
            let o = 4u * g;
{acc}        }}
{fin}    }}
}}
"#,
        dq = deq4(ty, "dq", "w"),
    )
}

/// RMS norm of the tokens' rows of length n (times a weight from the weight
/// buffer, unless NONE); may work in place. Grid (T).
/// args: 0 n, 1 in, 2 out, 3 token stride, 4 weight word base, 5 eps, 6 T
const RMSNORM: &str = r#"
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    if (t >= arg(6u)) { return; }
    let n = arg(0u);
    let xi = arg(1u) + t * arg(3u);
    let xo = arg(2u) + t * arg(3u);
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = act[xi + i]; s += v * v; }
    s = subgroupAdd(s);
    if (li % 32u == 0u) { red[li / 32u] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red[j]; }
    let inv = 1.0 / sqrt(tot / f32(n) + argf(5u));
    let wb = arg(4u);
    for (var i = li; i < n; i += 256u) {
        var m = inv;
        if (wb != 0xffffffffu) { m *= bitcast<f32>(w[wb + i]); }
        act[xo + i] = act[xi + i] * m;
    }
}
"#;

/// h = silu(min(g, lim)) * clamp(u, -lim, lim) * weight, per token row
/// (lim 0: no clamp). Grid (n/256, T).
/// args: 0 n, 1 g, 2 u, 3 h, 4 stride of g/u rows, 5 stride of h rows,
///       6 per-token weights (NONE: 1), 7 T, 8 lim
const SWIGLU: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= arg(0u) || t >= arg(7u)) { return; }
    var g = act[arg(1u) + t * arg(4u) + i];
    var u = act[arg(2u) + t * arg(4u) + i];
    let lim = argf(8u);
    if (lim > 0.0) { g = min(g, lim); u = clamp(u, -lim, lim); }
    var h = g / (1.0 + exp(-g)) * u;
    if (arg(6u) != 0xffffffffu) { h *= act[arg(6u) + t]; }
    act[arg(3u) + t * arg(5u) + i] = h;
}
"#;

/// Index of the largest of n values (the first on ties), written as bits.
/// One workgroup. args: 0 n, 1 in, 2 out
const ARGMAX: &str = r#"
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = arg(0u);
    var best = -3.0e38;
    var at = 0u;
    for (var i = li; i < n; i += 256u) {
        let v = act[arg(1u) + i];
        if (v > best) { best = v; at = i; }
    }
    bv[li] = best;
    bi[li] = at;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s / 2u) {
        if (li < s) {
            let v = bv[li + s];
            let j = bi[li + s];
            if (v > bv[li] || (v == bv[li] && j < bi[li])) { bv[li] = v; bi[li] = j; }
        }
        workgroupBarrier();
    }
    if (li == 0u) { act[arg(2u)] = bitcast<f32>(bi[0]); }
}
"#;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// the GPU and API, for display
    pub name: String,
    layout: wgpu::PipelineLayout,
    bgl: wgpu::BindGroupLayout,
    pipes: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    params: wgpu::Buffer,
    stride: u64,
    pub act: wgpu::Buffer,
    /// weight arenas; a weight lives at (arena, word)
    pub bufs: Vec<wgpu::Buffer>,
    groups: Vec<wgpu::BindGroup>,
    rb: wgpu::Buffer,
    /// largest buffer a binding may cover
    pub max_binding: u64,
    /// bytes allocated so far
    pub allocated: u64,
}

impl Gpu {
    /// The device (32-wide subgroups required) and an activation buffer of
    /// `act_words` f32.
    pub fn new(act_words: usize) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if !adapter.features().contains(wgpu::Features::SUBGROUP) || info.subgroup_min_size > 32 || info.subgroup_max_size < 32 {
            bail!("{} lacks 32-wide subgroups, which the GPU engine needs", info.name);
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pelica-ds4"),
            required_features: wgpu::Features::SUBGROUP,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;
        let entry = |binding: u32, ty: wgpu::BufferBindingType, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ds4"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, true),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }, false),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ds4"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let stride = (limits.min_uniform_buffer_offset_alignment as u64).max(64);
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("args"), size: stride * SLOTS as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if (act_words as u64) * 4 > max_binding {
            bail!("activations ({} MB) exceed the GPU's largest binding", act_words * 4 >> 20);
        }
        let mut g = Gpu {
            name: format!("{} ({:?})", info.name, info.backend),
            act: device.create_buffer(&wgpu::BufferDescriptor { label: Some("activations"), size: (act_words as u64 * 4).div_ceil(16) * 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            rb: device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: READ_CAP, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            device,
            queue,
            layout,
            bgl,
            pipes: Mutex::new(HashMap::new()),
            params,
            stride,
            bufs: Vec::new(),
            groups: Vec::new(),
            max_binding,
            allocated: 0,
        };
        g.allocated = g.act.size() + stride * SLOTS as u64;
        Ok(g)
    }

    /// A new weight arena of `bytes` (zeroed); returns its index.
    pub fn arena(&mut self, bytes: u64) -> Result<usize> {
        let size = bytes.max(16).div_ceil(16) * 16;
        if size > self.max_binding {
            bail!("an arena of {size} bytes exceeds the GPU's largest binding");
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("weights"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("allocating {:.2} GB of VRAM ({:.2} GB so far): {e}", size as f64 / 1e9, self.allocated as f64 / 1e9);
        }
        self.allocated += size;
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: std::num::NonZeroU64::new(64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: self.act.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
            ],
        });
        self.bufs.push(b);
        self.groups.push(group);
        Ok(self.bufs.len() - 1)
    }

    /// Writes bytes into an arena (word aligned; the tail padded).
    pub fn write(&self, arena: usize, word: u64, bytes: &[u8]) {
        let n4 = bytes.len() / 4 * 4;
        if n4 > 0 {
            self.queue.write_buffer(&self.bufs[arena], word * 4, &bytes[..n4]);
        }
        if n4 < bytes.len() {
            let mut tail = [0u8; 4];
            tail[..bytes.len() - n4].copy_from_slice(&bytes[n4..]);
            self.queue.write_buffer(&self.bufs[arena], word * 4 + n4 as u64, &tail);
        }
    }

    /// Waits for everything queued so far.
    pub fn flush(&self) -> Result<()> {
        let idx = self.queue.submit([]);
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
        Ok(())
    }

    fn pipe(&self, key: &str, src: impl FnOnce() -> String) -> wgpu::ComputePipeline {
        let mut m = self.pipes.lock().unwrap();
        if let Some(p) = m.get(key) {
            return p.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(key), source: wgpu::ShaderSource::Wgsl(src().into()) });
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: Some(&self.layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        m.insert(key.to_string(), p.clone());
        p
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch { g: self, args: Vec::new(), ops: Vec::new(), reads: Vec::new(), read_bytes: 0 }
    }
}

/// A weight matrix in an arena: rows of `cols` values, `wpr` words each.
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub buf: usize,
    pub base: u32,
    pub ty: Ty,
    pub rows: usize,
    pub cols: usize,
    pub wpr: u32,
}

impl Mat {
    /// rows [a, a + n) of this matrix
    pub fn rows(&self, a: usize, n: usize) -> Mat {
        Mat { base: self.base + (a as u32) * self.wpr, rows: n, ..*self }
    }
}

struct Op {
    pipe: wgpu::ComputePipeline,
    group: usize,
    grid: [u32; 3],
}

/// Dispatches, uploads (queued at once) and readbacks of one submission.
pub struct Batch<'a> {
    g: &'a Gpu,
    args: Vec<u8>,
    ops: Vec<Op>,
    /// (activation word, words)
    reads: Vec<(u32, usize)>,
    read_bytes: u64,
}

impl Batch<'_> {
    fn push(&mut self, pipe: wgpu::ComputePipeline, group: usize, args: &[u32], grid: [u32; 3]) {
        assert!(self.ops.len() < SLOTS, "too many dispatches in one batch");
        let at = self.args.len();
        self.args.resize(at + self.g.stride as usize, 0);
        for (i, a) in args.iter().enumerate() {
            self.args[at + 4 * i..at + 4 * i + 4].copy_from_slice(&a.to_le_bytes());
        }
        self.ops.push(Op { pipe, group, grid });
    }

    /// Writes f32 values into the activations at a word offset.
    pub fn upload(&mut self, at: u32, v: &[f32]) {
        if !v.is_empty() {
            let b: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            self.g.queue.write_buffer(&self.g.act, at as u64 * 4, b);
        }
    }

    /// y = M x for `t` tokens (x rows `xs` words apart, y rows `ys` apart),
    /// adding into y if asked; `group` = (rows per group, x step).
    pub fn mv(&mut self, m: &Mat, x: u32, xs: u32, y: u32, ys: u32, t: usize, add: bool, group: Option<(u32, u32)>) {
        if t == 0 || m.rows == 0 {
            return;
        }
        let ty = m.ty;
        let pipe = self.g.pipe(&format!("mv-{}", ty.name()), || mv(ty));
        let (gr, gs) = group.unwrap_or((0, 0));
        let args = [m.rows as u32, m.cols as u32, x, xs, y, ys, m.base, m.wpr, t as u32, add as u32, gr, gs];
        self.push(pipe, m.buf, &args, [m.rows.div_ceil(8) as u32, t.div_ceil(64) as u32, 1]);
    }

    /// RMS norm of `t` rows of n values (stride apart), times weight `wt`
    /// (an f32 vector at (arena, word)) if given.
    pub fn rmsnorm(&mut self, n: usize, x: u32, y: u32, stride: u32, wt: Option<(usize, u32)>, eps: f32, t: usize) {
        let pipe = self.g.pipe("rmsnorm", || format!("{COMMON}\n{RMSNORM}"));
        let (buf, wb) = wt.unwrap_or((0, NONE));
        self.push(pipe, buf, &[n as u32, x, y, stride, wb, eps.to_bits(), t as u32], [t as u32, 1, 1]);
    }

    /// h = silu(g) * u (clamped at `lim` if > 0), times per-token weights.
    pub fn swiglu(&mut self, n: usize, g: u32, u: u32, h: u32, gs: u32, hs: u32, wts: Option<u32>, t: usize, lim: f32) {
        let pipe = self.g.pipe("swiglu", || format!("{COMMON}\n{SWIGLU}"));
        self.push(pipe, 0, &[n as u32, g, u, h, gs, hs, wts.unwrap_or(NONE), t as u32, lim.to_bits()], [n.div_ceil(256) as u32, t as u32, 1]);
    }

    pub fn argmax(&mut self, n: usize, x: u32, out: u32) {
        let pipe = self.g.pipe("argmax", || format!("{COMMON}\n{ARGMAX}"));
        self.push(pipe, 0, &[n as u32, x, out], [1, 1, 1]);
    }

    /// Reads `n` words of the activations back once the batch has run.
    pub fn read(&mut self, at: u32, n: usize) {
        assert!(self.read_bytes + n as u64 * 4 <= READ_CAP, "readback too large");
        self.reads.push((at, n));
        self.read_bytes += n as u64 * 4;
    }

    /// Runs the batch; returns the readbacks in the order asked.
    pub fn run(self) -> Result<Vec<Vec<f32>>> {
        let g = self.g;
        if !self.args.is_empty() {
            g.queue.write_buffer(&g.params, 0, &self.args);
        }
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ds4") });
        if !self.ops.is_empty() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            for (i, op) in self.ops.iter().enumerate() {
                pass.set_pipeline(&op.pipe);
                pass.set_bind_group(0, &g.groups[op.group], &[(i as u64 * g.stride) as u32]);
                pass.dispatch_workgroups(op.grid[0], op.grid[1], op.grid[2]);
            }
        }
        let mut at = 0u64;
        for &(src, n) in &self.reads {
            enc.copy_buffer_to_buffer(&g.act, src as u64 * 4, &g.rb, at, n as u64 * 4);
            at += n as u64 * 4;
        }
        let idx = g.queue.submit([enc.finish()]);
        if self.reads.is_empty() {
            g.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
            return Ok(Vec::new());
        }
        let slice = g.rb.slice(..at);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
        let mut out = Vec::with_capacity(self.reads.len());
        {
            let v = slice.get_mapped_range().context("mapping the readback")?;
            let mut o = 0usize;
            for &(_, n) in &self.reads {
                let b = &v[o..o + n * 4];
                out.push(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
                o += n * 4;
            }
        }
        g.rb.unmap();
        Ok(out)
    }
}
