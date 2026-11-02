//! GPU side of the DeepSeek V4 engine (`ds4`): weights in VRAM arenas, one
//! activation buffer, and four kernels (matrix products for any GGUF type
//! the GPU decodes, RMS norm, clamped SwiGLU, argmax). Every dispatch reads
//! its arguments from its own slot of one uniform buffer (dynamic offset);
//! a `Batch` records dispatches, uploads and readbacks and runs them as one
//! submission.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// batches run, and time spent building + submitting them and waiting on them (ns)
pub static BATCH_PROF: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
use std::sync::{Arc, Mutex};

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
// the IQ3_S grid, for engines with such weights: activation words 0..512
fn iq3s_grid(i: u32) -> u32 { return bitcast<u32>(act[i]); }
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

/// `mv` for one token (decode): the workgroup's 8 rows share x, so x is
/// staged in workgroup memory 2048 values at a time and read as vec4s,
/// rather than every lane loading 4 floats from the activations for every 4
/// weights. Same arguments as `mv`; rows per group (arg 10) 0 or a multiple
/// of 8, so all 8 rows read the same x. Grid (rows/8, 1).
fn mv1(ty: Ty) -> String {
    let (gpb, bw) = (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"));
    format!(
        r#"{COMMON}
{DEQ_HELPERS}
{dq}
var<workgroup> xt: array<vec4<f32>, 512>;
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
    if (arg(10u) != 0u) {{ xb += ((wg.x * 8u) / arg(10u)) * arg(11u); }}
    var a = 0.0;
    for (var t0 = 0u; t0 < G; t0 += 512u) {{
        let n = min(512u, G - t0);
        for (var i = li; i < n; i += 256u) {{ xt[i] = x4(xb + 4u * (t0 + i)); }}
        workgroupBarrier();
        for (var j = lane; j < n; j += 32u) {{
            let g = t0 + j;
            a += dot(dq(rb + (g / {gpb}u) * {bw}u, g % {gpb}u), xt[j]);
        }}
        workgroupBarrier();
    }}
    let s = subgroupAdd(a);
    if (lane == 0u && valid) {{ let yo = arg(4u) + row; if (arg(9u) == 1u) {{ act[yo] += s; }} else {{ act[yo] = s; }} }}
}}
"#,
        dq = deq4(ty, "dq", "w"),
    )
}

/// one-token products through `mv1` (PELICA_MV1=0: the plain `mv`)
fn mv1_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("PELICA_MV1").map_or(true, |v| v != "0"))
}

/// The same product as `mv` for many tokens (prompt passes): a workgroup
/// works a 64-row by 64-token tile, staging 32 columns at a time of its
/// weight rows (decoded once) and of its tokens' vectors in workgroup
/// memory; each thread keeps 4 rows x 4 tokens of sums. Same arguments as
/// `mv` (a row group must not straddle a tile: rows per group a multiple
/// of 64), columns a multiple of 32. Grid (rows/64, T/64).
fn mm(ty: Ty) -> String {
    let (gpb, bw) = (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"));
    let mut acc = String::new();
    let mut fma = String::new();
    let mut out = String::new();
    for a in 0..4 {
        for b in 0..4 {
            acc += &format!("    var s{a}{b} = 0.0;\n");
            fma += &format!("            s{a}{b} = fma(w{a}, x{b}, s{a}{b});\n");
            out += &format!("    {{ let row = r0 + tr * 4u + {a}u; let t = t0 + tc + {b}u * 16u; if (row < R && t < T) {{ let yo = arg(4u) + t * arg(5u) + row; if (arg(9u) == 1u) {{ act[yo] += s{a}{b}; }} else {{ act[yo] = s{a}{b}; }} }} }}\n");
        }
    }
    format!(
        r#"{COMMON}
{DEQ_HELPERS}
{dq}
var<workgroup> Wt: array<f32, 2112>;
var<workgroup> Xt: array<f32, 2112>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let R = arg(0u);
    let K = arg(1u);
    let T = arg(8u);
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    var xb = arg(2u);
    if (arg(10u) != 0u) {{ xb += (r0 / arg(10u)) * arg(11u); }}
    let xs = arg(3u);
    let tr = li / 16u;
    let tc = li % 16u;
{acc}    for (var k0 = 0u; k0 < K; k0 += 32u) {{
        for (var i = 0u; i < 2u; i++) {{
            let id = li + i * 256u;
            let r = id / 8u;
            let q = id % 8u;
            let row = min(r0 + r, R - 1u);
            let g = k0 / 4u + q;
            let v = dq(arg(6u) + row * arg(7u) + (g / {gpb}u) * {bw}u, g % {gpb}u);
            let wo = r * 33u + q * 4u;
            Wt[wo] = v.x; Wt[wo + 1u] = v.y; Wt[wo + 2u] = v.z; Wt[wo + 3u] = v.w;
            let t = min(t0 + r, T - 1u);
            let xv = x4(xb + t * xs + k0 + q * 4u);
            Xt[wo] = xv.x; Xt[wo + 1u] = xv.y; Xt[wo + 2u] = xv.z; Xt[wo + 3u] = xv.w;
        }}
        workgroupBarrier();
        for (var k = 0u; k < 32u; k++) {{
            let w0 = Wt[(tr * 4u) * 33u + k];
            let w1 = Wt[(tr * 4u + 1u) * 33u + k];
            let w2 = Wt[(tr * 4u + 2u) * 33u + k];
            let w3 = Wt[(tr * 4u + 3u) * 33u + k];
            let x0 = Xt[tc * 33u + k];
            let x1 = Xt[(tc + 16u) * 33u + k];
            let x2 = Xt[(tc + 32u) * 33u + k];
            let x3 = Xt[(tc + 48u) * 33u + k];
{fma}        }}
        workgroupBarrier();
    }}
{out}}}
"#,
        dq = deq4(ty, "dq", "w"),
    )
}

/// \`mm\` on the tensor cores (\`coop_mat16x16\`: f16 inputs, f32 sums), for
/// devices that offer them: the same 64-row by 64-token tile, weights decoded
/// and tokens' vectors staged as f16, 32 columns at a time; subgroup w of
/// the 8 keeps the sums of rows (w / 2) * 16.. for tokens (w % 2) * 32..
/// (two 16x16 tiles). Same arguments and grid as \`mm\`.
/// (Only arrays in workgroup memory: a small variable beside the f16 tiles
/// shifts them off 16-byte alignment on NVIDIA's Vulkan driver, and the
/// matrix loads then read shifted rows.)
fn mm_tc(ty: Ty) -> String {
    let (gpb, bw) = (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"));
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
{COMMON}
{DEQ_HELPERS}
{dq}
var<workgroup> Ws: array<f16, 2560>;
var<workgroup> Xs: array<f16, 2560>;
var<workgroup> Cs: array<f32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let R = arg(0u);
    let K = arg(1u);
    let T = arg(8u);
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    var xb = arg(2u);
    if (arg(10u) != 0u) {{ xb += (r0 / arg(10u)) * arg(11u); }}
    let xs = arg(3u);
    let lane = li % 32u;
    let warp = li / 32u;
    let rb = (warp / 2u) * 16u;
    let cb = (warp % 2u) * 32u;
    let cs = warp * 256u;
    for (var i = li; i < 2048u; i += 256u) {{ Cs[i] = 0.0; }}
    workgroupBarrier();
    var s0 = coopLoad<coop_mat16x16<f32, C>>(&Cs[cs], 16u);
    var s1 = coopLoad<coop_mat16x16<f32, C>>(&Cs[cs], 16u);
    workgroupBarrier();
    for (var k0 = 0u; k0 < K; k0 += 32u) {{
        for (var i = 0u; i < 2u; i++) {{
            let id = li + i * 256u;
            let r = id / 8u;
            let q = id % 8u;
            let row = min(r0 + r, R - 1u);
            let g = k0 / 4u + q;
            let v = dq(arg(6u) + row * arg(7u) + (g / {gpb}u) * {bw}u, g % {gpb}u);
            let wo = r * 40u + q * 4u;
            Ws[wo] = f16(v.x); Ws[wo + 1u] = f16(v.y); Ws[wo + 2u] = f16(v.z); Ws[wo + 3u] = f16(v.w);
            let t = min(t0 + r, T - 1u);
            let xv = x4(xb + t * xs + k0 + q * 4u);
            Xs[wo] = f16(xv.x); Xs[wo + 1u] = f16(xv.y); Xs[wo + 2u] = f16(xv.z); Xs[wo + 3u] = f16(xv.w);
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk += 16u) {{
            let a = coopLoadT<coop_mat16x16<f16, A>>(&Ws[rb * 40u + kk], 40u);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&Xs[cb * 40u + kk], 40u);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&Xs[(cb + 16u) * 40u + kk], 40u);
            s0 = coopMultiplyAdd(a, b0, s0);
            s1 = coopMultiplyAdd(a, b1, s1);
        }}
        workgroupBarrier();
    }}
    // each sum tile through the subgroup's scratch (element (m, n) at n*16 + m)
    for (var h = 0u; h < 2u; h++) {{
        if (h == 0u) {{ coopStore(s0, &Cs[cs], 16u); }} else {{ coopStore(s1, &Cs[cs], 16u); }}
        workgroupBarrier();
        for (var i = lane; i < 256u; i += 32u) {{
            let row = r0 + rb + i % 16u;
            let t = t0 + cb + h * 16u + i / 16u;
            if (row < R && t < T) {{
                let yo = arg(4u) + t * arg(5u) + row;
                if (arg(9u) == 1u) {{ act[yo] += Cs[cs + i]; }} else {{ act[yo] = Cs[cs + i]; }}
            }}
        }}
        workgroupBarrier();
    }}
}}
"#,
        dq = deq4(ty, "dq", "w"),
    )
}

/// tokens from which a product uses the tensor cores (PELICA_TC_MIN), when
/// the device has them (PELICA_TC=0: never)
fn tc_min() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("PELICA_TC_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16))
}

/// tokens from which a product uses \`mm\` rather than \`mv\` (PELICA_MM_MIN;
/// 0 = never)
fn mm_min() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("PELICA_MM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64))
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
    /// tensor cores for prompt passes' products (`mm_tc`)
    pub tc: bool,
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
        let adapter = crate::gpu::adapter(true)?;
        let info = adapter.get_info();
        crate::gpu::need_subgroups32(&adapter)?;
        let limits = adapter.limits();
        // tensor cores, if offered (PELICA_TC=0: not)
        let tcf = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        let tc = adapter.features().contains(tcf)
            && adapter.cooperative_matrix_properties().iter().any(|c| c.m_size == 16 && c.n_size == 16 && c.k_size == 16 && c.ab_type == wgpu::CooperativeScalarType::F16 && c.cr_type == wgpu::CooperativeScalarType::F32)
            && std::env::var("PELICA_TC").map_or(true, |v| v != "0");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pelica-ds4"),
            required_features: wgpu::Features::SUBGROUP | if tc { tcf } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            // SAFETY: cooperative matrices are only used as wgpu documents
            // them (16x16 f16 x f16 + f32, a configuration the device lists)
            experimental_features: if tc { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("creating GPU device")?;
        crate::gpu::probe_subgroups32(&device, &queue)?;
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
            tc,
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

impl<'a> Batch<'a> {
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
        let (gr, gs) = group.unwrap_or((0, 0));
        let args = [m.rows as u32, m.cols as u32, x, xs, y, ys, m.base, m.wpr, t as u32, add as u32, gr, gs];
        let shape = m.cols % 32 == 0 && gr % 64 == 0;
        if self.g.tc && shape && t >= tc_min() {
            let pipe = self.g.pipe(&format!("mmtc-{}", ty.name()), || mm_tc(ty));
            self.push(pipe, m.buf, &args, [m.rows.div_ceil(64) as u32, t.div_ceil(64) as u32, 1]);
            return;
        }
        let tiled = mm_min() > 0 && t >= mm_min() && shape;
        if tiled {
            let pipe = self.g.pipe(&format!("mm-{}", ty.name()), || mm(ty));
            self.push(pipe, m.buf, &args, [m.rows.div_ceil(64) as u32, t.div_ceil(64) as u32, 1]);
        } else if t == 1 && gr % 8 == 0 && mv1_on() {
            let pipe = self.g.pipe(&format!("mv1-{}", ty.name()), || mv1(ty));
            self.push(pipe, m.buf, &args, [m.rows.div_ceil(8) as u32, 1, 1]);
        } else {
            let pipe = self.g.pipe(&format!("mv-{}", ty.name()), || mv(ty));
            self.push(pipe, m.buf, &args, [m.rows.div_ceil(8) as u32, t.div_ceil(64) as u32, 1]);
        }
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

    /// A dispatch of an engine's own kernel: WGSL `src` (an entry point
    /// `main` reading its arguments with `arg`/`argf`, the activations `act`
    /// and the weights `w` of arena `buf`), built once per `key`.
    pub fn kernel(&mut self, key: &str, src: impl FnOnce() -> String, buf: usize, args: &[u32], grid: [u32; 3]) {
        let pipe = self.g.pipe(key, || format!("{COMMON}\n{DEQ_HELPERS}\n{}", src()));
        self.push(pipe, buf, args, grid);
    }

    /// Reads `n` words of the activations back once the batch has run.
    pub fn read(&mut self, at: u32, n: usize) {
        assert!(self.read_bytes + n as u64 * 4 <= READ_CAP, "readback too large");
        self.reads.push((at, n));
        self.read_bytes += n as u64 * 4;
    }

    /// Runs the batch; returns the readbacks in the order asked.
    pub fn run(self) -> Result<Vec<Vec<f32>>> {
        self.submit().wait()
    }

    /// Submits the batch; `wait` on the result for the readbacks.
    pub fn submit(self) -> Pending<'a> {
        let t0 = std::time::Instant::now();
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
        let done = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(std::sync::Mutex::new(None));
        let (d, f) = (done.clone(), failed.clone());
        if !self.reads.is_empty() {
            g.rb.slice(..at).map_async(wgpu::MapMode::Read, move |r| {
                if let Err(e) = r {
                    *f.lock().unwrap() = Some(e.to_string());
                }
                d.store(true, Ordering::Release)
            });
        } else {
            g.queue.on_submitted_work_done(move || d.store(true, Ordering::Release));
        }
        BATCH_PROF[0].fetch_add(1, Ordering::Relaxed);
        BATCH_PROF[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Pending { g, idx, reads: self.reads, at, done, failed }
    }
}

/// A submitted batch.
pub struct Pending<'a> {
    g: &'a Gpu,
    idx: wgpu::SubmissionIndex,
    reads: Vec<(u32, usize)>,
    at: u64,
    done: Arc<AtomicBool>,
    /// why the readback couldn't be mapped (the device was lost)
    failed: Arc<std::sync::Mutex<Option<String>>>,
}

/// PELICA_SPIN=0 sleeps in the driver while the GPU works; by default the
/// wait polls, which wakes sooner (the CPU has nothing else to do then)
fn spin() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("PELICA_SPIN").map_or(true, |v| v != "0"))
}

impl Pending<'_> {
    fn block(&self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let r = self.block_();
        BATCH_PROF[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }
    fn block_(&self) -> Result<()> {
        if spin() {
            while !self.done.load(Ordering::Acquire) {
                self.g.device.poll(wgpu::PollType::Poll).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
                std::hint::spin_loop();
            }
            return Ok(());
        }
        self.g.device.poll(wgpu::PollType::Wait { submission_index: Some(self.idx.clone()), timeout: None }).map_err(|e| anyhow::anyhow!("GPU poll: {e}"))?;
        Ok(())
    }
}

impl Pending<'_> {
    pub fn wait(self) -> Result<Vec<Vec<f32>>> {
        self.block()?;
        let (g, at) = (self.g, self.at);
        if self.reads.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(e) = self.failed.lock().unwrap().take() {
            anyhow::bail!("the GPU stopped answering (its driver reset, or another program took its memory): {e}");
        }
        let slice = g.rb.slice(..at);
        let mut out = Vec::with_capacity(self.reads.len());
        {
            let v = slice.get_mapped_range().context("the GPU stopped answering (its driver reset, or another program took its memory)")?;
            let mut o = 0usize;
            for &(_, n) in &self.reads {
                let mut r = vec![0f32; n];
                // SAFETY: n * 4 bytes in the mapped range; any bits are a valid f32
                unsafe { std::ptr::copy_nonoverlapping(v[o..o + n * 4].as_ptr(), r.as_mut_ptr() as *mut u8, n * 4) };
                out.push(r);
                o += n * 4;
            }
        }
        g.rb.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-token products on Q8_0 weights at the attention shapes: GB/s of
    /// weights read, and the time of a batch holding a single product.
    /// `cargo test --release mv_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mv_speed() {
        let Ok(mut g) = Gpu::new(1 << 22) else { return };
        for (rows, cols) in [(4096usize, 8192usize), (32768, 1024), (4096, 2048)] {
            let mut wb = vec![0u8; rows * cols / 32 * 36];
            let mut s = 7u32;
            for blk in wb.chunks_mut(36) {
                blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                for v in blk[4..].iter_mut() {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    *v = (s >> 24) as u8;
                }
            }
            let a = g.arena(wb.len() as u64).unwrap();
            g.write(a, 0, &wb);
            let m = Mat { buf: a, base: 0, ty: Ty::Q8_0, rows, cols, wpr: (cols / 32 * 9) as u32 };
            let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
            let y = cols as u32;
            let time = |n: usize| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let mut b = g.batch();
                    b.upload(0, &x);
                    for _ in 0..n {
                        b.mv(&m, 0, cols as u32, y, rows as u32, 1, false, None);
                    }
                    b.read(y, rows);
                    let t = std::time::Instant::now();
                    b.run().unwrap();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best
            };
            let (one, many) = (time(1), time(64));
            let per = (many - one) / 63.0;
            eprintln!("Q8_0 {rows}x{cols} ({:.1} MB): {:.0} us a product, {:.0} GB/s; a batch of one: {:.0} us", wb.len() as f64 / 1e6, per * 1e6, wb.len() as f64 / per / 1e9, one * 1e6);
        }
    }

    /// The staged one-token kernel gives the same sums as `mv` (to rounding):
    /// Q8_0 and Q4_K, columns past one 2048-value tile, row groups, adding.
    #[test]
    fn mv1_matches_mv() {
        let Ok(mut g) = Gpu::new(1 << 22) else { return };
        let (rows, cols) = (200usize, 2560usize);
        let mut s = 11u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        for ty in [Ty::Q8_0, Ty::Q4K] {
            let (bpb, words) = if ty == Ty::Q8_0 { (32usize, 9usize) } else { (256, 36) };
            let mut wb = vec![0u8; rows * cols / bpb * words * 4];
            for blk in wb.chunks_mut(words * 4) {
                blk[0..2].copy_from_slice(&0x2400u16.to_le_bytes());
                blk[2..4].copy_from_slice(&if ty == Ty::Q8_0 { 0u16 } else { 0x2000u16 }.to_le_bytes());
                for v in blk[4..].iter_mut() {
                    *v = rnd() as u8;
                }
            }
            let a = g.arena(wb.len() as u64).unwrap();
            g.write(a, 0, &wb);
            let m = Mat { buf: a, base: 0, ty, rows, cols, wpr: (cols / bpb * words) as u32 };
            let x: Vec<f32> = (0..cols * 2).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
            let (xo, y) = (0u32, (cols * 2) as u32);
            for (group, add) in [(None, false), (Some((8u32, 8u32)), false), (None, true)] {
                let run = |g: &Gpu, staged: bool| -> Vec<f32> {
                    let mut b = g.batch();
                    b.upload(xo, &x);
                    b.upload(y, &vec![0.5f32; rows]);
                    let (gr, gs) = group.unwrap_or((0, 0));
                    let args = [m.rows as u32, m.cols as u32, xo, cols as u32, y, rows as u32, m.base, m.wpr, 1, add as u32, gr, gs];
                    let p = if staged { g.pipe(&format!("mv1-{}", ty.name()), || mv1(ty)) } else { g.pipe(&format!("mv-{}", ty.name()), || mv(ty)) };
                    b.push(p, m.buf, &args, [m.rows.div_ceil(8) as u32, 1, 1]);
                    b.read(y, rows);
                    b.run().unwrap().pop().unwrap()
                };
                let (r1, r2) = (run(&g, false), run(&g, true));
                let scale = r1.iter().fold(0f32, |m, v| m.max(v.abs()));
                let worst = r1.iter().zip(&r2).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
                eprintln!("{} group {group:?} add {add}: largest sum {scale}, worst difference {worst}", ty.name());
                assert!(worst <= 1e-5 * scale.max(1.0), "{} group {group:?} add {add}: worst difference {worst} of {scale}", ty.name());
            }
        }
    }

    /// Many-token `mv` gives each token the same sums as the one-token
    /// `mv1` on that token alone (tokens past one 8-token step and one
    /// 64-token workgroup, as prompt passes have). Direct3D 12 through the
    /// dxcompiler.dll 1.9 that Edge and Firefox ship fails it (only every 8th
    /// token right); 1.7 and 1.8 pass.
    #[test]
    fn mv_tokens_match_mv1() {
        let Ok(mut g) = Gpu::new(1 << 22) else { return };
        let (rows, cols, t) = (64usize, 512usize, 150usize);
        let mut s = 5u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        let mut wb = vec![0u8; rows * cols / 32 * 36];
        for blk in wb.chunks_mut(36) {
            blk[0..2].copy_from_slice(&0x2400u16.to_le_bytes());
            for v in blk[4..].iter_mut() {
                *v = rnd() as u8;
            }
        }
        let a = g.arena(wb.len() as u64).unwrap();
        g.write(a, 0, &wb);
        let m = Mat { buf: a, base: 0, ty: Ty::Q8_0, rows, cols, wpr: (cols / 32 * 9) as u32 };
        let x: Vec<f32> = (0..t * cols).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let y = (t * cols) as u32;
        let mut b = g.batch();
        b.upload(0, &x);
        let p = g.pipe("mv-Q8_0", || mv(Ty::Q8_0));
        b.push(p, m.buf, &[rows as u32, cols as u32, 0, cols as u32, y, rows as u32, 0, m.wpr, t as u32, 0, 0, 0], [(rows / 8) as u32, t.div_ceil(64) as u32, 1]);
        b.read(y, t * rows);
        let many = b.run().unwrap().pop().unwrap();
        let mut bad = Vec::new();
        for tok in 0..t {
            let mut b = g.batch();
            b.upload(0, &x[tok * cols..(tok + 1) * cols]);
            let p = g.pipe("mv1-Q8_0", || mv1(Ty::Q8_0));
            b.push(p, m.buf, &[rows as u32, cols as u32, 0, cols as u32, y, rows as u32, 0, m.wpr, 1, 0, 0, 0], [(rows / 8) as u32, 1, 1]);
            b.read(y, rows);
            let one = b.run().unwrap().pop().unwrap();
            let worst = one.iter().zip(&many[tok * rows..(tok + 1) * rows]).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            if worst > 1e-3 {
                bad.push((tok, worst));
            }
        }
        assert!(bad.is_empty(), "tokens whose sums differ: {bad:?}");
    }

    /// The tiled prompt-pass product gives the same sums as the one-token
    /// kernel (to rounding), plain and with row groups, adding and not.
    #[test]
    fn mm_matches_mv() {
        let Ok(mut g) = Gpu::new(1 << 22) else { return };
        let (rows, cols, t) = (192usize, 512usize, 150usize);
        let mut s = 3u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        let mut wb = vec![0u8; rows * cols / 256 * 144];
        for blk in wb.chunks_mut(144) {
            blk[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
            for v in blk[4..].iter_mut() {
                *v = rnd() as u8;
            }
        }
        let a = g.arena(wb.len() as u64).unwrap();
        g.write(a, 0, &wb);
        let m = Mat { buf: a, base: 0, ty: Ty::Q4K, rows, cols, wpr: (cols / 256 * 36) as u32 };
        let x: Vec<f32> = (0..t * cols * 3).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let (xo, y1, y2) = (0u32, (t * cols * 3) as u32, (t * cols * 3 + t * rows) as u32);
        for (group, add) in [(None, false), (Some((64u32, cols as u32)), false), (None, true)] {
            let run = |g: &Gpu, tiled: bool| -> Vec<f32> {
                let mut b = g.batch();
                b.upload(xo, &x);
                b.upload(y1, &vec![0.5f32; t * rows]);
                let (gr, gs) = group.unwrap_or((0, 0));
                let args = [m.rows as u32, m.cols as u32, xo, (cols * 3) as u32, y1, rows as u32, m.base, m.wpr, t as u32, add as u32, gr, gs];
                if tiled {
                    let p = g.pipe(&format!("mm-{}", m.ty.name()), || mm(m.ty));
                    b.push(p, m.buf, &args, [m.rows.div_ceil(64) as u32, t.div_ceil(64) as u32, 1]);
                } else {
                    let p = g.pipe(&format!("mv-{}", m.ty.name()), || mv(m.ty));
                    b.push(p, m.buf, &args, [m.rows.div_ceil(8) as u32, t.div_ceil(64) as u32, 1]);
                }
                b.read(y1, t * rows);
                b.run().unwrap().pop().unwrap()
            };
            let (r1, r2) = (run(&g, false), run(&g, true));
            let _ = y2;
            // summation order differs: compare against the size of the sums
            let scale = r1.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = r1.iter().zip(&r2).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            eprintln!("group {group:?} add {add}: largest sum {scale}, worst difference {worst}");
            assert!(worst < 1e-5 * scale, "group {group:?} add {add}: worst difference {worst} of {scale}");
        }
    }
}
