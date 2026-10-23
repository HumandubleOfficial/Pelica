//! WGSL kernels for the generic MoE engine (`moe`). Weights stay in their
//! GGUF formats, re-laid out so every field is word-aligned
//! (`gguf::gpu_layout`); each kernel decodes 4 consecutive values at a time
//! (`deq4`). Activations are f32, the KV cache f16 (packed in u32 words).
//!
//! Every kernel is its own module: COMMON (the parameter block and the
//! state buffer) plus what it binds.
//!
//! State words (`st`): 0 position of the pass's first token, 1 its token
//! count, 2 the token the pass produced, 3 its position, then the routing
//! plan (`PL_*`) and, from HIST, the token history by position.

use crate::gguf::Ty;

/// Tokens in a prompt pass, at most.
pub const PT: usize = 256;
/// Experts a token uses, at most.
pub const KMAX: usize = 8;
/// Experts a layer has, at most.
pub const NEMAX: usize = 512;
/// argmax partials
pub const ARGG: usize = 128;

/// routing plan in `st`: active expert count, the active experts (by
/// index), each one's first pair and pair count, every pair's token, every
/// (token, k)'s pair, every (token, k)'s expert
pub const PL_NACT: usize = 16;
pub const PL_ACT: usize = 32;
pub const PL_OFF: usize = PL_ACT + NEMAX;
pub const PL_CNT: usize = PL_OFF + NEMAX;
pub const PL_PTOK: usize = PL_CNT + NEMAX;
pub const PL_PPOS: usize = PL_PTOK + PT * KMAX;
pub const PL_SEL: usize = PL_PPOS + PT * KMAX;
pub const HIST: usize = PL_SEL + PT * KMAX;

/// "no buffer offset"
pub const NONE: u32 = 0xffff_ffff;

pub const COMMON: &str = r#"
struct P { a: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> p: P;
fn arg(i: u32) -> u32 { return p.a[i / 4u][i % 4u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
@group(0) @binding(1) var<storage, read_write> st: array<u32>;
// st[i] as a value the whole workgroup agrees on (16 bytes: see the
// alignment note in oss_batch)
var<workgroup> wu: vec4<u32>;
fn uload4(li: u32, i: u32) -> vec4<u32> {
    if (li == 0u) { wu = vec4<u32>(st[i], st[i + 1u], st[i + 2u], st[i + 3u]); }
    return workgroupUniformLoad(&wu);
}
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
"#;

fn consts() -> String {
    format!(
        "const PL_NACT: u32 = {PL_NACT}u;\nconst PL_ACT: u32 = {PL_ACT}u;\nconst PL_OFF: u32 = {PL_OFF}u;\nconst PL_CNT: u32 = {PL_CNT}u;\nconst PL_PTOK: u32 = {PL_PTOK}u;\nconst PL_PPOS: u32 = {PL_PPOS}u;\nconst PL_SEL: u32 = {PL_SEL}u;\nconst HIST: u32 = {HIST}u;\nconst NONE: u32 = 0xffffffffu;\n"
    )
}

/// `fn {name}(b: u32, g: u32) -> vec4<f32>`: values 4g..4g+3 of the block
/// at word `b` of buffer `buf`, for weights of type `ty`.
pub fn deq4(ty: Ty, name: &str, buf: &str) -> String {
    let body = match ty {
        Ty::F32 => "    return vec4<f32>(bitcast<f32>(B[b]), bitcast<f32>(B[b + 1u]), bitcast<f32>(B[b + 2u]), bitcast<f32>(B[b + 3u]));".to_string(),
        Ty::F16 => "    return vec4<f32>(unpack2x16float(B[b]), unpack2x16float(B[b + 1u]));".to_string(),
        Ty::BF16 => "    let w0 = B[b]; let w1 = B[b + 1u];\n    return vec4<f32>(bitcast<f32>(w0 << 16u), bitcast<f32>(w0 & 0xffff0000u), bitcast<f32>(w1 << 16u), bitcast<f32>(w1 & 0xffff0000u));".to_string(),
        Ty::Q8_0 => "    let d = unpack2x16float(B[b]).x;\n    let w = i32(B[b + 1u + g]);\n    return d * vec4<f32>(f32(extractBits(w, 0u, 8u)), f32(extractBits(w, 8u, 8u)), f32(extractBits(w, 16u, 8u)), f32(extractBits(w, 24u, 8u)));".to_string(),
        Ty::Q4_0 => "    let d = unpack2x16float(B[b]).x;\n    return d * (vec4<f32>(nib4(B[b + 1u + (g % 4u)], g >= 4u)) - 8.0);".to_string(),
        Ty::Q4_1 => "    let dm = unpack2x16float(B[b]);\n    return dm.x * vec4<f32>(nib4(B[b + 1u + (g % 4u)], g >= 4u)) + dm.y;".to_string(),
        Ty::Q5_0 => "    let d = unpack2x16float(B[b]).x;\n    let v = 4u * g;\n    let hb = (vec4<u32>(B[b + 1u]) >> (vec4<u32>(v) + vec4<u32>(0u, 1u, 2u, 3u))) & vec4<u32>(1u);\n    return d * (vec4<f32>(nib4(B[b + 2u + (g % 4u)], g >= 4u) | (hb << vec4<u32>(4u))) - 16.0);".to_string(),
        Ty::Q5_1 => "    let dm = unpack2x16float(B[b]);\n    let v = 4u * g;\n    let hb = (vec4<u32>(B[b + 1u]) >> (vec4<u32>(v) + vec4<u32>(0u, 1u, 2u, 3u))) & vec4<u32>(1u);\n    return dm.x * vec4<f32>(nib4(B[b + 2u + (g % 4u)], g >= 4u) | (hb << vec4<u32>(4u))) + dm.y;".to_string(),
        Ty::Q4K => "    let v = 4u * g;\n    let dm = unpack2x16float(B[b]);\n    let c = v / 64u;\n    let hi = (v % 64u) >= 32u;\n    let q = nib4(B[b + 4u + (32u * c + v % 32u) / 4u], hi);\n    let sm = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], v / 32u);\n    return dm.x * sm.x * vec4<f32>(q) - dm.y * sm.y;".to_string(),
        Ty::Q5K => "    let v = 4u * g;\n    let dm = unpack2x16float(B[b]);\n    let c = v / 64u;\n    let hi = (v % 64u) >= 32u;\n    let l = v % 32u;\n    let q = nib4(B[b + 12u + (32u * c + l) / 4u], hi);\n    let bit = 2u * c + select(0u, 1u, hi);\n    let hb = (vec4<u32>(B[b + 4u + l / 4u]) >> (vec4<u32>(0u, 8u, 16u, 24u) + vec4<u32>(bit))) & vec4<u32>(1u);\n    let sm = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], v / 32u);\n    return dm.x * sm.x * vec4<f32>(q | (hb << vec4<u32>(4u))) - dm.y * sm.y;".to_string(),
        Ty::Q6K => "    let v = 4u * g;\n    let n = v / 128u;\n    let r = v % 128u;\n    let qt = r / 32u;\n    let l = r % 32u;\n    let lo = nib4(B[b + (64u * n + l + (qt & 1u) * 32u) / 4u], qt >= 2u);\n    let hb = (vec4<u32>(B[b + 32u + (32u * n + l) / 4u]) >> (vec4<u32>(0u, 8u, 16u, 24u) + vec4<u32>(2u * qt))) & vec4<u32>(3u);\n    let k = 8u * n + l / 16u + 2u * qt;\n    let sc = f32(extractBits(i32(B[b + 48u + k / 4u]), (k % 4u) * 8u, 8u));\n    let d = unpack2x16float(B[b + 52u]).x;\n    return d * sc * (vec4<f32>(lo | (hb << vec4<u32>(4u))) - 32.0);".to_string(),
        Ty::Iq4Xs => "    let v = 4u * g;\n    let w0 = B[b];\n    let d = unpack2x16float(w0).x;\n    let ib = v / 32u;\n    let j = v % 32u;\n    let q = nib4(B[b + 2u + (16u * ib + j % 16u) / 4u], j >= 16u);\n    let sl = (B[b + 1u] >> (8u * (ib / 2u))) & 0xffu;\n    let ls = ((sl >> (4u * (ib % 2u))) & 15u) | (((w0 >> (16u + 2u * ib)) & 3u) << 4u);\n    let dl = d * (f32(ls) - 32.0);\n    return dl * vec4<f32>(KV_IQ4[q.x], KV_IQ4[q.y], KV_IQ4[q.z], KV_IQ4[q.w]);".to_string(),
        Ty::Mxfp4 => "    let e = B[b] & 0xffu;\n    var s: f32;\n    if (e < 2u) { s = bitcast<f32>(0x00200000u << e); } else { s = bitcast<f32>((e - 1u) << 23u); }\n    let q = nib4(B[b + 1u + (g % 4u)], g >= 4u);\n    return s * vec4<f32>(KV_MX[q.x], KV_MX[q.y], KV_MX[q.z], KV_MX[q.w]);".to_string(),
        Ty::Other(_) => panic!("no decoder for {}", ty.name()),
    };
    format!("fn {name}(b: u32, g: u32) -> vec4<f32> {{\n{}\n}}\n", body.replace("B[", &format!("{buf}[")))
}

/// Helpers the decoders use (no buffers).
pub const DEQ_HELPERS: &str = r#"
// the low (or high) nibbles of a word's 4 bytes
fn nib4(w: u32, hi: bool) -> vec4<u32> {
    let s = select(vec4<u32>(0u, 8u, 16u, 24u), vec4<u32>(4u, 12u, 20u, 28u), hi);
    return (vec4<u32>(w) >> s) & vec4<u32>(15u);
}
fn kbyte(w0: u32, w1: u32, w2: u32, k: u32) -> u32 {
    var w = w0;
    if (k >= 8u) { w = w2; } else if (k >= 4u) { w = w1; }
    return (w >> (8u * (k % 4u))) & 0xffu;
}
// Q4_K/Q5_K sub-block scale and min (6 bits each, packed in 12 bytes)
fn scmin_k4(w0: u32, w1: u32, w2: u32, s: u32) -> vec2<f32> {
    if (s < 4u) {
        return vec2<f32>(f32(kbyte(w0, w1, w2, s) & 63u), f32(kbyte(w0, w1, w2, s + 4u) & 63u));
    }
    let a = kbyte(w0, w1, w2, s + 4u);
    let sc = (a & 15u) | ((kbyte(w0, w1, w2, s - 4u) >> 6u) << 4u);
    let m = (a >> 4u) | ((kbyte(w0, w1, w2, s) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}
var<private> KV_IQ4: array<f32, 16> = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
var<private> KV_MX: array<f32, 16> = array<f32, 16>(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0);
"#;

/// Groups of 4 values per block, and words per block, of `ty`'s GPU layout.
fn geom(ty: Ty) -> (usize, usize) {
    (ty.gpu_values() / 4, ty.gpu_words().expect("decodable"))
}

fn header(binds: &str) -> String {
    format!("{binds}\n{COMMON}\n{}\n{DEQ_HELPERS}\n", consts())
}

const ACT: &str = "@group(0) @binding(2) var<storage, read_write> act: array<f32>;";
const FW: &str = "@group(0) @binding(5) var<storage, read> fw: array<f32>;";

/// Accumulating 8 tokens' dot products against one decoded vector `w`:
/// token j's vector starts at `xo{j}`.
fn acc8(acc: &str, w: &str) -> String {
    (0..8).map(|j| format!("            if (n > {j}u) {{ let o = xo{j} + 4u * g; {acc}{j} += dot({w}, vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u])); }}\n")).collect()
}

fn zeros(acc: &str) -> String {
    (0..8).map(|j| format!("        var {acc}{j} = 0.0;\n")).collect()
}

/// y[t][r] (+)= W[r] . x[t] (+ bias[r]) for the pass's tokens (or only its
/// last, arg 10), 8 rows per workgroup (a subgroup each), 8 tokens per
/// reading of the weights.  Grid (rows/8).
/// args: 0 rows, 1 cols, 2 x, 3 x stride, 4 y, 5 y stride, 6 weight word
///       base, 7 words per row, 8 bias (fw, NONE), 9 add into y, 10 last
///       token only
pub fn matvec(ty: Ty) -> String {
    let (gpb, bw) = geom(ty);
    format!(
        r#"{h}
{dq}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let T = uload4(li, 0u).y;
    let rows = arg(0u);
    let row0 = wg.x * 8u + li / 32u;
    let valid = row0 < rows;
    let row = min(row0, rows - 1u);
    let lane = li % 32u;
    let G = arg(1u) / 4u;
    let rb = arg(6u) + row * arg(7u);
    var t0 = 0u;
    if (arg(10u) == 1u) {{ t0 = T - 1u; }}
    for (var c = t0; c < T; c += 8u) {{
        let n = min(8u, T - c);
{z}{xo}        for (var g = lane; g < G; g += 32u) {{
            let w = dq(rb + (g / {gpb}u) * {bw}u, g % {gpb}u);
{acc}        }}
{fin}    }}
}}
"#,
        h = header(&format!("{ACT}\n@group(0) @binding(3) var<storage, read> wq: array<u32>;\n{FW}")),
        dq = deq4(ty, "dq", "wq"),
        z = zeros("a"),
        xo = (0..8).map(|j| format!("        let xo{j} = arg(2u) + (c + {j}u) * arg(3u);\n")).collect::<String>(),
        acc = acc8("a", "w"),
        fin = (0..8)
            .map(|j| format!(
                "        {{ let s = subgroupAdd(a{j}); if (lane == 0u && valid && n > {j}u) {{ var v = s; if (arg(8u) != NONE) {{ v += fw[arg(8u) + row]; }} var yo = arg(4u) + (c + {j}u) * arg(5u) + row; if (arg(10u) == 1u) {{ yo = arg(4u) + row; }} if (arg(9u) == 1u) {{ act[yo] += v; }} else {{ act[yo] = v; }} }} }}\n"
            ))
            .collect::<String>(),
    )
}

/// x[t] = row (token) of the embedding matrix, decoded.  Grid (cols/256, PT).
/// args: 0 cols, 1 out, 2 words per row, 3 weight word base, 4 scale (f32)
pub fn embed(ty: Ty) -> String {
    let (gpb, bw) = geom(ty);
    format!(
        r#"{h}
{dq}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let g = gid.x;
    let cols = arg(0u);
    if (4u * g >= cols) {{ return; }}
    let tok = st[HIST + st[0] + t];
    let v = dq(arg(3u) + tok * arg(2u) + (g / {gpb}u) * {bw}u, g % {gpb}u) * argf(4u);
    let o = arg(1u) + t * cols + 4u * g;
    act[o] = v.x; act[o + 1u] = v.y; act[o + 2u] = v.z; act[o + 3u] = v.w;
}}
"#,
        h = header(&format!("{ACT}\n@group(0) @binding(3) var<storage, read> wq: array<u32>;")),
        dq = deq4(ty, "dq", "wq"),
    )
}

/// RMS norm of rows of length n, times a weight of length n: for each of
/// the pass's tokens, `per` rows `rs` apart (one row: a token's vector;
/// heads: per-head norms).  May work in place.  Grid (per, PT).
/// args: 0 n, 1 in, 2 out, 3 token stride (in and out), 4 row stride,
///       5 weight (fw), 6 eps
pub fn rmsnorm() -> String {
    format!(
        r#"{h}
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let T = uload4(li, 0u).y;
    let t = wg.y;
    if (t >= T) {{ return; }}
    let n = arg(0u);
    let base = t * arg(3u) + wg.x * arg(4u);
    let xi = arg(1u) + base;
    let xo = arg(2u) + base;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) {{ let v = act[xi + i]; s += v * v; }}
    s = subgroupAdd(s);
    if (li % 32u == 0u) {{ red[li / 32u] = s; }}
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) {{ tot += red[j]; }}
    let inv = 1.0 / sqrt(tot / f32(n) + argf(6u));
    for (var i = li; i < n; i += 256u) {{ act[xo + i] = act[xi + i] * inv * fw[arg(5u) + i]; }}
}}
"#,
        h = header(&format!("{ACT}\n{FW}")),
    )
}

/// RoPE, in place, on the pass's q and k heads: pairs (m, m + rot/2)
/// (NEOX) or (2m, 2m+1). The angle of position p is that of 256*(p/256)
/// (table rows 0..) plus that of p%256 (rows from `lo`), composed here: a
/// table per position would be large at long contexts.
/// Grid ((n_head + n_kv) * rot/2 / 256, PT).
/// args: 0 q, 1 q stride, 2 k, 3 k stride, 4 n_head, 5 n_kv, 6 head dim,
///       7 rotated dims, 8 neox, 9 lo (first row of the fine table)
pub fn rope() -> String {
    format!(
        r#"{h}
@group(0) @binding(6) var<storage, read> rope: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let half = arg(7u) / 2u;
    let i = gid.x;
    let hh = i / half;
    let m = i % half;
    if (hh >= arg(4u) + arg(5u)) {{ return; }}
    let hd = arg(6u);
    var base: u32;
    if (hh < arg(4u)) {{ base = arg(0u) + t * arg(1u) + hh * hd; }}
    else {{ base = arg(2u) + t * arg(3u) + (hh - arg(4u)) * hd; }}
    var a = 2u * m;
    var b = 2u * m + 1u;
    if (arg(8u) == 1u) {{ a = m; b = m + half; }}
    let pos = st[0] + t;
    let hi = ((pos / 256u) * half + m) * 2u;
    let lo = ((arg(9u) + pos % 256u) * half + m) * 2u;
    let c1 = rope[hi]; let s1 = rope[hi + 1u];
    let c2 = rope[lo]; let s2 = rope[lo + 1u];
    let cs = c1 * c2 - s1 * s2;
    let sn = s1 * c2 + c1 * s2;
    let x0 = act[base + a];
    let x1 = act[base + b];
    act[base + a] = x0 * cs - x1 * sn;
    act[base + b] = x0 * sn + x1 * cs;
}}
"#,
        h = header(ACT),
    )
}

/// The cache on the GPU, all u32 words holding f16 pairs, a row of n_kv*hd
/// values per position: VRAM (positions below NV) and a staging copy of a
/// round of RK positions held by the CPU (staged row = position - R0).
const KV_BINDS: &str = r#"
@group(0) @binding(11) var<storage, read_write> vk: array<u32>;
@group(0) @binding(12) var<storage, read_write> vv: array<u32>;
"#;
const STAGE_BINDS: &str = r#"
@group(0) @binding(15) var<storage, read_write> sk: array<u32>;
@group(0) @binding(16) var<storage, read_write> sv: array<u32>;
"#;

/// Positions per decode split, keys per prompt split, prompt splits per
/// round, positions per round (and per staging buffer).
pub const CHUNK: usize = 256;
/// Positions per workgroup of decode attention in VRAM, and per slot of its
/// partials (the CPU part keeps CHUNK and fills the slots in between with
/// empty partials): small, so short contexts still spread over every SM.
pub const DCHUNK: usize = 64;
pub const SPLIT: usize = 1024;
pub const RSPL: usize = 16;
pub const RK: usize = SPLIT * RSPL;

/// The pass's k and v rows into VRAM; past VRAM (rows the CPU holds, which
/// prompt passes stage with `kv_stage`), a decode pass's row goes to `row`
/// (keys, then values) for the CPU.
/// Grid (words per position / 256, PT).
/// args: 0 k, 1 v, 2 k/v stride (floats), 3 words per position, 4 NV
/// Decoding: what follows the q, k, v projections, in one dispatch: a
/// workgroup per head (q heads, k heads, v heads), the per-head RMS norm (if
/// any), rope on q and k, and k and v as f16 into the cache (or `row` past
/// VRAM). Same arithmetic as `rmsnorm`, `rope` and `kv_home` one after
/// another. Grid (q heads + 2 kv heads).
/// args: 0 q, 1 k, 2 v, 3 q heads, 4 kv heads, 5 head dim, 6 rotary dims,
///       7 neox, 8 rope table's second part, 9 q norm (fw, NONE), 10 k norm,
///       11 eps, 12 words per cache row, 13 positions in VRAM
pub fn qk_dec() -> String {
    format!(
        r#"{h}
@group(0) @binding(6) var<storage, read> rope: array<f32>;
var<workgroup> red: array<f32, 8>;
var<workgroup> hv: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let pos = uload4(li, 0u).x;
    let nh = arg(3u);
    let nkv = arg(4u);
    let hd = arg(5u);
    let hh = wg.x;
    var base = arg(0u) + hh * hd;
    var nw = arg(9u);
    var kind = 0u;
    if (hh >= nh + nkv) {{ base = arg(2u) + (hh - nh - nkv) * hd; nw = NONE; kind = 2u; }}
    else if (hh >= nh) {{ base = arg(1u) + (hh - nh) * hd; nw = arg(10u); kind = 1u; }}
    var v = 0.0;
    if (li < hd) {{ v = act[base + li]; }}
    var s = 0.0;
    if (li < hd) {{ s = v * v; }}
    s = subgroupAdd(s);
    if (li % 32u == 0u) {{ red[li / 32u] = s; }}
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) {{ tot += red[j]; }}
    let inv = 1.0 / sqrt(tot / f32(hd) + argf(11u));
    if (nw != NONE && li < hd) {{ v = v * inv * fw[nw + li]; }}
    if (li < hd) {{ hv[li] = v; }}
    workgroupBarrier();
    let half = arg(6u) / 2u;
    if (kind < 2u && li < half) {{
        var a = 2u * li;
        var b = 2u * li + 1u;
        if (arg(7u) == 1u) {{ a = li; b = li + half; }}
        let hi = ((pos / 256u) * half + li) * 2u;
        let lo = ((arg(8u) + pos % 256u) * half + li) * 2u;
        let c1 = rope[hi]; let s1 = rope[hi + 1u];
        let c2 = rope[lo]; let s2 = rope[lo + 1u];
        let cs = c1 * c2 - s1 * s2;
        let sn = s1 * c2 + c1 * s2;
        let x0 = hv[a];
        let x1 = hv[b];
        hv[a] = x0 * cs - x1 * sn;
        hv[b] = x0 * sn + x1 * cs;
    }}
    workgroupBarrier();
    if (kind < 2u && li < hd) {{ act[base + li] = hv[li]; }}
    if (kind > 0u && li < hd / 2u) {{
        let hk = select(hh - nh - nkv, hh - nh, kind == 1u);
        let w = hk * (hd / 2u) + li;
        let x = pack2x16float(vec2<f32>(hv[2u * li], hv[2u * li + 1u]));
        let rw = arg(12u);
        if (pos < arg(13u)) {{
            if (kind == 1u) {{ vk[pos * rw + w] = x; }} else {{ vv[pos * rw + w] = x; }}
        }} else {{
            if (kind == 1u) {{ row[w] = x; }} else {{ row[rw + w] = x; }}
        }}
    }}
}}
"#,
        h = header(&format!("{ACT}\n{FW}\n{KV_BINDS}\n@group(0) @binding(13) var<storage, read_write> row: array<u32>;")),
    )
}

pub fn kv_home() -> String {
    format!(
        r#"{h}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let w = gid.x;
    let rw = arg(3u);
    if (w >= rw) {{ return; }}
    let pos = st[0] + t;
    let ko = arg(0u) + t * arg(2u) + 2u * w;
    let vo = arg(1u) + t * arg(2u) + 2u * w;
    let kx = pack2x16float(vec2<f32>(act[ko], act[ko + 1u]));
    let vx = pack2x16float(vec2<f32>(act[vo], act[vo + 1u]));
    let nv = arg(4u);
    if (pos < nv) {{
        vk[pos * rw + w] = kx;
        vv[pos * rw + w] = vx;
    }} else if (st[1] == 1u) {{
        row[w] = kx;
        row[rw + w] = vx;
    }}
}}
"#,
        h = header(&format!("{ACT}\n{KV_BINDS}\n@group(0) @binding(13) var<storage, read_write> row: array<u32>;")),
    )
}

/// The pass's rows that fall in round R0.. into the staging copy.
/// Grid (words per position / 256, PT).
/// args: 0 k, 1 v, 2 k/v stride, 3 words per position, 4 R0, 5 R1
pub fn kv_stage() -> String {
    format!(
        r#"{h}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let w = gid.x;
    let rw = arg(3u);
    if (w >= rw) {{ return; }}
    let pos = st[0] + t;
    if (pos < arg(4u) || pos >= arg(5u)) {{ return; }}
    let ko = arg(0u) + t * arg(2u) + 2u * w;
    let vo = arg(1u) + t * arg(2u) + 2u * w;
    let r = pos - arg(4u);
    sk[r * rw + w] = pack2x16float(vec2<f32>(act[ko], act[ko + 1u]));
    sv[r * rw + w] = pack2x16float(vec2<f32>(act[vo], act[vo + 1u]));
}}
"#,
        h = header(&format!("{ACT}\n{STAGE_BINDS}")),
    )
}

/// Constants of `pexp` (Cephes' expf), shared with the CPU's copy
/// (`cpu_attn`) so both round the same.
pub const EXP_LOG2E: f32 = 1.442_695_04;
pub const EXP_C1: f32 = 0.693_359_375;
pub const EXP_C2: f32 = -2.121_944_4e-4;
pub const EXP_P: [f32; 6] = [1.987_569_15e-4, 1.398_199_95e-3, 8.333_451_9e-3, 4.166_579_6e-2, 0.166_666_65, 0.500_000_01];
/// below this, `pexp` is 0
pub const EXP_MIN: f32 = -40.0;

/// `pexp(x)`: e^x for x <= 0 from adds, multiplies and explicit fmas only,
/// so the CPU can reproduce it bit for bit (the GPU's own exp is
/// approximate in its own way).
fn pexp_wgsl() -> String {
    let f = |x: f32| format!("bitcast<f32>({:#x}u)", x.to_bits());
    format!(
        r#"
fn pexp(x: f32) -> f32 {{
    if (x < {min}) {{ return 0.0; }}
    let n = floor(fma(x, {l2e}, 0.5));
    var r = fma(n, -{c1}, x);
    r = fma(n, -{c2}, r);
    var p = {p0};
    p = fma(p, r, {p1});
    p = fma(p, r, {p2});
    p = fma(p, r, {p3});
    p = fma(p, r, {p4});
    p = fma(p, r, {p5});
    var y = fma(p, r * r, r);
    y = y + 1.0;
    return y * bitcast<f32>(u32(i32(n) + 127) << 23u);
}}
"#,
        min = f(EXP_MIN),
        l2e = f(EXP_LOG2E),
        c1 = f(EXP_C1),
        c2 = f(EXP_C2),
        p0 = f(EXP_P[0]),
        p1 = f(EXP_P[1]),
        p2 = f(EXP_P[2]),
        p3 = f(EXP_P[3]),
        p4 = f(EXP_P[4]),
        p5 = f(EXP_P[5]),
    )
}

/// Decode attention of the pass's one token: workgroup (kv head g, chunk c
/// of DCHUNK positions) with all `hpk` query heads that share the kv head,
/// so each key is read once. 4 subgroups take every 4th key, each with an
/// online softmax per head; lane i holds dims i, i+32, ... Writes (max,
/// sum, sums of v) per (chunk, head) for `merge_dec`. Only chunks in VRAM:
/// `cpu_attn` does the rest.
///
/// The arithmetic is spelled out so that `cpu_attn` reproduces it bit for
/// bit: q scaled first, each lane's dot product in dim order with explicit
/// fmas, lanes summed by xor shuffles 16, 8, 4, 2, 1, `pexp`, and fmas
/// wherever a product meets a sum (so the compiler has nothing to fuse).
/// Grid (n_kv, chunks).
/// args: 0 q, 1 words per position, 2 NV, 3 scale, 4 n_head (partials'
///       stride)
pub fn attn_dec(hd: usize, hpk: usize) -> String {
    let d = hd / 32;
    let nw = hd / 2;
    let mut ql = String::new();
    let mut zs = String::new();
    for h in 0..hpk {
        for i in 0..d {
            ql += &format!("    let q{h}_{i} = act[qo + {h}u * {hd}u + lane + {o}u] * scale;\n", o = 32 * i);
            zs += &format!("    var o{h}_{i} = 0.0;\n");
        }
        zs += &format!("    var m{h} = -3.0e38;\n    var l{h} = 0.0;\n");
    }
    let lp = |kb: &str, vb: &str| -> String {
        let mut s = format!("        for (var j = j0 + sg; j < end; j += 4u) {{\n            let kr = j * rw + g * {nw}u + lane / 2u;\n");
        for i in 0..d {
            s += &format!("            let kw{i} = unpack2x16float({kb}[kr + {o}u]); let k{i} = select(kw{i}.x, kw{i}.y, odd);\n", o = 16 * i);
            s += &format!("            let vw{i} = unpack2x16float({vb}[kr + {o}u]); let v{i} = select(vw{i}.x, vw{i}.y, odd);\n", o = 16 * i);
        }
        for h in 0..hpk {
            s += &format!("            {{ var x = q{h}_0 * k0;");
            for i in 1..d {
                s += &format!(" x = fma(q{h}_{i}, k{i}, x);");
            }
            s += " x = x + subgroupShuffleXor(x, 16u); x = x + subgroupShuffleXor(x, 8u); x = x + subgroupShuffleXor(x, 4u); x = x + subgroupShuffleXor(x, 2u); x = x + subgroupShuffleXor(x, 1u);\n";
            s += &format!("              let mn = max(m{h}, x); let c = pexp(m{h} - mn); let p = pexp(x - mn); l{h} = fma(l{h}, c, p);\n");
            for i in 0..d {
                s += &format!("              o{h}_{i} = fma(o{h}_{i}, c, p * v{i});\n");
            }
            s += &format!("              m{h} = mn; }}\n");
        }
        s += "        }\n";
        s
    };
    let mut put = String::new();
    for h in 0..hpk {
        put += &format!("    if (lane == 0u) {{ sm[sg * {hpk}u + {h}u] = m{h}; sl[sg * {hpk}u + {h}u] = l{h}; }}\n");
        for i in 0..d {
            put += &format!("    sacc[(sg * {hpk}u + {h}u) * {hd}u + lane + {o}u] = o{h}_{i};\n", o = 32 * i);
        }
    }
    format!(
        r#"{h}
{pexp}
@group(0) @binding(17) var<storage, read_write> pt: array<f32>;
var<workgroup> sm: array<f32, {s4}>;
var<workgroup> sl: array<f32, {s4}>;
var<workgroup> sacc: array<f32, {sa}>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let pos = uload4(li, 0u).x;
    let g = wg.x;
    let c = wg.y;
    let j0 = c * {chunk}u;
    if (j0 > pos || j0 >= arg(2u)) {{ return; }}
    let end = min(j0 + {chunk}u, pos + 1u);
    let rw = arg(1u);
    let sg = li / 32u;
    let lane = li % 32u;
    let odd = (lane & 1u) == 1u;
    let scale = argf(3u);
    let qo = arg(0u) + g * {hpk}u * {hd}u;
{ql}{zs}{l0}{put}    workgroupBarrier();
    for (var x = li; x < {hpk}u * {hd}u; x += 128u) {{
        let h = x / {hd}u;
        let i = x % {hd}u;
        let s0 = h; let s1 = {hpk}u + h; let s2 = 2u * {hpk}u + h; let s3 = 3u * {hpk}u + h;
        let M = max(max(sm[s0], sm[s1]), max(sm[s2], sm[s3]));
        let w0 = pexp(sm[s0] - M); let w1 = pexp(sm[s1] - M); let w2 = pexp(sm[s2] - M); let w3 = pexp(sm[s3] - M);
        let pb = (c * arg(4u) + g * {hpk}u + h) * {hp}u;
        if (i == 0u) {{
            pt[pb] = M;
            var L = sl[s0] * w0; L = fma(sl[s1], w1, L); L = fma(sl[s2], w2, L); L = fma(sl[s3], w3, L);
            pt[pb + 1u] = L;
        }}
        var O = sacc[s0 * {hd}u + i] * w0; O = fma(sacc[s1 * {hd}u + i], w1, O); O = fma(sacc[s2 * {hd}u + i], w2, O); O = fma(sacc[s3 * {hd}u + i], w3, O);
        pt[pb + 2u + i] = O;
    }}
}}
"#,
        h = header(&format!("{ACT}\n{KV_BINDS}")),
        pexp = pexp_wgsl(),
        s4 = (4 * hpk).max(4),
        sa = 4 * hpk * hd,
        chunk = DCHUNK,
        hp = hd + 2,
        l0 = lp("vk", "vv"),
    )
}

/// Decode: the token's head h from `attn_dec`'s partials over the chunks
/// up to its position, in order.  Grid (n_head).
/// args: 0 out, 1 n_head (partials' stride)
pub fn merge_dec(hd: usize) -> String {
    format!(
        r#"{h}
@group(0) @binding(17) var<storage, read_write> pt: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let pos = uload4(li, 0u).x;
    let h = wg.x;
    let n = pos / {chunk}u + 1u;
    let nh = arg(1u);
    var M = -3.0e38;
    for (var c = 0u; c < n; c++) {{ M = max(M, pt[(c * nh + h) * {hp}u]); }}
    for (var i = li; i < {hd}u; i += 128u) {{
        var l = 0.0;
        var o = 0.0;
        for (var c = 0u; c < n; c++) {{
            let b = (c * nh + h) * {hp}u;
            let w = exp(pt[b] - M);
            l += pt[b + 1u] * w;
            o += pt[b + 2u + i] * w;
        }}
        act[arg(0u) + h * {hd}u + i] = o / l;
    }}
}}
"#,
        h = header(ACT),
        chunk = DCHUNK,
        hp = hd + 2,
    )
}

/// Whether `attn_dec_fast` handles this shape (a kv head's row in whole
/// 16-byte words that split 128 threads evenly).
pub fn attn_dec_fast_ok(hd: usize, hpk: usize) -> bool {
    DCHUNK == 64 && hd % 8 == 0 && hd <= 256 && 128 % (hd / 8) == 0 && (1..=16).contains(&hpk)
}

/// Decode attention, flash-decoding style: workgroup (kv head g, slot c of
/// DCHUNK positions), 128 threads. Scores: two threads per position, each
/// with half the query heads (queries in workgroup memory), reading the key
/// row 8 values at a time, so no lane shuffles; then one softmax per head
/// over the slot (hardware exp); then the values, each thread 8 dims of its
/// share of heads, rows in order. Writes the same (max, sum, sums of v)
/// partials as `attn_dec` for `merge_dec`, but not bit for bit what
/// `cpu_attn` computes (PELICA_ATTN=0 brings back `attn_dec` for that).
/// Grid (n_kv, chunks). args as `attn_dec`.
pub fn attn_dec_fast(hd: usize, hpk: usize) -> String {
    let n4 = hd / 8;
    let tpw = 128 / n4;
    let hh1 = hpk.div_ceil(2);
    let hh3 = hpk.div_ceil(tpw);
    format!(
        r#"{h}
@group(0) @binding(11) var<storage, read_write> vk: array<vec4<u32>>;
@group(0) @binding(12) var<storage, read_write> vv: array<vec4<u32>>;
@group(0) @binding(17) var<storage, read_write> pt: array<f32>;
var<workgroup> qs: array<vec4<f32>, {nq4}>;
var<workgroup> sc: array<f32, {ns}>;
var<workgroup> sm: array<f32, {hpk}>;
var<workgroup> sl: array<f32, {hpk}>;
fn h8(w: vec4<u32>, i: u32) -> vec4<f32> {{
    if (i == 0u) {{ return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }}
    return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w));
}}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let pos = uload4(li, 0u).x;
    let g = wg.x;
    let c = wg.y;
    let j0 = c * {ch}u;
    if (j0 > pos || j0 >= arg(2u)) {{ return; }}
    let end = min(j0 + {ch}u, pos + 1u);
    let n = end - j0;
    let rw4 = arg(1u) / 4u;
    let scale = argf(3u);
    let qo = arg(0u) + g * {hpk}u * {hd}u;
    for (var x = li; x < {nq4}u; x += 128u) {{
        let o = qo + 4u * x;
        qs[x] = vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u]) * scale;
    }}
    workgroupBarrier();
    // scores: thread (position jj, half p) over heads p*HH1..
    let jj = li % {ch}u;
    let p = li / {ch}u;
    var s: array<f32, {hh1}>;
    for (var t = 0u; t < {hh1}u; t++) {{ s[t] = 0.0; }}
    if (jj < n) {{
        let kb = (j0 + jj) * rw4 + g * {n4}u;
        for (var w = 0u; w < {n4}u; w++) {{
            let kw = vk[kb + w];
            let k0 = h8(kw, 0u);
            let k1 = h8(kw, 1u);
            for (var t = 0u; t < {hh1}u; t++) {{
                let h = min(p * {hh1}u + t, {hpk}u - 1u);
                let qb = h * {q4}u + 2u * w;
                s[t] = s[t] + dot(qs[qb], k0) + dot(qs[qb + 1u], k1);
            }}
        }}
    }}
    for (var t = 0u; t < {hh1}u; t++) {{
        let h = p * {hh1}u + t;
        if (h < {hpk}u) {{ sc[h * {ch}u + jj] = select(-3.0e38, s[t], jj < n); }}
    }}
    workgroupBarrier();
    // softmax per head over the slot: a subgroup per head
    let sg = li / 32u;
    let lane = li % 32u;
    for (var h = sg; h < {hpk}u; h += 4u) {{
        let a = sc[h * {ch}u + lane];
        let b = sc[h * {ch}u + lane + 32u];
        let m = subgroupMax(max(a, b));
        let pa = exp(a - m);
        let pb = exp(b - m);
        sc[h * {ch}u + lane] = pa;
        sc[h * {ch}u + lane + 32u] = pb;
        let l = subgroupAdd(pa + pb);
        if (lane == 0u) {{ sm[h] = m; sl[h] = l; }}
    }}
    workgroupBarrier();
    // values: thread (8 dims w, share q) over heads q*HH3..
    let w = li % {n4}u;
    let q = li / {n4}u;
    var o0: array<vec4<f32>, {hh3}>;
    var o1: array<vec4<f32>, {hh3}>;
    for (var t = 0u; t < {hh3}u; t++) {{ o0[t] = vec4<f32>(0.0); o1[t] = vec4<f32>(0.0); }}
    // all DCHUNK rows (a fixed count the compiler can unroll): rows past
    // the token are in the buffer and weigh 0
    for (var j = 0u; j < {ch}u; j++) {{
        let vw = vv[(j0 + j) * rw4 + g * {n4}u + w];
        let v0 = h8(vw, 0u);
        let v1 = h8(vw, 1u);
        for (var t = 0u; t < {hh3}u; t++) {{
            let h = min(q * {hh3}u + t, {hpk}u - 1u);
            let pr = sc[h * {ch}u + j];
            o0[t] = fma(vec4<f32>(pr), v0, o0[t]);
            o1[t] = fma(vec4<f32>(pr), v1, o1[t]);
        }}
    }}
    for (var t = 0u; t < {hh3}u; t++) {{
        let h = q * {hh3}u + t;
        if (h < {hpk}u) {{
            let pb = (c * arg(4u) + g * {hpk}u + h) * {hp}u + 2u + 8u * w;
            pt[pb] = o0[t].x; pt[pb + 1u] = o0[t].y; pt[pb + 2u] = o0[t].z; pt[pb + 3u] = o0[t].w;
            pt[pb + 4u] = o1[t].x; pt[pb + 5u] = o1[t].y; pt[pb + 6u] = o1[t].z; pt[pb + 7u] = o1[t].w;
        }}
    }}
    if (li < {hpk}u) {{
        let pb = (c * arg(4u) + g * {hpk}u + li) * {hp}u;
        pt[pb] = sm[li];
        pt[pb + 1u] = sl[li];
    }}
}}
"#,
        h = header(ACT),
        nq4 = hpk * hd / 4,
        q4 = hd / 4,
        ns = hpk * DCHUNK,
        ch = DCHUNK,
        hp = hd + 2,
    )
}

/// `merge_dec` with each slot's weight computed once (in workgroup memory,
/// a tile of slots at a time) instead of by every thread.
pub fn merge_dec_fast(hd: usize) -> String {
    format!(
        r#"{h}
@group(0) @binding(17) var<storage, read_write> pt: array<f32>;
var<workgroup> ws: array<f32, 256>;
var<workgroup> red: array<f32, 4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let pos = uload4(li, 0u).x;
    let h = wg.x;
    let n = pos / {chunk}u + 1u;
    let nh = arg(1u);
    let sg = li / 32u;
    let lane = li % 32u;
    var m = -3.0e38;
    for (var c = li; c < n; c += 128u) {{ m = max(m, pt[(c * nh + h) * {hp}u]); }}
    m = subgroupMax(m);
    if (lane == 0u) {{ red[sg] = m; }}
    workgroupBarrier();
    let M = max(max(red[0], red[1]), max(red[2], red[3]));
    workgroupBarrier();
    var lp = 0.0;
    var o0 = 0.0;
    var o1 = 0.0;
    for (var t0 = 0u; t0 < n; t0 += 256u) {{
        let tn = min(256u, n - t0);
        for (var x = li; x < tn; x += 128u) {{
            let b = ((t0 + x) * nh + h) * {hp}u;
            let w = exp(pt[b] - M);
            ws[x] = w;
            lp += pt[b + 1u] * w;
        }}
        // four slots a step (padded with zero weights), loads in flight together
        let t4 = (tn + 3u) / 4u * 4u;
        for (var x = tn + li; x < t4; x += 128u) {{ ws[x] = 0.0; }}
        workgroupBarrier();
        let s1 = nh * {hp}u;
        for (var x = 0u; x < t4; x += 4u) {{
            let b = ((t0 + x) * nh + h) * {hp}u + 2u + li;
            if (li < {hd}u) {{ o0 += pt[b] * ws[x] + pt[b + s1] * ws[x + 1u] + pt[b + 2u * s1] * ws[x + 2u] + pt[b + 3u * s1] * ws[x + 3u]; }}
            if ({hd}u > 128u && li + 128u < {hd}u) {{ o1 += pt[b + 128u] * ws[x] + pt[b + 128u + s1] * ws[x + 1u] + pt[b + 128u + 2u * s1] * ws[x + 2u] + pt[b + 128u + 3u * s1] * ws[x + 3u]; }}
        }}
        workgroupBarrier();
    }}
    lp = subgroupAdd(lp);
    if (lane == 0u) {{ red[sg] = lp; }}
    workgroupBarrier();
    let L = red[0] + red[1] + red[2] + red[3];
    if (li < {hd}u) {{ act[arg(0u) + h * {hd}u + li] = o0 / L; }}
    if ({hd}u > 128u && li + 128u < {hd}u) {{ act[arg(0u) + h * {hd}u + li + 128u] = o1 / L; }}
}}
"#,
        h = header(ACT),
        chunk = DCHUNK,
        hp = hd + 2,
    )
}

/// Query tokens per prompt-attention tile: 64 query rows (tokens x the
/// heads sharing a kv head).
pub fn attn_tq(hpk: usize) -> usize {
    (64 / hpk).max(1)
}

/// Prompt attention: workgroup (tile of query tokens, kv head g, split j of
/// round R0..), with every query head of g, so a key is read once per tile.
/// Key tiles of 32 go through workgroup memory; 4 threads per query row
/// share its dims (interleaved words, so no bank conflicts) and sum dot
/// products by shuffles. Keys below NV come from VRAM, the rest from the
/// round's staging copy. Writes (max, sum, sums of v) per (token, head,
/// split) for `attn_fold`.  Grid (PT / TQ, n_kv, RSPL).
/// args: 0 q, 1 q stride, 2 words per position, 3 NV, 4 R0, 5 scale,
///       6 n_head
pub fn attn_pre(hd: usize, hpk: usize) -> String {
    let tq = attn_tq(hpk);
    let rows = tq * hpk;
    let nw = hd / 2;
    let ni = nw / 4;
    let mut ql = String::new();
    let mut zs = String::new();
    for i in 0..ni {
        ql += &format!("    let q{i} = vec2<f32>(act[qb + 2u * ({i}u * 4u + part)], act[qb + 2u * ({i}u * 4u + part) + 1u]) * scale;\n");
        zs += &format!("    var o{i} = vec2<f32>(0.0);\n");
    }
    let mut sc = String::new();
    for kk in 0..32 {
        let dot: Vec<String> = (0..ni).map(|i| format!("dot(unpack2x16float(ks[{kk}u * {nw}u + {i}u * 4u + part]), q{i})")).collect();
        sc += &format!(
            "        var s{kk} = -3.0e38;\n        if (n > {kk}u) {{ var a = {}; a += subgroupShuffleXor(a, 1u); a += subgroupShuffleXor(a, 2u); if (valid && kt + {kk}u <= p) {{ s{kk} = a; }} }}\n",
            dot.join(" + ")
        );
    }
    let mx: Vec<String> = (0..32).map(|kk| format!("s{kk}")).collect();
    let mut pv = String::new();
    for kk in 0..32 {
        pv += &format!("        if (s{kk} > -1.0e38) {{ let pk = exp(s{kk} - mn); l += pk;");
        for i in 0..ni {
            pv += &format!(" o{i} += pk * unpack2x16float(vs[{kk}u * {nw}u + {i}u * 4u + part]);");
        }
        pv += " }\n";
    }
    let mut wr = String::new();
    for i in 0..ni {
        wr += &format!("        pp[pb + 2u + 2u * ({i}u * 4u + part)] = o{i}.x;\n        pp[pb + 3u + 2u * ({i}u * 4u + part)] = o{i}.y;\n");
    }
    // max of the 32 scores, as a tree
    let mut lvl = mx.clone();
    while lvl.len() > 1 {
        lvl = lvl.chunks(2).map(|c| format!("max({}, {})", c[0], c[1])).collect();
    }
    format!(
        r#"{h}
@group(0) @binding(18) var<storage, read_write> pp: array<f32>;
var<workgroup> ks: array<u32, {tile}>;
var<workgroup> vs: array<u32, {tile}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let u = uload4(li, 0u);
    let st0 = u.x;
    let T = u.y;
    let tile = wg.x;
    let g = wg.y;
    let j = wg.z;
    if (tile * {tq}u >= T) {{ return; }}
    let r0 = arg(4u);
    let a0 = r0 + j * {split}u;
    let pmax = st0 + min(T, (tile + 1u) * {tq}u) - 1u;
    if (a0 > pmax) {{ return; }}
    let e = min(a0 + {split}u, pmax + 1u);
    let nv = arg(3u);
    let rw = arg(2u);
    let scale = argf(5u);
    let r = li / 4u;
    let part = li % 4u;
    let t = tile * {tq}u + r / {hpk}u;
    let h = g * {hpk}u + r % {hpk}u;
    let valid = r < {rows}u && t < T;
    let tc = min(t, T - 1u);
    let p = st0 + tc;
    let qb = arg(0u) + tc * arg(1u) + min(h, arg(6u) - 1u) * {hd}u;
{ql}{zs}    var m = -3.0e38;
    var l = 0.0;
    for (var kt = a0; kt < e; kt += 32u) {{
        workgroupBarrier();
        for (var x = li; x < {tile}u; x += 256u) {{
            let key = kt + x / {nw}u;
            let w = x % {nw}u;
            var kw = 0u;
            var vw = 0u;
            if (key < e) {{
                if (key < nv) {{
                    kw = vk[key * rw + g * {nw}u + w];
                    vw = vv[key * rw + g * {nw}u + w];
                }} else {{
                    kw = sk[(key - r0) * rw + g * {nw}u + w];
                    vw = sv[(key - r0) * rw + g * {nw}u + w];
                }}
            }}
            ks[x] = kw;
            vs[x] = vw;
        }}
        workgroupBarrier();
        let n = min(32u, e - kt);
{sc}        let mn = max(m, {mx});
        let c = exp(m - mn);
        l = l * c;
{scale_o}{pv}        m = mn;
    }}
    if (valid) {{
        let pb = ((t * arg(6u) + h) * {rspl}u + j) * {hp}u;
        if (part == 0u) {{ pp[pb] = m; pp[pb + 1u] = l; }}
{wr}    }}
}}
"#,
        h = header(&format!("{ACT}\n{KV_BINDS}\n{STAGE_BINDS}")),
        tile = 32 * nw,
        tq = tq,
        split = SPLIT,
        rows = rows,
        rspl = RSPL,
        hp = hd + 2,
        mx = lvl[0],
        scale_o = (0..ni).map(|i| format!("        o{i} = o{i} * c;\n")).collect::<String>(),
    )
}

/// Prompt attention: folds a round's split partials into each (token,
/// head)'s running (max, sum, sums of v), split by split in key order (so
/// the result doesn't depend on rounds or tiers), and writes the output so
/// far.  Grid (n_head, PT).
/// args: 0 out, 1 out stride, 2 running state (act), 3 R0, 4 first round,
///       5 n_head
pub fn attn_fold(hd: usize) -> String {
    format!(
        r#"{h}
@group(0) @binding(18) var<storage, read_write> pp: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let u = uload4(li, 0u);
    let t = wg.y;
    if (t >= u.y) {{ return; }}
    let h = wg.x;
    let p = u.x + t;
    let nh = arg(5u);
    let ab = arg(2u) + (t * nh + h) * {hp}u;
    var M0 = -3.0e38;
    var L0 = 0.0;
    if (arg(4u) == 0u) {{ M0 = act[ab]; L0 = act[ab + 1u]; }}
    workgroupBarrier();
    var M = M0;
    var L = L0;
    for (var i = li; i < {hd}u; i += 128u) {{
        M = M0;
        L = L0;
        var O = 0.0;
        if (arg(4u) == 0u) {{ O = act[ab + 2u + i]; }}
        for (var j = 0u; j < {rspl}u; j++) {{
            if (arg(3u) + j * {split}u > p) {{ break; }}
            let pb = ((t * nh + h) * {rspl}u + j) * {hp}u;
            let pm = pp[pb];
            let mn = max(M, pm);
            let w0 = exp(M - mn);
            let w1 = exp(pm - mn);
            L = L * w0 + pp[pb + 1u] * w1;
            O = O * w0 + pp[pb + 2u + i] * w1;
            M = mn;
        }}
        act[ab + 2u + i] = O;
        act[arg(0u) + t * arg(1u) + h * {hd}u + i] = select(0.0, O / L, L > 0.0);
    }}
    if (li == 0u) {{ act[ab] = M; act[ab + 1u] = L; }}
}}
"#,
        h = header(ACT),
        hp = hd + 2,
        rspl = RSPL,
        split = SPLIT,
    )
}

/// Routing: each token's top-k experts (subgroup per token) by softmax or
/// sigmoid (+ a selection bias) of its router logits, the weights (optionally
/// renormalised, scaled), then the tokens grouped by expert (`PL_*`), and
/// each expert's use counted (`usage`).  Grid (1).
/// args: 0 logits (stride n_exp), 1 n_exp, 2 k, 3 renormalise, 4 scale
///       (f32), 5 sigmoid, 6 selection bias (fw, NONE), 7 layer, 8 weights out,
///       9 where in st the routes go (per layer PT tokens x K (expert, weight))
pub fn plan() -> String {
    format!(
        r#"{h}
@group(0) @binding(10) var<storage, read_write> usage: array<u32>;
var<workgroup> cnt: array<atomic<u32>, {ne}>;
var<workgroup> base: array<u32, {ne}>;
var<workgroup> sel: array<u32, {pairs}>;
var<workgroup> sa: array<vec4<u32>, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {{
    let T = uload4(li, 0u).y;
    let ne = arg(1u);
    let K = arg(2u);
    for (var i = li; i < {ne}u; i += 256u) {{ atomicStore(&cnt[i], 0u); }}
    workgroupBarrier();
    let sg = li / 32u;
    let lane = li % 32u;
    let nper = (ne + 31u) / 32u;
    for (var t = sg; t < T; t += 8u) {{
        let lg = arg(0u) + t * ne;
        var pr: array<f32, 16>;
        var sv: array<f32, 16>;
        var mx = -3.0e38;
        for (var i = 0u; i < nper; i++) {{
            let e = lane + 32u * i;
            var v = -3.0e38;
            if (e < ne) {{ v = act[lg + e]; }}
            pr[i] = v;
            mx = max(mx, v);
        }}
        mx = subgroupMax(mx);
        var sum = 0.0;
        for (var i = 0u; i < nper; i++) {{
            let e = lane + 32u * i;
            if (arg(5u) == 1u) {{
                pr[i] = 1.0 / (1.0 + exp(-pr[i]));
            }} else {{
                pr[i] = select(0.0, exp(pr[i] - mx), e < ne);
                sum += pr[i];
            }}
        }}
        if (arg(5u) == 0u) {{
            sum = subgroupAdd(sum);
            for (var i = 0u; i < nper; i++) {{ pr[i] = pr[i] / sum; }}
        }}
        for (var i = 0u; i < nper; i++) {{
            let e = lane + 32u * i;
            sv[i] = pr[i];
            if (arg(6u) != NONE && e < ne) {{ sv[i] += fw[arg(6u) + e]; }}
            if (e >= ne) {{ sv[i] = -3.0e38; }}
        }}
        var ws: array<f32, {km}>;
        var es: array<u32, {km}>;
        var wsum = 0.0;
        for (var k = 0u; k < K; k++) {{
            var bv = -3.0e38;
            var bi = 0xffffffffu;
            for (var i = 0u; i < nper; i++) {{
                if (sv[i] > bv) {{ bv = sv[i]; bi = lane + 32u * i; }}
            }}
            let m = subgroupMax(bv);
            let e = subgroupMin(select(0xffffffffu, bi, bv == m));
            var mine = 0.0;
            if (lane == e % 32u) {{ mine = pr[e / 32u]; sv[e / 32u] = -3.0e38; }}
            let w = subgroupAdd(mine);
            es[k] = e;
            ws[k] = w;
            wsum += w;
        }}
        if (lane == 0u) {{
            // and the route itself, per (layer, token), for tracing
            let tb = arg(9u) + (arg(7u) * {pt}u + t) * K * 2u;
            for (var k = 0u; k < K; k++) {{
                var w = ws[k];
                if (arg(3u) == 1u) {{ w = w / max(wsum, 6.103515625e-5); }}
                if (argf(4u) != 0.0) {{ w = w * argf(4u); }}
                act[arg(8u) + t * K + k] = w;
                sel[t * K + k] = es[k];
                atomicAdd(&cnt[es[k]], 1u);
                st[tb + 2u * k] = es[k];
                st[tb + 2u * k + 1u] = bitcast<u32>(w);
            }}
        }}
    }}
    workgroupBarrier();
    // the experts in use, in order, and where each one's tokens start: a
    // scan over the experts, 256 at a time (subgroup scans, then their sums)
    var run_n = 0u;
    var run_o = 0u;
    for (var e0 = 0u; e0 < ne; e0 += 256u) {{
        let e = e0 + li;
        var c = 0u;
        if (e < ne) {{ c = atomicLoad(&cnt[e]); }}
        let a = select(0u, 1u, c > 0u);
        let ia = subgroupInclusiveAdd(a);
        let ic = subgroupInclusiveAdd(c);
        if (lane == 31u) {{ sa[sg] = vec4<u32>(ia, ic, 0u, 0u); }}
        workgroupBarrier();
        var pa = 0u;
        var pc = 0u;
        var ta = 0u;
        var tc = 0u;
        for (var j = 0u; j < 8u; j++) {{
            let v = sa[j];
            if (j < sg) {{ pa += v.x; pc += v.y; }}
            ta += v.x;
            tc += v.y;
        }}
        if (c > 0u) {{
            let na = run_n + pa + ia - a;
            let o = run_o + pc + ic - c;
            st[PL_ACT + na] = e;
            st[PL_OFF + na] = o;
            st[PL_CNT + na] = c;
            base[e] = o;
            usage[arg(7u) * ne + e] += c;
        }}
        if (e < ne) {{ atomicStore(&cnt[e], 0u); }}
        run_n += ta;
        run_o += tc;
        workgroupBarrier();
    }}
    if (li == 0u) {{ st[PL_NACT] = run_n; }}
    workgroupBarrier();
    for (var i = li; i < T * K; i += 256u) {{
        let e = sel[i];
        let p = base[e] + atomicAdd(&cnt[e], 1u);
        st[PL_PPOS + i] = p;
        st[PL_PTOK + p] = i / K;
        st[PL_SEL + i] = e;
    }}
}}
"#,
        h = header(&format!("{ACT}\n{FW}")),
        ne = NEMAX,
        pairs = PT * KMAX,
        pt = PT,
        km = KMAX,
    )
}

/// The decoding loops of an expert kernel over the tokens routed to the
/// workgroup's expert, from VRAM (`v`) or host RAM (`h`) buffers.
fn expert_loop(two: bool, gpb: (usize, usize), bw: (usize, usize), suf: &str) -> String {
    let mut s = format!(
        "            for (var g = lane; g < G; g += 32u) {{\n                let wa = dqa_{suf}(ab + (g / {g0}u) * {b0}u, g % {g0}u);\n",
        g0 = gpb.0,
        b0 = bw.0
    );
    if two {
        s += &format!("                let wb = dqb_{suf}(bb + (g / {g1}u) * {b1}u, g % {g1}u);\n", g1 = gpb.1, b1 = bw.1);
    }
    s += &(0..8)
        .map(|j| {
            let b = if two { format!(" ab{j} += dot(wb, x);") } else { String::new() };
            format!("                if (n > {j}u) {{ let o = xo{j} + 4u * g; let x = vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u]); aa{j} += dot(wa, x);{b} }}\n")
        })
        .collect::<String>();
    s += "            }\n";
    s
}

/// The routed experts, a workgroup per (8 rows, active expert): gate and up
/// rows (types `ta`, `tb`) against each routed token's normed x, giving
/// silu(gate) * up per pair; or (`tb` None) down rows against each pair's
/// hidden vector, giving the pair's output. Experts live in VRAM or host
/// RAM, as `loc` says (bit 31: host; the rest: slot).
/// Grid (rows/8, active experts at most).
/// args: 0 rows, 1 cols, 2 x (per token, or per pair), 3 x stride, 4 out
///       (per pair), 5 layer, 6 n_exp, 7 words per expert (a), 8 words per
///       row (a), 9 words per expert (b), 10 words per row (b), 11 x by pair
pub fn expert(ta: Ty, tb: Option<Ty>) -> String {
    let two = tb.is_some();
    let (ga, ba) = geom(ta);
    let (gb, bb) = tb.map(geom).unwrap_or((1, 1));
    let mut dq = deq4(ta, "dqa_v", "av") + &deq4(ta, "dqa_h", "ah");
    let mut binds = format!("{ACT}\n@group(0) @binding(6) var<storage, read> av: array<u32>;\n@group(0) @binding(8) var<storage, read> ah: array<u32>;\n@group(0) @binding(10) var<storage, read> loc: array<u32>;");
    if let Some(tb) = tb {
        dq += &deq4(tb, "dqb_v", "bv");
        dq += &deq4(tb, "dqb_h", "bh");
        binds += "\n@group(0) @binding(7) var<storage, read> bv: array<u32>;\n@group(0) @binding(9) var<storage, read> bh: array<u32>;";
    }
    let fin: String = (0..8)
        .map(|j| {
            let val = if two { format!("silu(subgroupAdd(aa{j})) * subgroupAdd(ab{j})") } else { format!("subgroupAdd(aa{j})") };
            format!("        {{ let v = {val}; if (lane == 0u && valid && n > {j}u) {{ act[arg(4u) + (u.y + c + {j}u) * rows + row] = v; }} }}\n")
        })
        .collect();
    let xo: String = (0..8)
        .map(|j| format!("        var xo{j} = 0u;\n        if (n > {j}u) {{ let pp = u.y + c + {j}u; var r = pp; if (arg(11u) == 0u) {{ r = st[PL_PTOK + pp]; }} xo{j} = arg(2u) + r * arg(3u); }}\n"))
        .collect();
    let bdecl = if two { "        let bb = slot * arg(9u) + row * arg(10u);\n" } else { "" };
    format!(
        r#"{h}
{dq}
var<workgroup> we: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let na = uload4(li, {nact}u).x;
    let ui = wg.y;
    if (ui >= na) {{ return; }}
    if (li == 0u) {{
        let e = st[PL_ACT + ui];
        we = vec4<u32>(e, st[PL_OFF + ui], st[PL_CNT + ui], loc[arg(5u) * arg(6u) + e]);
    }}
    let u = workgroupUniformLoad(&we);
    let rows = arg(0u);
    let row0 = wg.x * 8u + li / 32u;
    let valid = row0 < rows;
    let row = min(row0, rows - 1u);
    let lane = li % 32u;
    let G = arg(1u) / 4u;
    let host = (u.w >> 31u) != 0u;
    let slot = u.w & 0x7fffffffu;
    let ab = slot * arg(7u) + row * arg(8u);
{bdecl}    for (var c = 0u; c < u.z; c += 8u) {{
        let n = min(8u, u.z - c);
{za}{zb}{xo}        if (host) {{
{lh}        }} else {{
{lv}        }}
{fin}    }}
}}
"#,
        h = header(&binds),
        nact = PL_NACT,
        za = zeros("aa"),
        zb = if two { zeros("ab") } else { String::new() },
        lh = expert_loop(two, (ga, gb), (ba, bb), "h"),
        lv = expert_loop(two, (ga, gb), (ba, bb), "v"),
    )
}

/// `fn {name}(rb: u32, cols: u32, xo: u32, lane: u32) -> f32`: lane `lane`'s
/// share of the dot product of the row at word `rb` of buffer `buf` (type
/// `ty`, `cols` values) with act[xo..]; summing the subgroup's shares gives
/// the dot product. K-quants go a block per subgroup step: each lane takes
/// one word of quants (8 values, one coalesced 128-byte read per block) and
/// the block's scales once; small-block types a block per lane; the rest 4
/// values at a time through `deq4`.
/// `rowdot` for rows of `cols` values read `XCH` at a time (known when the
/// kernel is made): the blocks of a call unrolled, all their loads issued
/// together, so a row costs one trip to memory instead of one per four
/// blocks. Falls back to `rowdot` for other types and widths.
pub fn rowdot_w(ty: Ty, name: &str, buf: &str, cols: usize) -> String {
    let base = rowdot(ty, name, buf);
    let (vals, words) = match ty {
        Ty::Q4K => (256, 36),
        Ty::Q5K => (256, 44),
        Ty::Q6K => (256, 53),
        _ => return base,
    };
    let per = cols.min(XCH as usize);
    if cols % per != 0 || per % vals != 0 || std::env::var("PELICA_ROWW").is_ok_and(|v| v == "0") {
        return base;
    }
    let nb = per / vals;
    // the block function from `rowdot`, then a straight-line row over it
    let unit_end = base.find(&format!("fn {name}(")).unwrap();
    let mut out = base[..unit_end].to_string();
    out += &format!("fn {name}(rb: u32, cols: u32, xo: u32, lane: u32) -> f32 {{\n");
    for j in 0..nb {
        out += &format!("    let p{j} = {name}_b(rb + {}u, xo + {}u, lane);\n", j * words, j * vals);
    }
    // pairwise sums
    let mut terms: Vec<String> = (0..nb).map(|j| format!("p{j}")).collect();
    while terms.len() > 1 {
        terms = terms.chunks(2).map(|c| if c.len() == 2 { format!("({} + {})", c[0], c[1]) } else { c[0].clone() }).collect();
    }
    out += &format!("    return {};\n}}\n", terms[0]);
    out
}

pub fn rowdot(ty: Ty, name: &str, buf: &str) -> String {
    // one block's share: (lane setup, body using b = block word, x0 = its
    // first value's offset), values and words per block
    let (setup, blk, vals, words): (&str, &str, u32, u32) = match ty {
        Ty::Q4K => (
            "    let c = lane / 8u;\n    let i = 4u * (lane % 8u);\n",
            "    let dm = unpack2x16float(B[b]);\n    let s0 = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], 2u * c);\n    let s1 = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], 2u * c + 1u);\n    let w = B[b + 4u + lane];\n    let xa = xv4(x0 + 64u * c + i);\n    let xb = xv4(x0 + 64u * c + i + 32u);\n    let lo = vec4<f32>(nib4(w, false));\n    let hi = vec4<f32>(nib4(w, true));\n    return dm.x * (s0.x * dot(lo, xa) + s1.x * dot(hi, xb)) - dm.y * (s0.y * dot(xa, vec4<f32>(1.0)) + s1.y * dot(xb, vec4<f32>(1.0)));\n",
            256,
            36,
        ),
        Ty::Q5K => (
            "    let c = lane / 8u;\n    let i = 4u * (lane % 8u);\n",
            "    let sh = vec4<u32>(0u, 8u, 16u, 24u);\n    let dm = unpack2x16float(B[b]);\n    let s0 = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], 2u * c);\n    let s1 = scmin_k4(B[b + 1u], B[b + 2u], B[b + 3u], 2u * c + 1u);\n    let w = B[b + 12u + lane];\n    let h = vec4<u32>(B[b + 4u + lane % 8u]);\n    let xa = xv4(x0 + 64u * c + i);\n    let xb = xv4(x0 + 64u * c + i + 32u);\n    let lo = vec4<f32>(nib4(w, false) | (((h >> (sh + vec4<u32>(2u * c))) & vec4<u32>(1u)) << vec4<u32>(4u)));\n    let hi = vec4<f32>(nib4(w, true) | (((h >> (sh + vec4<u32>(2u * c + 1u))) & vec4<u32>(1u)) << vec4<u32>(4u)));\n    return dm.x * (s0.x * dot(lo, xa) + s1.x * dot(hi, xb)) - dm.y * (s0.y * dot(xa, vec4<f32>(1.0)) + s1.y * dot(xb, vec4<f32>(1.0)));\n",
            256,
            44,
        ),
        Ty::Q6K => (
            "    let n = lane / 16u;\n    let o = (4u * lane) % 64u;\n    let qa = o / 32u;\n    let l = o % 32u;\n    let kl = 8u * n + l / 16u + 2u * qa;\n",
            "    let sh = vec4<u32>(0u, 8u, 16u, 24u);\n    let ql = B[b + lane];\n    let qh = vec4<u32>(B[b + 32u + 8u * n + l / 4u]);\n    let lo = vec4<f32>(nib4(ql, false) | (((qh >> (sh + vec4<u32>(2u * qa))) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0;\n    let hi = vec4<f32>(nib4(ql, true) | (((qh >> (sh + vec4<u32>(2u * qa + 4u))) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0;\n    let sl = f32(extractBits(i32(B[b + 48u + kl / 4u]), (kl % 4u) * 8u, 8u));\n    let shi = f32(extractBits(i32(B[b + 48u + (kl + 4u) / 4u]), ((kl + 4u) % 4u) * 8u, 8u));\n    let d = unpack2x16float(B[b + 52u]).x;\n    let xs = x0 + 128u * n + 32u * qa + l;\n    return d * (sl * dot(lo, xv4(xs)) + shi * dot(hi, xv4(xs + 64u)));\n",
            256,
            53,
        ),
        _ => ("", "", 0, 0),
    };
    if vals > 0 {
        // a block per subgroup step, 4 blocks in flight
        let unit = format!("fn {name}_b(b: u32, x0: u32, lane: u32) -> f32 {{\n{setup}{blk}}}\n").replace("B[", &format!("{buf}["));
        return format!(
            "{unit}fn {name}(rb: u32, cols: u32, xo: u32, lane: u32) -> f32 {{\n    let nb = cols / {vals}u;\n    var a0 = 0.0;\n    var a1 = 0.0;\n    var a2 = 0.0;\n    var a3 = 0.0;\n    var k = 0u;\n    for (; k + 4u <= nb; k += 4u) {{\n        let b = rb + k * {words}u;\n        let x = xo + k * {vals}u;\n        a0 += {name}_b(b, x, lane);\n        a1 += {name}_b(b + {words}u, x + {vals}u, lane);\n        a2 += {name}_b(b + {w2}u, x + {v2}u, lane);\n        a3 += {name}_b(b + {w3}u, x + {v3}u, lane);\n    }}\n    for (; k < nb; k++) {{ a0 += {name}_b(rb + k * {words}u, xo + k * {vals}u, lane); }}\n    return (a0 + a1) + (a2 + a3);\n}}\n",
            w2 = 2 * words,
            v2 = 2 * vals,
            w3 = 3 * words,
            v3 = 3 * vals
        );
    }
    let body = match ty {
        Ty::Q8_0 => r#"    var acc = 0.0;
    for (var k = lane; k < cols / 32u; k += 32u) {
        let b = rb + k * 9u;
        let x0 = xo + k * 32u;
        var s = 0.0;
        for (var j = 0u; j < 8u; j++) {
            let w = i32(B[b + 1u + j]);
            s += dot(vec4<f32>(f32(extractBits(w, 0u, 8u)), f32(extractBits(w, 8u, 8u)), f32(extractBits(w, 16u, 8u)), f32(extractBits(w, 24u, 8u))), xv4(x0 + 4u * j));
        }
        acc += unpack2x16float(B[b]).x * s;
    }
    return acc;"#
            .to_string(),
        Ty::Q4_0 => r#"    var acc = 0.0;
    for (var k = lane; k < cols / 32u; k += 32u) {
        let b = rb + k * 5u;
        let x0 = xo + k * 32u;
        var s = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let w = B[b + 1u + j];
            s += dot(vec4<f32>(nib4(w, false)) - 8.0, xv4(x0 + 4u * j)) + dot(vec4<f32>(nib4(w, true)) - 8.0, xv4(x0 + 16u + 4u * j));
        }
        acc += unpack2x16float(B[b]).x * s;
    }
    return acc;"#
            .to_string(),
        _ => {
            let (gpb, bw) = geom(ty);
            return format!(
                "{}fn {name}(rb: u32, cols: u32, xo: u32, lane: u32) -> f32 {{\n    var acc = 0.0;\n    for (var g = lane; g < cols / 4u; g += 32u) {{\n        acc += dot({name}_dq(rb + (g / {gpb}u) * {bw}u, g % {gpb}u), xv4(xo + 4u * g));\n    }}\n    return acc;\n}}\n",
                deq4(ty, &format!("{name}_dq"), buf)
            );
        }
    };
    format!("fn {name}(rb: u32, cols: u32, xo: u32, lane: u32) -> f32 {{\n{}\n}}\n", body.replace("B[", &format!("{buf}[")))
}

/// Decode kernels' input vector, staged in workgroup memory a chunk of `XCH`
/// values at a time (every row of the workgroup reads it, and global reads
/// of it would cost a load per lane per block): `xv4(o)` reads values o..o+4
/// of the chunk; `xload(o, n, li)` stages act[o..o+n].
const XCH: u32 = 2048;
/// Whether decode kernels read the input straight from `act` (it stays in
/// cache) instead of staging it in workgroup memory behind barriers.
pub fn x_direct() -> bool {
    std::env::var("PELICA_XDIRECT").map_or(false, |v| v != "0")
}

fn xs_decl() -> String {
    if x_direct() {
        return "fn xv4(o: u32) -> vec4<f32> { return vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u]); }\nfn xload(o: u32, n: u32, li: u32) {}\n".to_string();
    }
    format!(
        "var<workgroup> xs: array<vec4<f32>, {q}>;\nfn xv4(o: u32) -> vec4<f32> {{ return xs[o >> 2u]; }}\nfn xload(o: u32, n: u32, li: u32) {{\n    workgroupBarrier();\n    for (var i = li; i < n / 4u; i += 256u) {{ let p = o + 4u * i; xs[i] = vec4<f32>(act[p], act[p + 1u], act[p + 2u], act[p + 3u]); }}\n    workgroupBarrier();\n}}\n",
        q = XCH / 4
    )
}

/// Words of a row of type `ty` holding `XCH` values.
fn xch_words(ty: Ty) -> u32 {
    let (gpb, bw) = geom(ty);
    (XCH as usize / 4 / gpb * bw) as u32
}

/// Decoding (one token): y[r] (+)= W[r] . x (+ bias[r]), a subgroup per row
/// reading whole quant blocks (`rowdot`). Same arguments as `matvec` (the
/// token is the pass's only one, or its last with arg 10).  Grid (rows/8).
pub fn matvec_dec(ty: Ty, cols: usize) -> String {
    format!(
        r#"{h}
{xs}{rd}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let T = uload4(li, 0u).y;
    let rows = arg(0u);
    let cols = arg(1u);
    let row = wg.x * 8u + li / 32u;
    let lane = li % 32u;
    var t = 0u;
    if (arg(10u) == 1u) {{ t = T - 1u; }}
    let xo = arg(2u) + t * arg(3u);
    let rb = arg(6u) + min(row, rows - 1u) * arg(7u);
    var s = 0.0;
    for (var c0 = 0u; c0 < cols; c0 += {ch}u) {{
        let n = min({ch}u, cols - c0);
        xload(xo + c0, n, li);
        if (row < rows) {{ s += rd(rb + (c0 / {ch}u) * {wc}u, n, {xb}, lane); }}
    }}
    s = subgroupAdd(s);
    if (row < rows && lane == 0u) {{
        var v = s;
        if (arg(8u) != NONE) {{ v += fw[arg(8u) + row]; }}
        var yo = arg(4u) + t * arg(5u) + row;
        if (arg(10u) == 1u) {{ yo = arg(4u) + row; }}
        if (arg(9u) == 1u) {{ act[yo] += v; }} else {{ act[yo] = v; }}
    }}
}}
"#,
        h = header(&format!("{ACT}\n@group(0) @binding(3) var<storage, read> wq: array<u32>;\n{FW}")),
        xs = xs_decl(),
        rd = rowdot_w(ty, "rd", "wq", cols),
        ch = XCH,
        wc = xch_words(ty),
        xb = if x_direct() { "xo + c0" } else { "0u" },
    )
}

/// Decoding: the routed experts, a workgroup per (8·R rows, active expert),
/// a subgroup per R rows (R = `EXP_RPS`: more blocks in flight, the input
/// staged once for more rows) reading whole quant blocks; each expert has the
/// pass's one token. Same arguments and bindings as `expert`.
/// Grid (rows/(8·R), k).
pub fn exp_rps(two: bool) -> u32 {
    let v = |k: &str, d: u32| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    if two { v("PELICA_RPS_GU", 2) } else { v("PELICA_RPS_D", 2) }
}
pub fn expert_dec(ta: Ty, tb: Option<Ty>, cols: usize) -> String {
    let two = tb.is_some();
    let r = exp_rps(two);
    let mut rd = rowdot_w(ta, "rda_v", "av", cols) + &rowdot_w(ta, "rda_h", "ah", cols);
    let mut binds = format!("{ACT}\n@group(0) @binding(6) var<storage, read> av: array<u32>;\n@group(0) @binding(8) var<storage, read> ah: array<u32>;\n@group(0) @binding(10) var<storage, read> loc: array<u32>;");
    if let Some(tb) = tb {
        rd += &rowdot_w(tb, "rdb_v", "bv", cols);
        rd += &rowdot_w(tb, "rdb_h", "bh", cols);
        binds += "\n@group(0) @binding(7) var<storage, read> bv: array<u32>;\n@group(0) @binding(9) var<storage, read> bh: array<u32>;";
    }
    let wa = xch_words(ta);
    let wb = tb.map_or(0, xch_words);
    let xb = if x_direct() { "xo + c0" } else { "0u" };
    // per row j: its accumulators and offsets, then the calls, all unrolled
    let mut decl = String::new();
    let (mut host, mut vram, mut fin) = (String::new(), String::new(), String::new());
    for j in 0..r {
        decl += &format!("    let row{j} = wg.x * {rw}u + li / 32u + {o}u;\n    let ab{j} = slot * arg(7u) + min(row{j}, rows - 1u) * arg(8u);\n", rw = 8 * r, o = 8 * j);
        if two {
            decl += &format!("    let bb{j} = slot * arg(9u) + min(row{j}, rows - 1u) * arg(10u);\n");
        }
        host += &format!("ga{j} += rda_h(ab{j} + ch * {wa}u, n, {xb}, lane); ");
        vram += &format!("ga{j} += rda_v(ab{j} + ch * {wa}u, n, {xb}, lane); ");
        if two {
            host += &format!("ub{j} += rdb_h(bb{j} + ch * {wb}u, n, {xb}, lane); ");
            vram += &format!("ub{j} += rdb_v(bb{j} + ch * {wb}u, n, {xb}, lane); ");
        }
        let v = if two { format!("silu(subgroupAdd(ga{j})) * subgroupAdd(ub{j})") } else { format!("subgroupAdd(ga{j})") };
        fin += &format!("        let v{j} = {v};\n        if (row{j} < rows && lane == 0u) {{ act[arg(4u) + pp * rows + row{j}] = v{j}; }}\n");
    }
    let zero: String = (0..r).map(|j| format!("        var ga{j} = 0.0;\n        var ub{j} = 0.0;\n")).collect();
    format!(
        r#"{h}
{xs}{rd}
var<workgroup> we: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let na = uload4(li, {nact}u).x;
    let ui = wg.y;
    if (ui >= na) {{ return; }}
    if (li == 0u) {{
        let e = st[PL_ACT + ui];
        we = vec4<u32>(e, st[PL_OFF + ui], st[PL_CNT + ui], loc[arg(5u) * arg(6u) + e]);
    }}
    let u = workgroupUniformLoad(&we);
    let rows = arg(0u);
    let lane = li % 32u;
    let G4 = arg(1u);
    let host = (u.w >> 31u) != 0u;
    let slot = u.w & 0x7fffffffu;
{decl}    for (var c = 0u; c < u.z; c++) {{
        let pp = u.y + c;
        var r = pp;
        if (arg(11u) == 0u) {{ r = st[PL_PTOK + pp]; }}
        let xo = arg(2u) + r * arg(3u);
{zero}        for (var c0 = 0u; c0 < G4; c0 += {ch}u) {{
            let n = min({ch}u, G4 - c0);
            xload(xo + c0, n, li);
            let ch = c0 / {ch}u;
            if (host) {{ {host}}} else {{ {vram}}}
        }}
{fin}    }}
}}

"#,
        h = header(&binds),
        xs = xs_decl(),
        nact = PL_NACT,
        ch = XCH,
    )
}

/// Decoding: the attention's input in one dispatch: x RMS-normed (times the
/// norm weight) while staged, then its q, k and v projections, a subgroup
/// per row of the three matrices stacked (no biases).  Grid ((q+k+v rows)/8).
/// args: 0 q rows, 1 k rows, 2 v rows, 3 cols, 4 x, 5 norm weight (fw),
///       6 eps, 7 q out, 8 k out, 9 v out, 10-12 q/k/v word offsets in the
///       weight buffer, 13-15 q/k/v words per row
pub fn qkv_dec(tq: Ty, tk: Ty, tv: Ty, cols: usize) -> String {
    format!(
        r#"{h}
var<workgroup> xs: array<vec4<f32>, {q4}>;
fn xv4(o: u32) -> vec4<f32> {{ return xs[o >> 2u]; }}
{rq}{rk}{rv}
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = arg(3u);
    let xb = arg(4u);
    let nw = arg(5u);
    var s = 0.0;
    for (var i = li; i < n; i += 256u) {{ let v = act[xb + i]; s += v * v; }}
    s = subgroupAdd(s);
    if (li % 32u == 0u) {{ red[li / 32u] = s; }}
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) {{ tot += red[j]; }}
    let inv = 1.0 / sqrt(tot / f32(n) + argf(6u));
    let nq = arg(0u);
    let nk = arg(1u);
    let nv = arg(2u);
    let row = wg.x * 8u + li / 32u;
    let lane = li % 32u;
    // which matrix (q, k, v) and its row
    var m = 0u;
    var r = row;
    if (row >= nq + nk) {{ m = 2u; r = row - nq - nk; }} else if (row >= nq) {{ m = 1u; r = row - nq; }}
    if (row >= nq + nk + nv) {{ r = 0u; }}
    let rb = arg(10u + m) + r * arg(13u + m);
    var acc = 0.0;
    for (var c0 = 0u; c0 < n; c0 += {ch}u) {{
        let cn = min({ch}u, n - c0);
        workgroupBarrier();
        for (var i = li; i < cn / 4u; i += 256u) {{
            let p = c0 + 4u * i;
            let o = xb + p;
            let w = nw + p;
            xs[i] = vec4<f32>(act[o] * fw[w], act[o + 1u] * fw[w + 1u], act[o + 2u] * fw[w + 2u], act[o + 3u] * fw[w + 3u]) * inv;
        }}
        workgroupBarrier();
        let ch = c0 / {ch}u;
        if (m == 0u) {{ acc += rdq(rb + ch * {wq}u, cn, 0u, lane); }} else if (m == 1u) {{ acc += rdk(rb + ch * {wk}u, cn, 0u, lane); }} else {{ acc += rdv(rb + ch * {wv}u, cn, 0u, lane); }}
    }}
    let v = subgroupAdd(acc);
    if (lane == 0u && row < nq + nk + nv) {{ act[arg(7u + m) + r] = v; }}
}}
"#,
        h = header(&format!("{ACT}\n@group(0) @binding(3) var<storage, read> wq: array<u32>;\n{FW}")),
        q4 = XCH / 4,
        ch = XCH,
        wq = xch_words(tq),
        wk = xch_words(tk),
        wv = xch_words(tv),
        rq = rowdot_w(tq, "rdq", "wq", cols),
        rk = rowdot_w(tk, "rdk", "wq", cols),
        rv = rowdot_w(tv, "rdv", "wq", cols),
    )
}

fn lines(n: usize, f: impl Fn(usize) -> String) -> String {
    (0..n).map(f).collect()
}

/// Prompt-pass products on the tensor cores (`coop_mat16x16`, f16 inputs,
/// f32 sums). The workgroup stages 32 columns at a time of its weight rows
/// (decoded from any GGUF type) and of its tokens' vectors as f16 in
/// workgroup memory; subgroup (wy, wx) of its 8 keeps wm x wn 16x16 sum
/// tiles (gate and up: both) at rows wy*16*wm.. and tokens wx*16*wn... A
/// tile is 8/gc*16*wm rows by gc*16*wn tokens.
///
/// Dense (`experts` false, `tb` None): as `matvec` (same arguments, no
/// last-token mode), grid (rows / tile rows, PT / tile tokens).
/// Experts: as `expert` (same arguments), grid (rows / tile rows, active
/// experts at most, pairs per expert at most / tile tokens); with `tb`,
/// gate and up rows giving silu(gate) * up.
pub fn mm_tc(ta: Ty, tb: Option<Ty>, experts: bool, gc: usize, wm: usize, wn: usize) -> String {
    let gr = 8 / gc;
    let (tr, tt) = (gr * 16 * wm, gc * 16 * wn);
    let gu = tb.is_some();
    assert!(gr * gc == 8 && (!gu || experts));
    let wr = if gu { 2 * tr } else { tr };
    let wits = (wr * 8).div_ceil(256);
    let xits = (tt * 8).div_ceil(256);
    // halves per staged row: 32 columns and a pad (rows stay 16-byte aligned)
    const SA: usize = 40;
    let (ga, ba) = geom(ta);
    let (gb, bb) = tb.map(geom).unwrap_or((1, 1));

    let mut dq = String::new();
    let mut binds = format!("{ACT}\n");
    if experts {
        dq += &(deq4(ta, "dqa_v", "av") + &deq4(ta, "dqa_h", "ah"));
        binds += "@group(0) @binding(6) var<storage, read> av: array<u32>;\n@group(0) @binding(8) var<storage, read> ah: array<u32>;\n@group(0) @binding(10) var<storage, read> loc: array<u32>;\n";
        if let Some(tb) = tb {
            dq += &(deq4(tb, "dqb_v", "bv") + &deq4(tb, "dqb_h", "bh"));
            binds += "@group(0) @binding(7) var<storage, read> bv: array<u32>;\n@group(0) @binding(9) var<storage, read> bh: array<u32>;\n";
        }
    } else {
        dq += &deq4(ta, "dqa", "wq");
        binds += &format!("@group(0) @binding(3) var<storage, read> wq: array<u32>;\n{FW}\n");
    }

    // a staged token per item: its vector's offset, or none
    let xsrc = lines(xits, |it| {
        let src = if experts {
            "        let q = z0 + j;\n        if (q < u.z) { let pp = u.y + q; var r = pp; if (arg(11u) == 0u) { r = st[PL_PTOK + pp]; } xb = arg(2u) + r * arg(3u); ok = true; }\n"
        } else {
            "        let t = t0 + j;\n        if (t < T) { xb = arg(2u) + t * arg(3u); ok = true; }\n"
        };
        format!(
            "    var xok{it} = false;\n    var xb{it} = 0u;\n    let xq{it} = (li + {o}u) / {tt}u;\n    let xj{it} = (li + {o}u) % {tt}u;\n    var xv{it} = vec4<f32>(0.0);\n    {{\n        let j = xj{it};\n        var ok = false;\n        var xb = 0u;\n{src}        xok{it} = ok && li + {o}u < {n}u;\n        xb{it} = xb + 4u * xq{it};\n    }}\n",
            o = it * 256,
            n = tt * 8
        )
    });
    let xfetch = |k0: &str| lines(xits, |it| format!("        if (xok{it}) {{ let o = xb{it} + {k0}; xv{it} = vec4<f32>(act[o], act[o + 1u], act[o + 2u], act[o + 3u]); }}\n"));
    let xput = lines(xits, |it| {
        let guard = if (it + 1) * 256 > tt * 8 { format!("if (li + {}u < {}u) ", it * 256, tt * 8) } else { String::new() };
        let st: String = ["x", "y", "z", "w"].iter().enumerate().map(|(i, c)| format!(" Xs[xj{it} * {SA}u + xq{it} * 4u + {i}u] = f16(xv{it}.{c});")).collect();
        format!("        {guard}{{{st} }}\n")
    });
    // a staged weight row's 4 columns per item
    let wconst = lines(wits, |it| {
        let second = gu && it * 256 >= tr * 8;
        let rr = if second { format!("wr{it} - {tr}u") } else { format!("wr{it}") };
        let base = if !experts {
            format!("arg(6u) + min(r0 + {rr}, R - 1u) * arg(7u)")
        } else if second {
            format!("slot * arg(9u) + min(r0 + {rr}, R - 1u) * arg(10u)")
        } else {
            format!("slot * arg(7u) + min(r0 + {rr}, R - 1u) * arg(8u)")
        };
        format!("    let wr{it} = (li + {o}u) / 8u;\n    let wq{it} = (li + {o}u) % 8u;\n    let wb{it} = {base};\n    var wv{it} = vec4<f32>(0.0);\n", o = it * 256)
    });
    let wfetch = |k0: &str| {
        lines(wits, |it| {
            let second = gu && it * 256 >= tr * 8;
            let (n, g, b) = if second { ("b", gb, bb) } else { ("a", ga, ba) };
            let gx = format!("({k0}) / 4u + wq{it}");
            let call = |suf: &str| format!("dq{n}{suf}(wb{it} + ({gx}) / {g}u * {b}u, ({gx}) % {g}u)");
            let guard = if (it + 1) * 256 > wr * 8 { format!("if (li + {}u < {}u) ", it * 256, wr * 8) } else { String::new() };
            if experts {
                format!("        {guard}{{ if (host) {{ wv{it} = {}; }} else {{ wv{it} = {}; }} }}\n", call("_h"), call("_v"))
            } else {
                format!("        {guard}{{ wv{it} = {}; }}\n", call(""))
            }
        })
    };
    let wput = lines(wits, |it| {
        let guard = if (it + 1) * 256 > wr * 8 { format!("if (li + {}u < {}u) ", it * 256, wr * 8) } else { String::new() };
        let st: String = ["x", "y", "z", "w"].iter().enumerate().map(|(i, c)| format!(" Ws[wr{it} * {SA}u + wq{it} * 4u + {i}u] = f16(wv{it}.{c});")).collect();
        format!("        {guard}{{{st} }}\n")
    });

    let names: &[&str] = if gu { &["g", "u"] } else { &["s"] };
    // (the sums start from a zeroed scratch tile)
    let accs = lines(wm * wn, |k| names.iter().map(|n| format!("    var {n}{}_{} = coopLoad<coop_mat16x16<f32, C>>(&Cs[0], 16u);\n", k / wn, k % wn)).collect());
    let mmas = lines(2, |kk| {
        let mut s = String::new();
        for ni in 0..wn {
            s += &format!("            let b{ni} = coopLoad<coop_mat16x16<f16, B>>(&Xs[(cb + {}u) * {SA}u + {}u], {SA}u);\n", ni * 16, kk * 16);
        }
        for mi in 0..wm {
            s += &format!("            let a{mi} = coopLoadT<coop_mat16x16<f16, A>>(&Ws[(rb + {}u) * {SA}u + {}u], {SA}u);\n", mi * 16, kk * 16);
            if gu {
                s += &format!("            let c{mi} = coopLoadT<coop_mat16x16<f16, A>>(&Ws[(rb + {}u) * {SA}u + {}u], {SA}u);\n", tr + mi * 16, kk * 16);
            }
            for ni in 0..wn {
                if gu {
                    s += &format!("            g{mi}_{ni} = coopMultiplyAdd(a{mi}, b{ni}, g{mi}_{ni});\n            u{mi}_{ni} = coopMultiplyAdd(c{mi}, b{ni}, u{mi}_{ni});\n");
                } else {
                    s += &format!("            s{mi}_{ni} = coopMultiplyAdd(a{mi}, b{ni}, s{mi}_{ni});\n");
                }
            }
        }
        format!("        {{\n{s}        }}\n")
    });
    // each sum tile through the subgroup's scratch (column-major: element
    // (m, n) at n*16 + m), its lanes writing 8 elements each
    let epi = if !experts {
        "            let t = t0 + jj;\n            if (row < R && t < T) {\n                var v = Cs[cs + i];\n                if (arg(8u) != NONE) { v += fw[arg(8u) + row]; }\n                let o = arg(4u) + t * arg(5u) + row;\n                if (arg(9u) == 1u) { act[o] += v; } else { act[o] = v; }\n            }\n"
    } else if gu {
        "            if (row < R && z0 + jj < u.z) { act[arg(4u) + (u.y + z0 + jj) * R + row] = silu(Cs[cs + i]) * Cs[cs + 256u + i]; }\n"
    } else {
        "            if (row < R && z0 + jj < u.z) { act[arg(4u) + (u.y + z0 + jj) * R + row] = Cs[cs + i]; }\n"
    };
    let outs = lines(wm * wn, |k| {
        let (mi, ni) = (k / wn, k % wn);
        let st = if gu { format!("    coopStore(g{mi}_{ni}, &Cs[cs], 16u);\n    coopStore(u{mi}_{ni}, &Cs[cs + 256u], 16u);\n") } else { format!("    coopStore(s{mi}_{ni}, &Cs[cs], 16u);\n") };
        format!("{st}    workgroupBarrier();\n    for (var i = lane; i < 256u; i += 32u) {{\n        let row = r0 + rb + {}u + i % 16u;\n        let jj = cb + {}u + i / 16u;\n        {{\n{epi}        }}\n    }}\n    workgroupBarrier();\n", mi * 16, ni * 16)
    });
    // experts in host RAM: rows read contiguously by a subgroup each, as in
    // `expert` (the tiles' short scattered reads crawl over PCIe)
    let host_path = if experts {
        let fin: String = (0..8)
            .map(|j| {
                let val = if gu { format!("silu(subgroupAdd(aa{j})) * subgroupAdd(ab{j})") } else { format!("subgroupAdd(aa{j})") };
                format!("                {{ let v = {val}; if (lane == 0u && n > {j}u) {{ act[arg(4u) + (u.y + c + {j}u) * R + row] = v; }} }}
")
            })
            .collect();
        let xo: String = (0..8)
            .map(|j| format!("                var xo{j} = 0u;
                if (n > {j}u) {{ let pp = u.y + c + {j}u; var r = pp; if (arg(11u) == 0u) {{ r = st[PL_PTOK + pp]; }} xo{j} = arg(2u) + r * arg(3u); }}
"))
            .collect();
        let bdecl = if gu { "            let bb = slot * arg(9u) + row * arg(10u);
" } else { "" };
        format!(
            "    if (host) {{
        let G = KC / 4u;
        let zend = min(z0 + {tt}u, u.z);
        for (var rr = warp; rr < {tr}u; rr += 8u) {{
            let row = r0 + rr;
            if (row >= R) {{ break; }}
            let ab = slot * arg(7u) + row * arg(8u);
{bdecl}            for (var c = z0; c < zend; c += 8u) {{
                let n = min(8u, zend - c);
{za}{zb}{xo}{lp}{fin}            }}
        }}
        return;
    }}
",
            za = zeros("aa").replace("        var", "                var"),
            zb = if gu { zeros("ab").replace("        var", "                var") } else { String::new() },
            lp = expert_loop(gu, (ga, gb), (ba, bb), "h"),
        )
    } else {
        String::new()
    };
    let head = if experts {
        format!(
            "    let na = uload4(li, {nact}u).x;\n    if (wg.y >= na) {{ return; }}\n    if (li == 0u) {{\n        let e = st[PL_ACT + wg.y];\n        we = vec4<u32>(e, st[PL_OFF + wg.y], st[PL_CNT + wg.y], loc[arg(5u) * arg(6u) + e]);\n    }}\n    let u = workgroupUniformLoad(&we);\n    let z0 = wg.z * {tt}u;\n    if (z0 >= u.z) {{ return; }}\n    let host = (u.w >> 31u) != 0u;\n    let slot = u.w & 0x7fffffffu;\n",
            nact = PL_NACT
        )
    } else {
        format!("    let T = uload4(li, 0u).y;\n    let t0 = wg.y * {tt}u;\n    if (t0 >= T) {{ return; }}\n")
    };
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
{h}
{dq}
var<workgroup> we: vec4<u32>;
var<workgroup> Ws: array<f16, {wsz}>;
var<workgroup> Xs: array<f16, {xsz}>;
var<workgroup> Cs: array<f32, 4096>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
{head}    let R = arg(0u);
    let KC = arg(1u);
    let r0 = wg.x * {tr}u;
    let lane = li % 32u;
    let warp = li / 32u;
    let rb = (warp / {gc}u) * {rbs}u;
    let cb = (warp % {gc}u) * {cbs}u;
    let cs = warp * 512u;
{host_path}    Cs[li] = 0.0;
    workgroupBarrier();
{xsrc}{wconst}{accs}    workgroupBarrier();
{f0}    for (var k0 = 0u; k0 < KC; k0 += 32u) {{
{wput}{xput}        workgroupBarrier();
        if (k0 + 32u < KC) {{
{wf}{xf}        }}
{mmas}        workgroupBarrier();
    }}
{outs}}}
"#,
        h = header(&binds),
        wsz = wr * SA,
        xsz = tt * SA,
        rbs = 16 * wm,
        cbs = 16 * wn,
        f0 = wfetch("0u") + &xfetch("0u"),
        wf = wfetch("k0 + 32u"),
        xf = xfetch("k0 + 32u"),
    )
}

/// x[t] += sum over k of weight * the pair's expert output (fixed order),
/// plus a shared expert's output (times sigmoid of its gate logit, if any).
/// Grid (d/256, PT).
/// args: 0 d, 1 x, 2 x stride, 3 expert outputs (per pair), 4 weights, 5 k,
///       6 shared output (NONE), 7 shared gate logit (NONE)
pub fn combine() -> String {
    format!(
        r#"{h}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let c = gid.x;
    let d = arg(0u);
    if (c >= d) {{ return; }}
    let K = arg(5u);
    var s = 0.0;
    for (var k = 0u; k < K; k++) {{
        let p = st[PL_PPOS + t * K + k];
        s += act[arg(4u) + t * K + k] * act[arg(3u) + p * d + c];
    }}
    if (arg(6u) != NONE) {{
        var sv = act[arg(6u) + t * d + c];
        if (arg(7u) != NONE) {{ sv *= 1.0 / (1.0 + exp(-act[arg(7u) + t])); }}
        s += sv;
    }}
    act[arg(1u) + t * arg(2u) + c] += s;
}}
"#,
        h = header(ACT),
    )
}

/// out = silu(gate) * up, elementwise, for the pass's tokens (dense FFNs).
/// Grid (n/256, PT).  args: 0 n, 1 gate, 2 up, 3 out (all stride n)
pub fn swiglu() -> String {
    format!(
        r#"{h}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let t = wg.y;
    if (t >= st[1]) {{ return; }}
    let i = gid.x;
    let n = arg(0u);
    if (i >= n) {{ return; }}
    act[arg(3u) + t * n + i] = silu(act[arg(1u) + t * n + i]) * act[arg(2u) + t * n + i];
}}
"#,
        h = header(ACT),
    )
}

/// Argmax of the logits, part 1: workgroup w's chunk (ties: lowest index).
/// Grid (ARGG).  args: 0 n, 1 logits, 2 partials (act: value, index bits)
pub fn argmax1() -> String {
    format!(
        r#"{h}
var<workgroup> rv: array<f32, 8>;
var<workgroup> ri: array<u32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = arg(0u);
    let chunk = (n + {g}u - 1u) / {g}u;
    let lo = wg.x * chunk;
    let hi = min(lo + chunk, n);
    var bv = -3.0e38;
    var bi = 0xffffffffu;
    for (var i = lo + li; i < hi; i += 256u) {{
        let v = act[arg(1u) + i];
        if (v > bv) {{ bv = v; bi = i; }}
    }}
    let m = subgroupMax(bv);
    let mi = subgroupMin(select(0xffffffffu, bi, bv == m));
    if (li % 32u == 0u) {{ rv[li / 32u] = m; ri[li / 32u] = mi; }}
    workgroupBarrier();
    if (li == 0u) {{
        var v = rv[0];
        var i = ri[0];
        for (var j = 1u; j < 8u; j++) {{ if (rv[j] > v || (rv[j] == v && ri[j] < i)) {{ v = rv[j]; i = ri[j]; }} }}
        act[arg(2u) + 2u * wg.x] = v;
        act[arg(2u) + 2u * wg.x + 1u] = bitcast<f32>(i);
    }}
}}
"#,
        h = header(ACT),
        g = ARGG,
    )
}

/// Argmax part 2: the token, into the history after the pass's last
/// position; the next pass is that one token (decoding needs no CPU).
/// Grid (1).  args: 0 partials
pub fn argmax2() -> String {
    format!(
        r#"{h}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32) {{
    var bv = -3.0e38;
    var bi = 0xffffffffu;
    for (var j = li; j < {g}u; j += 32u) {{
        let v = act[arg(0u) + 2u * j];
        let i = bitcast<u32>(act[arg(0u) + 2u * j + 1u]);
        if (v > bv || (v == bv && i < bi)) {{ bv = v; bi = i; }}
    }}
    let m = subgroupMax(bv);
    let tok = subgroupMin(select(0xffffffffu, bi, bv == m));
    if (li == 0u) {{
        let pos = st[0] + st[1];
        st[HIST + pos] = tok;
        st[2] = tok;
        st[3] = pos;
        st[0] = pos;
        st[1] = 1u;
    }}
}}
"#,
        h = header(ACT),
        g = ARGG,
    )
}
