//! Kernels for prompt passes of up to PMAX tokens: tiled matrix products
//! (each workgroup stages 32 columns of a tile of weight rows and of
//! tokens in workgroup memory, and each thread keeps a small block of
//! row x token sums in registers), with the experts' tokens grouped by
//! expert, so a weight is read once per tile of tokens instead of once per
//! token.
//!
//! These passes only fill the KV cache: the prompt's last token still goes
//! through the batched kernels (`oss_batch`), which make the logits. Their
//! arithmetic is not the batched kernels' (the sums run in another order),
//! so a prompt's cache matches a token-at-a-time one only to rounding.
//!
//! Expert plan (`pl`): word 0 the tile count; PL_TILES + 3i: tile i's
//! expert, first list entry and pair count; PL_IDS + t*4 + k: token t's k-th
//! expert; PL_LIST..: the pairs (t*4 + k) of each expert in turn.

/// Tokens per prompt pass, at most.
pub const PMAX: usize = 512;
/// Expert pairs per tile of the expert kernels.
pub const MOE_TT: usize = 64;
pub const PL_TILES: usize = 64;
pub const PL_IDS: usize = 512;
pub const PL_LIST: usize = PL_IDS + 4 * PMAX;
pub const PL_WORDS: usize = PL_LIST + 4 * PMAX;

/// The tiles the expert kernels may need (each expert's pairs, MOE_TT at a time).
pub fn max_tiles(n_exp: usize) -> usize {
    4 * PMAX / MOE_TT + n_exp
}

/// Plan constants, for the kernels that use `pl`.
pub fn pcommon() -> String {
    format!("const PMAX: u32 = {PMAX}u;\nconst MOE_TT: u32 = {MOE_TT}u;\nconst PL_TILES: u32 = {PL_TILES}u;\nconst PL_IDS: u32 = {PL_IDS}u;\nconst PL_LIST: u32 = {PL_LIST}u;\n")
}

/// x[t] = embedding of the prompt token at position st[1] + t.
/// Grid (cols/256, PMAX).  args: 0 cols, 1 out, 2 history base
pub const EMBED_P: &str = r#"
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> wd: array<u32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.y;
    let c = gid.x;
    let cols = arg(0u);
    if (t >= ntok() || c >= cols) { return; }
    let tok = st[arg(2u) + st[1] + t];
    let qi = tok * cols + c;
    let v = f32(extractBits(i32(wq[qi / 4u]), (qi % 4u) * 8u, 8u));
    let si = tok * (cols / 32u) + c / 32u;
    act[arg(1u) + t * cols + c] = v * unpack2x16float(wd[si / 2u])[si % 2u];
}
"#;

/// xo[t] = x[t] / rms(x[t]) * w, after (if arg 5 is set) adding to x[t]
/// the previous layer's 4 expert outputs (already weighted).  Grid (PMAX).
/// args: 0 n, 1 x (act), 2 out (xo), 3 w (fw), 4 eps, 5 expert outputs (act, or !0)
pub const NORM_P: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
BCOMMON
var<workgroup> red8: array<f32, 8>;
fn ld4(i: u32) -> vec4<f32> { return vec4<f32>(act[i], act[i + 1u], act[i + 2u], act[i + 3u]); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    if (t >= uload(li, 3u)) { return; }
    let n = arg(0u);
    let xi = arg(1u) + t * n;
    let oi = arg(2u) + t * n;
    let pb = arg(5u);
    var s = 0.0;
    for (var i = li; i < n / 4u; i += 256u) {
        let c = xi + 4u * i;
        var v = ld4(c);
        if (pb != 0xffffffffu) {
            let q = pb + t * 4u * n + 4u * i;
            v += (ld4(q) + ld4(q + n)) + (ld4(q + 2u * n) + ld4(q + 3u * n));
            act[c] = v.x; act[c + 1u] = v.y; act[c + 2u] = v.z; act[c + 3u] = v.w;
        }
        s += dot(v, v);
    }
    s = subgroupAdd(s);
    if (li % 32u == 0u) { red8[li / 32u] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red8[j]; }
    let inv = 1.0 / sqrt(tot / f32(n) + argf(4u));
    for (var i = li; i < n / 4u; i += 256u) {
        for (var j = 0u; j < 4u; j++) {
            let c = 4u * i + j;
            xo[oi + c] = act[xi + c] * inv * fw[arg(3u) + c];
        }
    }
}
"#;

/// Token t's post-attention norm, router logits and top 4 experts (one
/// workgroup per token).  Grid (PMAX).
/// args: 0 n, 1 x (act), 2 normed out (xo), 3 norm w (fw), 4 eps, 5 router
///       w (fw), 6 router bias (fw), 7 logits (act, stride 32), 8 n_exp,
///       9 expert weights out (act, stride 4)
pub const ROUTE_P: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
@group(0) @binding(11) var<storage, read_write> pl: array<u32>;
BCOMMON
PCOMMON
TOPK
var<workgroup> red8: array<f32, 8>;
var<workgroup> xs: array<f32, 4096>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    if (t >= uload(li, 3u)) { return; }
    let n = arg(0u);
    let xi = arg(1u) + t * n;
    var ss = 0.0;
    for (var i = li; i < n; i += 256u) { let v = act[xi + i]; ss += v * v; }
    ss = subgroupAdd(ss);
    if (li % 32u == 0u) { red8[li / 32u] = ss; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red8[j]; }
    let inv = 1.0 / sqrt(tot / f32(n) + argf(4u));
    for (var i = li; i < n; i += 256u) {
        let v = act[xi + i] * inv * fw[arg(3u) + i];
        xs[i] = v;
        xo[arg(2u) + t * n + i] = v;
    }
    workgroupBarrier();
    let ne = arg(8u);
    let lane = li % 32u;
    let lg = arg(7u) + t * 32u;
    for (var q = 0u; q < 4u; q++) {
        let e = li / 32u + 8u * q;
        if (e < ne) {
            let wb = arg(5u) + e * n;
            var s = 0.0;
            for (var i = lane; i < n; i += 32u) { s += xs[i] * fw[wb + i]; }
            s = subgroupAdd(s);
            if (lane == 0u) { act[lg + e] = s + fw[arg(6u) + e]; }
        }
    }
    storageBarrier();
    workgroupBarrier();
    if (li < 32u) {
        let top = topk(li, lg, ne);
        if (li == 0u) {
            for (var k = 0u; k < 4u; k++) {
                pl[PL_IDS + t * 4u + k] = top.ids[k];
                act[arg(9u) + t * 4u + k] = top.w[k];
            }
        }
    }
}
"#;

/// Groups the pass's (token, expert) pairs by expert and cuts each
/// expert's list into tiles of MOE_TT.  Grid (1).  args: 0 n_exp
pub const PLAN_P: &str = r#"
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(11) var<storage, read_write> pl: array<u32>;
BCOMMON
PCOMMON
var<workgroup> ids: array<u32, 2048>;
var<workgroup> cnt: array<u32, 32>;
var<workgroup> off: array<u32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = uload(li, 3u) * 4u;
    let ne = arg(0u);
    for (var p = li; p < n; p += 256u) { ids[p] = pl[PL_IDS + p]; }
    workgroupBarrier();
    if (li < ne) {
        var c = 0u;
        for (var p = 0u; p < n; p++) { if (ids[p] == li) { c += 1u; } }
        cnt[li] = c;
    }
    workgroupBarrier();
    if (li == 0u) {
        var o = 0u;
        var nt = 0u;
        for (var e = 0u; e < ne; e++) {
            off[e] = o;
            for (var s = 0u; s < cnt[e]; s += MOE_TT) {
                pl[PL_TILES + 3u * nt] = e;
                pl[PL_TILES + 3u * nt + 1u] = o + s;
                pl[PL_TILES + 3u * nt + 2u] = min(MOE_TT, cnt[e] - s);
                nt += 1u;
            }
            o += cnt[e];
        }
        pl[0] = nt;
    }
    workgroupBarrier();
    if (li < ne) {
        var o = off[li];
        for (var p = 0u; p < n; p++) {
            if (ids[p] == li) { pl[PL_LIST + o] = p; o += 1u; }
        }
    }
}
"#;

/// What a tiled product multiplies and where its sums go.
#[derive(Clone, Copy, PartialEq)]
pub enum Mm {
    /// out[t][r] (=, or +=) Q8_0 row r . x[t] (+ bias).  Grid (rows/TR, PMAX/TT).
    /// args: 0 rows, 1 cols, 2 x (xi, in vec4s; token stride cols/4), 3 out
    ///       (act), 4 out stride, 5 bias (fw, or !0), 6 add
    Q8,
    /// An expert tile's gate and up rows for its pairs, then the clamped
    /// SwiGLU into xo[p*R + r] (p = t*4 + k).  Grid (R/TR, tiles).
    /// args: 0 R, 1 C, 2 x (xi, normed, in vec4s), 3 out (xo), 4 gate bias, 5 up bias
    GateUp,
    /// An expert tile's down rows for its pairs: out[p*R + r] = weight[p] *
    /// (sum + bias).  Grid (R/TR, tiles).
    /// args: 0 R, 1 C, 2 a (xi, in vec4s; pair stride C/4), 3 out (act),
    ///       4 down bias, 5 expert weights (act)
    Down,
}

fn lines(n: usize, f: impl Fn(usize) -> String) -> String {
    (0..n).map(f).collect()
}

/// A tiled product: a workgroup computes `tr` rows x `tt` tokens (or
/// pairs), each thread `rt` rows x `ct` tokens (rt a multiple of 4).
pub fn mm(kind: Mm, tr: usize, tt: usize, rt: usize, ct: usize) -> String {
    let (nx, ny) = (tt / ct, tr / rt);
    assert!(nx * ny == 256 && rt % 4 == 0 && [1, 2, 4].contains(&ct));
    let gu = kind == Mm::GateUp;
    let q8 = kind == Mm::Q8;
    // rows staged (gate and up: both), and words per row per 32 columns
    let wr = if gu { 2 * tr } else { tr };
    let wpr = if q8 { 8 } else { 4 };
    assert!((wr * wpr) % 256 == 0);
    let wits = wr * wpr / 256;
    let xits = (tt * 8).div_ceil(256);
    let xt = ["f32", "vec2<f32>", "", "vec4<f32>"][ct - 1];
    let (wr4, rt4) = (wr / 4, rt / 4);

    // where each thread's x loads come from (fixed over the columns)
    let xsrc = lines(xits, |it| {
        let src = match kind {
            Mm::Q8 => "        let t = t0 + j;\n        ok = t < T;\n        base = arg(2u) + t * c4;\n".to_string(),
            Mm::GateUp => "        ok = j < cnt;\n        if (ok) { base = arg(2u) + (pl[PL_LIST + start + j] / 4u) * c4; }\n".to_string(),
            Mm::Down => "        ok = j < cnt;\n        if (ok) { base = arg(2u) + pl[PL_LIST + start + j] * c4; }\n".to_string(),
        };
        format!("    var xok{it} = false;\n    var xb{it} = 0u;\n    {{\n        let j = (li + {}u) % {tt}u;\n        var ok = false;\n        var base = 0u;\n{src}        xok{it} = ok && li + {}u < {}u;\n        xb{it} = base;\n    }}\n", it * 256, it * 256, tt * 8)
    });
    let xstore = |k: &str, j: &str, v: &str| -> String {
        if ct == 1 {
            format!("Xs[({k}) * {tt}u + {j}] = {v};")
        } else {
            format!("Xs[(({k}) * {tt}u + {j}) / {ct}u][({j}) % {ct}u] = {v};")
        }
    };
    let xguard = |it: usize| if (it + 1) * 256 > tt * 8 { format!("if (li + {}u < {}u) ", it * 256, tt * 8) } else { String::new() };
    // per thread: which x row and columns it stages, and its register for them
    let xconst = lines(xits, |it| format!("    let xj{it} = (li + {o}u) % {tt}u;\n    let xq{it} = (li + {o}u) / {tt}u;\n    var xv{it} = vec4<f32>(0.0);\n", o = it * 256));
    let xfetch = |kn: &str| lines(xits, |it| format!("        if (xok{it}) {{ xv{it} = xi[xb{it} + ({kn}) / 4u + xq{it}]; }}\n"));
    let xput = lines(xits, |it| {
        let comps = ["x", "y", "z", "w"];
        let st: String = (0..4).map(|i| format!(" {}", xstore(&format!("xq{it} * 4u + {i}u"), &format!("xj{it}"), &format!("xv{it}.{}", comps[i])))).collect();
        format!("        {}{{{st} }}\n", xguard(it))
    });
    // a column's rows, padded by a vec4 so lanes storing different columns
    // of the same rows spread over the banks
    let wst = wr4 + 1;
    let wstore = |it: usize, k: &str, v: &str| format!("Ws[({k}) * {wst}u + wr{it} / 4u][wr{it} % 4u] = {v};");
    // per thread: which weight row and word it stages (fixed over the columns)
    let wconst = lines(wits, |it| {
        if q8 {
            format!("    let wr{it} = (li + {o}u) / 8u;\n    let ww{it} = (li + {o}u) % 8u;\n    let wrow{it} = min(r0 + wr{it}, R - 1u);\n    var wv{it}: u32;\n    var wd{it}: f32;\n", o = it * 256)
        } else {
            let rr = if gu && it * 64 >= tr { format!("wr{it} - {tr}u") } else { format!("wr{it}") };
            format!("    let wr{it} = (li + {o}u) / 4u;\n    let ww{it} = (li + {o}u) % 4u;\n    let wb{it} = (e * R + min(r0 + {rr}, R - 1u)) * nb;\n    var wv{it}: u32;\n    var wd{it}: f32;\n", o = it * 256)
        }
    });
    let wfetch = |kn: &str| {
        lines(wits, |it| {
            if q8 {
                format!("        {{\n            wv{it} = wq[(wrow{it} * C + ({kn})) / 4u + ww{it}];\n            let s = wrow{it} * nb + ({kn}) / 32u;\n            wd{it} = unpack2x16float(wd[s / 2u])[s % 2u];\n        }}\n")
            } else {
                // rows [it*64, it*64 + 64): for gate/up, the first tr are gate rows
                let (q, e) = if gu && it * 64 >= tr { ("uq", "ue") } else if gu { ("gq", "ge") } else { ("dq", "de") };
                format!("        {{\n            let i = wb{it} + ({kn}) / 32u;\n            wd{it} = e8(({e}[i / 4u] >> ((i % 4u) * 8u)) & 255u);\n            wv{it} = {q}[i * 4u + ww{it}];\n        }}\n")
            }
        })
    };
    let wput = lines(wits, |it| {
        if q8 {
            let st: String = (0..4).map(|i| format!("            {}\n", wstore(it, &format!("ww{it} * 4u + {i}u"), &format!("v[{i}]")))).collect();
            format!("        {{\n            let v = vec4<f32>(unpack4xI8(wv{it})) * wd{it};\n{st}        }}\n")
        } else {
            let st: String = (0..4).map(|i| format!("            {}\n            {}\n", wstore(it, &format!("ww{it} * 4u + {i}u"), &format!("lo[{i}]")), wstore(it, &format!("16u + ww{it} * 4u + {i}u"), &format!("hi[{i}]")))).collect();
            format!("        {{\n            let lo = lo4(wv{it}) * wd{it};\n            let hi = hi4(wv{it}) * wd{it};\n{st}        }}\n")
        }
    });

    // sums: s (or g and u) {row}_{token}
    let names: &[&str] = if gu { &["g", "u"] } else { &["s"] };
    let accs = lines(rt * ct, |k| names.iter().map(|n| format!("    var {n}{}_{} = 0.0;\n", k / ct, k % ct)).collect());
    let aload = |k: usize| {
        lines(rt4, |q| {
            let mut s = format!("            let a{q} = Ws[ab + {}u];\n", k * wst + q);
            if gu {
                s += &format!("            let c{q} = Ws[ab + {}u];\n", k * wst + tr / 4 + q);
            }
            s
        })
    };
    let bcomp = |j: usize| if ct == 1 { "b".to_string() } else { format!("b[{j}]") };
    let fmas = lines(rt * ct, |k| {
        let (i, j) = (k / ct, k % ct);
        let a = format!("a{}[{}]", i / 4, i % 4);
        let mut s = format!("            s{i}_{j} += {a} * {};\n", bcomp(j));
        if gu {
            s = format!("            g{i}_{j} += {a} * {b};\n            u{i}_{j} += c{}[{}] * {b};\n", i / 4, i % 4, b = bcomp(j));
        }
        s
    });
    let outs = lines(rt * ct, |k| {
        let (i, j) = (k / ct, k % ct);
        match kind {
            Mm::Q8 => format!(
                "    {{\n        let row = r0 + ty * {rt}u + {i}u;\n        let t = t0 + tx * {ct}u + {j}u;\n        if (row < R && t < T) {{\n            var v = s{i}_{j};\n            if (arg(5u) != 0xffffffffu) {{ v += fw[arg(5u) + row]; }}\n            let o = arg(3u) + t * arg(4u) + row;\n            if (arg(6u) == 1u) {{ act[o] += v; }} else {{ act[o] = v; }}\n        }}\n    }}\n"
            ),
            Mm::GateUp => format!(
                "    {{\n        let row = r0 + ty * {rt}u + {i}u;\n        let jj = tx * {ct}u + {j}u;\n        if (row < R && jj < cnt) {{\n            let p = pl[PL_LIST + start + jj];\n            let x = min(g{i}_{j} + fw[arg(4u) + e * R + row], 7.0);\n            let y = clamp(u{i}_{j} + fw[arg(5u) + e * R + row], -7.0, 7.0);\n            xo[arg(3u) + p * R + row] = x / (1.0 + exp(-1.702 * x)) * (y + 1.0);\n        }}\n    }}\n"
            ),
            Mm::Down => format!(
                "    {{\n        let row = r0 + ty * {rt}u + {i}u;\n        let jj = tx * {ct}u + {j}u;\n        if (row < R && jj < cnt) {{\n            let p = pl[PL_LIST + start + jj];\n            act[arg(3u) + p * R + row] = act[arg(5u) + p] * (s{i}_{j} + fw[arg(4u) + e * R + row]);\n        }}\n    }}\n"
            ),
        }
    });
    let binds = match kind {
        Mm::Q8 => "@group(0) @binding(1) var<storage, read> wq: array<u32>;\n@group(0) @binding(2) var<storage, read> wd: array<u32>;\n@group(0) @binding(3) var<storage, read_write> act: array<f32>;\n@group(0) @binding(4) var<storage, read_write> st: array<u32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n",
        Mm::GateUp => "@group(0) @binding(1) var<storage, read> gq: array<u32>;\n@group(0) @binding(2) var<storage, read> ge: array<u32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(6) var<storage, read> uq: array<u32>;\n@group(0) @binding(7) var<storage, read> ue: array<u32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n@group(0) @binding(10) var<storage, read_write> xo: array<f32>;\n@group(0) @binding(11) var<storage, read_write> pl: array<u32>;\n",
        Mm::Down => "@group(0) @binding(1) var<storage, read> dq: array<u32>;\n@group(0) @binding(2) var<storage, read> de: array<u32>;\n@group(0) @binding(3) var<storage, read_write> act: array<f32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n@group(0) @binding(11) var<storage, read_write> pl: array<u32>;\n",
    };
    // the tile: rows from wg.x; tokens from wg.y, or an expert tile
    let head = if q8 {
        format!("    let T = uload(li, 3u);\n    let t0 = wg.y * {tt}u;\n    if (t0 >= T) {{ return; }}\n")
    } else {
        "    if (wg.y >= ploadu(li, 0u)) { return; }\n    lut_init(li);\n    let tile = PL_TILES + 3u * wg.y;\n    let e = pl[tile];\n    let start = pl[tile + 1u];\n    let cnt = pl[tile + 2u];\n".to_string()
    };
    let pload = if q8 { "" } else { "var<workgroup> pu: u32;\nfn ploadu(li: u32, i: u32) -> u32 {\n    if (li == 0u) { pu = pl[i]; }\n    return workgroupUniformLoad(&pu);\n}\n" };
    format!(
        r#"
{binds}{pc}var<workgroup> Ws: array<vec4<f32>, {wsz}>;
var<workgroup> Xs: array<{xt}, {xsz}>;
{pload}@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
{head}    let R = arg(0u);
    let C = arg(1u);
    let nb = C / 32u;
    let c4 = C / 4u;
    let r0 = wg.x * {tr}u;
    let tx = li % {nx}u;
    let ty = li / {nx}u;
    let ab = ty * {rt4}u;
{xsrc}{xconst}{wconst}{accs}{f0}    for (var k0 = 0u; k0 < C; k0 += 32u) {{
{wput}{xput}        workgroupBarrier();
        // the next columns' loads, in flight while these are summed
        if (k0 + 32u < C) {{
{wf}{xf}        }}
{cols}        workgroupBarrier();
    }}
{outs}}}
"#,
        pc = if q8 { "BCOMMON\n" } else { "PCOMMON\n" },
        f0 = wfetch("0u") + &xfetch("0u"),
        wf = wfetch("k0 + 32u"),
        xf = xfetch("k0 + 32u"),
        wsz = 32 * wst,
        cols = lines(32, |k| format!("        {{\n{}            let b = Xs[{}u + tx];\n{fmas}        }}\n", aload(k), k * nx)),
        xsz = 32 * tt / ct,
    )
}

/// Attention for a prompt pass, flash style: a workgroup takes one kv head
/// and 64 queries (its hpk query heads for 64/hpk tokens), walks the keys
/// they see 32 at a time through workgroup memory (each key and value read
/// once for all 64), and keeps each query's running max, sum and output
/// (the sink counted in the sum).  Thread 4q + i holds query q's vector,
/// keys i*8.. of each block and output dims i*16...  Grid (n_kv, PMAX/(64/hpk)).
/// Keys of a sliding-window layer (arg 11 = !0) come from its ring; a full
/// layer's below arg 11 from VRAM, the rest from the far buffers (a staging
/// copy of the ones in host RAM).
/// args: 0 q, 1 out (xo), 2 kv base, 3 cap, 4 window (0 = all), 5 sinks (fw),
///       6 scale, 7 n_kv, 8 heads per kv head, 9 q stride, 10 out stride,
///       11 resident positions (!0: ring), 12 far rows
pub fn attn_p() -> String {
    let qload = lines(16, |d| format!("    let q{d} = vec4<f32>(act[qo + {o}u], act[qo + {o1}u], act[qo + {o2}u], act[qo + {o3}u]);\n", o = 4 * d, o1 = 4 * d + 1, o2 = 4 * d + 2, o3 = 4 * d + 3));
    let scores = lines(8, |k| {
        let dots: Vec<String> = (0..16).map(|d| format!("dot(q{d}, Ks[kb{k} + {d}u])")).collect();
        format!(
            "        let kb{k} = (part * 8u + {k}u) * 17u;\n        var s{k} = (({}) + ({})) * scale;\n        {{ let j = j0 + part * 8u + {k}u; if (j > pos || (win > 0u && j + win <= pos)) {{ s{k} = -3.0e38; }} }}\n",
            dots[..8].join(" + "),
            dots[8..].join(" + ")
        )
    });
    format!(
        r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(7) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> vc: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
@group(0) @binding(11) var<storage, read> fk: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> fv: array<vec4<f32>>;
BCOMMON
var<workgroup> Ks: array<vec4<f32>, 544>;
var<workgroup> Vs: array<vec4<f32>, 544>;
var<workgroup> Ps: array<f32, 2112>;
var<workgroup> wp: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let T = uload(li, 3u);
    if (li == 0u) {{ wp = st[1]; }}
    let pos0 = workgroupUniformLoad(&wp);
    let hpk = arg(8u);
    let bq = 64u / hpk;
    let t0 = wg.y * bq;
    if (t0 >= T) {{ return; }}
    let kvh = wg.x;
    let q = li / 4u;
    let part = li % 4u;
    let h = kvh * hpk + q % hpk;
    let tq = t0 + q / hpk;
    let t = min(tq, T - 1u);
    let pos = pos0 + t;
    let cap = arg(3u);
    let win = arg(4u);
    let res = arg(11u);
    let scale = argf(6u);
    let row4 = arg(7u) * 16u;
    let kv4 = arg(2u) / 4u + kvh * 16u;
    let qo = arg(0u) + t * arg(9u) + h * 64u;
{qload}    // the keys any of the 64 queries sees
    let first = pos0 + t0;
    let last = pos0 + min(t0 + bq, T) - 1u;
    var jlo = 0u;
    if (win > 0u && first + 1u > win) {{ jlo = first + 1u - win; }}
    var m = fw[arg(5u) + h];
    var l = 1.0;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    for (var j0 = jlo; j0 <= last; j0 += 32u) {{
        for (var i = li; i < 512u; i += 256u) {{
            let kl = i / 16u;
            let dv = i % 16u;
            let j = j0 + kl;
            var kk = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (res == 0xffffffffu) {{
                let src = kv4 + (j % cap) * row4 + dv;
                kk = kc[src];
                vv = vc[src];
            }} else if (j < res) {{
                let src = kv4 + j * row4 + dv;
                kk = kc[src];
                vv = vc[src];
            }} else if (j - res < arg(12u)) {{
                let src = (j - res) * row4 + kvh * 16u + dv;
                kk = fk[src];
                vv = fv[src];
            }}
            Ks[kl * 17u + dv] = kk;
            Vs[kl * 17u + dv] = vv;
        }}
        workgroupBarrier();
{scores}        var mt = max(max(max(s0, s1), max(s2, s3)), max(max(s4, s5), max(s6, s7)));
        mt = max(mt, subgroupShuffleXor(mt, 1u));
        mt = max(mt, subgroupShuffleXor(mt, 2u));
        let mn = max(m, mt);
        let sc = exp(m - mn);
        let p0 = exp(s0 - mn); let p1 = exp(s1 - mn); let p2 = exp(s2 - mn); let p3 = exp(s3 - mn);
        let p4 = exp(s4 - mn); let p5 = exp(s5 - mn); let p6 = exp(s6 - mn); let p7 = exp(s7 - mn);
        var ls = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
        ls += subgroupShuffleXor(ls, 1u);
        ls += subgroupShuffleXor(ls, 2u);
        l = l * sc + ls;
        m = mn;
        let pb = q * 33u + part * 8u;
        Ps[pb] = p0; Ps[pb + 1u] = p1; Ps[pb + 2u] = p2; Ps[pb + 3u] = p3;
        Ps[pb + 4u] = p4; Ps[pb + 5u] = p5; Ps[pb + 6u] = p6; Ps[pb + 7u] = p7;
        o0 *= sc; o1 *= sc; o2 *= sc; o3 *= sc;
        workgroupBarrier();
        for (var k = 0u; k < 32u; k++) {{
            let p = Ps[q * 33u + k];
            let vb = k * 17u + part * 4u;
            o0 += p * Vs[vb];
            o1 += p * Vs[vb + 1u];
            o2 += p * Vs[vb + 2u];
            o3 += p * Vs[vb + 3u];
        }}
        workgroupBarrier();
    }}
    if (tq < T) {{
        let ob = arg(1u) + tq * arg(10u) + h * 64u + part * 16u;
        let r = 1.0 / l;
        for (var i = 0u; i < 4u; i++) {{
            xo[ob + i] = o0[i] * r;
            xo[ob + 4u + i] = o1[i] * r;
            xo[ob + 8u + i] = o2[i] * r;
            xo[ob + 12u + i] = o3[i] * r;
        }}
    }}
}}
"#
    )
}

/// MXFP4 values (doubled e2m1, as MX's table holds them) of the low
/// nibbles of a word's 4 bytes, by arithmetic, and e8m0 scales (halved).
const MX4: &str = "fn e8(e: u32) -> f32 {
    if (e < 2u) { return bitcast<f32>(0x00200000u << e); }
    return bitcast<f32>((e - 1u) << 23u);
}
fn mx1(n: u32) -> f32 {
    let m = n & 7u;
    var v = f32(m);
    if (m >= 4u) { v = f32((2u + (m & 1u)) << ((m >> 1u) - 1u)); }
    return select(v, -v, (n & 8u) != 0u && m != 0u);
}
fn mx4(w: u32) -> vec4<f32> { return vec4<f32>(mx1(w), mx1(w >> 8u), mx1(w >> 16u), mx1(w >> 24u)); }
";

/// A tiled product on the tensor cores (`coop_mat16x16`): the workgroup
/// stages 32 columns of its weight rows and tokens as f16 in workgroup
/// memory, and subgroup (wy, wx) of its 8 keeps wm x wn 16x16 f32 sum tiles
/// (gate and up: both), at rows wy*16*wm.. and tokens wx*16*wn... A tile is
/// 8/gc*16*wm rows by gc*16*wn tokens. Same arguments and grid as `mm`.
pub fn mm_tc(kind: Mm, gc: usize, wm: usize, wn: usize) -> String {
    let gr = 8 / gc;
    let (tr, tt) = (gr * 16 * wm, gc * 16 * wn);
    let gu = kind == Mm::GateUp;
    let q8 = kind == Mm::Q8;
    let wr = if gu { 2 * tr } else { tr };
    let wpr = if q8 { 8 } else { 4 };
    assert!((wr * wpr) % 256 == 0 && gr * gc == 8);
    let wits = wr * wpr / 256;
    let xits = (tt * 8).div_ceil(256);
    // halves per staged row: 32 columns and a pad (rows stay 16-byte aligned)
    const SA: usize = 40;

    let xsrc = lines(xits, |it| {
        let src = match kind {
            Mm::Q8 => "        let t = t0 + j;\n        ok = t < T;\n        base = arg(2u) + t * c4;\n",
            Mm::GateUp => "        ok = j < cnt;\n        if (ok) { base = arg(2u) + (pl[PL_LIST + start + j] / 4u) * c4; }\n",
            Mm::Down => "        ok = j < cnt;\n        if (ok) { base = arg(2u) + pl[PL_LIST + start + j] * c4; }\n",
        };
        format!("    var xok{it} = false;\n    var xb{it} = 0u;\n    {{\n        let j = (li + {o}u) % {tt}u;\n        var ok = false;\n        var base = 0u;\n{src}        xok{it} = ok && li + {o}u < {n}u;\n        xb{it} = base;\n    }}\n", o = it * 256, n = tt * 8)
    });
    let xguard = |it: usize| if (it + 1) * 256 > tt * 8 { format!("if (li + {}u < {}u) ", it * 256, tt * 8) } else { String::new() };
    let xconst = lines(xits, |it| format!("    let xj{it} = (li + {o}u) % {tt}u;\n    let xq{it} = (li + {o}u) / {tt}u;\n    var xv{it} = vec4<f32>(0.0);\n", o = it * 256));
    let xfetch = |kn: &str| lines(xits, |it| format!("        if (xok{it}) {{ xv{it} = xi[xb{it} + ({kn}) / 4u + xq{it}]; }}\n"));
    let xput = lines(xits, |it| {
        let st: String = ["x", "y", "z", "w"].iter().enumerate().map(|(i, c)| format!(" Xs[xj{it} * {SA}u + xq{it} * 4u + {i}u] = f16(xv{it}.{c});")).collect();
        format!("        {}{{{st} }}\n", xguard(it))
    });
    let wconst = lines(wits, |it| {
        if q8 {
            format!("    let wr{it} = (li + {o}u) / 8u;\n    let ww{it} = (li + {o}u) % 8u;\n    let wrow{it} = min(r0 + wr{it}, R - 1u);\n    var wv{it}: u32;\n    var wd{it}: f32;\n", o = it * 256)
        } else {
            let rr = if gu && it * 64 >= tr { format!("wr{it} - {tr}u") } else { format!("wr{it}") };
            format!("    let wr{it} = (li + {o}u) / 4u;\n    let ww{it} = (li + {o}u) % 4u;\n    let wb{it} = (e * R + min(r0 + {rr}, R - 1u)) * nb;\n    var wv{it}: u32;\n    var wd{it}: f32;\n", o = it * 256)
        }
    });
    let wfetch = |kn: &str| {
        lines(wits, |it| {
            if q8 {
                format!("        {{\n            wv{it} = wq[(wrow{it} * KC + ({kn})) / 4u + ww{it}];\n            let s = wrow{it} * nb + ({kn}) / 32u;\n            wd{it} = unpack2x16float(wd[s / 2u])[s % 2u];\n        }}\n")
            } else {
                let (q, e) = if gu && it * 64 >= tr { ("uq", "ue") } else if gu { ("gq", "ge") } else { ("dq", "de") };
                format!("        {{\n            let i = wb{it} + ({kn}) / 32u;\n            wd{it} = e8(({e}[i / 4u] >> ((i % 4u) * 8u)) & 255u);\n            wv{it} = {q}[i * 4u + ww{it}];\n        }}\n")
            }
        })
    };
    let wput = lines(wits, |it| {
        if q8 {
            let st: String = (0..4).map(|i| format!(" Ws[wr{it} * {SA}u + ww{it} * 4u + {i}u] = f16(v[{i}]);")).collect();
            format!("        {{\n            let v = vec4<f32>(unpack4xI8(wv{it})) * wd{it};\n           {st}\n        }}\n")
        } else {
            // (a block's byte j holds columns j and j + 16)
            let st: String = (0..4).map(|i| format!(" Ws[wr{it} * {SA}u + ww{it} * 4u + {i}u] = f16(lo[{i}]); Ws[wr{it} * {SA}u + 16u + ww{it} * 4u + {i}u] = f16(hi[{i}]);")).collect();
            format!("        {{\n            let lo = mx4(wv{it}) * wd{it};\n            let hi = mx4(wv{it} >> 4u) * wd{it};\n           {st}\n        }}\n")
        }
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
    let epi = match kind {
        Mm::Q8 => "            let t = t0 + jj;\n            if (row < R && t < T) {\n                var v = Cs[cs + i];\n                if (arg(5u) != 0xffffffffu) { v += fw[arg(5u) + row]; }\n                let o = arg(3u) + t * arg(4u) + row;\n                if (arg(6u) == 1u) { act[o] += v; } else { act[o] = v; }\n            }\n",
        Mm::GateUp => "            if (row < R && jj < cnt) {\n                let p = pl[PL_LIST + start + jj];\n                let x = min(Cs[cs + i] + fw[arg(4u) + e * R + row], 7.0);\n                let y = clamp(Cs[cs + 256u + i] + fw[arg(5u) + e * R + row], -7.0, 7.0);\n                xo[arg(3u) + p * R + row] = x / (1.0 + exp(-1.702 * x)) * (y + 1.0);\n            }\n",
        Mm::Down => "            if (row < R && jj < cnt) {\n                let p = pl[PL_LIST + start + jj];\n                act[arg(3u) + p * R + row] = act[arg(5u) + p] * (Cs[cs + i] + fw[arg(4u) + e * R + row]);\n            }\n",
    };
    let outs = lines(wm * wn, |k| {
        let (mi, ni) = (k / wn, k % wn);
        let st = if gu { format!("    coopStore(g{mi}_{ni}, &Cs[cs], 16u);\n    coopStore(u{mi}_{ni}, &Cs[cs + 256u], 16u);\n") } else { format!("    coopStore(s{mi}_{ni}, &Cs[cs], 16u);\n") };
        format!("{st}    workgroupBarrier();\n    for (var i = lane; i < 256u; i += 32u) {{\n        let row = r0 + rb + {}u + i % 16u;\n        let jj = cb + {}u + i / 16u;\n        {{\n{epi}        }}\n    }}\n    workgroupBarrier();\n", mi * 16, ni * 16)
    });
    let binds = match kind {
        Mm::Q8 => "@group(0) @binding(1) var<storage, read> wq: array<u32>;\n@group(0) @binding(2) var<storage, read> wd: array<u32>;\n@group(0) @binding(3) var<storage, read_write> act: array<f32>;\n@group(0) @binding(4) var<storage, read_write> st: array<u32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n",
        Mm::GateUp => "@group(0) @binding(1) var<storage, read> gq: array<u32>;\n@group(0) @binding(2) var<storage, read> ge: array<u32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(6) var<storage, read> uq: array<u32>;\n@group(0) @binding(7) var<storage, read> ue: array<u32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n@group(0) @binding(10) var<storage, read_write> xo: array<f32>;\n@group(0) @binding(11) var<storage, read_write> pl: array<u32>;\n",
        Mm::Down => "@group(0) @binding(1) var<storage, read> dq: array<u32>;\n@group(0) @binding(2) var<storage, read> de: array<u32>;\n@group(0) @binding(3) var<storage, read_write> act: array<f32>;\n@group(0) @binding(5) var<storage, read> fw: array<f32>;\n@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;\n@group(0) @binding(11) var<storage, read_write> pl: array<u32>;\n",
    };
    let head = if q8 {
        format!("    let T = uload(li, 3u);\n    let t0 = wg.y * {tt}u;\n    if (t0 >= T) {{ return; }}\n")
    } else {
        "    if (wg.y >= ploadu(li, 0u)) { return; }\n    let tile = PL_TILES + 3u * wg.y;\n    let e = pl[tile];\n    let start = pl[tile + 1u];\n    let cnt = pl[tile + 2u];\n".to_string()
    };
    // (16 bytes: a 4-byte workgroup value can leave the f16 tiles off the
    // 16-byte alignment the cooperative loads assume, and they then read
    // shifted rows)
    let pload = if q8 { "" } else { "var<workgroup> pu: vec4<u32>;\nfn ploadu(li: u32, i: u32) -> u32 {\n    if (li == 0u) { pu.x = pl[i]; }\n    return workgroupUniformLoad(&pu).x;\n}\n" };
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
{binds}{mx4}{pc}var<workgroup> Ws: array<f16, {wsz}>;
var<workgroup> Xs: array<f16, {xsz}>;
var<workgroup> Cs: array<f32, 4096>;
{pload}@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
{head}    let R = arg(0u);
    let KC = arg(1u);
    let nb = KC / 32u;
    let c4 = KC / 4u;
    let r0 = wg.x * {tr}u;
    let lane = li % 32u;
    let warp = li / 32u;
    let rb = (warp / {gc}u) * {rbs}u;
    let cb = (warp % {gc}u) * {cbs}u;
    let cs = warp * 512u;
    Cs[li] = 0.0;
    workgroupBarrier();
{xsrc}{xconst}{wconst}{accs}    workgroupBarrier();
{f0}    for (var k0 = 0u; k0 < KC; k0 += 32u) {{
{wput}{xput}        workgroupBarrier();
        if (k0 + 32u < KC) {{
{wf}{xf}        }}
{mmas}        workgroupBarrier();
    }}
{outs}}}
"#,
        pc = if q8 { "BCOMMON\n" } else { "PCOMMON\n" },
        mx4 = if q8 { "" } else { MX4 },
        wsz = wr * SA,
        xsz = tt * SA,
        rbs = 16 * wm,
        cbs = 16 * wn,
        f0 = wfetch("0u") + &xfetch("0u"),
        wf = wfetch("k0 + 32u"),
        xf = xfetch("k0 + 32u"),
    )
}
