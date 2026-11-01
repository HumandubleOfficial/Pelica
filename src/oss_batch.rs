//! Batched WGSL kernels for the GPU-resident engine: the same forward pass
//! for up to TMAX tokens at once (prompt chunks, and checking speculative
//! drafts). Weights are read once per pass for all the tokens, and each
//! token's arithmetic is the same whatever the batch size, so a token comes
//! out the same whether it was checked alone or alongside a draft.
//!
//! The pass's token count, positions and token ids live in the state buffer
//! (`st`), so every dispatch's parameters stay fixed. Grids are sized for
//! TMAX; workgroups for tokens past the count return at once.
//!
//! State words: 1 position of the pass's first token, 3 token count, 4
//! tokens accepted by the last pass, 5 mode (1: the pass ends the prompt),
//! 6 longest draft, 7 longest draft on offer, 32.. the pass's tokens, 40..
//! each token's argmax, 48.. the expert plan; 8 experts used by the pass
//! (summed over layers), 9 its tokens, 10 the length of the match behind
//! the next draft; the token history after that.

/// Tokens per batched pass, at most.
pub const TMAX: usize = 8;

/// Shared by all batched kernels (after COMMON).
pub const BCOMMON: &str = r#"
const TMAX: u32 = 8u;
fn ntok() -> u32 { return st[3]; }
// st[i], as a value every invocation agrees on (so barriers may follow a
// branch on it); one call per kernel (16 bytes, so f16 tiles declared
// next to it keep the alignment cooperative matrix loads need)
var<workgroup> wu: vec4<u32>;
fn uload(li: u32, i: u32) -> u32 {
    if (li == 0u) { wu.x = st[i]; }
    return workgroupUniformLoad(&wu).x;
}
"#;

/// x[t] = embedding of the pass's token t.  Grid (cols/256, TMAX).
/// args: 0 cols, 1 out
pub const EMBED_T: &str = r#"
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
    // per-pass counters for the CPU: experts used (summed over layers), tokens
    if (c == 0u && t == 0u) { st[8] = 0u; st[9] = ntok(); }
    if (t >= ntok() || c >= cols) { return; }
    let tok = st[32u + t];
    let qi = tok * cols + c;
    let v = f32(extractBits(i32(wq[qi / 4u]), (qi % 4u) * 8u, 8u));
    let si = tok * (cols / 32u) + c / 32u;
    act[arg(1u) + t * cols + c] = v * unpack2x16float(wd[si / 2u])[si % 2u];
}
"#;

/// xo[t] = x[t] / rms(x[t]) * w, after (if arg 5 is set) adding to x[t]
/// the previous layer's experts: the sum over token t's 4 of weight *
/// (down output + bias).  Grid (TMAX).
/// args: 0 n, 1 x (act), 2 out (xo), 3 w (fw), 4 eps, 5 down outputs (act,
///       or !0), 6 down bias (fw), 7 expert weights (act)
pub const NORM_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
BCOMMON
var<workgroup> red8: array<f32, 8>;
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
        var v = vec4<f32>(act[c], act[c + 1u], act[c + 2u], act[c + 3u]);
        if (pb != 0xffffffffu) {
            for (var j = 0u; j < 4u; j++) {
                let col = 4u * i + j;
                var m = 0.0;
                for (var k = 0u; k < 4u; k++) {
                    let e = st[113u + t * 4u + k];
                    m += act[arg(7u) + t * 4u + k] * (act[pb + (t * 4u + k) * n + col] + fw[arg(6u) + e * n + col]);
                }
                v[j] += m;
                act[c + j] = v[j];
            }
        }
        s += dot(v, v);
    }
    s = subgroupAdd(s);
    if (li % 32u == 0u) { red8[li / 32u] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var j = 0u; j < 8u; j++) { tot += red8[j]; }
    let inv = 1.0 / sqrt(tot / f32(n) + argf(4u));
    // (each thread re-reads only the values it wrote)
    for (var i = li; i < n / 4u; i += 256u) {
        for (var j = 0u; j < 4u; j++) {
            let c = 4u * i + j;
            xo[oi + c] = act[xi + c] * inv * fw[arg(3u) + c];
        }
    }
}
"#;

/// RoPE on q and k of every token, and k, v into the cache slots: a
/// sliding-window layer's ring (arg 6 = !0), or a full layer's position,
/// in VRAM below arg 6 and in the far (host RAM) buffers from there on.
/// Grid (threads/256, TMAX).
/// args: 0 q (stride arg 5), 1 n_head, 2 n_kv, 3 kv base, 4 cap, 5 q stride,
///       6 positions resident in VRAM (full layers; !0: ring)
pub const ROPE_KV_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(6) var<storage, read> rope: array<f32>;
@group(0) @binding(7) var<storage, read_write> kc: array<f32>;
@group(0) @binding(8) var<storage, read_write> vc: array<f32>;
@group(0) @binding(11) var<storage, read_write> fk: array<f32>;
@group(0) @binding(12) var<storage, read_write> fv: array<f32>;
BCOMMON
fn put_k(far: bool, i: u32, v: f32) { if (far) { fk[i] = v; } else { kc[i] = v; } }
fn put_v(far: bool, i: u32, v: f32) { if (far) { fv[i] = v; } else { vc[i] = v; } }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.y;
    if (t >= ntok()) { return; }
    let i = gid.x;
    let pos = st[1] + t;
    let qo = arg(0u) + t * arg(5u);
    let nq = arg(1u) * 32u;
    let nkv = arg(2u);
    let nk = nkv * 32u;
    let res = arg(6u);
    let far = res != 0xffffffffu && pos >= res;
    var kvb: u32;
    if (res == 0xffffffffu) { kvb = arg(3u) + (pos % arg(4u)) * nkv * 64u; }
    else if (far) { kvb = (pos - res) * nkv * 64u; }
    else { kvb = arg(3u) + pos * nkv * 64u; }
    if (i < nq + nk) {
        var h: u32;
        var j: u32;
        var base: u32;
        if (i < nq) { h = i / 32u; j = i % 32u; base = qo + h * 64u; }
        else { h = (i - nq) / 32u; j = (i - nq) % 32u; base = qo + arg(1u) * 64u + h * 64u; }
        let cs = rope[pos * 64u + 2u * j];
        let sn = rope[pos * 64u + 2u * j + 1u];
        let x0 = act[base + j];
        let x1 = act[base + j + 32u];
        let y0 = x0 * cs - x1 * sn;
        let y1 = x0 * sn + x1 * cs;
        if (i < nq) {
            act[base + j] = y0;
            act[base + j + 32u] = y1;
        } else {
            put_k(far, kvb + h * 64u + j, y0);
            put_k(far, kvb + h * 64u + j + 32u, y1);
        }
    } else if (i < nq + nk + nkv * 64u) {
        let v = i - nq - nk;
        put_v(far, kvb + v, act[qo + arg(1u) * 64u + nkv * 64u + v]);
    }
}
"#;

/// Sliding-window attention of the pass's token t (at position pos + t, so
/// it sees the tokens before it in the pass too), one workgroup per (head,
/// token): scores in workgroup memory, softmax with the head's sink.
/// args: 0 q, 1 out (xo), 2 kv base, 3 cap, 4 window (0 = all), 5 sinks (fw),
///       6 scale, 7 n_kv, 8 heads per kv head, 9 q stride, 10 out stride
pub const ATTN_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(7) var<storage, read_write> kc: array<f32>;
@group(0) @binding(8) var<storage, read_write> vc: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
BCOMMON
var<workgroup> sc: array<f32, MAXN>;
var<workgroup> qv: array<vec4<f32>, 16>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    if (t >= uload(li, 3u)) { return; }
    let h = wg.x;
    let nkv = arg(7u);
    let kvh = h / arg(8u);
    let pos = st[1] + t;
    let cap = arg(3u);
    let win = arg(4u);
    var lo = 0u;
    if (win > 0u && pos + 1u > win) { lo = pos + 1u - win; }
    let n = pos + 1u - lo;
    let qo = arg(0u) + t * arg(9u) + h * 64u;
    let row = nkv * 64u;
    let kvo = arg(2u) + kvh * 64u;
    if (li < 16u) {
        qv[li] = vec4<f32>(act[qo + 4u * li], act[qo + 4u * li + 1u], act[qo + 4u * li + 2u], act[qo + 4u * li + 3u]);
    }
    workgroupBarrier();
    let scale = argf(6u);
    var mx = -3.0e38;
    for (var j = li; j < n; j += 256u) {
        let kb = kvo + ((lo + j) % cap) * row;
        var s = 0.0;
        for (var d = 0u; d < 16u; d++) {
            let k = vec4<f32>(kc[kb + 4u * d], kc[kb + 4u * d + 1u], kc[kb + 4u * d + 2u], kc[kb + 4u * d + 3u]);
            s += dot(qv[d], k);
        }
        s *= scale;
        sc[j] = s;
        mx = max(mx, s);
    }
    mx = subgroupMax(mx);
    if (li % 32u == 0u) { red[li / 32u] = mx; }
    workgroupBarrier();
    var m = fw[arg(5u) + h];
    for (var i = 0u; i < 8u; i++) { m = max(m, red[i]); }
    workgroupBarrier();
    var sum = 0.0;
    for (var j = li; j < n; j += 256u) {
        let e = exp(sc[j] - m);
        sc[j] = e;
        sum += e;
    }
    sum = subgroupAdd(sum);
    if (li % 32u == 0u) { red[8u + li / 32u] = sum; }
    workgroupBarrier();
    var total = exp(fw[arg(5u) + h] - m);
    for (var i = 0u; i < 8u; i++) { total += red[8u + i]; }
    let d = li % 64u;
    let g = li / 64u;
    var o = 0.0;
    for (var j = g; j < n; j += 4u) {
        o += sc[j] * vc[kvo + ((lo + j) % cap) * row + d];
    }
    workgroupBarrier();
    red[li] = o;
    workgroupBarrier();
    if (li < 64u) {
        xo[arg(1u) + t * arg(10u) + h * 64u + li] = (red[li] + red[li + 64u] + red[li + 128u] + red[li + 192u]) / total;
    }
}
"#;

/// Values a partial attention result takes in ATTN_S's output: max, sum,
/// 64 weighted sums of v.
pub const PART: usize = 66;

/// Full-layer attention of a pass's tokens, split over the keys: workgroup
/// (kv head, s) takes keys s*S.. for the kv head's query heads and all the
/// pass's tokens (64 queries: 64/hpk tokens x hpk heads; queries past the
/// token count repeat the last token), and writes each query's partial
/// softmax (max, sum, sums of v) for ATTN_M. Keys below arg 3 are in VRAM
/// (kc, vc), the rest in the far buffers (host RAM, read in place). Each
/// key block of 32 is read once for all 64 queries. A key split, its blocks
/// and each query's arithmetic depend only on key positions, so a token's
/// result is the same whatever the pass's size. Thread 4q + i holds query
/// q's vector, keys i*8.. of each block and output dims i*16...
/// Grid (n_kv, splits).
/// args: 0 q, 1 splits (partials per query), 2 kv base, 3 resident, 4 split
///       size, 5 n_head, 6 scale, 7 n_kv, 8 heads per kv head, 9 q stride,
///       10 far rows
pub fn attn_s() -> String {
    let lines = |n: usize, f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<String>();
    let qload = lines(16, &|d| format!("    let q{d} = vec4<f32>(act[qo + {o}u], act[qo + {o1}u], act[qo + {o2}u], act[qo + {o3}u]);\n", o = 4 * d, o1 = 4 * d + 1, o2 = 4 * d + 2, o3 = 4 * d + 3));
    let scores = lines(8, &|k| {
        let dots: Vec<String> = (0..16).map(|d| format!("dot(q{d}, Ks[kb{k} + {d}u])")).collect();
        format!(
            "        let kb{k} = (part * 8u + {k}u) * 17u;\n        var s{k} = (({}) + ({})) * scale;\n        if (j0 + part * 8u + {k}u > pos) {{ s{k} = -3.0e38; }}\n",
            dots[..8].join(" + "),
            dots[8..].join(" + ")
        )
    });
    format!(
        r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(7) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> vc: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> fk: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> fv: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> pt: array<f32>;
BCOMMON
var<workgroup> Ks: array<vec4<f32>, 544>;
var<workgroup> Vs: array<vec4<f32>, 544>;
var<workgroup> Ps: array<f32, 2112>;
var<workgroup> wp: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let T = uload(li, 3u);
    if (li == 0u) {{ wp.x = st[1]; }}
    let pos0 = workgroupUniformLoad(&wp).x;
    let hpk = arg(8u);
    let kvh = wg.x;
    let sp = wg.y;
    let S = arg(4u);
    let jstart = sp * S;
    let last = pos0 + T - 1u;
    if (jstart > last) {{ return; }}
    let jend = min(jstart + S, last + 1u);
    let q = li / 4u;
    let part = li % 4u;
    let h = kvh * hpk + q % hpk;
    let tq = q / hpk;
    let t = min(tq, T - 1u);
    let pos = pos0 + t;
    let res = arg(3u);
    let scale = argf(6u);
    let row4 = arg(7u) * 16u;
    let kv4 = arg(2u) / 4u + kvh * 16u;
    let qo = arg(0u) + t * arg(9u) + h * 64u;
{qload}    var m = -3.0e38;
    var l = 0.0;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    for (var j0 = jstart; j0 < jend; j0 += 32u) {{
        for (var i = li; i < 512u; i += 256u) {{
            let kl = i / 16u;
            let dv = i % 16u;
            let j = j0 + kl;
            var kk = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (j < res) {{
                let src = kv4 + j * row4 + dv;
                kk = kc[src];
                vv = vc[src];
            }} else if (j - res < arg(10u)) {{
                let src = (j - res) * row4 + kvh * 16u + dv;
                kk = fk[src];
                vv = fv[src];
            }}
            Ks[kl * 17u + dv] = kk;
            Vs[kl * 17u + dv] = vv;
        }}
        workgroupBarrier();
        // (a subgroup holds one token's queries: past the token count it
        // only helps load)
        if (tq < T) {{
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
        }}
        workgroupBarrier();
        if (tq < T) {{
        for (var k = 0u; k < 32u; k++) {{
            let p = Ps[q * 33u + k];
            let vb = k * 17u + part * 4u;
            o0 += p * Vs[vb];
            o1 += p * Vs[vb + 1u];
            o2 += p * Vs[vb + 2u];
            o3 += p * Vs[vb + 3u];
        }}
        }}
        workgroupBarrier();
    }}
    if (tq < T) {{
        let pb = ((tq * arg(5u) + h) * arg(1u) + sp) * {PART}u;
        if (part == 0u) {{ pt[pb] = m; pt[pb + 1u] = l; }}
        let ob = pb + 2u + part * 16u;
        for (var i = 0u; i < 4u; i++) {{
            pt[ob + i] = o0[i];
            pt[ob + 4u + i] = o1[i];
            pt[ob + 8u + i] = o2[i];
            pt[ob + 12u + i] = o3[i];
        }}
    }}
}}
"#
    )
}

/// Token t's head h from ATTN_S's partials: the splits holding its keys
/// (0..=pos/S), combined in order with the head's sink.  Grid (n_head, TMAX).
/// args: 0 out (xo), 1 splits, 2 split size, 3 sinks (fw), 4 out stride,
///       5 n_head
pub const ATTN_M: &str = r#"
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
@group(0) @binding(13) var<storage, read_write> pt: array<f32>;
BCOMMON
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    if (t >= ntok()) { return; }
    let h = wg.x;
    let ns = (st[1] + t) / arg(2u) + 1u;
    let base = (t * arg(5u) + h) * arg(1u) * 66u;
    let sink = fw[arg(3u) + h];
    var mx = sink;
    for (var s = 0u; s < ns; s++) { mx = max(mx, pt[base + s * 66u]); }
    var l = exp(sink - mx);
    var o = 0.0;
    for (var s = 0u; s < ns; s++) {
        let b = base + s * 66u;
        let w = exp(pt[b] - mx);
        l += pt[b + 1u] * w;
        o += pt[b + 2u + li] * w;
    }
    xo[arg(0u) + t * arg(4u) + h * 64u + li] = o / l;
}
"#;

/// The post-attention norm and router logits: workgroup e computes expert
/// e's logit for every token, norming the tokens itself (with NORM_T's
/// arithmetic; workgroup 0 also stores the normed x for the experts).
/// args: 0 n, 1 x (act), 2 normed out (xo), 3 norm w (fw), 4 eps, 5 router
///       w (fw), 6 router bias (fw), 7 logits (act, stride 32)
pub const ROUTE_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
BCOMMON
var<workgroup> red: array<f32, 8>;
var<workgroup> red2: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let e = wg.x;
    let T = uload(li, 3u);
    let n = arg(0u);
    let n4 = n / 4u;
    let wb = arg(5u) + e * n;
    for (var t = 0u; t < T; t++) {
        let xi = arg(1u) + t * n;
        var ss = 0.0;
        for (var i = li; i < n4; i += 256u) {
            let c = xi + 4u * i;
            let v = vec4<f32>(act[c], act[c + 1u], act[c + 2u], act[c + 3u]);
            ss += dot(v, v);
        }
        ss = subgroupAdd(ss);
        if (li % 32u == 0u) { red[li / 32u] = ss; }
        workgroupBarrier();
        var tot = 0.0;
        for (var j = 0u; j < 8u; j++) { tot += red[j]; }
        let inv = 1.0 / sqrt(tot / f32(n) + argf(4u));
        var s = 0.0;
        for (var i = li; i < n4; i += 256u) {
            var x: vec4<f32>;
            for (var j = 0u; j < 4u; j++) {
                let c = 4u * i + j;
                x[j] = act[xi + c] * inv * fw[arg(3u) + c];
                if (e == 0u) { xo[arg(2u) + t * n + c] = x[j]; }
            }
            let w = wb + 4u * i;
            s += dot(x, vec4<f32>(fw[w], fw[w + 1u], fw[w + 2u], fw[w + 3u]));
        }
        s = subgroupAdd(s);
        if (li % 32u == 0u) { red2[li / 32u] = s; }
        workgroupBarrier();
        if (li == 0u) {
            var v = 0.0;
            for (var j = 0u; j < 8u; j++) { v += red2[j]; }
            act[arg(7u) + t * 32u + e] = v + fw[arg(6u) + e];
        }
        workgroupBarrier();
    }
}
"#;

/// Each token's top 4 experts and weights (subgroup t does token t), and
/// the union of experts the pass needs, each with the tokens routed to it.
/// args: 0 logits (stride 32), 1 n_exp, 2 weights out (act, stride 4)
/// st[48]: union size; 49 + u: expert; 81 + u: token mask; 113 + t*4 + k: chosen ids
pub const PLAN_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
TOPK
var<workgroup> ids: array<u32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let T = ntok();
    let t = li / 32u;
    let top = topk(li, arg(0u) + t * 32u, arg(1u));
    if (li % 32u == 0u && t < T) {
        for (var k = 0u; k < 4u; k++) {
            ids[t * 4u + k] = top.ids[k];
            st[113u + t * 4u + k] = top.ids[k];
            act[arg(2u) + t * 4u + k] = top.w[k];
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        var masks: array<u32, 32>;
        for (var e = 0u; e < 32u; e++) { masks[e] = 0u; }
        for (var i = 0u; i < T * 4u; i++) { masks[ids[i]] |= 1u << (i / 4u); }
        var count = 0u;
        for (var e = 0u; e < 32u; e++) {
            if (masks[e] != 0u) {
                st[49u + count] = e;
                st[81u + count] = masks[e];
                count += 1u;
            }
        }
        st[48] = count;
        st[8] += count;
    }
}
"#;

/// Shared by the batched expert kernels: union entry u's expert and the
/// tokens routed to it (with which of their 4 slots it is).
pub const ROUTED: &str = r#"
struct Routed { e: u32, m: u32, toks: array<u32, 8>, ks: array<u32, 8> }
fn routed(u: u32) -> Routed {
    var r: Routed;
    r.e = st[49u + u];
    let mask = st[81u + u];
    r.m = 0u;
    for (var t = 0u; t < TMAX; t++) {
        if (((mask >> t) & 1u) != 0u) {
            var k = 0u;
            for (var j = 0u; j < 4u; j++) { if (st[113u + t * 4u + j] == r.e) { k = j; } }
            r.toks[r.m] = t;
            r.ks[r.m] = k;
            r.m += 1u;
        }
    }
    return r;
}
"#;

/// Per-workgroup argmax of each token's logits.  Grid (groups, TMAX).
/// args: 0 n, 1 logits (stride n), 2 partials (stride 512), 3 chunk
pub const ARGMAX1_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    if (t >= uload(li, 3u)) { return; }
    let n = arg(0u);
    let lb = arg(1u) + t * n;
    let start = wg.x * arg(3u);
    let end = min(start + arg(3u), n);
    var best = -3.0e38;
    var bi = 0xffffffffu;
    for (var i = start + li; i < end; i += 256u) {
        let v = act[lb + i];
        if (v > best) { best = v; bi = i; }
    }
    rv[li] = best;
    ri[li] = bi;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (li < k) {
            let v = rv[li + k];
            let j = ri[li + k];
            if (v > rv[li] || (v == rv[li] && j < ri[li])) { rv[li] = v; ri[li] = j; }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let pb = arg(2u) + t * 512u;
        act[pb + 2u * wg.x] = rv[0];
        act[pb + 2u * wg.x + 1u] = bitcast<f32>(ri[0]);
    }
}
"#;

/// Each token's argmax into st[40 + t].  Grid (TMAX).  args: 0 partials, 1 count
pub const ARGMAX2_T: &str = r#"
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    if (t >= uload(li, 3u)) { return; }
    let pb = arg(0u) + t * 512u;
    var best = -3.0e38;
    var bi = 0xffffffffu;
    if (li < arg(1u)) { best = act[pb + 2u * li]; bi = bitcast<u32>(act[pb + 2u * li + 1u]); }
    rv[li] = best;
    ri[li] = bi;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (li < k) {
            let v = rv[li + k];
            let j = ri[li + k];
            if (v > rv[li] || (v == rv[li] && j < ri[li])) { rv[li] = v; ri[li] = j; }
        }
        workgroupBarrier();
    }
    if (li == 0u) { st[40u + t] = ri[0]; }
}
"#;

/// After a pass: keep the drafted tokens the model agrees with, plus the
/// model's own next token, append them to the history and move on.
/// (Mode 1, the prompt's last chunk: only the next token.)  args: 0 history base
pub const ACCEPT: &str = r#"
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
@compute @workgroup_size(1)
fn main() {
    let T = ntok();
    if (T == 0u) { st[4] = 0u; return; }
    let pos = st[1];
    let h = arg(0u);
    if (st[5] == 1u) {
        let tok = st[40u + T - 1u];
        st[h + pos + T] = tok;
        st[40] = tok;
        st[4] = 1u;
        st[1] = pos + T;
        st[32] = tok;
        st[5] = 0u;
        return;
    }
    // token t + 1 of the pass was drafted; the pass's answer for it is out[t]
    var a = 0u;
    while (a + 1u < T && st[32u + a + 1u] == st[40u + a]) { a += 1u; }
    for (var i = 0u; i <= a; i++) { st[h + pos + 1u + i] = st[40u + i]; }
    st[4] = a + 1u;
    st[1] = pos + a + 1u;
    st[32] = st[40u + a];
}
"#;

/// Prompt-lookup draft: find the latest earlier place where the history
/// ends the same way it does now (the longest match of nmin..nmax tokens)
/// and propose the tokens that followed it, up to st[6] of them (st[7]:
/// how many were on offer, st[10]: the match's length).
/// args: 0 history base, 1 nmax, 2 nmin, 3 ctx
pub const DRAFT: &str = r#"
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
BCOMMON
var<workgroup> best: array<atomic<u32>, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let h = arg(0u);
    let nmax = arg(1u);
    let nmin = arg(2u);
    let pos = st[1];
    if (li < 8u) { atomicStore(&best[li], 0u); }
    workgroupBarrier();
    // candidate j: the end of an earlier match, j < pos
    for (var j = li; j < pos; j += 256u) {
        var n = 0u;
        while (n < nmax && n <= j && st[h + j - n] == st[h + pos - n]) { n += 1u; }
        if (n >= nmin) { atomicMax(&best[n - nmin], j + 1u); }
    }
    workgroupBarrier();
    if (li == 0u) {
        let ctx = arg(3u);
        if (pos >= ctx) { st[3] = 0u; return; }
        var at = 0u;
        var used = 0u;
        for (var n = nmax; n >= nmin; n -= 1u) {
            let a = atomicLoad(&best[n - nmin]);
            if (a > 0u) { at = a; used = n; break; }
        }
        // what is on offer (st[7], for the CPU's choice of kernels), and what
        // fits; past the present the continuation repeats with the match's
        // period (a run of one token offers more of it)
        var offer = 0u;
        if (at > 0u) { offer = min(TMAX - 1u, ctx - 1u - pos); }
        let k = min(st[6], offer);
        let period = pos + 1u - at;
        for (var i = 0u; i < k; i++) { st[33u + i] = st[h + at + i % period]; }
        st[3] = 1u + k;
        st[7] = offer;
        st[10] = used;
    }
}
"#;

// ---- generated kernels: the matrix kernels, unrolled for a build ----
//
// Each (row, token) sum is a scalar of its own, and the code is written out
// for the build's token count `tb` and rows per subgroup, so the sums stay
// in registers. Every build makes each sum with the same expression in the
// same order (a lane's columns, ascending), so builds agree bit for bit.

fn lines(n: usize, f: impl Fn(usize) -> String) -> String {
    (0..n).map(f).collect()
}

/// out[t][r] (=, or +=) Q8_0 row r . x[t] (+ bias) for every token t, `mvr`
/// rows per subgroup summed together (sharing the x loads). Each lane takes
/// 16 columns (one 16-byte load per row) of every 512.
/// args: 0 rows, 1 cols, 2 x (xi, in vec4s; token stride cols/4), 3 out (act),
///       4 out stride, 5 bias (fw, or !0), 6 add
pub fn mv_q8(tb: usize, mvr: usize) -> String {
    let rows = lines(mvr, |r| {
        format!(
            "        let q{r} = wq[min(row0 + {r}u, rows - 1u) * n16 + i];\n        let s{r} = min(row0 + {r}u, rows - 1u) * nb + i / 2u;\n        let d{r} = unpack2x16float(wd[s{r} / 2u])[s{r} % 2u];\n        let w{r}0 = i8f(q{r}.x); let w{r}1 = i8f(q{r}.y); let w{r}2 = i8f(q{r}.z); let w{r}3 = i8f(q{r}.w);\n"
        )
    });
    let toks = lines(tb, |t| {
        let sums = lines(mvr, |r| format!("            a{r}_{t} += d{r} * (dot(w{r}0, x0) + dot(w{r}1, x1) + dot(w{r}2, x2) + dot(w{r}3, x3));\n"));
        format!("        if ({t}u < T) {{\n            let x = arg(2u) + {t}u * c4 + i * 4u;\n            let x0 = xi[x]; let x1 = xi[x + 1u]; let x2 = xi[x + 2u]; let x3 = xi[x + 3u];\n{sums}        }}\n")
    });
    let accs = lines(mvr * tb, |k| format!("    var a{}_{} = 0.0;\n", k / tb, k % tb));
    let outs = lines(mvr * tb, |k| {
        let (r, t) = (k / tb, k % tb);
        format!(
            "    {{\n        let tot = subgroupAdd(a{r}_{t});\n        let row = row0 + {r}u;\n        if (lane == 0u && {t}u < T && row < rows) {{\n            var v = tot;\n            if (arg(5u) != 0xffffffffu) {{ v += fw[arg(5u) + row]; }}\n            let o = arg(3u) + {t}u * arg(4u) + row;\n            if (arg(6u) == 1u) {{ act[o] += v; }} else {{ act[o] = v; }}\n        }}\n    }}\n"
        )
    });
    format!(
        r#"
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wd: array<u32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;
BCOMMON
fn i8f(w: u32) -> vec4<f32> {{ return vec4<f32>(unpack4xI8(w)); }}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {{
    let rows = arg(0u);
    let cols = arg(1u);
    let T = ntok();
    let lane = li % 32u;
    let n16 = cols / 16u;
    let nb = cols / 32u;
    let c4 = cols / 4u;
    let row0 = ((wg.x + wg.y * nw.x) * 8u + li / 32u) * {mvr}u;
    if (row0 >= rows) {{ return; }}
{accs}    for (var i = lane; i < n16; i += 32u) {{
{rows}{toks}    }}
{outs}}}
"#
    )
}

/// Gate and up rows of one union expert (grid y = union entry) for the
/// tokens routed to it, then the clamped SwiGLU into xo[(t*4+k)*R + r].
/// Each subgroup has 4 rows, `nr` at a time; lanes work in quads: a quad
/// takes a block, each lane one 4-byte word of it (8 values: x[4w..4w+4]
/// from the low nibbles, x[16+4w..] from the high).
/// args: 0 R, 1 C, 2 x (xi, normed, in vec4s), 3 out (xo), 4 gate bias, 5 up bias
pub fn gateup(tb: usize, nr: usize) -> String {
    let xbs = lines(tb, |t| format!("    let xb{t} = arg(2u) + rt.toks[{t}u] * c4;\n"));
    let mut groups = String::new();
    for j0 in (0..4).step_by(nr) {
        let accs = lines(nr * tb, |k| format!("        var g{}_{} = 0.0;\n        var u{}_{} = 0.0;\n", k / tb, k % tb, k / tb, k % tb));
        let rows = lines(nr, |r| {
            format!(
                "            let i{r} = (e * R + min(g0 + {j}u, R - 1u)) * nb + b;\n            let sg{r} = e8((ge[i{r} / 4u] >> ((i{r} % 4u) * 8u)) & 255u);\n            let su{r} = e8((ue[i{r} / 4u] >> ((i{r} % 4u) * 8u)) & 255u);\n            let gw{r} = gq[i{r} * 4u + w];\n            let uw{r} = uq[i{r} * 4u + w];\n            let gl{r} = lo4(gw{r}); let gh{r} = hi4(gw{r});\n            let ul{r} = lo4(uw{r}); let uh{r} = hi4(uw{r});\n",
                j = j0 + r
            )
        });
        let toks = lines(tb, |t| {
            let sums = lines(nr, |r| format!("                g{r}_{t} += sg{r} * (dot(gl{r}, x0) + dot(gh{r}, x1));\n                u{r}_{t} += su{r} * (dot(ul{r}, x0) + dot(uh{r}, x1));\n"));
            format!("            if ({t}u < m) {{\n                let x0 = xi[xb{t} + xo4];\n                let x1 = xi[xb{t} + xo4 + 4u];\n{sums}            }}\n")
        });
        let outs = lines(nr * tb, |k| {
            let (r, t) = (k / tb, k % tb);
            format!(
                "        {{\n            let sg = subgroupAdd(g{r}_{t});\n            let su = subgroupAdd(u{r}_{t});\n            let row = g0 + {j}u;\n            if (lane == 0u && {t}u < m && row < R) {{\n                let x = min(sg + fw[arg(4u) + e * R + row], 7.0);\n                let y = clamp(su + fw[arg(5u) + e * R + row], -7.0, 7.0);\n                xo[arg(3u) + (rt.toks[{t}u] * 4u + rt.ks[{t}u]) * R + row] = x / (1.0 + exp(-1.702 * x)) * (y + 1.0);\n            }}\n        }}\n",
                j = j0 + r
            )
        });
        groups += &format!("    {{\n{accs}        for (var b = lane / 4u; b < nb; b += 8u) {{\n            let xo4 = b * 8u + w;\n{rows}{toks}        }}\n{outs}    }}\n");
    }
    format!(
        r#"
@group(0) @binding(1) var<storage, read> gq: array<u32>;
@group(0) @binding(2) var<storage, read> ge: array<u32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<storage, read> fw: array<f32>;
@group(0) @binding(6) var<storage, read> uq: array<u32>;
@group(0) @binding(7) var<storage, read> ue: array<u32>;
@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> xo: array<f32>;
BCOMMON
ROUTED
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let u = wg.y;
    if (u >= uload(li, 48u)) {{ return; }}
    lut_init(li);
    let R = arg(0u);
    let C = arg(1u);
    let rt = routed(u);
    let e = rt.e;
    let m = rt.m;
    let lane = li % 32u;
    let nb = C / 32u;
    let c4 = C / 4u;
    let g0 = (wg.x * 8u + li / 32u) * 4u;
    let w = lane % 4u;
{xbs}{groups}}}
"#
    )
}

/// Down rows of one union expert for its tokens: out[(t*4+k)*R + r];
/// layout as `gateup`.
/// args: 0 R, 1 C, 2 a (xi, in vec4s; index (t*4+k)*C/4), 3 out (act)
pub fn down(tb: usize, nr: usize) -> String {
    let xbs = lines(tb, |t| format!("    let xb{t} = arg(2u) + (rt.toks[{t}u] * 4u + rt.ks[{t}u]) * c4;\n"));
    let mut groups = String::new();
    for j0 in (0..4).step_by(nr) {
        let accs = lines(nr * tb, |k| format!("        var a{}_{} = 0.0;\n", k / tb, k % tb));
        let rows = lines(nr, |r| {
            format!(
                "            let i{r} = (e * R + r0 + {j}u) * nb + b;\n            let s{r} = e8((de[i{r} / 4u] >> ((i{r} % 4u) * 8u)) & 255u);\n            let dw{r} = dq[i{r} * 4u + w];\n            let l{r} = lo4(dw{r}); let h{r} = hi4(dw{r});\n",
                j = j0 + r
            )
        });
        let toks = lines(tb, |t| {
            let sums = lines(nr, |r| format!("                a{r}_{t} += s{r} * (dot(l{r}, x0) + dot(h{r}, x1));\n"));
            format!("            if ({t}u < m) {{\n                let x0 = xi[xb{t} + xo4];\n                let x1 = xi[xb{t} + xo4 + 4u];\n{sums}            }}\n")
        });
        let outs = lines(nr * tb, |k| {
            let (r, t) = (k / tb, k % tb);
            format!(
                "        {{\n            let v = subgroupAdd(a{r}_{t});\n            let row = r0 + {j}u;\n            if (lane == 0u && {t}u < m && row < R) {{\n                act[arg(3u) + (rt.toks[{t}u] * 4u + rt.ks[{t}u]) * R + row] = v;\n            }}\n        }}\n",
                j = j0 + r
            )
        });
        groups += &format!("    {{\n{accs}        for (var b = lane / 4u; b < nb; b += 8u) {{\n            let xo4 = b * 8u + w;\n{rows}{toks}        }}\n{outs}    }}\n");
    }
    format!(
        r#"
@group(0) @binding(1) var<storage, read> dq: array<u32>;
@group(0) @binding(2) var<storage, read> de: array<u32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(9) var<storage, read> xi: array<vec4<f32>>;
BCOMMON
ROUTED
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let u = wg.y;
    if (u >= uload(li, 48u)) {{ return; }}
    lut_init(li);
    let R = arg(0u);
    let C = arg(1u);
    let rt = routed(u);
    let e = rt.e;
    let m = rt.m;
    let lane = li % 32u;
    let nb = C / 32u;
    let c4 = C / 4u;
    let r0 = (wg.x * 8u + li / 32u) * 4u;
    let w = lane % 4u;
{xbs}{groups}}}
"#
    )
}
