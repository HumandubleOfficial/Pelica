//! WGSL pieces shared by the GPU-resident engine's kernels (`oss_batch`):
//! the parameter block, top-k routing and MXFP4 decoding. Each kernel is
//! its own module (these pieces + its bindings) so pipeline layouts stay
//! exact.

pub const COMMON: &str = r#"
struct P { a: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> p: P;
fn arg(i: u32) -> u32 { return p.a[i / 4u][i % 4u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
"#;

/// The 4 experts with the largest router logits (ties: lowest index) and
/// their softmax weights, made by one subgroup.
pub const TOPK: &str = r#"
struct Top { ids: vec4<u32>, w: vec4<f32> }
fn topk(li: u32, lg: u32, ne: u32) -> Top {
    let lane = li % 32u;
    var v = -3.0e38;
    if (lane < ne) { v = act[lg + lane]; }
    var t: Top;
    var vals = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let m = subgroupMax(v);
        let b = subgroupBallot(v == m);
        let i = firstTrailingBit(b.x);
        t.ids[k] = i;
        vals[k] = m;
        if (lane == i) { v = -3.0e38; }
    }
    let e = exp(vals - vec4<f32>(vals.x));
    t.w = e / (e.x + e.y + e.z + e.w);
    return t;
}
"#;

/// MXFP4 decoding: e2m1 nibbles (doubled, as ggml stores them) times
/// 2^(e-128). The 16 nibble values come from a table in workgroup memory
/// (`lut_init` fills it); each entry has a bank of its own, so lanes never
/// wait on each other, and a value costs a shift, a mask and a load.
pub const MX: &str = r#"
fn e8(e: u32) -> f32 {
    if (e < 2u) { return bitcast<f32>(0x00200000u << e); }
    return bitcast<f32>((e - 1u) << 23u);
}
var<workgroup> e2m1: array<f32, 16>;
fn lut_init(li: u32) {
    if (li < 16u) {
        let v = array<f32, 8>(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0);
        e2m1[li] = select(v[li & 7u], -v[li & 7u], li >= 8u);
    }
    workgroupBarrier();
}
fn lo4(w: u32) -> vec4<f32> { return vec4<f32>(e2m1[w & 15u], e2m1[(w >> 8u) & 15u], e2m1[(w >> 16u) & 15u], e2m1[(w >> 24u) & 15u]); }
fn hi4(w: u32) -> vec4<f32> { return vec4<f32>(e2m1[(w >> 4u) & 15u], e2m1[(w >> 12u) & 15u], e2m1[(w >> 20u) & 15u], e2m1[w >> 28u]); }
"#;
