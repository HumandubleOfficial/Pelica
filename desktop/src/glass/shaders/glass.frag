#version 300 es
// The liquid glass material. One full-screen pass draws the backdrop and then
// composites every glass group over it in z order.
//
// Shape data lives in a float texture (4 texels per row):
//   rows [0, MAXS)        shapes: (cx, cy, hw, hh) (radius, blur, refraction, bevel)
//                                 (tint.rgb, tint.a) (glow, opacity, lift, dispersion)
//   rows [MAXS, 2*MAXS)   groups: (bbox min.xy, max.xy) (start, end, mergeK, z)
// All lengths are canvas pixels, y pointing down.
precision highp float;
precision highp int;
#define MAXS 64

in vec2 v_uv;
layout(location = 0) out vec4 o;
// Refraction map for HTML content drawn after this pass: offset (canvas px,
// y down) of the topmost glass here, its z, and its dispersion.
layout(location = 1) out vec4 o2;
uniform float u_contentDepth;  // content sits inside the slab: a share of the backdrop's bend

uniform vec2 u_res;
uniform float u_scale;         // canvas px per CSS px
uniform sampler2D u_scene;
uniform sampler2D u_b1;
uniform sampler2D u_b2;
uniform sampler2D u_b3;
uniform highp sampler2D u_data;
uniform int u_groups;
uniform vec3 u_levels;         // blur sigma of b1..b3, canvas px

uniform float u_refraction;
uniform float u_dispersion;
uniform float u_ior;
uniform float u_specular;
uniform float u_fresnel;
uniform float u_saturation;
uniform float u_brightness;
uniform float u_shadow;
uniform float u_legibility;    // 0..1 local darkening over bright backdrops
uniform float u_lift;          // 0..1 grey lift over dark backdrops
uniform vec2 u_light;          // unit vector pointing toward the light (screen space, y down)

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Rounded rectangle with superellipse ("continuous") corners. Capsules fall
// back to circular ends so pills stay perfectly round.
float sdShape(vec2 p, vec4 r, float rad) {
  vec2 q = abs(p - r.xy) - r.zw + rad;
  float n = mix(3.4, 2.0, smoothstep(0.7, 1.0, rad / max(min(r.z, r.w), 1e-3)));
  vec2 m = max(q, 0.0) + 1e-4;
  float l = pow(pow(m.x, n) + pow(m.y, n), 1.0 / n);
  return l + min(max(q.x, q.y), 0.0) - rad;
}

vec3 blurAt(vec2 uv, float b) {
  vec3 s0 = texture(u_scene, uv).rgb;
  if (b < 0.25) return s0;
  if (b < u_levels.x) return mix(s0, texture(u_b1, uv).rgb, b / u_levels.x);
  if (b < u_levels.y) return mix(texture(u_b1, uv).rgb, texture(u_b2, uv).rgb,
                                 (b - u_levels.x) / (u_levels.y - u_levels.x));
  return mix(texture(u_b2, uv).rgb, texture(u_b3, uv).rgb,
             clamp((b - u_levels.y) / (u_levels.z - u_levels.y), 0.0, 1.0));
}

vec3 composite(vec3 col, vec2 uv, float d, float dx, float dy, vec4 P1, vec4 P2, vec4 P3, float gz, inout vec4 o2) {
  float opacity = P3.y;
  vec2 g = vec2(dx - d, dy - d);
  float gl = length(g);
  vec2 outward = gl > 1e-5 ? g / gl : vec2(0.0);

  // Soft contact shadow, nudged downward with a first-order SDF shift.
  float dsh = d - 5.0 * u_scale * g.y;
  float shadow = u_shadow * opacity * (1.0 - smoothstep(-6.0 * u_scale, 26.0 * u_scale, dsh));
  col *= 1.0 - shadow * smoothstep(-1.0, 1.0, d);

  float cover = clamp(0.5 - d, 0.0, 1.0) * opacity;
  if (cover <= 0.0) return col;

  // Surface: flat top, convex circular-arc rim of width `bevel`.
  float bevel = max(P1.w, 1.0);
  float t = clamp(-d / bevel, 0.0, 1.0);   // 0 at the edge, 1 on the flat top
  float u = 1.0 - t;
  float slope = min(u / sqrt(max(1.0 - u * u, 0.03)), 5.0);
  vec3 n = normalize(vec3(outward * slope, 1.0));

  // Refract a straight-down view ray through the top surface. The bend is
  // largest on the rim and vanishes on the flat top, like a real slab lens.
  vec3 r = refract(vec3(0.0, 0.0, -1.0), n, 1.0 / u_ior);
  vec2 off = r.xy / max(-r.z, 0.3) * bevel * u_refraction * P1.z;
  vec2 offUv = vec2(off.x, -off.y) / u_res;

  // Chromatic dispersion: red bends least, blue most.
  float blur = P1.y;
  float disp = u_dispersion * P3.w;   // per-shape multiplier: lenses in motion fringe harder
  if (cover > 0.5) o2 = vec4(off * u_contentDepth, gz, disp);
  vec3 refr = vec3(blurAt(uv + offUv * (1.0 - disp), blur).r,
                   blurAt(uv + offUv, blur).g,
                   blurAt(uv + offUv * (1.0 + disp), blur).b);

  // Vibrancy: blurring washes colors toward grey, so push saturation back up.
  const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
  float l = dot(refr, LUMA);
  refr = max(mix(vec3(l), refr, u_saturation), 0.0) * (1.0 + u_brightness) + u_brightness * 0.25;

  // Legibility: over bright backdrops the glass darkens locally so content on
  // it keeps its contrast. Judged on the heavily blurred backdrop so it varies
  // smoothly across the shape. Opaque tints don't need it.
  float bgL = dot(texture(u_b3, uv).rgb, LUMA);
  refr *= 1.0 - 0.38 * u_legibility * smoothstep(0.45, 0.95, bgL) * (1.0 - P2.a);
  // ...and over dark backdrops it lifts toward a soft grey instead of going
  // see-through black (P3.z: 1 for frosted "regular" glass, less for "clear").
  float lift = u_lift * P3.z * (1.0 - smoothstep(0.0, 0.4, bgL)) * (1.0 - P2.a);
  refr += (1.0 - refr) * lift;
  refr = mix(refr, P2.rgb, P2.a);

  // Environment reflection with Schlick's Fresnel: the steep rim mirrors the
  // blurred surroundings just outside the shape, so edges pick up nearby
  // color instead of glowing plain white. Head-on, glass reflects ~4%.
  vec3 rf = reflect(vec3(0.0, 0.0, -1.0), n);
  vec2 roff = rf.xy * bevel * 1.6;
  vec3 env = texture(u_b2, uv + vec2(roff.x, -roff.y) / u_res).rgb;
  float f0 = pow((u_ior - 1.0) / (u_ior + 1.0), 2.0);
  float F = f0 + (1.0 - f0) * pow(1.0 - n.z, 5.0);
  refr = mix(refr, env * 1.2 + 0.06, clamp(F * u_fresnel * 1.8, 0.0, 0.85));

  // Lighting: a key highlight facing the light, a dimmer one on the far rim
  // (light passing through the slab), and a hairline edge.
  float rim = u * u * u;
  float lit = dot(outward, u_light);
  float spec = (pow(max(lit, 0.0), 2.5) + 0.42 * pow(max(-lit, 0.0), 2.5)) * rim * u_specular;
  float edge = (1.0 - smoothstep(0.0, 1.5 * u_scale, -d)) * (0.22 + 0.5 * abs(lit));
  float glow = P3.x;
  vec3 glass = refr + vec3(1.0) * (spec * (1.0 + glow) + F * u_fresnel * 0.12 + edge * 0.5)
             + glow * 0.10 * (0.4 + 0.6 * t);

  // Faint monochrome grain makes the surface read as a physical material.
  glass += (hash(gl_FragCoord.xy * 0.73 + 11.0) - 0.5) * 0.022;
  return mix(col, glass, cover);
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_res;
  vec2 px = vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y);
  vec3 col = texture(u_scene, uv).rgb;
  o2 = vec4(0.0, 0.0, -1000.0, 0.0);

  for (int gi = 0; gi < MAXS; gi++) {
    if (gi >= u_groups) break;
    vec4 bb = texelFetch(u_data, ivec2(0, MAXS + gi), 0);
    if (px.x < bb.x || px.y < bb.y || px.x > bb.z || px.y > bb.w) continue;
    vec4 info = texelFetch(u_data, ivec2(1, MAXS + gi), 0);
    int s = int(info.x + 0.5);
    int e = int(info.y + 0.5);
    float k = max(info.z, 1e-3);

    float d = 1e6, dx = 1e6, dy = 1e6;
    vec4 P1 = vec4(0.0), P2 = vec4(0.0), P3 = vec4(0.0);
    bool first = true;
    for (int i = s; i < e; i++) {
      vec4 r = texelFetch(u_data, ivec2(0, i), 0);
      vec4 p1 = texelFetch(u_data, ivec2(1, i), 0);
      vec4 p2 = texelFetch(u_data, ivec2(2, i), 0);
      vec4 p3 = texelFetch(u_data, ivec2(3, i), 0);
      float rad = min(p1.x, min(r.z, r.w));
      float s0 = sdShape(px, r, rad);
      float sx = sdShape(px + vec2(1.0, 0.0), r, rad);
      float sy = sdShape(px + vec2(0.0, 1.0), r, rad);
      if (first) {
        d = s0; dx = sx; dy = sy; P1 = p1; P2 = p2; P3 = p3;
        first = false;
      } else {
        // Polynomial smooth-min: shapes in one group melt into each other.
        float h = clamp(0.5 + 0.5 * (s0 - d) / k, 0.0, 1.0);
        d = mix(s0, d, h) - k * h * (1.0 - h);
        float hx = clamp(0.5 + 0.5 * (sx - dx) / k, 0.0, 1.0);
        dx = mix(sx, dx, hx) - k * hx * (1.0 - hx);
        float hy = clamp(0.5 + 0.5 * (sy - dy) / k, 0.0, 1.0);
        dy = mix(sy, dy, hy) - k * hy * (1.0 - hy);
        P1 = mix(p1, P1, h); P2 = mix(p2, P2, h); P3 = mix(p3, P3, h);
      }
    }
    col = composite(col, uv, d, dx, dy, P1, P2, P3, info.w, o2);
  }

  col += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
  o = vec4(col, 1.0);
}
