#version 300 es
// Draws the backdrop the glass refracts: a procedural wallpaper or a
// cover-fitted image/video, plus an optional overlay canvas (e.g. a clock).
precision highp float;
in vec2 v_uv;
out vec4 o;

uniform vec2 u_res;
uniform float u_time;
uniform int u_preset;          // 0 bloom, 1 dunes, 2 tiles, 3 texture, 4 night
uniform sampler2D u_image;
uniform vec2 u_imageSize;
uniform sampler2D u_overlay;
uniform float u_hasOverlay;
uniform float u_exposure;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 17.1; a *= 0.5; }
  return v;
}
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) { return a + b * cos(6.28318 * (c * t + d)); }

// Layered wobbly blobs with crisp edges and soft cast shadows.
vec3 bloom(vec2 p, float aspect, float px) {
  float t = u_time * 0.06;
  vec3 col = mix(vec3(0.04, 0.05, 0.16), vec3(0.16, 0.05, 0.26), p.y);
  vec3 C[6] = vec3[6](
    vec3(0.98, 0.42, 0.30), vec3(1.00, 0.74, 0.26), vec3(0.32, 0.52, 1.00),
    vec3(0.95, 0.34, 0.68), vec3(0.22, 0.82, 0.78), vec3(0.56, 0.36, 1.00));
  for (int i = 0; i < 6; i++) {
    float fi = float(i);
    vec2 c = vec2(aspect * (0.12 + 0.76 * fract(0.37 * fi + 0.11)) + 0.10 * sin(t * 1.3 + fi * 2.1),
                  0.12 + 0.76 * fract(0.61 * fi + 0.30) + 0.08 * cos(t * 1.1 + fi * 1.7));
    float r = 0.21 + 0.07 * sin(fi * 3.1);
    float w = (fbm(p * 2.2 + fi * 3.7 + t * 0.8) - 0.5) * 0.24;
    float d = length(p - c) - r - w;
    float ds = length(p - c - vec2(0.014, -0.02)) - r - w;
    float a = 1.0 - smoothstep(-px, px, d);
    col *= 1.0 - 0.38 * (1.0 - smoothstep(0.0, 0.09, ds)) * (1.0 - a);
    vec2 n = (p - c) / r;
    vec3 lc = C[i] * (0.9 + 0.14 * dot(n, vec2(-0.6, 0.8)) + 0.1 * smoothstep(0.0, -0.2, d));
    col = mix(col, lc, a);
  }
  return col;
}

// Sunset dunes: stacked ridges with sharp silhouettes.
vec3 dunes(vec2 p, float aspect, float px) {
  float t = u_time * 0.03;
  vec3 col = mix(vec3(1.0, 0.62, 0.38), vec3(0.40, 0.28, 0.68), smoothstep(0.3, 1.0, p.y));
  vec2 sc = vec2(aspect * 0.68, 0.72);
  float sd = length(p - sc) - 0.11;
  col += vec3(1.0, 0.7, 0.4) * 0.28 * exp(-max(sd, 0.0) * 7.0);
  col = mix(col, vec3(1.0, 0.94, 0.78), 1.0 - smoothstep(-px, px, sd));
  for (int i = 0; i < 7; i++) {
    float fi = float(i);
    float base = 0.62 - fi * 0.085;
    float h = base + 0.14 * (fbm(vec2(p.x * (1.1 + fi * 0.25) + fi * 11.0 + t * (1.0 + fi * 0.3), fi)) - 0.5)
                   + 0.03 * sin(p.x * 3.0 + fi);
    float d = p.y - h;
    float a = 1.0 - smoothstep(-px, px, d);
    vec3 lc = mix(vec3(0.96, 0.46, 0.36), vec3(0.16, 0.07, 0.26), fi / 6.0);
    lc *= 0.82 + 0.26 * smoothstep(-0.14, 0.0, d);
    col = mix(col, lc, a);
  }
  return col;
}

// Drifting grid of colored rounded tiles: the best preset for seeing refraction.
vec3 tiles(vec2 p, float px) {
  float t = u_time * 0.04;
  vec2 q = p * 7.0 + vec2(t, t * 0.6);
  vec2 id = floor(q);
  vec2 f = fract(q) - 0.5;
  float h = hash(id + 3.0);
  vec3 c = pal(h, vec3(0.55), vec3(0.45), vec3(1.0), vec3(0.0, 0.33, 0.67));
  float d = (length(max(abs(f) - 0.30, 0.0)) - 0.10) / 7.0;
  float a = 1.0 - smoothstep(-px, px, d);
  vec3 col = vec3(0.05, 0.06, 0.09);
  float ring = abs(length(f) - 0.18) / 7.0 - 0.006;
  c = mix(c, vec3(1.0), 0.55 * (1.0 - smoothstep(-px, px, ring)));
  return mix(col, c, a);
}

float sdBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Night: soft glowing outlines of rounded shapes on deep navy, a warm glow below.
vec3 night(vec2 p, float aspect, float px) {
  vec3 col = mix(vec3(0.005, 0.008, 0.02), vec3(0.02, 0.03, 0.07), smoothstep(1.0, 0.0, p.y));
  vec2 g = p - vec2(aspect * 0.42, 0.18);
  col += vec3(0.36, 0.2, 0.12) * exp(-dot(g * vec2(0.9, 1.6), g * vec2(0.9, 1.6)) * 5.0);
  vec2 C[5] = vec2[5](vec2(0.30, 0.80), vec2(0.78, 0.55), vec2(0.18, 0.30), vec2(0.62, 0.16), vec2(0.95, 0.88));
  vec2 B[5] = vec2[5](vec2(0.26, 0.13), vec2(0.20, 0.10), vec2(0.14, 0.12), vec2(0.24, 0.08), vec2(0.18, 0.09));
  for (int i = 0; i < 5; i++) {
    vec2 c = vec2(C[i].x * aspect, C[i].y);
    float d = sdBox(p - c, B[i] * vec2(aspect * 0.62, 1.0), 0.06);
    // a navy pool inside each shape, brightest at its rim
    float inside = 1.0 - smoothstep(-px, px, d);
    col = mix(col, col + vec3(0.03, 0.06, 0.16) * exp(d * 14.0), inside);
    float line = exp(-abs(d) / 0.0016) * 0.95 + exp(-abs(d) / 0.009) * 0.32 + exp(-abs(d) / 0.04) * 0.08;
    vec3 lc = mix(vec3(0.5, 0.62, 1.0), vec3(0.95, 0.97, 1.0), exp(-abs(d) / 0.002));
    col += lc * line;
  }
  return col;
}

void main() {
  vec2 frag = v_uv * u_res;
  float aspect = u_res.x / u_res.y;
  vec2 p = frag / u_res.y;
  float px = 1.25 / u_res.y;
  vec3 col;
  if (u_preset == 0) col = bloom(p, aspect, px);
  else if (u_preset == 1) col = dunes(p, aspect, px);
  else if (u_preset == 2) col = tiles(p, px);
  else if (u_preset == 4) col = night(p, aspect, px);
  else {
    float s = max(u_res.x / u_imageSize.x, u_res.y / u_imageSize.y);
    vec2 iuv = (frag - 0.5 * u_res) / (u_imageSize * s) + 0.5;
    col = texture(u_image, iuv).rgb;
  }
  if (u_hasOverlay > 0.5) {
    vec4 ov = texture(u_overlay, v_uv);
    col = mix(col, ov.rgb, ov.a);
  }
  col *= u_exposure;
  col += (hash(frag + 0.37) - 0.5) / 255.0;
  o = vec4(col, 1.0);
}
