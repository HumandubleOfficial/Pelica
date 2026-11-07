#version 300 es
// Separable Gaussian (9-tap kernel folded into 5 bilinear fetches).
// Run horizontally then vertically; downsampling happens for free because the
// target is smaller than the source and the source is linearly filtered.
precision highp float;
in vec2 v_uv;
out vec4 o;
uniform sampler2D u_tex;
uniform vec2 u_dir; // direction * spread / target size, in UV units

void main() {
  vec3 c = texture(u_tex, v_uv).rgb * 0.2270270270;
  vec2 o1 = u_dir * 1.3846153846;
  vec2 o2 = u_dir * 3.2307692308;
  c += texture(u_tex, v_uv + o1).rgb * 0.3162162162;
  c += texture(u_tex, v_uv - o1).rgb * 0.3162162162;
  c += texture(u_tex, v_uv + o2).rgb * 0.0702702703;
  c += texture(u_tex, v_uv - o2).rgb * 0.0702702703;
  o = vec4(c, 1.0);
}
