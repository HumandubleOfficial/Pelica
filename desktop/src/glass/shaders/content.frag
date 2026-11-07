#version 300 es
// Draws a rasterized HTML element *through* the glass it sits in. The glass
// pass leaves a per-pixel map of (refraction offset, top z, dispersion); we
// look the element's texture up at the refracted position, split channels
// for dispersion, and hide content that higher glass covers.
precision highp float;

uniform vec2 u_res;
uniform vec4 u_rect;            // element texture placement: x, y, w, h (canvas px, y down)
uniform float u_opacity;
uniform float u_z;              // z of the glass that owns this content
uniform sampler2D u_tex;        // premultiplied
uniform highp sampler2D u_off;  // xy offset px, z top glass z, w dispersion
uniform float u_useOff;

out vec4 o;

vec4 at(vec2 c) {
  vec2 uv = (c - u_rect.xy) / u_rect.zw;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
  return texture(u_tex, uv);
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y);
  vec4 f = u_useOff > 0.5 ? texelFetch(u_off, ivec2(gl_FragCoord.xy), 0) : vec4(0.0, 0.0, -1000.0, 0.0);
  if (f.z > u_z + 0.5) discard;
  vec4 g = at(p + f.xy);
  float r = at(p + f.xy * (1.0 - f.w)).r;
  float b = at(p + f.xy * (1.0 + f.w)).b;
  o = vec4(r, g.g, b, g.a) * u_opacity;
}
