#version 300 es
// One quad per content element, in canvas pixels (y down). The quad is the
// element's box grown by the largest refraction offset (so content pulled in
// from outside its box still shows) and clipped to its scroll containers.
uniform vec2 u_res;
uniform vec4 u_geom; // x0, y0, x1, y1

void main() {
  int i = gl_VertexID;
  vec2 c = vec2((i == 1 || i == 2 || i == 4) ? 1.0 : 0.0, (i == 2 || i == 4 || i == 5) ? 1.0 : 0.0);
  vec2 p = mix(u_geom.xy, u_geom.zw, c);
  gl_Position = vec4(p.x / u_res.x * 2.0 - 1.0, 1.0 - p.y / u_res.y * 2.0, 0.0, 1.0);
}
