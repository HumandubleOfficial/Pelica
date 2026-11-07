#version 300 es
// Copies the finished composite to the screen.
precision highp float;
in vec2 v_uv;
out vec4 o;
uniform sampler2D u_tex;
void main() {
  o = vec4(texture(u_tex, v_uv).rgb, 1.0);
}
