// The neural map: a model drawn as a network you can orbit. Every expert of
// every layer is a node (attention heads stand in for models without
// experts); attention and the router run down the middle as a spine. Two
// layouts morph into each other: a brain-shaped cloud (layers front to back,
// experts over two hemispheres) and the layered network (one ring per layer).
// Nodes fire as the model routes tokens to them (live activity from the
// engine), synapses connect each layer's active experts to the next, and a
// traced token's route lights up end to end. Bigger models get a richer
// picture (see `grandeur`): a connection web, drifting particles, a denser
// cortex, orbiting data rings and a nebula. Plain WebGL2, no libraries.
import { useEffect, useRef, useState } from "react";
import { experts as live } from "../state";
import { Icon } from "./Icons";

export interface MapPick {
  kind: "expert" | "attn" | "router" | "embed" | "output";
  layer: number;
  expert?: number;
}

interface Props {
  layers: number;
  /** experts per layer; 0 for dense models (heads are drawn instead) */
  experts: number;
  heads: number;
  usage: number[] | null;
  /** the engine is running this model, so live activity applies */
  isLive: boolean;
  title: string;
  sel: MapPick | null;
  /** a traced token's route: per layer, its experts and their weights */
  route: [number, number][][] | null;
  onPick(p: MapPick): void;
  /** right side of the toolbar (e.g. the "watch it think" button) */
  actions?: React.ReactNode;
  caption?: string;
  /** total parameters (how grand the picture gets) */
  params?: number;
}

/** 0 (a few billion parameters or fewer) .. 1 (500B and up), on a log scale. */
export function grandeur(params: number): number {
  const b = params / 1e9;
  return b <= 2 ? 0 : Math.min(1, Math.log10(b / 2) / Math.log10(250));
}

export function sizeClass(params: number): string {
  const b = params / 1e9;
  return b >= 100 ? "Frontier-scale" : b >= 20 ? "Large" : b >= 4 ? "Mid-size" : "Small";
}

const rnd = (i: number, s: number) => {
  const x = Math.sin(i * 12.9898 + s * 78.233) * 43758.5453;
  return x - Math.floor(x);
};

/** Node order: L×E experts, then L attention nodes, L routers, the embedding, the output. */
function layouts(L: number, E: number): { brain: Float32Array; net: Float32Array; kind: Float32Array } {
  const N = L * E + 2 * L + 2;
  const brain = new Float32Array(N * 3);
  const net = new Float32Array(N * 3);
  const kind = new Float32Array(N);
  const half = Math.max(1, Math.ceil(E / 2));
  const set = (a: Float32Array, i: number, x: number, y: number, z: number) => ((a[i * 3] = x), (a[i * 3 + 1] = y), (a[i * 3 + 2] = z));
  for (let l = 0; l < L; l++) {
    const t = L > 1 ? l / (L - 1) : 0.5;
    for (let e = 0; e < E; e++) {
      const k = l * E + e;
      // brain: layers front to back, each hemisphere's experts over its dome,
      // with folds and jitter so it reads as tissue rather than a grid
      const side = e < half ? -1 : 1;
      const j = e % half;
      const x0 = -0.95 + 1.9 * t + (rnd(k, 1) - 0.5) * (1.9 / Math.max(1, L)) * 0.9;
      const xa = Math.min(0.97, Math.abs(x0));
      const ry = 0.6 * Math.sqrt(1 - xa * xa) * (0.9 + 0.14 * x0);
      const rz = 0.5 * Math.sqrt(1 - xa * xa * 0.85);
      const u = (j + 0.25 + rnd(k, 3) * 0.5) / half;
      const phi = -0.34 * Math.PI + u * 0.84 * Math.PI;
      const fold = 1 + 0.05 * Math.sin(phi * 9 + x0 * 13) + (rnd(k, 2) - 0.5) * 0.07;
      set(brain, k, x0, ry * Math.sin(phi) * fold + 0.05, side * (0.04 + rz * Math.cos(phi) * fold));
      // network: one ring per layer, slightly twisted along the stack
      const a = (e / Math.max(1, E)) * Math.PI * 2 + l * 0.11;
      set(net, k, -1.3 + 2.6 * t, 0.46 * Math.sin(a), 0.46 * Math.cos(a));
    }
    const ia = L * E + l;
    const ir = L * E + L + l;
    set(brain, ia, -0.9 + 1.8 * t, 0.06, 0);
    set(brain, ir, -0.9 + 1.8 * t, -0.13, 0);
    set(net, ia, -1.3 + 2.6 * t, 0, 0);
    set(net, ir, -1.3 + 2.6 * t - 1.3 / Math.max(1, L), 0, 0);
    kind[ia] = 1;
    kind[ir] = 2;
  }
  const ie = L * E + 2 * L;
  set(brain, ie, -1.12, -0.08, 0);
  set(brain, ie + 1, 0.98, -0.5, 0);
  set(net, ie, -1.58, 0, 0);
  set(net, ie + 1, 1.58, 0, 0);
  kind[ie] = 3;
  kind[ie + 1] = 3;
  return { brain, net, kind };
}

// ---------- tiny matrix helpers (column-major, as GL wants) ----------

type M4 = Float32Array<ArrayBuffer>;
function mul(a: M4, b: M4): M4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function perspective(fovy: number, aspect: number, near: number, far: number): M4 {
  const f = 1 / Math.tan(fovy / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = (far + near) / (near - far);
  o[11] = -1;
  o[14] = (2 * far * near) / (near - far);
  return o;
}
function orbit(yaw: number, pitch: number, dist: number): M4 {
  // camera on a sphere around the origin, looking at it
  const ex = dist * Math.cos(pitch) * Math.sin(yaw);
  const ey = dist * Math.sin(pitch);
  const ez = dist * Math.cos(pitch) * Math.cos(yaw);
  const f = [-ex / dist, -ey / dist, -ez / dist];
  let s = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0];
  const sl = Math.hypot(s[0], s[1], s[2]) || 1;
  s = s.map((v) => v / sl);
  const u = [s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]];
  const o = new Float32Array(16);
  o[0] = s[0];
  o[4] = s[1];
  o[8] = s[2];
  o[1] = u[0];
  o[5] = u[1];
  o[9] = u[2];
  o[2] = -f[0];
  o[6] = -f[1];
  o[10] = -f[2];
  o[12] = -(s[0] * ex + s[1] * ey + s[2] * ez);
  o[13] = -(u[0] * ex + u[1] * ey + u[2] * ez);
  o[14] = f[0] * ex + f[1] * ey + f[2] * ez;
  o[15] = 1;
  return o;
}

// ---------- shaders ----------

const POINT_VS = `#version 300 es
in vec3 a_from;
in vec3 a_to;
in float a_heat;
in float a_kind;
in float a_base;
uniform mat4 u_mvp;
uniform float u_mix;
uniform float u_px;
uniform int u_sel;
uniform int u_hover;
uniform float u_dim;
out float v_heat;
out float v_kind;
out float v_base;
out float v_mark;
void main() {
  vec3 p = mix(a_from, a_to, u_mix);
  vec4 c = u_mvp * vec4(p, 1.0);
  gl_Position = c;
  float mark = (gl_VertexID == u_sel) ? 1.0 : ((gl_VertexID == u_hover) ? 0.6 : 0.0);
  float size = (a_kind > 0.5 ? (a_kind > 2.5 ? 16.0 : 9.0) : 3.4 + 2.5 * a_base) + a_heat * 15.0 + mark * 10.0;
  gl_PointSize = size * u_px / c.w;
  v_heat = max(a_heat, mark);
  v_kind = a_kind;
  v_base = a_base * u_dim;
  v_mark = mark;
}`;

const POINT_FS = `#version 300 es
precision highp float;
in float v_heat;
in float v_kind;
in float v_base;
in float v_mark;
uniform float u_dim;
uniform float u_grand;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r = dot(d, d);
  if (r > 1.0) discard;
  float glow = exp(-r * 3.5);
  float core = smoothstep(0.35, 0.0, r);
  vec3 cold = mix(vec3(0.42, 0.40, 0.52), vec3(0.36, 0.52, 0.92), u_grand * 0.7);
  vec3 hot = vec3(1.0, 0.52, 0.22);
  vec3 col = v_kind < 0.5 ? mix(cold, hot, clamp(v_heat * 1.2, 0.0, 1.0))
           : v_kind < 1.5 ? vec3(0.62, 0.50, 1.0)
           : v_kind < 2.5 ? vec3(1.0, 0.76, 0.36)
           : vec3(0.42, 0.88, 0.80);
  float a = ((v_kind < 0.5 ? 0.16 + 0.08 * u_grand : 0.45) * u_dim + 0.35 * v_base + (1.1 + 0.5 * u_grand) * v_heat) * glow + core * (0.25 * u_dim + v_heat * 0.9);
  if (v_mark > 0.0) a += (1.0 - smoothstep(0.55, 0.7, abs(r - 0.62) + 0.55)) * 0.9;
  o = vec4(col * a, a);
}`;

const LINE_VS = `#version 300 es
in vec3 a_pos;
in vec4 a_col;
uniform mat4 u_mvp;
out vec4 v_col;
void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  v_col = a_col;
}`;

const LINE_FS = `#version 300 es
precision mediump float;
in vec4 v_col;
uniform float u_a;
out vec4 o;
void main() { float a = v_col.a * u_a; o = vec4(v_col.rgb * a, a); }`;

// drifting particles (k 0) and tilted data rings (k 1), moved on the GPU
const DUST_VS = `#version 300 es
in vec4 a_p;
in float a_k;
uniform mat4 u_mvp;
uniform float u_time;
uniform float u_px;
out float v_a;
out float v_k;
void main() {
  float ang = a_p.y + u_time * a_p.w;
  vec3 p;
  if (a_k < 0.5) {
    p = vec3(cos(ang) * a_p.x * 1.3, a_p.z + 0.04 * sin(u_time * 0.7 + a_p.y * 5.0), sin(ang) * a_p.x);
  } else {
    vec3 q = vec3(cos(ang) * a_p.x * 1.15, 0.0, sin(ang) * a_p.x * 0.85);
    float t = a_p.z;
    p = vec3(q.x * cos(t * 0.6) - q.y * sin(t * 0.6), q.y * cos(t) - q.z * sin(t), q.y * sin(t) + q.z * cos(t));
  }
  vec4 c = u_mvp * vec4(p, 1.0);
  gl_Position = c;
  gl_PointSize = (a_k < 0.5 ? 2.0 : 2.6) * u_px / c.w;
  v_a = 0.55 + 0.45 * sin(u_time * 1.9 + a_p.y * 17.0);
  v_k = a_k;
}`;

const DUST_FS = `#version 300 es
precision mediump float;
in float v_a;
in float v_k;
uniform float u_alpha;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r = dot(d, d);
  if (r > 1.0) discard;
  vec3 col = v_k < 0.5 ? vec3(0.55, 0.62, 1.0) : vec3(0.45, 0.85, 1.0);
  float a = u_alpha * v_a * (v_k < 0.5 ? 1.0 : 1.6) * exp(-r * 3.0);
  o = vec4(col * a, a);
}`;

// sparks (signals travelling along synapses) and the faint cortex shell
const SPARK_VS = `#version 300 es
in vec3 a_pos;
in float a_a;
uniform mat4 u_mvp;
uniform float u_px;
uniform float u_size;
out float v_a;
void main() {
  vec4 c = u_mvp * vec4(a_pos, 1.0);
  gl_Position = c;
  gl_PointSize = u_size * u_px / c.w;
  v_a = a_a;
}`;

const SPARK_FS = `#version 300 es
precision mediump float;
in float v_a;
uniform vec3 u_col;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r = dot(d, d);
  if (r > 1.0) discard;
  float a = v_a * (exp(-r * 4.0) + 0.6 * smoothstep(0.2, 0.0, r));
  o = vec4(u_col * a, a);
}`;

/** Points on two hemispheres around the brain layout: the cortex, drawn faintly. */
function shell(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const side = i % 2 ? 1 : -1;
    const u = rnd(i, 11) * 2 - 1;
    const th = rnd(i, 12) * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    let x = u * 1.08;
    let y = r * Math.sin(th) * 0.7;
    let z = r * Math.cos(th) * 0.58;
    // one hemisphere per side, a flatter base, and folds (gyri)
    z = side * (Math.abs(z) + 0.035);
    if (y < -0.3) y = -0.3 - (y + 0.3) * 0.25;
    const g = 1 + 0.045 * Math.sin(x * 17 + y * 11) * Math.cos(z * 13 + x * 5);
    x *= g;
    y = y * g + 0.05;
    z *= g;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = z;
  }
  return out;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || "link");
  return p;
}

const COL = { spine: [0.62, 0.5, 1.0], hot: [1.0, 0.55, 0.25], route: [1.0, 0.85, 0.45], path: [0.55, 0.5, 0.75] };

export function NeuralMap(p: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"brain" | "net">("brain");
  const [hover, setHover] = useState<{ pick: MapPick; x: number; y: number } | null>(null);
  const [glErr, setGlErr] = useState("");
  const props = useRef(p);
  props.current = p;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const view = useRef({ yaw: 0.9, pitch: 0.32, dist: 3.0, idle: 0, reset: false });

  const L = Math.max(1, p.layers);
  const E = p.experts > 0 ? p.experts : Math.max(1, p.heads);

  useEffect(() => {
    const c = canvas.current!;
    const gl = c.getContext("webgl2", { antialias: true, premultipliedAlpha: true, alpha: true });
    if (!gl) {
      setGlErr("WebGL2 isn't available, so the map can't be drawn.");
      return;
    }
    let pp: WebGLProgram, lp: WebGLProgram, sp: WebGLProgram, dp: WebGLProgram;
    try {
      pp = program(gl, POINT_VS, POINT_FS);
      lp = program(gl, LINE_VS, LINE_FS);
      sp = program(gl, SPARK_VS, SPARK_FS);
      dp = program(gl, DUST_VS, DUST_FS);
    } catch (e: any) {
      setGlErr(String(e?.message || e));
      return;
    }
    const { brain, net, kind } = layouts(L, E);
    const N = kind.length;
    const NE = L * E;
    const ia = (l: number) => NE + l;
    const ir = (l: number) => NE + L + l;
    const iEmb = NE + 2 * L;
    const iOut = iEmb + 1;
    // how much each expert was used before (log scale), the resting glow
    const base = new Float32Array(N);
    const u = props.current.usage;
    const topUsed: number[][] = [];
    if (u && u.length === NE && p.experts > 0) {
      const mx = Math.log1p(Math.max(1, ...u));
      for (let k = 0; k < NE; k++) base[k] = Math.log1p(u[k]) / mx;
      for (let l = 0; l < L; l++) {
        const row = Array.from({ length: E }, (_, e) => e).sort((a, b) => u[l * E + b] - u[l * E + a]);
        topUsed.push(row.slice(0, 2));
      }
    }
    const heat = new Float32Array(N);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const attr = (prog: WebGLProgram, name: string, data: Float32Array, size: number, usage: number) => {
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      const loc = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return b;
    };
    attr(pp, "a_from", brain, 3, gl.STATIC_DRAW);
    attr(pp, "a_to", net, 3, gl.STATIC_DRAW);
    attr(pp, "a_kind", kind, 1, gl.STATIC_DRAW);
    attr(pp, "a_base", base, 1, gl.STATIC_DRAW);
    const heatBuf = attr(pp, "a_heat", heat, 1, gl.DYNAMIC_DRAW);

    const lvao = gl.createVertexArray();
    gl.bindVertexArray(lvao);
    const lineBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    const lpos = gl.getAttribLocation(lp, "a_pos");
    const lcol = gl.getAttribLocation(lp, "a_col");
    gl.enableVertexAttribArray(lpos);
    gl.enableVertexAttribArray(lcol);
    gl.vertexAttribPointer(lpos, 3, gl.FLOAT, false, 28, 0);
    gl.vertexAttribPointer(lcol, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);

    // the spark program's two buffers: the shell (fixed) and the sparks (each frame)
    const sparkVao = (data: Float32Array | null) => {
      const v = gl.createVertexArray()!;
      gl.bindVertexArray(v);
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const lp0 = gl.getAttribLocation(sp, "a_pos");
      const la = gl.getAttribLocation(sp, "a_a");
      gl.enableVertexAttribArray(lp0);
      gl.enableVertexAttribArray(la);
      gl.vertexAttribPointer(lp0, 3, gl.FLOAT, false, 16, 0);
      gl.vertexAttribPointer(la, 1, gl.FLOAT, false, 16, 12);
      gl.bindVertexArray(null);
      return { v, b };
    };
    const G = grandeur(props.current.params || 0);
    const SHELL = Math.round(2400 + 9600 * G);
    const sh = shell(SHELL);
    const shellData = new Float32Array(SHELL * 4);
    for (let i = 0; i < SHELL; i++) {
      shellData.set(sh.subarray(i * 3, i * 3 + 3), i * 4);
      shellData[i * 4 + 3] = 0.5 + 0.5 * rnd(i, 13);
    }
    const shellV = sparkVao(shellData);
    const sparkV = sparkVao(null);
    const sparks: number[] = [];

    // the connection web (models from a few billion parameters): each node to
    // its neighbour in the layer and to the same node one layer on, in both
    // layouts; drawn faintly and blended as the layouts morph
    const webOf = (lay: Float32Array) => {
      const out: number[] = [];
      const stride = Math.max(1, Math.ceil((NE * 2) / 26000));
      const col = [0.45, 0.55, 0.95];
      const push = (a: number, b: number, al: number) => out.push(lay[a * 3], lay[a * 3 + 1], lay[a * 3 + 2], col[0], col[1], col[2], al, lay[b * 3], lay[b * 3 + 1], lay[b * 3 + 2], col[0], col[1], col[2], al);
      const halfE = Math.max(1, Math.ceil(E / 2));
      for (let k = 0; k < NE; k += stride) {
        const l = Math.floor(k / E);
        const e = k % E;
        const al = 0.05 + 0.06 * rnd(k, 21);
        if ((e + 1) % halfE !== 0 && e + 1 < E) push(k, k + 1, al);
        if (l + 1 < L) push(k, k + E, al * 0.8);
      }
      return new Float32Array(out);
    };
    const webVao = (data: Float32Array) => {
      const v = gl.createVertexArray()!;
      gl.bindVertexArray(v);
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const p0 = gl.getAttribLocation(lp, "a_pos");
      const c0 = gl.getAttribLocation(lp, "a_col");
      gl.enableVertexAttribArray(p0);
      gl.enableVertexAttribArray(c0);
      gl.vertexAttribPointer(p0, 3, gl.FLOAT, false, 28, 0);
      gl.vertexAttribPointer(c0, 4, gl.FLOAT, false, 28, 12);
      gl.bindVertexArray(null);
      return { v, n: data.length / 7 };
    };
    const web = G > 0.12 ? { brain: webVao(webOf(brain)), net: webVao(webOf(net)) } : null;

    // drifting particles (from a few billion) and data rings (from ~100B)
    const DUST = Math.round(3500 * Math.max(0, (G - 0.12) / 0.88));
    const RINGS = G > 0.55 ? 3 : 0;
    const RING_N = 700;
    const dustData = new Float32Array((DUST + RINGS * RING_N) * 5);
    for (let i = 0; i < DUST; i++) {
      const r = 0.25 + 1.05 * Math.sqrt(rnd(i, 31));
      dustData.set([r, rnd(i, 32) * Math.PI * 2, (rnd(i, 33) - 0.5) * 1.3 * (1.2 - r * 0.4), (0.04 + 0.1 * rnd(i, 34)) * (rnd(i, 35) > 0.5 ? 1 : -1), 0], i * 5);
    }
    for (let ri = 0; ri < RINGS; ri++)
      for (let j = 0; j < RING_N; j++) {
        const i = DUST + ri * RING_N + j;
        dustData.set([1.12 + ri * 0.13 + (rnd(i, 36) - 0.5) * 0.025, (j / RING_N) * Math.PI * 2, 0.35 + ri * 0.75, (ri % 2 ? -1 : 1) * (0.05 + ri * 0.025), 1], i * 5);
      }
    const dustV = gl.createVertexArray()!;
    gl.bindVertexArray(dustV);
    const dustB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, dustB);
    gl.bufferData(gl.ARRAY_BUFFER, dustData, gl.STATIC_DRAW);
    const dP = gl.getAttribLocation(dp, "a_p");
    const dK = gl.getAttribLocation(dp, "a_k");
    gl.enableVertexAttribArray(dP);
    gl.enableVertexAttribArray(dK);
    gl.vertexAttribPointer(dP, 4, gl.FLOAT, false, 20, 0);
    gl.vertexAttribPointer(dK, 1, gl.FLOAT, false, 20, 16);
    gl.bindVertexArray(null);
    const DN = DUST + RINGS * RING_N;

    const U = (prog: WebGLProgram, n: string) => gl.getUniformLocation(prog, n);
    const uSMvp = U(sp, "u_mvp"), uSPx = U(sp, "u_px"), uSSize = U(sp, "u_size"), uSCol = U(sp, "u_col");
    const uMvp = U(pp, "u_mvp"), uMix = U(pp, "u_mix"), uPx = U(pp, "u_px"), uSel = U(pp, "u_sel"), uHover = U(pp, "u_hover"), uDim = U(pp, "u_dim");
    const uLMvp = U(lp, "u_mvp"), uLA = U(lp, "u_a");
    const uGrand = U(pp, "u_grand");
    const uDMvp = U(dp, "u_mvp"), uDTime = U(dp, "u_time"), uDPx = U(dp, "u_px"), uDAlpha = U(dp, "u_alpha");

    let mix = modeRef.current === "net" ? 1 : 0;
    let raf = 0;
    let last = performance.now();
    let mouse: { x: number; y: number } | null = null;
    let hoverIdx = -1;
    let drag: { x: number; y: number; moved: number } | null = null;
    let mvp: M4 = new Float32Array(16);
    const lines: number[] = [];
    const pos = (i: number, m: number) => [brain[i * 3] + (net[i * 3] - brain[i * 3]) * m, brain[i * 3 + 1] + (net[i * 3 + 1] - brain[i * 3 + 1]) * m, brain[i * 3 + 2] + (net[i * 3 + 2] - brain[i * 3 + 2]) * m];
    /** A signal on its way from node a to node b: `t` of the way there. */
    const spark = (a: number, b: number, t: number, alpha: number) => {
      const pa = pos(a, mix);
      const pb = pos(b, mix);
      for (let k = 0; k < 3; k++) {
        // a bright head and a short fading tail
        const tt = Math.max(0, t - k * 0.045);
        sparks.push(pa[0] + (pb[0] - pa[0]) * tt, pa[1] + (pb[1] - pa[1]) * tt, pa[2] + (pb[2] - pa[2]) * tt, alpha * (1 - k * 0.38));
      }
    };
    const seg = (a: number, b: number, col: number[], alpha: number) => {
      if (alpha < 0.01) return;
      const pa = pos(a, mix);
      const pb = pos(b, mix);
      lines.push(pa[0], pa[1], pa[2], col[0], col[1], col[2], alpha, pb[0], pb[1], pb[2], col[0], col[1], col[2], alpha);
    };
    const toPick = (i: number): MapPick => (i < NE ? { kind: "expert", layer: Math.floor(i / E), expert: i % E } : i < NE + L ? { kind: "attn", layer: i - NE } : i < NE + 2 * L ? { kind: "router", layer: i - NE - L } : { kind: i === iEmb ? "embed" : "output", layer: i === iEmb ? 0 : L - 1 });
    const fromPick = (s: MapPick | null): number => {
      if (!s) return -1;
      if (s.kind === "expert" && s.expert !== undefined) return s.layer * E + s.expert;
      if (s.kind === "attn") return ia(s.layer);
      if (s.kind === "router") return ir(s.layer);
      return s.kind === "embed" ? iEmb : iOut;
    };
    const project = (i: number, W: number, H: number): [number, number, number] | null => {
      const q = pos(i, mix);
      const x = mvp[0] * q[0] + mvp[4] * q[1] + mvp[8] * q[2] + mvp[12];
      const y = mvp[1] * q[0] + mvp[5] * q[1] + mvp[9] * q[2] + mvp[13];
      const w = mvp[3] * q[0] + mvp[7] * q[1] + mvp[11] * q[2] + mvp[15];
      if (w <= 0.01) return null;
      return [(x / w * 0.5 + 0.5) * W, (1 - (y / w * 0.5 + 0.5)) * H, w];
    };
    const nearest = (mx: number, my: number, W: number, H: number) => {
      let best = -1;
      let bd = 15 * 15;
      for (let i = 0; i < N; i++) {
        const s = project(i, W, H);
        if (!s) continue;
        const d = (s[0] - mx) ** 2 + (s[1] - my) ** 2 - (kind[i] > 0 ? 40 : 0);
        if (d < bd) (bd = d), (best = i);
      }
      return best;
    };

    let drawn = 0;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const P = props.current;
      const v = view.current;
      // at rest (nothing firing, nobody touching it) 30 frames a second is
      // plenty, and leaves the GPU to the model
      const calm = !drag && !v.reset && now - live.last > 1500 && !P.route && Math.abs(mix - (modeRef.current === "net" ? 1 : 0)) < 0.002;
      if (calm && now - drawn < 31) return;
      drawn = now;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (v.reset) {
        v.yaw += (0.9 - v.yaw) * 0.15;
        v.pitch += (0.32 - v.pitch) * 0.15;
        v.dist += (3.0 - v.dist) * 0.15;
        if (Math.abs(v.yaw - 0.9) < 0.002 && Math.abs(v.dist - 3) < 0.005) v.reset = false;
      } else if (!drag && now - v.idle > 2500) v.yaw += dt * 0.07;
      mix += ((modeRef.current === "net" ? 1 : 0) - mix) * (1 - Math.exp(-dt * 5));

      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth;
      const H = c.clientHeight;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
        c.width = Math.round(W * dpr);
        c.height = Math.round(H * dpr);
      }
      gl.viewport(0, 0, c.width, c.height);
      mvp = mul(perspective(0.78, W / Math.max(1, H), 0.05, 40), orbit(v.yaw, v.pitch, v.dist));

      // heat: live activity (when this is the running model), or a traced route
      const liveOn = P.isLive && live.layers === L && live.experts === E && p.experts > 0;
      heat.fill(0);
      const route = P.route;
      const recent = now - live.last < 1500;
      if (route) {
        for (let l = 0; l < Math.min(L, route.length); l++) for (const [e, w] of route[l]) if (e < E) heat[l * E + e] = Math.max(heat[l * E + e], 0.35 + 0.65 * Math.min(1, w * 2));
      } else if (liveOn) {
        for (let k = 0; k < NE; k++) heat[k] = live.heat(k, now);
      }
      // the spine: each layer's attention and router glow with its experts;
      // while tokens flow, a pulse runs from input to output
      const pulse = (now / 900) % 1;
      for (let l = 0; l < L; l++) {
        let m = 0;
        for (let e = 0; e < E; e++) m = Math.max(m, heat[l * E + e]);
        const wave = recent || route ? Math.exp(-((l / L - pulse) ** 2) / 0.004) * 0.7 : 0;
        heat[ia(l)] = Math.min(1, m * 0.6 + wave);
        heat[ir(l)] = Math.min(1, m * 0.5 + wave * 0.6);
      }
      heat[iEmb] = recent || route ? 0.6 + 0.4 * Math.exp(-(pulse ** 2) / 0.003) : 0.15;
      heat[iOut] = recent || route ? 0.6 + 0.4 * Math.exp(-((1 - pulse) ** 2) / 0.003) : 0.15;
      gl.bindBuffer(gl.ARRAY_BUFFER, heatBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, heat);

      // synapses: attention → the layer's hottest experts → the next layer
      lines.length = 0;
      sparks.length = 0;
      seg(iEmb, ia(0), COL.spine, 0.35);
      for (let l = 0; l + 1 < L; l++) seg(ia(l), ia(l + 1), COL.spine, 0.22);
      seg(ia(L - 1), iOut, COL.spine, 0.35);
      for (let l = 0; l < L; l++) seg(ir(l), ia(l), COL.spine, 0.12);
      if (p.experts > 0) {
        for (let l = 0; l < L; l++) {
          const next = l + 1 < L ? ia(l + 1) : iOut;
          if (route) {
            for (const [e, w] of route[l] || []) {
              const a = 0.25 + 0.75 * Math.min(1, w * 2);
              seg(ir(l), l * E + e, COL.route, a);
              seg(l * E + e, next, COL.route, a * 0.8);
              // the token's signal, travelling down the layers in step with the pulse
              const ph = (pulse * L - l + L) % L;
              if (ph < 1) ph < 0.5 ? spark(ir(l), l * E + e, ph * 2, a) : spark(l * E + e, next, ph * 2 - 1, a);
            }
            continue;
          }
          const hot: [number, number][] = [];
          for (let e = 0; e < E; e++) {
            const h = heat[l * E + e];
            if (h > 0.12) hot.push([h, e]);
          }
          hot.sort((a, b) => b[0] - a[0]);
          for (const [h, e] of hot.slice(0, 4 + Math.round(4 * G))) {
            seg(ir(l), l * E + e, COL.hot, h * 0.75);
            seg(l * E + e, next, COL.hot, h * 0.5);
            // signals firing through the experts in use
            if (h > 0.3) {
              const ph = (now / 650 + rnd(l * E + e, 7)) % 1;
              ph < 0.5 ? spark(ir(l), l * E + e, ph * 2, h) : spark(l * E + e, next, ph * 2 - 1, h * 0.8);
            }
          }
          if (!hot.length && topUsed[l]) for (const e of topUsed[l]) (seg(ir(l), l * E + e, COL.path, 0.07), seg(l * E + e, next, COL.path, 0.05));
        }
      }

      if (mouse && !drag) {
        const i = nearest(mouse.x, mouse.y, W, H);
        if (i !== hoverIdx) {
          hoverIdx = i;
          setHover(i >= 0 ? { pick: toPick(i), x: mouse.x, y: mouse.y } : null);
        }
      }

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(lp);
      gl.uniformMatrix4fv(uLMvp, false, mvp);
      if (web) {
        const wa = (route ? 0.4 : 1) * (0.45 + 0.3 * G);
        if (mix < 0.995) (gl.uniform1f(uLA, wa * (1 - mix)), gl.bindVertexArray(web.brain.v), gl.drawArrays(gl.LINES, 0, web.brain.n));
        if (mix > 0.005) (gl.uniform1f(uLA, wa * mix), gl.bindVertexArray(web.net.v), gl.drawArrays(gl.LINES, 0, web.net.n));
      }
      gl.uniform1f(uLA, 1);
      if (lines.length) {
        gl.useProgram(lp);
        gl.bindVertexArray(lvao);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STREAM_DRAW);
        gl.uniformMatrix4fv(uLMvp, false, mvp);
        gl.drawArrays(gl.LINES, 0, lines.length / 7);
      }
      gl.blendFunc(gl.ONE, gl.ONE);
      // the cortex, fading out as the layers unfold
      gl.useProgram(sp);
      gl.uniformMatrix4fv(uSMvp, false, mvp);
      gl.uniform1f(uSPx, (H * dpr) / 150);
      const shellA = (1 - mix) * (route ? 0.05 : 0.085) * (1 + 0.2 * G);
      if (shellA > 0.004) {
        gl.uniform1f(uSSize, 2.6);
        gl.uniform3f(uSCol, 0.55 * shellA * 10, 0.5 * shellA * 10, 0.75 * shellA * 10);
        gl.bindVertexArray(shellV.v);
        gl.drawArrays(gl.POINTS, 0, SHELL);
      }
      if (DN) {
        gl.useProgram(dp);
        gl.uniformMatrix4fv(uDMvp, false, mvp);
        gl.uniform1f(uDTime, now / 1000);
        gl.uniform1f(uDPx, (H * dpr) / 150);
        gl.uniform1f(uDAlpha, (route ? 0.12 : 0.22) * (0.5 + G));
        gl.bindVertexArray(dustV);
        gl.drawArrays(gl.POINTS, 0, DN);
        gl.useProgram(sp);
      }
      if (sparks.length) {
        gl.uniform1f(uSSize, 7 + 3 * G);
        gl.uniform3f(uSCol, route ? 1.0 : 1.0, route ? 0.88 : 0.7, route ? 0.55 : 0.4);
        gl.bindVertexArray(sparkV.v);
        gl.bindBuffer(gl.ARRAY_BUFFER, sparkV.b);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sparks), gl.STREAM_DRAW);
        gl.drawArrays(gl.POINTS, 0, sparks.length / 4);
      }
      gl.useProgram(pp);
      gl.bindVertexArray(vao);
      gl.uniformMatrix4fv(uMvp, false, mvp);
      gl.uniform1f(uMix, mix);
      gl.uniform1f(uPx, (H * dpr) / 150);
      gl.uniform1i(uSel, fromPick(P.sel));
      gl.uniform1i(uHover, hoverIdx);
      gl.uniform1f(uDim, route ? 0.35 : 1);
      gl.uniform1f(uGrand, G);
      gl.drawArrays(gl.POINTS, 0, N);
      gl.bindVertexArray(null);

      // labels that follow the nodes
      const box = labels.current;
      if (box) {
        const place = (el: Element, i: number) => {
          const s = project(i, W, H);
          (el as HTMLElement).style.transform = s ? `translate(${s[0]}px, ${s[1]}px)` : "translate(-999px,0)";
        };
        const els = box.children;
        if (els.length >= 4) {
          place(els[0], iEmb);
          place(els[1], iOut);
          place(els[2], ia(0));
          place(els[3], ia(L - 1));
        }
      }
    };
    raf = requestAnimationFrame(frame);

    const rect = () => c.getBoundingClientRect();
    const down = (e: PointerEvent) => {
      c.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, moved: 0 };
      view.current.idle = performance.now();
      view.current.reset = false;
    };
    const move = (e: PointerEvent) => {
      const r = rect();
      mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = e.clientX;
      drag.y = e.clientY;
      const v = view.current;
      v.yaw -= dx * 0.006;
      v.pitch = Math.max(-1.35, Math.min(1.35, v.pitch + dy * 0.006));
      v.idle = performance.now();
    };
    const up = (e: PointerEvent) => {
      const d = drag;
      drag = null;
      view.current.idle = performance.now();
      if (d && d.moved < 5) {
        const r = rect();
        const i = nearest(e.clientX - r.left, e.clientY - r.top, c.clientWidth, c.clientHeight);
        if (i >= 0) props.current.onPick(toPick(i));
      }
    };
    const leave = () => {
      mouse = null;
      hoverIdx = -1;
      setHover(null);
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const v = view.current;
      v.dist = Math.max(1.3, Math.min(6, v.dist * Math.exp(e.deltaY * 0.0012)));
      v.idle = performance.now();
    };
    const dbl = () => (view.current.reset = true);
    c.addEventListener("pointerdown", down);
    c.addEventListener("pointermove", move);
    c.addEventListener("pointerup", up);
    c.addEventListener("pointerleave", leave);
    c.addEventListener("wheel", wheel, { passive: false });
    c.addEventListener("dblclick", dbl);
    return () => {
      cancelAnimationFrame(raf);
      c.removeEventListener("pointerdown", down);
      c.removeEventListener("pointermove", move);
      c.removeEventListener("pointerup", up);
      c.removeEventListener("pointerleave", leave);
      c.removeEventListener("wheel", wheel);
      c.removeEventListener("dblclick", dbl);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
    // the usage is read once per model: a new model remounts the map
  }, [L, E, p.experts, p.params]);

  const hp = hover?.pick;
  const usageOf = (l: number, e: number) => (p.usage && p.experts > 0 ? p.usage[l * p.experts + e] || 0 : 0);
  const tip = !hp
    ? ""
    : hp.kind === "expert"
      ? p.experts > 0
        ? `Layer ${hp.layer} · expert ${hp.expert} — picked ${usageOf(hp.layer, hp.expert!).toLocaleString()} times before${live.count.length === p.layers * p.experts && live.count[hp.layer * p.experts + hp.expert!] ? `, ${live.count[hp.layer * p.experts + hp.expert!].toLocaleString()} this session` : ""}`
        : `Layer ${hp.layer} · attention head ${hp.expert}`
      : hp.kind === "attn"
        ? `Layer ${hp.layer} · attention: each token gathers context from the earlier ones`
        : hp.kind === "router"
          ? `Layer ${hp.layer} · router: picks which experts each token goes to`
          : hp.kind === "embed"
            ? "Input: tokens become vectors here"
            : "Output: the next token is scored here";

  return (
    <div className={`nmap grand-${p.params ? Math.min(3, Math.floor(grandeur(p.params) * 4)) : 0}`} style={{ ["--grand" as any]: p.params ? grandeur(p.params).toFixed(2) : 0 }}>
      <canvas ref={canvas} className="nmap-canvas" />
      <div ref={labels} className="nmap-labels" aria-hidden>
        <span>Input</span>
        <span>Output</span>
        <span>Layer 0</span>
        <span>Layer {L - 1}</span>
      </div>
      <div className="nmap-top">
        <div className="nmap-title">
          <Icon name="brain" size={15} /> {p.title}
          {p.isLive && <span className="live-dot">live</span>}
          {!!p.params && (
            <span className="nmap-size" title={`${p.params.toLocaleString()} parameters`}>
              {p.params >= 1e9 ? `${(p.params / 1e9).toFixed(p.params >= 1e11 ? 0 : 1)}B` : `${Math.round(p.params / 1e6)}M`} · {sizeClass(p.params)}
            </span>
          )}
        </div>
        <div className="seg small">
          <button className={mode === "brain" ? "sel" : ""} onClick={() => setMode("brain")}>
            Brain
          </button>
          <button className={mode === "net" ? "sel" : ""} onClick={() => setMode("net")}>
            Layers
          </button>
        </div>
        <span className="grow" />
        {p.actions}
        <button className="nmap-btn" title="Reset the view (or double-click the map)" onClick={() => (view.current.reset = true)}>
          <Icon name="reload" size={13} />
        </button>
      </div>
      <div className="nmap-legend">
        <span>
          <i style={{ background: "#ff8a3d" }} /> {p.experts > 0 ? "Expert (lit when used)" : "Attention head"}
        </span>
        <span>
          <i style={{ background: "#9e80ff" }} /> Attention
        </span>
        {p.experts > 0 && (
          <span>
            <i style={{ background: "#ffc25c" }} /> Router
          </span>
        )}
        <span className="muted">Drag to turn · scroll to zoom · click a node</span>
      </div>
      {p.caption && <div className="nmap-caption">{p.caption}</div>}
      {hover && (
        <div className="nmap-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {tip}
        </div>
      )}
      {glErr && <div className="nmap-err">{glErr}</div>}
    </div>
  );
}
