// The brain inspector: a model's insides from its GGUF file. The
// architecture as a map you can click through, every tensor with what it
// does and what it reads and writes, weight plots for the small ones, and
// how often each expert has been picked (from Pelica's usage counts).
import { useEffect, useMemo, useRef, useState } from "react";
import { engine, type LocalModel } from "../lib/engine";
import { app, experts as live, useApp } from "../state";
import { Icon } from "./Icons";
import { NeuralMap, type MapPick } from "./NeuralMap";

interface T {
  name: string;
  dims: number[];
  type: string;
  bytes: number;
}

interface Info {
  arch: string;
  meta: Record<string, any>;
  tensors: T[];
  usage: number[] | null;
  bytes: number;
}

const fmtN = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));
const fmtB = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} KB`);
const prod = (d: number[]) => d.reduce((a, b) => a * b, 1);

interface Hyper {
  layers: number;
  d: number;
  heads: number;
  kv: number;
  experts: number;
  used: number;
  ff: number;
  ffe: number;
  ctx: number;
  vocab: number;
  name: string;
}

function hyper(i: Info): Hyper {
  const m = i.meta;
  const a = i.arch;
  const g = (k: string) => Number(m[`${a}.${k}`] ?? 0);
  const emb = i.tensors.find((t) => t.name === "token_embd.weight");
  return {
    layers: g("block_count"),
    d: g("embedding_length"),
    heads: g("attention.head_count"),
    kv: g("attention.head_count_kv") || g("attention.head_count"),
    experts: g("expert_count"),
    used: g("expert_used_count"),
    ff: g("feed_forward_length"),
    ffe: g("expert_feed_forward_length") || g("feed_forward_length"),
    ctx: g("context_length"),
    vocab: emb ? emb.dims[1] : 0,
    name: String(m["general.name"] || a),
  };
}

/** What a tensor does, and what it reads and writes. */
function role(name: string, t: T, h: Hyper): { what: string; io?: string; group: string } {
  const n = name.replace(/^blk\.\d+\./, "").replace(/\.weight$|\.bias$/, "");
  const bias = name.endsWith(".bias");
  const [c0, c1, c2] = t.dims;
  const io = c1 ? `reads ${c0} → writes ${c1}${c2 ? `, × ${c2} experts` : ""}` : `${c0} values`;
  const hd = h.heads ? Math.round(h.d / h.heads) : 0;
  const R: Record<string, [string, string]> = {
    token_embd: ["embed", `Token embedding: one learned ${c0}-number vector for each of the ${fmtN(c1 || 0)} tokens. Text enters the model here.`],
    output_norm: ["head", "Final RMS norm: rescales the hidden state's channels before the output head."],
    output: ["head", `Output head (unembedding): scores all ${fmtN(c1 || 0)} tokens from the final hidden state; the highest is the next token.`],
    attn_norm: ["attn", `Attention input norm (RMS): rescales each of the ${c0} channels before attention reads them.`],
    attn_q: ["attn", `Query projection: for each token, what it is looking for in the earlier ones. ${h.heads} heads × ${hd} dims.`],
    attn_k: ["attn", `Key projection: what each token offers to be found by.${h.kv && h.kv !== h.heads ? ` ${h.kv} key heads shared by ${h.heads / h.kv} query heads each (grouped-query attention).` : ""}`],
    attn_v: ["attn", "Value projection: the content a token passes on when another token attends to it."],
    attn_qkv: ["attn", "Query, key and value projections fused in one matrix."],
    attn_output: ["attn", `Attention output: mixes the heads' results back into the ${h.d}-wide residual stream.`],
    attn_q_norm: ["attn", "Query norm (QK-norm): per-head RMS norm on queries, keeps attention scores in range."],
    attn_k_norm: ["attn", "Key norm (QK-norm): per-head RMS norm on keys."],
    attn_sinks: ["attn", "Attention sinks: a learned extra score per head that can soak up attention."],
    ffn_norm: ["ffn", "Feed-forward input norm (RMS) before the experts."],
    post_attention_norm: ["ffn", "Norm applied after attention, before the feed-forward part."],
    ffn_gate_inp: ["router", `Router: scores all ${h.experts} experts for each token; the top ${h.used} run and their outputs are mixed by those scores.`],
    exp_probs_b: ["router", "Router bias: nudges which experts get picked (load balancing)."],
    ffn_gate_exps: ["experts", `Experts' gate matrices (${h.experts} experts): with the up matrices they form each expert's SwiGLU, deciding which of its ${c1} features fire.`],
    ffn_up_exps: ["experts", `Experts' up matrices: expand the token's ${c0} numbers to ${c1} features per expert.`],
    ffn_down_exps: ["experts", `Experts' down matrices: bring each expert's ${c0} features back to the ${c1}-wide stream.`],
    ffn_gate_shexp: ["shared", "Shared expert (gate): runs for every token alongside the routed experts."],
    ffn_up_shexp: ["shared", "Shared expert (up)."],
    ffn_down_shexp: ["shared", "Shared expert (down)."],
    ffn_gate_inp_shexp: ["shared", "Shared expert's gate: how much of the shared expert to mix in."],
    ffn_gate: ["ffn", "Dense feed-forward gate (SwiGLU)."],
    ffn_up: ["ffn", "Dense feed-forward up projection."],
    ffn_down: ["ffn", "Dense feed-forward down projection."],
    rope_freqs: ["embed", "Rotary position frequencies: how token positions are encoded into queries and keys."],
  };
  const r = R[n];
  if (!r) return { what: "A tensor this inspector doesn't describe yet.", io, group: "other" };
  return { what: (bias ? "Bias of the " : "") + r[1], io, group: r[0] };
}

const GROUPS: { key: string; label: string; icon: string; color: string }[] = [
  { key: "embed", label: "Embedding", icon: "layers", color: "#7aa2f7" },
  { key: "attn", label: "Attention", icon: "search", color: "#bb9af7" },
  { key: "router", label: "Router", icon: "spark", color: "#ff9e64" },
  { key: "experts", label: "Experts", icon: "cpu", color: "#e5533d" },
  { key: "shared", label: "Shared expert", icon: "cpu", color: "#e0af68" },
  { key: "ffn", label: "Feed-forward", icon: "bolt", color: "#9ece6a" },
  { key: "head", label: "Output", icon: "send", color: "#73daca" },
  { key: "other", label: "Other", icon: "info", color: "#888" },
];

function Spark({ path, name }: { path: string; name: string }) {
  const [v, setV] = useState<{ buckets: [number, number][]; mean: number; sd: number; n: number } | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setV(null);
    setErr("");
    engine.tensor(path, name).then(setV).catch((e) => setErr(e.message));
  }, [path, name]);
  if (err) return <div className="muted small">{err}</div>;
  if (!v) return <div className="muted small">Reading values…</div>;
  const lo = Math.min(...v.buckets.map((b) => b[0]));
  const hi = Math.max(...v.buckets.map((b) => b[1]));
  const W = 520;
  const H = 90;
  const y = (x: number) => H - ((x - lo) / (hi - lo || 1)) * (H - 4) - 2;
  const bw = W / v.buckets.length;
  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="spark">
        {lo < 0 && hi > 0 && <line x1="0" x2={W} y1={y(0)} y2={y(0)} stroke="var(--line)" />}
        {v.buckets.map((b, i) => (
          <rect key={i} x={i * bw} width={Math.max(1, bw - 0.5)} y={y(b[1])} height={Math.max(1, y(b[0]) - y(b[1]))} fill="var(--accent)" opacity=".8" />
        ))}
      </svg>
      <div className="muted small">
        {v.n.toLocaleString()} values · min {lo.toFixed(3)} · max {hi.toFixed(3)} · mean {v.mean.toFixed(4)} · sd {v.sd.toFixed(4)}
      </div>
    </div>
  );
}

function HeatMap({ usage, layers, experts, layer, onPick }: { usage: number[]; layers: number; experts: number; layer: number; onPick: (l: number, e: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<string>("");
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const g = c.getContext("2d")!;
    c.width = experts * 4;
    c.height = layers * 4;
    const max = Math.max(1, ...usage);
    for (let l = 0; l < layers; l++)
      for (let e = 0; e < experts; e++) {
        const v = Math.log1p(usage[l * experts + e] || 0) / Math.log1p(max);
        g.fillStyle = `hsl(${20 - v * 15}, ${40 + v * 55}%, ${8 + v * 55}%)`;
        g.fillRect(e * 4, l * 4, 4, 4);
      }
    g.strokeStyle = "rgba(255,255,255,.6)";
    g.strokeRect(0.5, layer * 4 + 0.5, experts * 4 - 1, 3);
  }, [usage, layers, experts, layer]);
  return (
    <div className="heat-wrap">
      <canvas
        ref={ref}
        className="heat"
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const ex = Math.floor(((e.clientX - r.left) / r.width) * experts);
          const ly = Math.floor(((e.clientY - r.top) / r.height) * layers);
          setHover(`layer ${ly} · expert ${ex} · picked ${(usage[ly * experts + ex] || 0).toLocaleString()} times`);
        }}
        onMouseLeave={() => setHover("")}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onPick(Math.floor(((e.clientY - r.top) / r.height) * layers), Math.floor(((e.clientX - r.left) / r.width) * experts));
        }}
      />
      <div className="muted small">{hover || "Rows: layers (top = first). Columns: experts. Brighter = picked more often. Click a row to inspect that layer."}</div>
    </div>
  );
}

/** One expert: its slices of the layer's expert tensors and how much it's used. */
function ExpertCard({ info, h, layer, expert, trace }: { info: Info; h: Hyper; layer: number; expert: number; trace: { tokens: string[]; routes: [number, number][][][] } | null }) {
  // the traced tokens routed here, strongest first
  const seen = trace
    ? trace.routes
        .map((r, i) => [i, r[layer]?.find((x) => x[0] === expert)?.[1] ?? 0] as [number, number])
        .filter((x) => x[1] > 0)
        .sort((a, b) => b[1] - a[1])
    : [];
  const ts = info.tensors.filter((t) => t.name.startsWith(`blk.${layer}.`) && /_exps\.weight$/.test(t.name));
  const bytes = ts.reduce((a, t) => a + t.bytes / h.experts, 0);
  const row = info.usage ? info.usage.slice(layer * h.experts, layer * h.experts + h.experts) : null;
  const n = row ? row[expert] : 0;
  const total = row ? row.reduce((a, b) => a + b, 0) : 0;
  const rank = row ? row.filter((x) => x > n).length + 1 : 0;
  const now = live.count.length === h.layers * h.experts ? live.count[layer * h.experts + expert] : 0;
  const part = (t: T) => (/gate/.test(t.name) ? "gate" : /up/.test(t.name) ? "up" : "down");
  return (
    <>
      <div className="mono">
        layer {layer} · expert {expert}
      </div>
      <p>
        One of {h.experts} specialists in this layer. When the router picks it (with {h.used - 1} others) for a token, it reads the token&apos;s {h.d} numbers, lights up {h.ffe} features of its own, and writes its answer back, weighted by how much the router
        trusts it.
      </p>
      <div className="kv">
        <span>Picked before</span>
        <b>{n.toLocaleString()} times</b>
        <span>Share of layer</span>
        <b>{total ? ((100 * n) / total).toFixed(2) : "0"}% (even split: {(100 / h.experts).toFixed(2)}%)</b>
        <span>Rank in layer</span>
        <b>{row ? `${rank} of ${h.experts}` : "?"}</b>
        <span>This session</span>
        <b>{now ? `${now.toLocaleString()} picks` : "not yet"}</b>
        <span>Size</span>
        <b>{fmtB(bytes)}</b>
        {ts.map((t) => (
          <span key={t.name} style={{ display: "contents" }}>
            <span>{part(t)}</span>
            <b>
              {t.dims[0]} → {t.dims[1]} · {t.type}
            </b>
          </span>
        ))}
      </div>
      {trace && (
        <div className="expert-seen">
          <div className="muted small">{seen.length ? `Traced tokens sent here (${seen.length}), strongest first:` : "None of the traced tokens came to this expert."}</div>
          <div className="trace-toks small">
            {seen.slice(0, 40).map(([i, w]) => (
              <span key={i} className="tok" style={{ opacity: 0.45 + 0.55 * Math.min(1, w * 2) }} title={`weight ${w.toFixed(3)}`}>
                {trace.tokens[i].replace(/\n/g, "⏎") || "·"}
              </span>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

export function Brain() {
  const st = useApp((s) => s.status);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [path, setPath] = useState<string>("");
  const [info, setInfo] = useState<Info | null>(null);
  const [err, setErr] = useState("");
  const [group, setGroup] = useState<string>("experts");
  const [layer, setLayer] = useState(0);
  const [sel, setSel] = useState<T | null>(null);
  const [pick, setPick] = useState<MapPick | null>(null);
  const [caption, setCaption] = useState("");
  const [watching, setWatching] = useState(false);
  const [traceText, setTraceText] = useState("The cat sat on the mat. def add(a, b): return a + b");
  const [trace, setTrace] = useState<{ tokens: string[]; routes: [number, number][][][] } | null>(null);
  const [traceErr, setTraceErr] = useState("");
  const [tracing, setTracing] = useState(false);
  const [tokSel, setTokSel] = useState<number | null>(null);
  const [tokHover, setTokHover] = useState<number | null>(null);
  const running = useApp((s) => s.running);

  useEffect(() => {
    engine.models().then((m) => {
      setModels(m.models);
      setPath((p) => p || st?.path || m.models[0]?.path || "");
    });
  }, []);
  useEffect(() => {
    if (!path) return;
    setInfo(null);
    setErr("");
    setSel(null);
    setPick(null);
    engine
      .gguf(path)
      .then(setInfo)
      .catch((e) => setErr(e.message));
  }, [path]);

  const h = useMemo(() => (info ? hyper(info) : null), [info]);
  const params = useMemo(() => (info ? info.tensors.reduce((a, t) => a + prod(t.dims), 0) : 0), [info]);
  const active = useMemo(() => {
    if (!info || !h?.experts) return params;
    const exp = info.tensors.filter((t) => /_exps\./.test(t.name)).reduce((a, t) => a + prod(t.dims), 0);
    return params - exp + (exp * h.used) / h.experts;
  }, [info, h, params]);
  const byGroup = useMemo(() => {
    const m = new Map<string, T[]>();
    if (info && h)
      for (const t of info.tensors) {
        const g = role(t.name, t, h).group;
        if (!m.has(g)) m.set(g, []);
        m.get(g)!.push(t);
      }
    return m;
  }, [info, h]);
  const isLive = st?.state === "ready" && !!st.path && st.path === path;

  /** A node clicked on the map: show its part of the model below. */
  function onPick(p: MapPick) {
    setPick(p);
    setSel(null);
    setLayer(p.layer);
    setGroup(p.kind === "expert" ? (h?.experts ? "experts" : "attn") : p.kind === "attn" ? "attn" : p.kind === "router" ? "router" : p.kind === "embed" ? "embed" : "head");
  }

  /** Runs a short prompt through the loaded model so the map lights up. */
  async function watch() {
    if (watching || running || !isLive) return;
    setWatching(true);
    setCaption("");
    let text = "";
    try {
      await engine.chat(
        { messages: [{ role: "user", content: "In three short sentences: how does a mixture-of-experts model decide which experts handle each word?" }], tools: [], max_tokens: 160, experts: true },
        (e) => {
          if (e.t === "experts") {
            if (e.layers && e.experts) live.grid(e.layers, e.experts);
            if (e.hits?.length) live.hit(e.hits);
          } else if (e.t === "text") {
            text += e.d;
            setCaption(text.replace(/\s+/g, " ").slice(-180));
          }
        },
      );
    } catch (e: any) {
      setCaption(String(e?.message || e));
    } finally {
      setWatching(false);
      setTimeout(() => setCaption(""), 6000);
    }
  }

  async function runTrace() {
    if (tracing || running || !isLive || !traceText.trim()) return;
    setTracing(true);
    setTraceErr("");
    try {
      const r = await engine.trace(traceText);
      setTrace(r);
      setTokSel(0);
    } catch (e: any) {
      setTraceErr(String(e?.message || e));
    } finally {
      setTracing(false);
    }
  }
  const tokShown = tokHover ?? tokSel;
  const route = trace && tokShown !== null ? trace.routes[tokShown] || null : null;

  const inLayer = (t: T) => !t.name.startsWith("blk.") || t.name.startsWith(`blk.${layer}.`);
  const list = (byGroup.get(group) || []).filter(inLayer);
  const groupBytes = (g: string) => (byGroup.get(g) || []).reduce((a, t) => a + t.bytes, 0);

  return (
    <div className="page brain">
      <div className="page-head">
        <h1>
          <Icon name="brain" size={22} /> Brain
        </h1>
        <select className="text-input" value={path} onChange={(e) => setPath(e.target.value)}>
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.name}
            </option>
          ))}
        </select>
      </div>
      {err && <div className="err-text">{err}</div>}
      {!info && !err && <div className="muted">Reading the model file…</div>}
      {info && h && (
        <>
          <NeuralMap
            key={path}
            params={params}
            layers={h.layers}
            experts={h.experts}
            heads={h.heads}
            usage={info.usage && info.usage.length === h.layers * h.experts ? info.usage : null}
            isLive={isLive}
            title={`${h.name} · ${h.layers} layers${h.experts ? ` × ${h.experts} experts` : ` × ${h.heads} heads`}`}
            sel={pick}
            route={route}
            onPick={onPick}
            caption={caption}
            actions={
              isLive ? (
                <button className="nmap-btn wide" disabled={watching || running} onClick={watch} title="Run a short prompt and watch the experts fire">
                  <Icon name={watching ? "spark" : "play"} size={12} /> {watching ? "Thinking…" : "Watch it think"}
                </button>
              ) : (
                <button className="nmap-btn wide" onClick={() => app.set({ view: "chat" })} title="Load this model from the model menu under the message box to see it work live">
                  Not loaded
                </button>
              )
            }
          />
          {h.experts > 0 && (
            <div className="trace-card">
              <div className="trace-head">
                <div>
                  <b>Trace a sentence</b>
                  <span className="muted small"> See which experts every token is sent to, layer by layer. Click a token to light its route on the map.</span>
                </div>
              </div>
              <div className="trace-row">
                <input className="text-input" value={traceText} onChange={(e) => setTraceText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runTrace()} placeholder="Type anything: prose, code, another language…" disabled={!isLive} />
                <button className="btn primary" onClick={runTrace} disabled={!isLive || tracing || running || !traceText.trim()}>
                  <Icon name={tracing ? "spark" : "scan"} size={14} /> {tracing ? "Tracing…" : "Trace"}
                </button>
              </div>
              {!isLive && <div className="muted small">Load this model (model menu under the message box) to trace it.</div>}
              {traceErr && <div className="err-text">{traceErr}</div>}
              {trace && (
                <div className="trace-toks" onMouseLeave={() => setTokHover(null)}>
                  {trace.tokens.map((t, i) => (
                    <button key={i} className={`tok ${i === tokSel ? "sel" : ""}`} onMouseEnter={() => setTokHover(i)} onClick={() => setTokSel(i === tokSel ? null : i)} title={`token ${i}`}>
                      {t.replace(/\n/g, "⏎") || "·"}
                    </button>
                  ))}
                </div>
              )}
              {trace && tokShown !== null && trace.routes[tokShown] && (
                <div className="muted small trace-sum">
                  “{trace.tokens[tokShown]}” → top expert per layer:{" "}
                  <span className="mono">
                    {trace.routes[tokShown]
                      .map((l) => l.slice().sort((a, b) => b[1] - a[1])[0]?.[0])
                      .slice(0, 16)
                      .join(" · ")}
                    {trace.routes[tokShown].length > 16 ? " …" : ""}
                  </span>
                </div>
              )}
            </div>
          )}
          <div className="brain-stats">
            {[
              ["Model", h.name],
              ["Architecture", info.arch],
              ["Parameters", fmtN(params)],
              ["Active per token", fmtN(active)],
              ["File", fmtB(info.bytes)],
              ["Layers", h.layers],
              ["Width", h.d],
              ["Heads", `${h.heads} q / ${h.kv} kv`],
              ...(h.experts ? [["Experts", `${h.experts} per layer, ${h.used} active`]] : []),
              ["Context", h.ctx ? fmtN(h.ctx) : "?"],
              ["Vocabulary", fmtN(h.vocab)],
            ].map(([k, v]) => (
              <div key={String(k)} className="stat">
                <span>{k}</span>
                <b>{v}</b>
              </div>
            ))}
          </div>
          <div className="brain-main">
            <div className="arch">
              <div className="arch-title">How a token flows</div>
              {GROUPS.filter((g) => byGroup.has(g.key)).map((g) => (
                <button key={g.key} className={`arch-block ${group === g.key ? "sel" : ""}`} style={{ "--gc": g.color } as React.CSSProperties} onClick={() => (setGroup(g.key), setSel(null))}>
                  <Icon name={g.icon} size={15} />
                  <span>{g.label}</span>
                  <span className="muted small">{fmtB(groupBytes(g.key))}</span>
                </button>
              ))}
              <div className="arch-layer">
                <label>
                  Layer {layer} of {h.layers - 1}
                </label>
                <input type="range" min={0} max={Math.max(0, h.layers - 1)} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
              </div>
            </div>
            <div className="tensor-list">
              {list.map((t) => {
                const r = role(t.name, t, h);
                return (
                  <button key={t.name} className={`tensor-row ${sel?.name === t.name ? "sel" : ""}`} onClick={() => setSel(t)}>
                    <span className="mono">{t.name}</span>
                    <span className="muted small">
                      {t.dims.join(" × ")} · {t.type} · {fmtB(t.bytes)}
                    </span>
                    <span className="tensor-what">{r.what}</span>
                  </button>
                );
              })}
              {!list.length && <div className="empty-note">Nothing of this kind in layer {layer}.</div>}
            </div>
            <div className="tensor-detail">
              {sel ? (
                <>
                  <div className="mono">{sel.name}</div>
                  <p>{role(sel.name, sel, h).what}</p>
                  <div className="kv">
                    <span>Shape</span>
                    <b>{sel.dims.join(" × ")}</b>
                    <span>Targets</span>
                    <b>{role(sel.name, sel, h).io}</b>
                    <span>Numbers</span>
                    <b>{fmtN(prod(sel.dims))}</b>
                    <span>Format</span>
                    <b>{sel.type}</b>
                    <span>Size</span>
                    <b>{fmtB(sel.bytes)}</b>
                    <span>Bits per number</span>
                    <b>{((sel.bytes * 8) / prod(sel.dims)).toFixed(2)}</b>
                  </div>
                  {(sel.type === "F32" || sel.type === "F16") && <Spark path={path} name={sel.name} />}
                  {/_exps\./.test(sel.name) && info.usage && h.experts > 0 && (
                    <div className="expert-bars">
                      <div className="muted small">How often each expert of layer {layer} was picked</div>
                      <div className="bars">
                        {Array.from({ length: h.experts }, (_, e) => {
                          const u = info.usage!.slice(layer * h.experts, layer * h.experts + h.experts);
                          const mx = Math.max(1, ...u);
                          return <span key={e} title={`expert ${e}: ${u[e]}`} style={{ height: `${(u[e] / mx) * 100}%` }} />;
                        })}
                      </div>
                    </div>
                  )}
                </>
              ) : pick?.kind === "expert" && h.experts > 0 && pick.expert !== undefined ? (
                <ExpertCard info={info} h={h} layer={pick.layer} expert={pick.expert} trace={trace} />
              ) : (
                <div className="muted">Pick a tensor, or a node on the map, to see what it does.</div>
              )}
            </div>
          </div>
          {info.usage && h.experts > 0 && info.usage.length === h.layers * h.experts && (
            <div className="brain-heat">
              <div className="arch-title">Expert heat: every expert's picks so far (Pelica keeps the hottest in VRAM)</div>
              <HeatMap usage={info.usage} layers={h.layers} experts={h.experts} layer={layer} onPick={(l) => (setLayer(l), setGroup("experts"))} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
