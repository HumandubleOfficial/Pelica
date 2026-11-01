// Model comparison: a local model's published benchmark scores next to the
// flagship models' (GPT-6 Astra, Claude Fable 5.1, Claude Opus 5.5, Kimi K3,
// GLM 5.3). Every number has a source; a score nobody published shows as
// n/a rather than a guess.
import { useEffect, useMemo, useState } from "react";
import { engine, type LocalModel } from "../lib/engine";
import { app, openLink, useApp } from "../state";
import { Icon } from "./Icons";

type Bench = "vals" | "gpqa" | "mmlupro" | "swe" | "lcb" | "tb4" | "ioi";
type Src = "vals" | "vendor";

interface Entry {
  id: string;
  name: string;
  maker: string;
  /** the source of every score in this entry */
  src: Src;
  url: string;
  scores: Partial<Record<Bench, number>>;
  /** what the vendor's numbers were measured as (when src is vendor) */
  note?: string;
  /** matches a local model's file or display name */
  match?: RegExp;
}

const BENCHES: { id: Bench; label: string; what: string }[] = [
  { id: "vals", label: "Vals Index", what: "Vals AI's overall score across its real-world tasks (finance, law, coding, medicine…)" },
  { id: "gpqa", label: "GPQA Diamond", what: "Graduate-level science questions written to be hard to look up" },
  { id: "mmlupro", label: "MMLU-Pro", what: "Broad knowledge and reasoning across 14 subjects" },
  { id: "swe", label: "SWE-bench", what: "Fixing real GitHub issues in Python projects (Verified set)" },
  { id: "lcb", label: "LiveCodeBench", what: "Fresh competitive-programming problems" },
  { id: "tb4", label: "Terminal-Bench 4.0", what: "Long multi-step tasks done in a terminal" },
  { id: "ioi", label: "IOI", what: "International Olympiad in Informatics problems" },
];

const VALS = "https://www.vals.ai/models/";

export const FLAGSHIPS: Entry[] = [
  { id: "astra", name: "GPT-6 Astra", maker: "OpenAI", src: "vals", url: VALS + "openai_gpt-6-astra", scores: { vals: 65.02, tb4: 59.6, ioi: 100 } },
  { id: "fable", name: "Claude Fable 5.1", maker: "Anthropic", src: "vals", url: VALS + "anthropic_claude-fable-5-1", scores: { vals: 67.87, gpqa: 93.43, mmlupro: 92.38, lcb: 90.52, tb4: 58.08, ioi: 90.78 } },
  { id: "opus", name: "Claude Opus 5.5", maker: "Anthropic", src: "vals", url: VALS + "anthropic_claude-opus-5-5", scores: { vals: 70.18, tb4: 65.15, ioi: 95.06 } },
  { id: "kimi", name: "Kimi K3", maker: "Moonshot AI", src: "vals", url: VALS + "kimi_kimi-k3", scores: { vals: 52.95, gpqa: 92.93, mmlupro: 87.97, swe: 93.4, lcb: 87.19, tb4: 17.17, ioi: 48.94 } },
  { id: "glm", name: "GLM 5.3", maker: "Z.ai", src: "vals", url: VALS + "zai_glm-5.3", scores: { vals: 59.73, gpqa: 88.13, mmlupro: 86.77, swe: 95.4, lcb: 80.53, tb4: 38.89, ioi: 68.44 } },
];

/** Models Pelica runs, with the scores published for them. */
const LOCAL: Entry[] = [
  { id: "glm53flash", name: "GLM 5.3 Flash", maker: "Z.ai", src: "vals", url: VALS + "zai_glm-5.3-flash", match: /glm.?5\.?3.?flash|glm53.?flash/i, scores: { vals: 50.86, gpqa: 86.36, mmlupro: 86.06, swe: 92.0, lcb: 80.51, tb4: 37.37, ioi: 52.5 } },
  { id: "qwen3-30b-2507", name: "Qwen3-30B-A3B-Instruct-2507", maker: "Alibaba Qwen", src: "vendor", url: "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507", match: /qwen3.?30b.?a3b.?instruct.?2507/i, note: "Qwen's model card for Instruct-2507; LiveCodeBench v6", scores: { gpqa: 70.4, mmlupro: 78.4, lcb: 43.2 } },
  { id: "qwen3-30b", name: "Qwen3-30B-A3B", maker: "Alibaba Qwen", src: "vendor", url: "https://arxiv.org/abs/2505.09388", match: /qwen3.?30b.?a3b(?!.*2507)/i, note: "Qwen3 technical report, thinking mode (Table 15); LiveCodeBench v5", scores: { gpqa: 65.8, lcb: 62.6 } },
  { id: "gpt-oss-20b", name: "gpt-oss-20b", maker: "OpenAI", src: "vendor", url: "https://arxiv.org/abs/2508.10925", match: /gpt.?oss.?20b/i, note: "OpenAI's model card, high reasoning, no tools (Table 3)", scores: { gpqa: 71.5, swe: 60.7 } },
  { id: "olmoe", name: "OLMoE 1B-7B", maker: "Ai2", src: "vendor", url: "https://huggingface.co/allenai/OLMoE-1B-7B-0924", match: /olmoe/i, note: "No scores on these benchmarks were published", scores: {} },
  { id: "dsv4flash", name: "DeepSeek V4 Flash", maker: "DeepSeek", src: "vendor", url: "https://huggingface.co/deepseek-ai", match: /deepseek.?v4.?flash/i, note: "No independent scores for this release (and a modified, abliterated copy may score differently)", scores: {} },
];

const CHECKED = "23 Oct 2026";

function localEntry(name: string): Entry | null {
  return LOCAL.find((e) => e.match!.test(name)) || null;
}

const COLORS = ["var(--cmp-0)", "var(--cmp-1)", "var(--cmp-2)", "var(--cmp-3)", "var(--cmp-4)", "var(--cmp-5)"];

export function Compare() {
  const which = useApp((s) => s.compare);
  const st = useApp((s) => s.status);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [mine, setMine] = useState<string>("");
  const [picked, setPicked] = useState<string[]>(["opus"]);
  useEffect(() => {
    if (which === null) return;
    engine.models().then((m) => setModels(m.models)).catch(() => {});
    setMine(which || st?.model || "");
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ compare: null });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [which]);
  const options = useMemo(() => {
    const names = new Set<string>(models.map((m) => m.name));
    if (mine) names.add(mine);
    return [...names];
  }, [models, mine]);
  if (which === null) return null;

  const me = localEntry(mine);
  const rows: { e: Entry; label: string; color: string; local: boolean }[] = [];
  if (me && Object.keys(me.scores).length) rows.push({ e: me, label: mine && me.name.toLowerCase() !== mine.toLowerCase() ? `${mine}` : me.name, color: COLORS[0], local: true });
  FLAGSHIPS.forEach((f, i) => picked.includes(f.id) && rows.push({ e: f, label: f.name, color: COLORS[i + 1], local: false }));
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const shown = BENCHES.filter((b) => rows.some((r) => r.e.scores[b.id] !== undefined));

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ compare: null })}>
      <div className="modal compare">
        <div className="pm-head">
          <span className="pm-badge">
            <Icon name="compare" size={16} />
          </span>
          <div>
            <h2>Compare models</h2>
            <div className="muted small">Published benchmark scores, side by side. Higher is better.</div>
          </div>
          <span className="grow" />
          <button className="ghost-btn" onClick={() => app.set({ compare: null })} title="Close (Esc)">
            <Icon name="x" size={16} />
          </button>
        </div>

        <div className="cmp-pick">
          <label className="cmp-mine">
            <span className="muted small">Your model</span>
            <select value={mine} onChange={(e) => setMine(e.target.value)}>
              {!mine && <option value="">Choose a model</option>}
              {options.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <div className="cmp-vs">vs</div>
          <div className="cmp-flags">
            {FLAGSHIPS.map((f, i) => (
              <button key={f.id} className={`cmp-flag ${picked.includes(f.id) ? "sel" : ""}`} style={{ ["--c" as any]: COLORS[i + 1] }} onClick={() => toggle(f.id)}>
                <span className="cmp-swatch" />
                <span>
                  <b>{f.name}</b>
                  <small className="muted">{f.maker}</small>
                </span>
              </button>
            ))}
          </div>
        </div>

        {!me && mine && <div className="agent-hint">No published scores found for {mine} on these benchmarks. The flagships are shown on their own.</div>}
        {me && !Object.keys(me.scores).length && <div className="agent-hint">{me.name}: {me.note}.</div>}

        <div className="cmp-chart">
          {shown.length === 0 && <div className="muted pad">Pick a flagship above to see scores.</div>}
          {shown.map((b) => (
            <div key={b.id} className="cmp-bench">
              <div className="cmp-bench-head" title={b.what}>
                <b>{b.label}</b>
                <span className="muted small">{b.what}</span>
              </div>
              {rows.map((r) => {
                const v = r.e.scores[b.id];
                return (
                  <div key={r.e.id} className={`cmp-bar-row ${r.local ? "local" : ""}`}>
                    <span className="cmp-name">{r.label}</span>
                    <span className="cmp-track">
                      {v !== undefined ? <span className={`cmp-bar ${r.e.src === "vendor" ? "vendor" : ""}`} style={{ width: `${Math.max(1.5, v)}%`, background: r.color }} /> : <span className="cmp-na">n/a</span>}
                    </span>
                    <span className="cmp-val">{v !== undefined ? `${v.toFixed(1)}%` : ""}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="cmp-foot muted small">
          <p>
            Scores from <a onClick={() => openLink("https://www.vals.ai")}>Vals AI</a>&apos;s independent runs where they exist (checked {CHECKED}).{" "}
            {rows.some((r) => r.e.src === "vendor" && Object.keys(r.e.scores).length) && (
              <>
                <span className="cmp-legend vendor" /> Striped bars are the model maker&apos;s own numbers ({rows.filter((r) => r.e.src === "vendor").map((r) => r.e.note).join("; ")}): a different setup, so treat the gap as rough.{" "}
              </>
            )}
            n/a means no score was published there. Pelica runs quantized copies, which can score a little lower than the originals.
          </p>
          <div className="cmp-sources">
            {rows.map((r) => (
              <a key={r.e.id} onClick={() => openLink(r.e.url)}>
                <Icon name="external" size={11} /> {r.e.name}
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
