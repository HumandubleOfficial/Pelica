// The model library: models on this PC (load them), and GGUF models on
// Hugging Face to browse and download.
import { useEffect, useState } from "react";
import { bridge, type Download } from "../lib/bridge";
import { engine, type LocalModel } from "../lib/engine";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const SUGGEST = ["Qwen3-30B-A3B", "OLMoE", "gpt-oss-20b", "Mixtral", "DeepSeek-V2-Lite", "granite-3.1 a800m", "Qwen3-Coder-30B-A3B", "ERNIE-4.5-21B-A3B", "Ling-lite", "SmallThinker"];
const moeish = (id: string) => /moe|a\d+(\.\d+)?b|mixtral|olmoe|gpt-oss|deepseek|qwen3-(30|235)|ernie-4\.5-21b|ling|granite.*a\d|smallthinker|phi-3\.5-moe|jamba|dbrx|grok|kimi|glm-4\.5/i.test(id);

function fit(bytes: number): [string, string] {
  if (bytes < 13e9) return ["fits", "Fits in VRAM: fastest"];
  if (bytes < 42e9) return ["split", "VRAM + RAM: experts that don't fit stay in RAM"];
  return ["big", "Larger than VRAM + RAM: may not load"];
}

function Local() {
  const st = useApp((s) => s.status);
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [dir, setDir] = useState("");
  useEffect(() => {
    engine.models().then((m) => (setModels(m.models), setDir(m.dir))).catch(() => {});
  }, [st?.state]);
  return (
    <div className="lib-list">
      <div className="muted small">Looking in {dir || "…"}</div>
      {models.map((m) => (
        <div key={m.path} className={`lib-card ${st?.path === m.path ? "sel" : ""}`}>
          <div className="lib-card-main">
            <div className="lib-name">{m.name}</div>
            <div className="muted small mono">{m.path}</div>
            <div className="lib-tags">
              <span className="pill">{m.kind === "moe" ? "Mixture of experts" : m.kind === "gpu" ? "gpt-oss" : "Tiered (SSD)"}</span>
              <span className="pill">{m.gb.toFixed(1)} GB</span>
              {m.max_ctx && <span className="pill">context up to {Math.round(m.max_ctx / 1024)}K</span>}
              <span className={`pill fit-${fit(m.gb * 1e9)[0]}`}>{fit(m.gb * 1e9)[1].split(":")[0]}</span>
            </div>
          </div>
          <div className="lib-actions">
            {st?.path === m.path && st.state === "ready" ? (
              <span className="pill on">Loaded</span>
            ) : (
              <button
                className="btn"
                disabled={st?.state === "loading"}
                onClick={async () => {
                  try {
                    await engine.load(m.path, Math.min(lastCtx, m.max_ctx || lastCtx));
                    setSettings({ lastModel: m.path });
                    app.set({ view: "chat" });
                  } catch (e: any) {
                    toast(e.message);
                  }
                }}
              >
                {st?.state === "loading" && st.path === m.path ? "Loading…" : "Load"}
              </button>
            )}
            <button className="btn ghost small" onClick={() => app.set({ compare: m.name })} title="Compare with flagship models">
              <Icon name="compare" size={14} />
            </button>
            <button className="btn ghost small" onClick={() => app.set({ view: "brain" })} title="Look inside">
              <Icon name="brain" size={14} />
            </button>
          </div>
        </div>
      ))}
      {!models.length && <div className="empty-note">No models yet. Find one under Discover.</div>}
    </div>
  );
}

function Discover() {
  const dl = useApp((s) => s.settings.downloadDir);
  const [q, setQ] = useState("Qwen3-30B-A3B");
  const [hits, setHits] = useState<{ id: string; downloads: number; likes: number; updated: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [repo, setRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<{ path: string; size: number }[] | null>(null);
  const [downloads, setDownloads] = useState<Download[]>([]);

  async function search(text = q) {
    if (!bridge) return setErr("Browsing needs the Pelica desktop app.");
    setBusy(true);
    setErr("");
    try {
      setHits(await bridge.lib.search(text));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    search();
    if (!bridge) return;
    bridge.lib.downloads(dl).then(setDownloads);
    return bridge.lib.onProgress(setDownloads);
  }, []);
  useEffect(() => {
    setFiles(null);
    if (repo && bridge) bridge.lib.files(repo).then(setFiles).catch((e) => setErr(e.message));
  }, [repo]);

  return (
    <div className="discover">
      <div className="lib-search">
        <Icon name="search" size={16} />
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Search GGUF models on Hugging Face" />
        <button className="btn small" onClick={() => search()} disabled={busy}>
          {busy ? "…" : "Search"}
        </button>
      </div>
      <div className="lib-suggest">
        {SUGGEST.map((s) => (
          <button key={s} className="chip" onClick={() => (setQ(s), search(s))}>
            {s}
          </button>
        ))}
      </div>
      <p className="muted small">Pelica runs mixture-of-experts models (and gpt-oss). Experts that don't fit in your GPU's memory stay in RAM. Downloads go to {dl}.</p>
      {err && <div className="err-text">{err}</div>}
      {downloads.length > 0 && (
        <div className="downloads">
          {downloads.map((d) => (
            <div key={d.id} className={`dl-row ${d.state}`}>
              <div className="dl-name">
                {d.file.split("/").pop()} <span className="muted small">{d.repo}</span>
              </div>
              <div className="dl-bar">
                <span style={{ width: `${d.total ? (d.got / d.total) * 100 : 0}%` }} />
              </div>
              <div className="muted small">
                {d.state === "running"
                  ? `${gb(d.got)} / ${gb(d.total)}`
                  : d.state === "paused"
                    ? `Paused at ${gb(d.got)}${d.total ? ` / ${gb(d.total)}` : ""}`
                    : d.state === "done"
                      ? "Done: find it under On this PC"
                      : d.state === "error"
                        ? d.error
                        : "Cancelled"}
              </div>
              {d.state === "running" && (
                <button className="ghost-btn" onClick={() => bridge?.lib.pause(d.id)} title="Pause (resume any time, even after closing Pelica)">
                  <Icon name="pause" size={14} />
                </button>
              )}
              {(d.state === "paused" || d.state === "error") && (
                <button className="ghost-btn" onClick={() => bridge?.lib.resume(d.id)} title="Resume">
                  <Icon name="play" size={14} />
                </button>
              )}
              {(d.state === "running" || d.state === "paused" || d.state === "error") && (
                <button className="ghost-btn" onClick={() => bridge?.lib.cancel(d.id)} title="Cancel (the partial file goes to the Recycle Bin)">
                  <Icon name="x" size={14} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
      <div className="lib-split">
        <div className="lib-list">
          {hits.map((h) => (
            <button key={h.id} className={`repo-row ${repo === h.id ? "sel" : ""}`} onClick={() => setRepo(h.id)}>
              <span className="repo-id">{h.id}</span>
              <span className="muted small">
                ↓ {h.downloads.toLocaleString()} · ♥ {h.likes}
                {moeish(h.id) ? " · MoE" : ""}
              </span>
            </button>
          ))}
          {!hits.length && !busy && <div className="empty-note">No results.</div>}
        </div>
        <div className="lib-files">
          {!repo && <div className="empty-note">Pick a model to see its files.</div>}
          {repo && (
            <>
              <div className="lib-name">{repo}</div>
              {!moeish(repo) && <div className="warn-note">This doesn't look like a mixture-of-experts model; Pelica may not run it.</div>}
              {!files && <div className="muted">Loading files…</div>}
              {files?.map((f) => {
                const [cls, why] = fit(f.size);
                return (
                  <div key={f.path} className="file-row">
                    <span className="mono">{f.path}</span>
                    <span className={`pill fit-${cls}`} title={why}>
                      {gb(f.size)}
                    </span>
                    <button className="btn small" onClick={() => bridge?.lib.download(repo, f.path, dl).catch((e: any) => toast(e.message))}>
                      <Icon name="download" size={13} /> Get
                    </button>
                  </div>
                );
              })}
              {files && !files.length && <div className="muted">No GGUF files in this repository.</div>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function Library() {
  const [tab, setTab] = useState<"local" | "discover">("local");
  return (
    <div className="page">
      <div className="page-head">
        <h1>Models</h1>
        <div className="seg">
          <button className={tab === "local" ? "sel" : ""} onClick={() => setTab("local")}>
            On this PC
          </button>
          <button className={tab === "discover" ? "sel" : ""} onClick={() => setTab("discover")}>
            Discover
          </button>
        </div>
      </div>
      {tab === "local" ? <Local /> : <Discover />}
    </div>
  );
}
