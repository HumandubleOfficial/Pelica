// The model library: models on this PC (load them), and GGUF models on
// Hugging Face to browse and download.
import { useEffect, useState } from "react";
import { bridge, type Download, type Hardware, type Repack } from "../lib/bridge";
import { engine, type LocalModel } from "../lib/engine";
import { bestFor, getFiles, gib, gpuOf, STARTERS, type Starter } from "../lib/starters";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const SUGGEST = ["Qwen3-30B-A3B", "OLMoE", "gpt-oss-20b", "Mixtral", "DeepSeek-V2-Lite", "granite-3.1 a800m", "Qwen3-Coder-30B-A3B", "ERNIE-4.5-21B-A3B", "Ling-lite", "SmallThinker"];
const moeish = (id: string) => /moe|a\d+(\.\d+)?b|mixtral|olmoe|gpt-oss|deepseek|qwen3-(30|235)|ernie-4\.5-21b|ling|granite.*a\d|smallthinker|phi-3\.5-moe|jamba|dbrx|grok|kimi|glm-4\.5/i.test(id);

/** This PC's GPU memory and RAM (bytes), once known. */
function useHardware(): { hw: Hardware | null; vram: number; ram: number } {
  const [hw, setHw] = useState<Hardware | null>(null);
  const gpu = useApp((s) => s.status?.gpu);
  useEffect(() => {
    bridge?.lib
      .hardware()
      .then(setHw)
      .catch(() => {});
  }, []);
  // (until it's known: a 16 GB GPU and 32 GB of RAM)
  return { hw, vram: hw ? gpuOf(hw, gpu).vram : 16e9, ram: hw?.ram || 32e9 };
}

/** How a model of `bytes` sits on this PC: all in the GPU's memory, shared with RAM, or too big. */
function fit(bytes: number, vram: number, ram: number): [string, string] {
  if (bytes < vram - 3e9) return ["fits", "Fits in VRAM: fastest"];
  if (bytes < vram + ram * 0.75) return ["split", "VRAM + RAM: experts that don't fit stay in RAM"];
  return ["big", "Larger than VRAM + RAM: may not load"];
}

const samePath = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
const mins = (s: number) => (s < 90 ? `${s} s` : `${Math.round(s / 60)} min`);
const cleanErr = (e: any) => String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");

/** Speeding a DeepSeek V4 file up (pelica repack), or putting it back: the
 *  main process asks first, unloads it and runs it in the background. */
async function repackModel(path: string, undo: boolean) {
  if (!bridge) return toast("Speeding a model up needs the Pelica desktop app.");
  try {
    await bridge.lib.repack(path, undo);
  } catch (e) {
    toast(cleanErr(e));
  }
}

/** Speed-ups running or finished this session; `onEnd` runs when one stops. */
function useRepacks(onEnd?: () => void): Repack[] {
  const [list, setList] = useState<Repack[]>([]);
  useEffect(() => {
    if (!bridge) return;
    bridge.lib
      .repacks()
      .then(setList)
      .catch(() => {});
    let prev: Repack[] = [];
    return bridge.lib.onRepack((r) => {
      if (r.some((x) => x.state !== "running" && prev.find((p) => p.path === x.path)?.state === "running")) onEnd?.();
      prev = r;
      setList(r);
    });
  }, []);
  return list;
}

function RepackProgress({ r }: { r: Repack }) {
  return (
    <div className="repack-prog">
      <div className="dl-bar">
        <span style={{ width: `${r.pct}%` }} />
      </div>
      <span className="muted small">
        {r.undo ? "Undoing the speed-up" : "Speeding up"}: {r.pct}%{r.left != null && r.pct > 0 ? ` · about ${mins(r.left)} left` : ""}
      </span>
    </div>
  );
}

function Local() {
  const brainMenu = useApp((s) => s.settings.brainMenu);
  const st = useApp((s) => s.status);
  const { vram, ram } = useHardware();
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [dir, setDir] = useState("");
  const [removing, setRemoving] = useState("");
  const refresh = () =>
    engine
      .models()
      .then((m) => (setModels(m.models), setDir(m.dir)))
      .catch(() => {});
  const repacks = useRepacks(refresh);
  useEffect(() => {
    refresh();
  }, [st?.state]);
  // asks first (a Windows dialog), unloads it if it's loaded, then its files go to the Recycle Bin
  async function uninstall(m: LocalModel) {
    if (!bridge) return toast("Uninstalling needs the Pelica desktop app.");
    setRemoving(m.path);
    try {
      const r = await bridge.lib.uninstall(m.path);
      if (!r) return;
      if (app.get().settings.lastModel === m.path) setSettings({ lastModel: "" });
      toast(r === "deleted" ? `Deleted ${m.name}: the space is free.` : `Uninstalled ${m.name}: it's in the Recycle Bin. Empty it to free the space.`);
      await refresh();
    } catch (e: any) {
      toast(cleanErr(e));
    } finally {
      setRemoving("");
    }
  }
  return (
    <div className="lib-list">
      <div className="muted small">Looking in {dir || "…"}</div>
      {models.map((m) => {
        const r = repacks.find((x) => samePath(x.path, m.path));
        const running = r?.state === "running";
        return (
          <div key={m.path} className={`lib-card ${st?.path === m.path ? "sel" : ""}`}>
            <div className="lib-card-main">
              <div className="lib-name">{m.name}</div>
              <div className="muted small mono">{m.path}</div>
              <div className="lib-tags">
                <span className="pill">{m.kind === "moe" ? "Mixture of experts" : m.kind === "gpu" ? "gpt-oss" : "Tiered (SSD)"}</span>
                <span className="pill">{gb(m.gb * 1e9)}</span>
                {m.max_ctx && <span className="pill">context up to {Math.round(m.max_ctx / 1024)}K</span>}
                <span className={`pill fit-${fit(m.gb * 1e9, vram, ram)[0]}`}>{fit(m.gb * 1e9, vram, ram)[1].split(":")[0]}</span>
                {m.repack === "yes" && !running && (
                  <span className="pill on" title="Rearranged with pelica repack: each expert loads in one read">
                    Sped up
                  </span>
                )}
                {(m.repack === "partial" || m.repack === "unfinished") && !running && (
                  <span className="pill fit-split" title={m.repack === "unfinished" ? "It was stopped partway. It can't be loaded until it's finished." : "Only some layers are sped up. It loads fine; finishing makes the rest fast too."}>
                    {m.repack === "unfinished" ? "Speed-up not finished" : "Partly sped up"}
                  </span>
                )}
              </div>
              {running && <RepackProgress r={r} />}
              {r?.state === "error" && <div className="err-text small">{r.error}</div>}
            </div>
            <div className="lib-actions">
              {st?.path === m.path && st.state === "ready" ? (
                <span className="pill on">Loaded</span>
              ) : (
                <button
                  className="btn"
                  disabled={st?.state === "loading" || running || m.repack === "unfinished"}
                  title={running ? "Wait for the speed-up to finish" : m.repack === "unfinished" ? "Finish the speed-up first" : undefined}
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
              {m.repack &&
                (running ? (
                  <button className="btn ghost small" onClick={() => bridge?.lib.repackStop(m.path)} title="Stop. It's safe: start it again later to finish.">
                    Stop
                  </button>
                ) : m.repack === "yes" ? (
                  <button className="btn ghost small" onClick={() => repackModel(m.path, true)} title="Put the file back the way it was, so other programs can open it">
                    Undo speed-up
                  </button>
                ) : (
                  <button className="btn small" onClick={() => repackModel(m.path, m.repack === "unfinished" && !!r?.undo)} title="Rearrange the file so each expert loads in one read: answers about 20% faster">
                    <Icon name="bolt" size={13} /> {m.repack === "partial" || m.repack === "unfinished" ? (r?.undo ? "Finish undoing" : "Finish speed-up") : "Speed up"}
                  </button>
                ))}
              <button className="btn ghost small" onClick={() => app.set({ compare: m.name })} title="Compare with flagship models">
                <Icon name="compare" size={14} />
              </button>
              {brainMenu && (
                <button className="btn ghost small" onClick={() => app.set({ view: "brain" })} title="Look inside">
                  <Icon name="brain" size={14} />
                </button>
              )}
              <button className="btn ghost small lib-uninstall" disabled={!!removing || running || (st?.state === "loading" && st.path === m.path)} onClick={() => uninstall(m)} title="Uninstall (moves it to the Recycle Bin)">
                <Icon name="trash" size={14} />
              </button>
            </div>
          </div>
        );
      })}
      {!models.length && (
        <div className="empty-note">
          No models yet.{" "}
          <button className="btn small primary" onClick={() => app.set({ libTab: "discover" })}>
            See the ones suggested for this PC
          </button>
        </div>
      )}
    </div>
  );
}

/** One suggested model: how it runs here, and Get / its download / Load. */
function StarterCard({ m, best, vram, ram, have, downloads, onGet }: { m: Starter; best: boolean; vram: number; ram: number; have?: LocalModel; downloads: Download[]; onGet: (m: Starter) => void }) {
  const st = useApp((s) => s.status);
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const wont = m.wont(vram, ram);
  const mine = downloads.filter((d) => d.repo === m.repo && m.files.includes(d.file));
  const going = mine.filter((d) => d.state === "running" || d.state === "paused" || d.state === "error");
  const got = mine.reduce((n, d) => n + d.got, 0);
  return (
    <div className={`starter ${best ? "best" : ""} ${wont ? "wont" : ""}`}>
      <div className="starter-head">
        <b>{m.name}</b>
        {best && <span className="pill on">Best fit</span>}
      </div>
      <div className="muted small">{m.blurb}</div>
      <div className="lib-tags">
        <span className="pill">{gb(m.size)}</span>
        <span className="pill">{m.tools ? "Uses tools and files" : "Chat only"}</span>
        {m.vision && <span className="pill">Sees pictures</span>}
      </div>
      {wont && <div className="warn-note small">Too big for this PC: {wont}.</div>}
      <div className="starter-foot">
        {have ? (
          st?.path === have.path && st.state === "ready" ? (
            <span className="pill on">Loaded</span>
          ) : (
            <button
              className="btn small"
              disabled={st?.state === "loading"}
              onClick={async () => {
                try {
                  await engine.load(have.path, Math.min(lastCtx, have.max_ctx || lastCtx));
                  setSettings({ lastModel: have.path });
                  app.set({ view: "chat" });
                } catch (e: any) {
                  toast(cleanErr(e));
                }
              }}
            >
              {st?.state === "loading" && st.path === have.path ? "Loading…" : "Load"}
            </button>
          )
        ) : going.length ? (
          <span className="muted small">
            {going.some((d) => d.state === "running") ? "Downloading" : going.some((d) => d.state === "error") ? "Download stopped" : "Paused"}: {gb(got)} of {gb(m.size)} (below)
          </span>
        ) : (
          <button className={`btn small ${best ? "primary" : ""}`} onClick={() => onGet(m)}>
            <Icon name="download" size={13} /> Get
          </button>
        )}
      </div>
    </div>
  );
}

function Discover() {
  const { hw, vram, ram } = useHardware();
  const dl = hw?.downloadDir || "…";
  const [q, setQ] = useState("Qwen3-30B-A3B");
  const [hits, setHits] = useState<{ id: string; downloads: number; likes: number; updated: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [repo, setRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<{ path: string; size: number }[] | null>(null);
  const [downloads, setDownloads] = useState<Download[]>([]);
  const auto = useApp((s) => s.settings.autoRepack);
  const repacks = useRepacks();
  // (to offer a speed-up on a finished DeepSeek V4 download)
  const [models, setModels] = useState<LocalModel[]>([]);
  const finished = downloads.filter((d) => d.state === "done").length + repacks.filter((r) => r.state !== "running").length;
  useEffect(() => {
    engine
      .models()
      .then((m) => setModels(m.models))
      .catch(() => {});
  }, [finished]);
  const best = bestFor(vram, ram);
  // a suggested model already here: its first file is among the models
  const haveOf = (m: Starter) => models.find((x) => m.files.some((f) => samePath(x.path.split(/[\\/]/).pop() || "", f.split("/").pop()!)));
  async function get(repo: string, files: { path: string; size: number }[]) {
    const why = await getFiles(repo, files, downloads).catch((e) => cleanErr(e));
    if (why) toast(why);
  }
  // a starter's real sizes come from Hugging Face
  async function getStarter(m: Starter) {
    const listed = await bridge!.lib.files(m.repo).catch(() => [] as { path: string; size: number }[]);
    await get(
      m.repo,
      m.files.map((f) => ({ path: f, size: listed.find((x) => x.path === f)?.size || m.size / m.files.length })),
    );
  }

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
    bridge.lib.hardware().then((h) => bridge!.lib.downloads(h.downloadDir).then(setDownloads));
    return bridge.lib.onProgress(setDownloads);
  }, []);
  useEffect(() => {
    setFiles(null);
    if (repo && bridge)
      bridge.lib
        .files(repo)
        .then(setFiles)
        .catch((e) => setErr(e.message));
  }, [repo]);

  return (
    <div className="discover">
      <div className="set-title starters-title">Suggested for this PC</div>
      <div className="muted small">
        {hw ? `${gpuOf(hw, app.get().status?.gpu).name}, ${gib(vram)} of GPU memory, ${gib(ram)} of RAM. ` : ""}These are the models Pelica is tested with.
      </div>
      <div className="starters">
        {STARTERS.map((m) => (
          <StarterCard key={m.id} m={m} best={m === best} vram={vram} ram={ram} have={haveOf(m)} downloads={downloads} onGet={getStarter} />
        ))}
      </div>
      <div className="set-title starters-title">More on Hugging Face</div>
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
      <label className="switch-row" title="Rearranges the file so each expert loads in one read. Other programs can't open it afterwards; Undo speed-up under On this PC puts it back.">
        <button className={`toggle small ${auto ? "on" : ""}`} role="switch" aria-checked={auto} onClick={() => setSettings({ autoRepack: !auto })} />
        <span>Speed up DeepSeek V4 and Qwen3.8 models when they finish downloading (answers about 20% faster)</span>
      </label>
      {err && <div className="err-text">{err}</div>}
      {downloads.length > 0 && (
        <div className="downloads">
          {downloads.map((d) => {
            const m = d.state === "done" ? models.find((x) => samePath(x.path, d.dest)) : undefined;
            const r = repacks.find((x) => samePath(x.path, d.dest));
            return (
              <div key={d.id} className={`dl-row ${d.state}`}>
                <div className="dl-name">
                  {d.file.split("/").pop()} <span className="muted small">{d.repo}</span>
                </div>
                <div className="dl-bar">
                  <span
                    style={{
                      width: `${d.total ? (d.got / d.total) * 100 : 0}%`,
                    }}
                  />
                </div>
                <div className="muted small">
                  {d.state === "running"
                    ? `${gb(d.got)} / ${gb(d.total)}`
                    : d.state === "paused"
                      ? `Paused at ${gb(d.got)}${d.total ? ` / ${gb(d.total)}` : ""}`
                      : d.state === "done"
                        ? r?.state === "running"
                          ? `Done · speeding up: ${r.pct}%`
                          : m?.repack === "yes"
                            ? "Done and sped up: find it under On this PC"
                            : "Done: find it under On this PC"
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
                {m?.repack === "no" && r?.state !== "running" && (
                  <button className="btn small" onClick={() => repackModel(m.path, false)} title="Rearrange the file so each expert loads in one read: answers about 20% faster">
                    <Icon name="bolt" size={13} /> Speed up
                  </button>
                )}
              </div>
            );
          })}
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
                const [cls, why] = fit(f.size, vram, ram);
                return (
                  <div key={f.path} className="file-row">
                    <span className="mono">{f.path}</span>
                    <span className={`pill fit-${cls}`} title={why}>
                      {gb(f.size)}
                    </span>
                    <button className="btn small" onClick={() => get(repo, [f])}>
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
  const tab = useApp((s) => s.libTab);
  const setTab = (libTab: "local" | "discover") => app.set({ libTab });
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
