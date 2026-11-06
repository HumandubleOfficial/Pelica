// The model library: models on this PC (load them), and GGUF models on
// Hugging Face to browse and download.
import { useEffect, useRef, useState } from "react";
import { bridge, type Download, type Hardware, type Repack } from "../lib/bridge";
import { engine, type LocalModel } from "../lib/engine";
import { bestFor, getFiles, getStarter as fetchStarter, gib, gpuOf, STARTERS, type Starter } from "../lib/starters";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";
import { Menu, type MenuItem } from "./Menu";

const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const SUGGEST = ["Qwen3-30B-A3B", "OLMoE", "gpt-oss-20b", "Mixtral", "DeepSeek-V2-Lite", "granite-3.1 a800m", "Qwen3-Coder-30B-A3B", "ERNIE-4.5-21B-A3B", "Ling-lite", "SmallThinker"];
const shortN = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n));
const KINDS: Record<string, string> = { moe: "Mixture of experts", gpu: "gpt-oss" };
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

/** Loads a model and goes to the chat. */
async function loadModel(m: LocalModel) {
  const ctx = app.get().settings.lastCtx;
  try {
    await engine.load(m.path, Math.min(ctx, m.max_ctx || ctx));
    setSettings({ lastModel: m.path });
    app.set({ view: "chat" });
  } catch (e) {
    toast(cleanErr(e));
  }
}

/** What this PC has to run models with. */
function PcStrip({ hw, vram, ram, children }: { hw: Hardware | null; vram: number; ram: number; children?: React.ReactNode }) {
  const gpu = useApp((s) => s.status?.gpu);
  return (
    <div className="pc-strip">
      <Icon name="monitor" size={15} />
      <span className="pc-strip-gpu">{hw ? gpuOf(hw, gpu).name : gpu || "This PC"}</span>
      <span className="pc-strip-sep" />
      <span>{gib(vram)} GPU memory</span>
      <span className="pc-strip-sep" />
      <span>{gib(ram)} RAM</span>
      {children}
    </div>
  );
}

/** Downloads, kept up to date. */
function useDownloads(): Download[] {
  const [list, setList] = useState<Download[]>([]);
  useEffect(() => {
    if (!bridge) return;
    bridge.lib
      .hardware()
      .then((h) => bridge!.lib.downloads(h.downloadDir).then(setList))
      .catch(() => {});
    return bridge.lib.onProgress(setList);
  }, []);
  return list;
}

function DownloadRow({ d, m, r }: { d: Download; m?: LocalModel; r?: Repack }) {
  return (
    <div className={`dl-row ${d.state}`}>
      <div className="dl-name">
        {d.file.split("/").pop()} <span className="muted small">{d.repo}</span>
      </div>
      <div className="dl-bar">
        <span style={{ width: `${d.total ? (d.got / d.total) * 100 : 0}%` }} />
      </div>
      <div className="muted small">
        {d.state === "running"
          ? `${gb(d.got)} of ${gb(d.total)}`
          : d.state === "paused"
            ? `Paused at ${gb(d.got)}${d.total ? ` of ${gb(d.total)}` : ""}`
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
      {d.state === "paused" && (
        <button className="ghost-btn" onClick={() => bridge?.lib.resume(d.id)} title="Resume">
          <Icon name="play" size={14} />
        </button>
      )}
      {d.state === "error" && (
        <button className="btn small" onClick={() => bridge?.lib.resume(d.id)}>
          Try again
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
}

/** One model on this PC: what it is, how it fits, Load, and the rest in its menu. */
function ModelRow({ m, vram, ram, repack, busy, onUninstall }: { m: LocalModel; vram: number; ram: number; repack?: Repack; busy: boolean; onUninstall: (m: LocalModel) => void }) {
  const st = useApp((s) => s.status);
  const brainMenu = useApp((s) => s.settings.brainMenu);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const more = useRef<HTMLButtonElement>(null);
  const running = repack?.state === "running";
  const loaded = st?.path === m.path && st.state === "ready";
  const loading = st?.state === "loading" && st.path === m.path;
  const [cls, why] = fit(m.gb * 1e9, vram, ram);
  const canSpeed = !!m.repack && m.repack !== "yes";
  const items: MenuItem[] = [
    ...(bridge ? [{ label: "Show in folder", icon: "folder", on: () => bridge!.sys.open(m.path.replace(/[\\/][^\\/]*$/, ""), "").catch((e: any) => toast(cleanErr(e))) }] : []),
    { label: "Compare with flagship models", icon: "compare", on: () => app.set({ compare: m.name }) },
    ...(brainMenu ? [{ label: "Look inside", icon: "brain", on: () => app.set({ view: "brain" }) }] : []),
    ...(m.repack === "yes" && !running ? [{ label: "Undo speed-up", icon: "reload", desc: "So other programs can open the file", on: () => repackModel(m.path, true) }] : []),
    "sep",
    { label: "Uninstall", icon: "trash", danger: true, disabled: busy || running || loading, on: () => onUninstall(m) },
  ];
  return (
    <div className={`mrow ${loaded ? "on" : ""}`}>
      <span className="mrow-ico">
        <Icon name="library" size={17} />
      </span>
      <div className="mrow-main">
        <div className="mrow-name" title={m.path}>
          <span>{m.name}</span>
          {m.repack === "yes" && !running && <span className="model-tag" title="Rearranged with pelica repack: each expert loads in one read">Sped up</span>}
          {(m.repack === "partial" || m.repack === "unfinished") && !running && (
            <span className="model-tag warn" title={m.repack === "unfinished" ? "It was stopped partway. It can't be loaded until it's finished." : "Only some layers are sped up. It loads fine; finish it for the full speed-up."}>
              {m.repack === "unfinished" ? "Speed-up not finished" : "Partly sped up"}
            </span>
          )}
        </div>
        <div className="mrow-meta">
          {KINDS[m.kind] || "Tiered (SSD)"} · {gb(m.gb * 1e9)}
          {m.max_ctx ? ` · ${Math.round(m.max_ctx / 1024)}K context` : ""}
        </div>
        {running && <RepackProgress r={repack!} />}
        {repack?.state === "error" && <div className="err-text small">{repack.error}</div>}
      </div>
      <span className={`model-fit ${cls}`} title={why}>
        {why.split(":")[0]}
      </span>
      <div className="mrow-actions">
        {running ? (
          <button className="btn small ghost" onClick={() => bridge?.lib.repackStop(m.path)} title="Stop. It's safe: start it again later to finish.">
            Stop
          </button>
        ) : (
          canSpeed && (
            <button className="btn small ghost" onClick={() => repackModel(m.path, m.repack === "unfinished" && !!repack?.undo)} title="Rearrange the file so each expert loads in one read: answers about 20% faster">
              <Icon name="bolt" size={13} /> {m.repack === "no" ? "Speed up" : repack?.undo ? "Finish undoing" : "Finish speed-up"}
            </button>
          )
        )}
        {loaded ? (
          <button className="btn small" onClick={() => engine.unload().catch((e: any) => toast(cleanErr(e)))} title="Free the GPU memory and RAM it's using">
            Unload
          </button>
        ) : (
          <button
            className="btn small primary"
            disabled={st?.state === "loading" || running || m.repack === "unfinished"}
            title={running ? "Wait for the speed-up to finish" : m.repack === "unfinished" ? "Finish the speed-up first" : undefined}
            onClick={() => loadModel(m)}
          >
            {loading ? "Loading…" : "Load"}
          </button>
        )}
        <button
          ref={more}
          className={`ghost-btn ${menu ? "menu-open" : ""}`}
          title="More"
          aria-haspopup="menu"
          onClick={() => {
            const r = more.current!.getBoundingClientRect();
            setMenu(menu ? null : { x: r.right - 240, y: r.bottom + 4 });
          }}
        >
          <Icon name="more" size={16} />
        </button>
        {menu && <Menu at={menu} ignore={more} minWidth={220} close={() => setMenu(null)} items={items} />}
      </div>
    </div>
  );
}

function Local() {
  const st = useApp((s) => s.status);
  const { hw, vram, ram } = useHardware();
  const [models, setModels] = useState<LocalModel[]>([]);
  const [known, setKnown] = useState(false);
  const [dir, setDir] = useState("");
  const [removing, setRemoving] = useState("");
  const [q, setQ] = useState("");
  const refresh = () =>
    engine
      .models()
      .then((m) => (setModels(m.models), setDir(m.dir), setKnown(true)))
      .catch(() => setKnown(true));
  const repacks = useRepacks(refresh);
  const downloads = useDownloads().filter((d) => d.state === "running" || d.state === "paused" || d.state === "error");
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
  const needle = q.trim().toLowerCase();
  // the loaded one first, then by name
  const shown = models
    .filter((m) => !needle || m.name.toLowerCase().includes(needle))
    .sort((a, b) => Number(b.path === st?.path) - Number(a.path === st?.path) || a.name.localeCompare(b.name));
  const total = models.reduce((n, m) => n + m.gb, 0);
  return (
    <div className="models">
      <PcStrip hw={hw} vram={vram} ram={ram}>
        {models.length > 0 && (
          <span className="pc-strip-end" title={dir}>
            {models.length} {models.length === 1 ? "model" : "models"} · {gb(total * 1e9)}
          </span>
        )}
      </PcStrip>
      {downloads.length > 0 && (
        <div className="downloads">
          {downloads.map((d) => (
            <DownloadRow key={d.id} d={d} />
          ))}
        </div>
      )}
      {models.length > 6 && (
        <label className="page-filter">
          <Icon name="search" size={15} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter your models" spellCheck={false} />
        </label>
      )}
      {shown.length > 0 && (
        <div className="model-list">
          {shown.map((m) => (
            <ModelRow key={m.path} m={m} vram={vram} ram={ram} repack={repacks.find((x) => samePath(x.path, m.path))} busy={!!removing} onUninstall={uninstall} />
          ))}
        </div>
      )}
      {needle && !shown.length && <div className="empty-note">No models match “{q.trim()}”.</div>}
      {known && !models.length && (
        <div className="empty-state">
          <span className="empty-icon">
            <Icon name="library" size={20} />
          </span>
          <b>No models yet</b>
          <span className="muted small">Pelica runs models on this PC, so nothing you write leaves it. Get one to start chatting.</span>
          <button className="btn primary small" onClick={() => app.set({ libTab: "discover" })}>
            See the ones suggested for this PC
          </button>
        </div>
      )}
      {dir && <div className="muted small models-dir">Looking in {dir}. Change it in Settings → Models.</div>}
    </div>
  );
}

/** One suggested model: how it runs here, and Get / its download / Load. */
function StarterCard({ m, best, vram, ram, have, downloads, onGet }: { m: Starter; best: boolean; vram: number; ram: number; have?: LocalModel; downloads: Download[]; onGet: (m: Starter) => void }) {
  const st = useApp((s) => s.status);
  const wont = m.wont(vram, ram);
  const mine = downloads.filter((d) => d.repo === m.repo && m.files.includes(d.file));
  const going = mine.filter((d) => d.state === "running" || d.state === "paused" || d.state === "error");
  const got = mine.reduce((n, d) => n + d.got, 0);
  return (
    <div className={`starter ${best ? "best" : ""} ${wont ? "wont" : ""}`}>
      <div className="starter-head">
        <b>{m.name}</b>
        {best && <span className="model-tag on">Best fit</span>}
      </div>
      <div className="muted small starter-blurb">{m.blurb}</div>
      {wont && <div className="warn-note small">Too big for this PC: {wont}.</div>}
      <div className="starter-foot">
        <span className="muted small">
          {gb(m.size)} · {m.tools ? (m.vision ? "Tools, pictures" : "Tools and files") : "Chat only"}
        </span>
        {have ? (
          st?.path === have.path && st.state === "ready" ? (
            <span className="model-tag on">Loaded</span>
          ) : (
            <button className="btn small" disabled={st?.state === "loading"} onClick={() => loadModel(have)}>
              {st?.state === "loading" && st.path === have.path ? "Loading…" : "Load"}
            </button>
          )
        ) : going.length ? (
          <span className="muted small">
            {going.some((d) => d.state === "running") ? "Downloading" : going.some((d) => d.state === "error") ? "Stopped" : "Paused"}: {Math.round((got / m.size) * 100)}%
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

/** "Q4_K_M" out of a file's name. */
const quantOf = (f: string) => f.split("/").pop()!.match(/(?:UD-)?(?:I?Q\d[\w]*?|MXFP4|BF16|F16|F32)(?=[.-]|$)/i)?.[0] || "";

type HfFile = { path: string; size: number };
/** A model split over several files (name-00001-of-00003.gguf) is one choice. */
function wholeModels(files: HfFile[]) {
  const all = new Map<string, HfFile[]>();
  for (const f of files) {
    const key = f.path.replace(/-\d{5}-of-\d{5}(?=\.gguf$)/i, "");
    all.set(key, [...(all.get(key) || []), f]);
  }
  return [...all].map(([key, parts]) => {
    const name = key.split("/").pop()!;
    const quant = quantOf(key);
    return { key, parts, size: parts.reduce((n, f) => n + f.size, 0), label: /^mmproj/i.test(name) ? `Pictures add-on ${quant}`.trim() : quant || name };
  });
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
  const downloads = useDownloads();
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
  async function getStarter(m: Starter) {
    const why = await fetchStarter(m, downloads).catch((e) => cleanErr(e));
    if (why) toast(why);
  }

  // (the answer to an older search arriving late mustn't replace a newer one)
  const asked = useRef(0);
  async function search(text = q) {
    if (!bridge) return setErr("Browsing needs the Pelica desktop app.");
    const mine = ++asked.current;
    setBusy(true);
    setErr("");
    try {
      const found = await bridge.lib.search(text);
      if (mine === asked.current) setHits(found);
    } catch (e: any) {
      if (mine === asked.current) setErr(cleanErr(e));
    } finally {
      if (mine === asked.current) setBusy(false);
    }
  }
  useEffect(() => {
    search();
  }, []);
  useEffect(() => {
    setFiles(null);
    if (repo && bridge)
      bridge.lib
        .files(repo)
        .then(setFiles)
        .catch((e) => setErr(cleanErr(e)));
  }, [repo]);

  return (
    <div className="discover">
      <PcStrip hw={hw} vram={vram} ram={ram} />
      {downloads.length > 0 && (
        <div className="downloads">
          {downloads.map((d) => (
            <DownloadRow key={d.id} d={d} m={d.state === "done" ? models.find((x) => samePath(x.path, d.dest)) : undefined} r={repacks.find((x) => samePath(x.path, d.dest))} />
          ))}
        </div>
      )}
      <h2 className="page-sub">Suggested for this PC</h2>
      <p className="page-sub-note">The models Pelica is tested with.</p>
      <div className="starters">
        {STARTERS.map((m) => (
          <StarterCard key={m.id} m={m} best={m === best} vram={vram} ram={ram} have={haveOf(m)} downloads={downloads} onGet={getStarter} />
        ))}
      </div>
      <h2 className="page-sub">Search Hugging Face</h2>
      <p className="page-sub-note">Pelica runs mixture-of-experts models and gpt-oss. Experts that don't fit in your GPU's memory stay in RAM.</p>
      <div className="lib-search">
        <Icon name="search" size={16} />
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Search GGUF models" spellCheck={false} />
        <button className="btn small" onClick={() => search()} disabled={busy}>
          {busy ? "Searching…" : "Search"}
        </button>
      </div>
      <div className="lib-suggest">
        {SUGGEST.map((s) => (
          <button key={s} className={`chip ${q === s ? "on" : ""}`} onClick={() => (setQ(s), search(s))}>
            {s}
          </button>
        ))}
      </div>
      {err && <div className="err-text">{err}</div>}
      <div className="hf-split">
        <div className="hf-pane">
          {hits.map((h) => {
            const [owner, name] = h.id.includes("/") ? h.id.split("/") : ["", h.id];
            return (
              <button key={h.id} className={`hf-repo ${repo === h.id ? "sel" : ""}`} title={h.id} onClick={() => setRepo(h.id)}>
                <span className="hf-repo-name">{name}</span>
                <span className="muted small">
                  {owner} · {shortN(h.downloads)} downloads · {shortN(h.likes)} likes
                </span>
              </button>
            );
          })}
          {!hits.length && !busy && <div className="empty-note">No results.</div>}
        </div>
        <div className="hf-pane">
          {!repo && <div className="empty-note">Pick a model to see its files.</div>}
          {repo && (
            <>
              <div className="hf-head">
                <span className="hf-repo-name" title={repo}>
                  {repo.split("/").pop()}
                </span>
                <button className="ghost-btn" title="Open on Hugging Face" onClick={() => bridge?.sys.external(`https://huggingface.co/${repo}`)}>
                  <Icon name="external" size={14} />
                </button>
              </div>
              {!moeish(repo) && <div className="warn-note small">This doesn't look like a mixture-of-experts model, so Pelica may not run it.</div>}
              {!files && <div className="muted small hf-note">Loading files…</div>}
              {files &&
                wholeModels(files).map((f) => {
                const [cls, why] = fit(f.size, vram, ram);
                return (
                  <div key={f.key} className="hf-file">
                    <span className="hf-file-name" title={f.key}>
                      {f.label}
                    </span>
                    {f.parts.length > 1 && <span className="muted small">{f.parts.length} parts</span>}
                    <span className={`model-fit ${cls}`} title={why}>
                      {gb(f.size)}
                    </span>
                    <button className="btn small" onClick={() => get(repo, f.parts)}>
                      <Icon name="download" size={13} /> Get
                    </button>
                  </div>
                );
              })}
              {files && !files.length && <div className="muted small hf-note">No GGUF files in this repository.</div>}
            </>
          )}
        </div>
      </div>
      <div className="hf-foot">
        <span className="muted small" title={dl}>
          Downloads go to {dl}
        </span>
        <label className="switch-row" title="Rearranges the file so each expert loads in one read. Other programs can't open it afterwards; Undo speed-up under On this PC puts it back.">
          <button className={`toggle small ${auto ? "on" : ""}`} role="switch" aria-checked={auto} onClick={() => setSettings({ autoRepack: !auto })} />
          <span>Speed up DeepSeek V4 and Qwen3.8 models when they finish (answers about 20% faster)</span>
        </label>
      </div>
    </div>
  );
}

export function Library() {
  const tab = useApp((s) => s.libTab);
  const setTab = (libTab: "local" | "discover") => app.set({ libTab });
  return (
    <div className="page">
      <div className="page-in">
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
    </div>
  );
}
