// Settings → Models → Storage: each drive's space and how much of it is
// models, and each model with Move (to another drive's models folder, or any
// folder) and Delete. Moving goes on in the main process (electron/library.ts
// moveModel) with progress here; deleting is the library's uninstall, which
// asks first and says when a model is too big for the Recycle Bin.
import { useEffect, useRef, useState } from "react";
import { bridge, type DriveInfo, type Hardware, type ModelMove } from "../lib/bridge";
import { engine, type LocalModel } from "../lib/engine";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

const gb = (n: number) => (n >= 10e9 ? `${Math.round(n / 1e9)} GB` : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`);
const rootOf = (p: string) => (p.match(/^[a-z]:/i)?.[0] || "").toUpperCase() + "\\";
const cleanErr = (e: any) => String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");

/** Where a move to `drive` puts a model: a models folder Pelica already uses there, else <drive>models. */
function folderOn(drive: string, hw: Hardware): string {
  return hw.dirs.find((d) => rootOf(d) === drive) || `${drive}models`;
}

function MoveMenu({ m, drives, hw, onPick }: { m: LocalModel; drives: DriveInfo[]; hw: Hardware; onPick: (target: string) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, []);
  const here = rootOf(m.path);
  const size = m.gb * 1e9;
  return (
    <div className="store-move" ref={ref}>
      <button className="btn small" onClick={() => setOpen(!open)}>
        Move <Icon name="chevronDown" size={12} />
      </button>
      {open && (
        <div className="store-menu">
          {drives
            .filter((d) => d.root !== here)
            .map((d) => {
              const room = d.free > size + 1e9;
              return (
                <button key={d.root} disabled={!room} onClick={() => (setOpen(false), onPick(folderOn(d.root, hw)))} title={folderOn(d.root, hw)}>
                  <span>To {d.root.slice(0, 2)}</span>
                  <span className="muted small">{room ? `${gb(d.free)} free` : "not enough space"}</span>
                </button>
              );
            })}
          <button
            onClick={async () => {
              setOpen(false);
              const d = await bridge?.lib.pickDir();
              if (d) onPick(d);
            }}
          >
            <span>Choose a folder…</span>
          </button>
        </div>
      )}
    </div>
  );
}

export function StorageRows() {
  const st = useApp((s) => s.status);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [drives, setDrives] = useState<DriveInfo[]>([]);
  const [hw, setHw] = useState<Hardware | null>(null);
  const [moves, setMoves] = useState<ModelMove[]>([]);
  const [busy, setBusy] = useState("");
  const refresh = async () => {
    if (!bridge) return;
    const [m, d, h] = await Promise.all([engine.models().catch(() => ({ models: [] as LocalModel[] })), bridge.lib.drives(), bridge.lib.hardware()]);
    setModels(m.models);
    setDrives(d);
    setHw(h);
    // a move may have added a models folder
    const set = await bridge.settings.get().catch(() => null);
    if (set?.modelDirs) app.set({ settings: { ...app.get().settings, modelDirs: set.modelDirs, lastModel: set.lastModel ?? app.get().settings.lastModel } });
  };
  useEffect(() => {
    refresh();
    if (!bridge) return;
    bridge.lib.moves().then(setMoves);
    let prev: ModelMove[] = [];
    return bridge.lib.onMove((list) => {
      for (const x of list) {
        const was = prev.find((p) => p.path === x.path);
        if (was?.state === "running" && x.state !== "running") {
          if (x.state === "done") toast(`Moved ${x.name} to ${x.to}`);
          else if (x.state === "error") toast(`Couldn't move ${x.name}: ${x.error}`);
          refresh();
        }
      }
      prev = list;
      setMoves(list);
    });
  }, [st?.state]);
  if (!bridge) return <p className="set-desc">Storage needs the Pelica desktop app.</p>;

  async function move(m: LocalModel, target: string) {
    try {
      const to = await bridge!.lib.move(m.path, target);
      if (to && app.get().settings.lastModel === m.path) setSettings({ lastModel: to });
    } catch (e) {
      toast(cleanErr(e));
    }
  }
  async function remove(m: LocalModel) {
    setBusy(m.path);
    try {
      const r = await bridge!.lib.uninstall(m.path);
      if (!r) return;
      if (app.get().settings.lastModel === m.path) setSettings({ lastModel: "" });
      toast(r === "deleted" ? `Deleted ${m.name}: the space is free.` : `Uninstalled ${m.name}: it's in the Recycle Bin. Empty it to free the space.`);
      await refresh();
    } catch (e) {
      toast(cleanErr(e));
    } finally {
      setBusy("");
    }
  }

  const onDrive = (root: string) => models.filter((m) => rootOf(m.path) === root).reduce((n, m) => n + m.gb * 1e9, 0);
  // the drives that matter: ones with models or a models folder
  const shown = drives.filter((d) => onDrive(d.root) > 0 || hw?.dirs.some((x) => rootOf(x) === d.root));
  return (
    <>
      {shown.map((d) => {
        const mine = onDrive(d.root);
        const used = d.total - d.free;
        return (
          <div key={d.root} className="store-drive">
            <div className="store-drive-head">
              <b>Drive {d.root.slice(0, 2)}</b>
              <span className="muted">
                {gb(d.free)} free of {gb(d.total)}
              </span>
            </div>
            <div className="store-bar" title={`Models ${gb(mine)} · other files ${gb(used - mine)} · free ${gb(d.free)}`}>
              <span className="models" style={{ width: `${(mine / d.total) * 100}%` }} />
              <span className="other" style={{ width: `${(Math.max(0, used - mine) / d.total) * 100}%` }} />
            </div>
            <div className="store-legend">
              <span>
                <i style={{ background: "var(--accent)" }} />
                Models {gb(mine)}
              </span>
              <span>
                <i style={{ background: "color-mix(in srgb, var(--text) 28%, transparent)" }} />
                Other files {gb(Math.max(0, used - mine))}
              </span>
            </div>
          </div>
        );
      })}
      {models.map((m) => {
        const mv = moves.find((x) => x.path === m.path && x.state === "running");
        const loaded = st?.path === m.path && (st.state === "ready" || st.state === "loading");
        return (
          <div key={m.path} className="store-model">
            <div className="store-model-text">
              <span>
                {m.name} <span className="muted small">· {gb(m.gb * 1e9)}</span>
              </span>
              {mv ? (
                <span className="muted small">
                  Moving to {mv.to}… {mv.total ? Math.floor((mv.got / mv.total) * 100) : 0}% ({gb(mv.got)} of {gb(mv.total)})
                </span>
              ) : (
                <span className="muted small mono" title={m.path}>
                  {m.path}
                </span>
              )}
            </div>
            {mv ? (
              <button className="btn small ghost" onClick={() => bridge?.lib.moveStop(m.path)} title="Stop. The model stays where it was.">
                Cancel
              </button>
            ) : (
              <>
                {hw && <MoveMenu m={m} drives={drives} hw={hw} onPick={(t) => move(m, t)} />}
                <button className="btn small ghost danger" disabled={!!busy || st?.state === "loading"} onClick={() => remove(m)} title={loaded ? "It's unloaded first" : undefined}>
                  <Icon name="trash" size={13} /> Delete
                </button>
              </>
            )}
          </div>
        );
      })}
      {!models.length && (
        <p className="set-note">
          No models on this PC yet.{" "}
          <button className="btn small" onClick={() => app.set({ settingsOpen: false, view: "library", libTab: "discover" })}>
            Get a model
          </button>
        </p>
      )}
    </>
  );
}
