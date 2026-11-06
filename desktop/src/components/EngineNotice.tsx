// Over the message box: how far a model has got in loading, and what to do
// when the engine stops or a model won't load.
import { useRef, useState } from "react";
import { bridge } from "../lib/bridge";
import { engine } from "../lib/engine";
import { app, toast, useApp } from "../state";
import { Icon } from "./Icons";

export const shortModel = (m?: string | null) => (m || "the model").replace(/\s*\((?:I?Q\d|F16|BF16|F32)[^)]*\)\s*$/i, "");

/** "about 40 seconds left", from how fast the percentage has been moving. */
function left(s: number) {
  if (s < 5) return "almost there";
  if (s < 60) return `about ${Math.round(s / 5) * 5} seconds left`;
  return `about ${Math.round(s / 60)} min left`;
}

function Loading({ name, pct }: { name: string; pct: number }) {
  const first = useRef<{ t: number; p: number } | null>(null);
  const now = performance.now();
  if (!first.current || pct < first.current.p) first.current = { t: now, p: pct };
  const done = pct - first.current.p;
  const secs = (now - first.current.t) / 1000;
  const eta = done >= 3 && secs > 2 ? left(((100 - pct) / done) * secs) : "";
  return (
    <div className="engine-note" role="status">
      <div className="engine-note-text">
        <b>Loading {name}</b>
        <span className="muted small">
          {Math.floor(pct)}%{eta ? ` · ${eta}` : ""}
        </span>
      </div>
      <div className="setup-bar">
        <span style={{ transform: `scaleX(${pct / 100})` }} />
      </div>
    </div>
  );
}

function Problem() {
  const st = useApp((s) => s.status);
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const [busy, setBusy] = useState(false);
  if (st?.state !== "error") return null;
  const gone = /answering/.test(st.error || "");
  async function fix() {
    setBusy(true);
    try {
      if (gone) await bridge?.engine.restart();
      else await engine.load(st!.path!, st!.ctx || lastCtx);
    } catch (e: any) {
      toast(String(e?.message || e));
    }
    setBusy(false);
  }
  return (
    <div className="engine-note bad" role="alert">
      <Icon name="info" size={15} />
      <div className="engine-note-text">
        <b>{gone ? "Pelica's engine stopped" : `${shortModel(st.model)} didn't load`}</b>
        <span className="muted small">{gone ? "Your chats are safe. Restart it to carry on." : st.error}</span>
      </div>
      {(gone ? !!bridge : !!st.path) && (
        <button className="btn small" disabled={busy} onClick={fix}>
          {busy ? "Working…" : gone ? "Restart" : "Try again"}
        </button>
      )}
      {!gone && (
        <button className="btn small" onClick={() => app.set({ view: "library", libTab: "local" })}>
          Models
        </button>
      )}
    </div>
  );
}

export function EngineNotice() {
  const st = useApp((s) => s.status);
  if (st?.state === "loading") return <Loading key={st.path} name={shortModel(st.model)} pct={st.progress || 0} />;
  return <Problem />;
}
