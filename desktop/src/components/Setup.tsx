// First-run setup: with no model on this PC, look at what the PC has, suggest
// the one model that runs best on it, download it and load it. The Home
// page's "Get a model" opens this again for someone who said Not now.
import { useEffect, useRef, useState } from "react";
import { bridge, type Download, type DriveInfo, type Hardware } from "../lib/bridge";
import { engine } from "../lib/engine";
import { bestFor, getStarter, gib, gpuOf, type Starter } from "../lib/starters";
import { app, setSettings, toast, useApp } from "../state";
import { Icon, Logo } from "./Icons";

const gb = (b: number) => `${(b / 1e9).toFixed(b < 1e10 ? 1 : 0)} GB`;
const clean = (e: any) => String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");

/** At start: no model here yet, so offer the setup (once). */
export async function checkSetup() {
  const s = app.get();
  if (!bridge || s.settings.setupDone) return;
  for (let i = 0; i < 20; i++) {
    const m = await engine.models().catch(() => null);
    if (m) {
      if (m.models.length) setSettings({ setupDone: true });
      else app.set({ setup: true });
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

function leave() {
  setSettings({ setupDone: true });
  app.set({ setup: false });
}

/** "about 12 min left" from what the last few seconds averaged. */
function timeLeft(bytes: number, perSec: number) {
  if (perSec < 1) return "";
  const s = bytes / perSec;
  if (s < 60) return "less than a minute left";
  if (s < 3600) return `about ${Math.round(s / 60)} min left`;
  return `about ${(s / 3600).toFixed(1)} hours left`;
}

function CheckRow({ icon, label, value, delay }: { icon: string; label: string; value: string; delay: number }) {
  return (
    <div className="setup-check" style={{ animationDelay: `${delay}ms` }}>
      <span className="setup-check-icon">
        <Icon name={icon} size={15} />
      </span>
      <span className="setup-check-label">{label}</span>
      <span className="setup-check-value">{value}</span>
      <span className="setup-tick" style={{ animationDelay: `${delay + 140}ms` }}>
        <Icon name="check" size={13} />
      </span>
    </div>
  );
}

function Progress({ m, mine, onDone }: { m: Starter; mine: Download[]; onDone: () => void }) {
  const got = mine.reduce((n, d) => n + d.got, 0);
  const total = Math.max(mine.reduce((n, d) => n + d.total, 0), 1);
  const pct = Math.min(100, (got / total) * 100);
  const failed = mine.find((d) => d.state === "error");
  const paused = !failed && mine.some((d) => d.state === "paused");
  // the speed over the last several updates
  const marks = useRef<{ t: number; got: number }[]>([]);
  const now = performance.now();
  if (!marks.current.length || now - marks.current[marks.current.length - 1].t > 500) marks.current = [...marks.current.slice(-20), { t: now, got }];
  const first = marks.current[0];
  const perSec = now > first.t + 1000 ? ((got - first.got) / (now - first.t)) * 1000 : 0;
  const stateWord = failed ? "Download stopped" : paused ? "Paused" : `${gb(got)} of ${gb(total)}`;
  return (
    <div className="setup-dl">
      <div className="setup-dl-head">
        <b>Downloading {m.name}</b>
        <span className="setup-pct">{Math.floor(pct)}%</span>
      </div>
      <div className={`setup-bar ${failed ? "bad" : paused ? "paused" : ""}`}>
        <span style={{ transform: `scaleX(${pct / 100})` }} />
      </div>
      <div className="setup-dl-foot muted small">
        <span>{stateWord}</span>
        <span>{failed ? failed.error : paused ? "" : timeLeft(total - got, perSec)}</span>
      </div>
      <div className="setup-actions">
        {failed || paused ? (
          <button className="btn primary small" onClick={() => mine.filter((d) => d.state === "error" || d.state === "paused").forEach((d) => bridge!.lib.resume(d.id).catch((e) => toast(clean(e))))}>
            {failed ? "Try again" : "Resume"}
          </button>
        ) : (
          <button className="btn small" onClick={() => mine.filter((d) => d.state === "running").forEach((d) => bridge!.lib.pause(d.id).catch((e) => toast(clean(e))))}>
            Pause
          </button>
        )}
        <button className="btn small" onClick={onDone} title="It keeps downloading. See it under Models.">
          Keep going in the background
        </button>
      </div>
    </div>
  );
}

export function Setup() {
  const show = useApp((s) => s.setup && !s.signUp && !s.whatsNew && !s.settingsOpen);
  const st = useApp((s) => s.status);
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const [hw, setHw] = useState<Hardware | null>(null);
  const [drive, setDrive] = useState<DriveInfo | null>(null);
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!show || !bridge) return;
    bridge.lib
      .hardware()
      .then(async (h) => {
        setHw(h);
        setDrive(await bridge!.lib.space(h.downloadDir).catch(() => null));
        bridge!.lib.downloads(h.downloadDir).then(setDownloads).catch(() => {});
      })
      .catch((e) => setErr(clean(e)));
    const off = bridge.lib.onProgress(setDownloads);
    // (tests stage a download without starting one)
    (window as any).__setupDownloads = setDownloads;
    // (the recommendation comes up after the checks have)
    const t = setTimeout(() => setReady(true), 820);
    const f = (e: KeyboardEvent) => e.key === "Escape" && leave();
    window.addEventListener("keydown", f);
    return () => {
      off();
      clearTimeout(t);
      window.removeEventListener("keydown", f);
    };
  }, [show]);

  if (!show) return null;
  const gpu = hw ? gpuOf(hw, st?.gpu) : null;
  const vram = gpu?.vram || 0;
  const ram = hw?.ram || 0;
  const best = hw ? bestFor(vram, ram) : undefined;
  const mine = best ? downloads.filter((d) => d.repo === best.repo && best.files.includes(d.file)) : [];
  const finished = !!best && mine.length >= best.files.length && mine.every((d) => d.state === "done");
  const going = mine.some((d) => d.state === "running" || d.state === "paused" || d.state === "error");
  const short = best && drive ? drive.free < best.size + 1e9 : false;

  async function get() {
    setBusy(true);
    setErr("");
    const why = await getStarter(best!, downloads).catch(clean);
    if (why) setErr(why);
    setBusy(false);
  }
  async function start() {
    setBusy(true);
    setErr("");
    try {
      const { models } = await engine.models();
      const name = (f: string) => f.split(/[\\/]/).pop()!.toLowerCase();
      const m = models.find((x) => best!.files.some((f) => name(f) === name(x.path)));
      if (!m) throw new Error("The download finished, but the model isn't in the list yet. Look under Models.");
      await engine.load(m.path, Math.min(lastCtx, m.max_ctx || lastCtx));
      setSettings({ lastModel: m.path });
      app.set({ view: "chat" });
      leave();
    } catch (e) {
      setErr(clean(e));
    }
    setBusy(false);
  }
  const browse = () => {
    leave();
    app.set({ view: "library", libTab: "discover" });
  };

  return (
    <div className="modal-back signup-back">
      <div className="modal setup" role="dialog" aria-label="Set up Pelica">
        <Logo size={44} />
        <h2>{finished ? "Ready to go" : "Let's set up Pelica"}</h2>
        <p className="signup-lead">{finished ? `${best!.name} is on this PC.` : "Pelica runs on your PC, so it needs a model that fits it. Here's what yours has."}</p>
        {!finished && (
          <div className="setup-checks">
            {gpu && <CheckRow icon="cpu" label={gpu.name} value={vram ? `${gib(vram)} memory` : ""} delay={0} />}
            {hw && <CheckRow icon="layers" label="RAM" value={gib(ram)} delay={200} />}
            {drive && <CheckRow icon="download" label={`Space on ${drive.root.slice(0, 2)}`} value={`${gb(drive.free)} free`} delay={400} />}
          </div>
        )}
        {ready && hw && !finished && !going && best && (
          <div className="setup-pick">
            <div className="setup-pick-head">
              <span className="setup-eyebrow">Best fit for this PC</span>
              <span className="pill">{gb(best.size)}</span>
            </div>
            <b className="setup-pick-name">{best.name}</b>
            <div className="muted small">{best.blurb}</div>
            {short && <div className="warn-note small">There isn't room for it on {drive!.root.slice(0, 2)}. Change the download folder in Settings → Models, or free some space.</div>}
            <button className="btn primary setup-get" disabled={busy || short} onClick={get}>
              <Icon name="download" size={14} /> {busy ? "Starting…" : `Download ${gb(best.size)}`}
            </button>
          </div>
        )}
        {ready && hw && !best && (
          <div className="setup-pick">
            <b className="setup-pick-name">No suggested model fits this PC</b>
            <div className="muted small">Smaller models are on Hugging Face. Models can search them and shows each one's size.</div>
            <button className="btn primary setup-get" onClick={browse}>
              Find a smaller model
            </button>
          </div>
        )}
        {best && going && <Progress m={best} mine={mine} onDone={() => app.set({ setup: false })} />}
        {finished && (
          <button className="btn primary setup-get" disabled={busy || st?.state === "loading"} onClick={start}>
            {busy || st?.state === "loading" ? "Loading the model…" : "Start chatting"}
          </button>
        )}
        {err && <div className="warn-note small setup-err">{err}</div>}
        {!going && !finished && ready && (
          <div className="setup-links">
            {best && (
              <button className="signup-skip" onClick={browse}>
                See all models
              </button>
            )}
            <button className="signup-skip" onClick={leave}>
              Not now
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
