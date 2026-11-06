// Updates (electron/updates.ts): asking the feed, by hand or by itself
// (Settings → Updates: at start, once a day, or never), and installing. The
// page itself is UpdatesTab in Settings.tsx; a download shows there and in the
// sidebar, so it can be followed with Settings closed.
import { bridge, type UpdateCheck, type UpdateFeed } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";

const DAY = 24 * 3600e3;
const openUpdates = () => app.set({ settingsOpen: true, settingsTab: "Updates" });

/** Asks the feed now; remembers when, and the newer version if there is one. */
export async function checkNow(): Promise<UpdateCheck> {
  const r = await bridge!.update.check();
  if (!r.error) setSettings({ updateLast: Date.now() });
  app.set({ update: r.newer ? r : null });
  return r;
}

/** Downloads, checks and installs: Pelica closes, and starts again when it's done.
 *  What went wrong, or "" (stopped by hand counts as nothing wrong). */
export async function installUpdate(f: UpdateFeed): Promise<string> {
  if (app.get().updating) return "";
  app.set({ updating: { version: f.version, phase: "download", got: 0, total: 0, speed: 0 } });
  try {
    await bridge!.update.install(f);
    return "";
  } catch (e: any) {
    app.set({ updating: null });
    const m = String(e?.message || e).replace(/^Error invoking remote method '[^']*': (Error: )?/, "");
    return /abort/i.test(m) ? "" : m;
  }
}

/** Follows the main process's progress, with how fast it's going. */
function followProgress() {
  let at = 0;
  let last = 0;
  let speed = 0;
  bridge!.update.onProgress((p) => {
    const u = app.get().updating;
    if (!u) return;
    const now = performance.now();
    if (p.phase === "download" && at && now > at && p.got > last) {
      const v = (p.got - last) / ((now - at) / 1000);
      speed = speed ? speed * 0.85 + v * 0.15 : v;
    }
    if (p.phase === "download") (at = now), (last = p.got);
    // (checking and starting carry no sizes: the download's stay on show)
    app.set({ updating: { ...u, phase: p.phase, got: p.got || u.got, total: p.total || u.total, speed } });
  });
}

/** From start: the checks the user chose, and a note when a version they didn't skip is out. */
export function checkForUpdates() {
  if (!bridge?.update) return;
  followProgress();
  bridge.update.attempt().then((v) => {
    if (v) toast(`The update to Pelica ${v} didn't finish. Pelica is still on the version you had.`, { label: "Try again", run: openUpdates });
  });
  const tick = async (atStart: boolean) => {
    const s = app.get().settings;
    if (s.updateCheck === "off") return;
    if (s.updateCheck === "start" && !atStart) return;
    if (s.updateCheck === "daily" && Date.now() - (s.updateLast || 0) < DAY - 60e3) return;
    const r = await checkNow().catch(() => null);
    if (!r?.newer || !r.latest || r.latest.version === app.get().settings.updateSkip) return;
    toast(`Pelica ${r.latest.version} is available`, { label: "See it", run: openUpdates });
  };
  setTimeout(() => tick(true), 15000);
  // once a day: looked at every hour, asked when a day has gone by
  setInterval(() => tick(false), 3600e3);
}

const mb = (n: number) => `${Math.round(n / 1e6)} MB`;

function timeLeft(s: number) {
  if (s < 5) return "almost done";
  if (s < 60) return `about ${Math.ceil(s / 5) * 5} seconds left`;
  return `about ${Math.round(s / 60)} min left`;
}

/** The bar: it fills while downloading; a piece slides along it while nothing
 *  can be measured. */
function Bar({ value }: { value: number | null }) {
  return (
    <div className={`upd-bar ${value === null ? "busy" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value === null ? undefined : Math.round(value * 100)}>
      <span style={value === null ? undefined : { transform: `scaleX(${value})` }} />
    </div>
  );
}

/** A running update: what it's doing, how far, how fast and how long is left. */
export function UpdateProgress({ compact = false }: { compact?: boolean }) {
  const u = useApp((s) => s.updating);
  if (!u) return null;
  const known = u.phase === "download" && u.total > 0;
  const part = known ? Math.min(1, u.got / u.total) : null;
  const title = u.phase === "start" ? "Closing Pelica to install" : u.phase === "check" ? "Checking the download" : compact ? "Downloading update" : `Downloading Pelica ${u.version}`;
  const detail = !known
    ? u.phase === "start"
      ? "It starts again by itself when it's done."
      : u.phase === "check"
        ? "Making sure it's the real file."
        : u.got
          ? `${mb(u.got)} so far`
          : "Connecting…"
    : [compact ? "" : `${mb(u.got)} of ${mb(u.total)}`, !compact && u.speed ? `${(u.speed / 1e6).toFixed(1)} MB/s` : "", u.speed ? timeLeft((u.total - u.got) / u.speed) : ""].filter(Boolean).join(" · ");
  return (
    <div className={`upd ${compact ? "compact" : ""}`}>
      <div className="upd-head">
        <b>{title}</b>
        {part !== null && <span className="upd-pct">{Math.floor(part * 100)}%</span>}
      </div>
      <Bar value={part} />
      <div className="upd-foot">
        <span>{detail}</span>
        {u.phase === "download" && (
          <button className="upd-stop" onClick={() => bridge?.update.cancel()}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

/** The sidebar's note: a new version, or the download under way. */
export function SideUpdate() {
  const running = useApp((s) => !!s.updating);
  const latest = useApp((s) => s.update?.latest);
  const skip = useApp((s) => s.settings.updateSkip);
  if (running)
    return (
      <div className="side-update">
        <UpdateProgress compact />
      </div>
    );
  if (!latest || latest.version === skip) return null;
  return (
    <div className="side-update row">
      <button className="side-update-text" title="See what's new" onClick={openUpdates}>
        Pelica {latest.version} is available
      </button>
      <button
        className="btn small"
        onClick={async () => {
          const m = await installUpdate(latest);
          if (m) toast(m);
        }}
      >
        Install
      </button>
    </div>
  );
}
