// Updates (electron/updates.ts): asking the feed, by hand or by itself
// (Settings → Updates: every few hours, once a day, or never), and installing.
// A new version shows on the title bar's button, whatever page is open and
// with the sidebar hidden too; a download shows there, in the sidebar and in
// Settings → Updates (UpdatesTab in Settings.tsx).
import { useEffect, useRef } from "react";
import { bridge, type UpdateCheck, type UpdateFeed } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

const DAY = 24 * 3600e3;
const OFTEN = 3 * 3600e3;
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

/** There's a version to offer: newer, and not skipped. */
export const useUpdateReady = () => useApp((s) => !!s.update?.latest && s.update.latest.version !== s.settings.updateSkip);

/** From start: the checks the user chose, and a note the first time a version they didn't skip is out. */
export function checkForUpdates() {
  if (!bridge?.update) return;
  followProgress();
  bridge.update.attempt().then((v) => {
    if (v) toast(`The update to Pelica ${v} didn't finish. Pelica is still on the version you had.`, { label: "Try again", run: openUpdates });
  });
  let asking = false;
  const tick = async (atStart: boolean) => {
    const s = app.get().settings;
    if (s.updateCheck === "off" || asking) return;
    if (!atStart && Date.now() - (s.updateLast || 0) < (s.updateCheck === "daily" ? DAY : OFTEN) - 60e3) return;
    asking = true;
    const r = await checkNow().catch(() => null);
    asking = false;
    const now = app.get().settings;
    if (!r?.newer || !r.latest) return;
    const v = r.latest.version;
    if (v === now.updateSkip || v === now.updateTold) return;
    setSettings({ updateTold: v });
    toast(`Pelica ${v} is ready to install`, { label: "See what's new", run: () => app.set({ updatePop: true }) });
    bridge?.app.notify({ title: `Pelica ${v} is ready to install`, body: r.latest.title || "Open Pelica to see what's new." }).catch(() => {});
  };
  // every start asks, soon after the window is up: it's one small file
  setTimeout(() => tick(true), 3000);
  // and while Pelica stays open: looked at now and then, and on coming back to it
  setInterval(() => tick(false), 1800e3);
  window.addEventListener("focus", () => tick(false));
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

/** The sidebar's note: the download under way. */
export function SideUpdate() {
  const running = useApp((s) => !!s.updating);
  if (!running) return null;
  return (
    <div className="side-update">
      <UpdateProgress compact />
    </div>
  );
}

/** The title bar's button: a new version is out (or on its way in). It opens
 *  what's new with Install right there. */
export function UpdatePill() {
  const u = useApp((s) => s.updating);
  const latest = useApp((s) => s.update?.latest);
  const ready = useUpdateReady();
  const open = useApp((s) => s.updatePop);
  // (a download shows in the sidebar when that's on screen)
  const sideShows = useApp((s) => s.sidebar && !s.compact);
  const box = useRef<HTMLDivElement>(null);
  const close = () => app.set({ updatePop: false });
  useEffect(() => {
    if (!open) return;
    const down = (e: MouseEvent) => !box.current?.contains(e.target as Node) && close();
    const key = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("mousedown", down);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", down);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  if (u ? sideShows : !ready) return null;
  const pct = u?.phase === "download" && u.total ? ` ${Math.floor((u.got / u.total) * 100)}%` : "";
  const notes = latest?.notes || [];
  return (
    <div className="pop-wrap upd-wrap nodrag" ref={box}>
      <button className={`upd-pill ${open ? "open" : ""}`} title={u ? "See the update" : "See what's new, and install"} onClick={() => app.set({ updatePop: !open })}>
        <Icon name="download" size={13} />
        <span>{u ? `Updating${pct}` : `Update to ${latest!.version}`}</span>
      </button>
      {open && (
        <div className="popover upd-pop">
          {u ? (
            <UpdateProgress />
          ) : (
            <>
              <div className="upd-pop-head">
                <img src="logo.svg" alt="" />
                <div>
                  <b>Pelica {latest!.version} is ready</b>
                  {latest!.title && <span>{latest!.title}</span>}
                </div>
              </div>
              {notes.length > 0 && (
                <ul className="upd-pop-notes">
                  {notes.slice(0, 5).map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              )}
              {notes.length > 5 && (
                <button className="link-btn small" onClick={() => (close(), openUpdates())}>
                  And {notes.length - 5} more
                </button>
              )}
              <div className="upd-pop-foot">
                <button className="link-btn small" title="Don't offer this version again. A newer one will be." onClick={() => (close(), setSettings({ updateSkip: latest!.version }))}>
                  Skip this version
                </button>
                <button className="btn small" onClick={close}>
                  Later
                </button>
                <button
                  className="btn small primary"
                  onClick={async () => {
                    const m = await installUpdate(latest!);
                    if (m) toast(m);
                  }}
                >
                  Install now
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
