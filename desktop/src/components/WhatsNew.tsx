// What's new: the release notes (src/changelog.json) for the versions since
// the one the user last saw. Opens once after an update; Settings → About and
// the palette show every release.
import { useEffect } from "react";
import log from "../changelog.json";
import { app, setSettings, useApp } from "../state";
import { Icon } from "./Icons";
import { Markdown } from "./Markdown";

export interface Release {
  version: string;
  date: string;
  title?: string;
  notes: string[];
}

export const RELEASES = log as Release[];
/** this build's version ("next" entries are unreleased notes being written) */
export const VERSION = RELEASES.find((r) => r.version !== "next")?.version || "0.0.0";

/** -1, 0 or 1, comparing dotted version numbers */
export function cmpVersion(a: string, b: string): number {
  const x = a.split(".").map(Number);
  const y = b.split(".").map(Number);
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] || 0) - (y[i] || 0);
    if (d) return Math.sign(d);
  }
  return 0;
}

/** Releases newer than `seen` (just the latest one when nothing was seen yet). */
export function unseen(seen: string): Release[] {
  const rel = RELEASES.filter((r) => r.version !== "next");
  return seen ? rel.filter((r) => cmpVersion(r.version, seen) > 0) : rel.slice(0, 1);
}

/** At start: open What's new when this version's notes haven't been seen. */
export function checkWhatsNew() {
  const seen = app.get().settings.lastSeenVersion;
  if (cmpVersion(VERSION, seen || "0") > 0 && unseen(seen).length) app.set({ whatsNew: "new" });
}

export function ReleaseNotes({ releases }: { releases: Release[] }) {
  return (
    <div className="release-notes">
      {releases.map((r) => (
        <section key={r.version} className="release">
          <div className="release-head">
            <b>Pelica {r.version}</b>
            <span className="muted small">{r.date}</span>
          </div>
          {r.title && <div className="release-title">{r.title}</div>}
          <Markdown text={r.notes.map((n) => `- ${n}`).join("\n")} quiet />
        </section>
      ))}
    </div>
  );
}

export function WhatsNew() {
  const mode = useApp((s) => s.whatsNew);
  const seen = useApp((s) => s.settings.lastSeenVersion);
  const close = () => {
    setSettings({ lastSeenVersion: VERSION });
    app.set({ whatsNew: null });
  };
  useEffect(() => {
    if (!mode) return;
    const f = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [mode]);
  if (!mode) return null;
  const list = mode === "all" ? RELEASES.filter((r) => r.version !== "next") : unseen(seen);
  return (
    <div className="modal-back whatsnew-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal whatsnew" role="dialog" aria-label="What's new">
        <div className="whatsnew-head">
          <h2>{mode === "all" ? "Release notes" : `What's new in Pelica ${VERSION}`}</h2>
          <button className="ghost-btn" title="Close" onClick={close}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="whatsnew-body">
          <ReleaseNotes releases={list} />
        </div>
        <div className="whatsnew-foot">
          <button className="btn" onClick={close}>
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}
