// Updates (electron/updates.ts): asking the feed, by hand or by itself
// (Settings → Updates: at start, once a day, or never). The page itself is
// UpdatesTab in Settings.tsx.
import { bridge, type UpdateCheck } from "../lib/bridge";
import { app, setSettings, toast } from "../state";

const DAY = 24 * 3600e3;

/** Asks the feed now; remembers when, and the newer version if there is one. */
export async function checkNow(): Promise<UpdateCheck> {
  const r = await bridge!.update.check();
  if (!r.error) setSettings({ updateLast: Date.now() });
  app.set({ update: r.newer ? r : null });
  return r;
}

/** From start: the checks the user chose, and a note when a version they didn't skip is out. */
export function checkForUpdates() {
  if (!bridge?.update) return;
  const tick = async (atStart: boolean) => {
    const s = app.get().settings;
    if (s.updateCheck === "off") return;
    if (s.updateCheck === "start" && !atStart) return;
    if (s.updateCheck === "daily" && Date.now() - (s.updateLast || 0) < DAY - 60e3) return;
    const r = await checkNow().catch(() => null);
    if (!r?.newer || !r.latest || r.latest.version === app.get().settings.updateSkip) return;
    toast(`Pelica ${r.latest.version} is available`, { label: "See it", run: () => app.set({ settingsOpen: true, settingsTab: "Updates" }) });
  };
  setTimeout(() => tick(true), 15000);
  // once a day: looked at every hour, asked when a day has gone by
  setInterval(() => tick(false), 3600e3);
}
