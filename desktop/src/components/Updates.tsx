// Updates (electron/updates.ts): the check at start. The rows in
// Settings → About are UpdateRows in Settings.tsx.
import { bridge } from "../lib/bridge";
import { app, toast } from "../state";

/** At start: when checking is on, say once if a newer version is out. */
export function checkForUpdates() {
  const s = app.get().settings;
  if (!bridge?.update || !s.updateAuto) return;
  setTimeout(async () => {
    const r = await bridge!.update.check().catch(() => null);
    if (!r?.newer || !r.latest) return;
    app.set({ update: r });
    toast(`Pelica ${r.latest.version} is available`, { label: "See it", run: () => app.set({ settingsOpen: true, settingsTab: "About" }) });
  }, 15000);
}
