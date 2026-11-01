// Settings, laid out like the Claude app's: a title bar, a nav with a
// search box and groups, and pages of headed sections whose rows are split
// by thin lines (settings.css).
import { useEffect, useState } from "react";
import { bridge, type UpdateCheck } from "../lib/bridge";
import { app, PERM_MODES, refreshChats, setSettings, toast, useApp, type Settings as S } from "../state";
import { ConnectorsTab } from "./Connectors";
import { DevicesTab } from "./Devices";
import { Diff } from "./ToolCard";
import { Icon } from "./Icons";
import { setMaterial } from "./Palette";
import { RELEASES, ReleaseNotes, VERSION } from "./WhatsNew";
import { UsageTab } from "./Usage";
import { PromptList } from "./Prompts";
import { checkNow } from "./Updates";
import { StorageRows } from "./Storage";

const PREVIEW_A = 'function greet(name: string) {\n  return "Hello, " + name;\n}\n';
const PREVIEW_B = "function greet(name: string) {\n  return `Hello, ${name}!`;\n}\n";

/** The nav: groups as in Claude's settings; `keys` are what the search box also finds a page by. */
const TABS: { name: string; icon: string; group: string; keys: string }[] = [
  { name: "General", icon: "user", group: "Preferences", keys: "name instructions send enter reply length saved prompts back up restore chats" },
  { name: "Appearance", icon: "palette", group: "Preferences", keys: "theme dark light warm midnight neural background backdrop mica acrylic text size font serif code animations" },
  { name: "Capabilities", icon: "bolt", group: "Preferences", keys: "web search steps thinking effort" },
  { name: "Permissions", icon: "shield", group: "Preferences", keys: "manual accept edits plan bypass folders deleting recycle shell cmd powershell" },
  { name: "Context", icon: "gauge", group: "Preferences", keys: "auto-compact compact trim tool output" },
  { name: "CQS", icon: "scan", group: "Preferences", keys: "code quality scanner fix checkers" },
  { name: "Usage", icon: "spark", group: "Preferences", keys: "tokens days models" },
  { name: "Desktop", icon: "monitor", group: "This computer", keys: "quick ask shortcut hotkey tray start with windows notifications computer tools" },
  { name: "Models", icon: "library", group: "This computer", keys: "storage disk space drive move delete uninstall model folders load download speed up repack local api openai port engine restart" },
  { name: "Devices", icon: "monitor", group: "This computer", keys: "bridge ram lend pairing network" },
  { name: "Developer", icon: "terminal", group: "This computer", keys: "versions electron engine log data folder" },
  { name: "Connectors", icon: "plug", group: "Customize", keys: "mcp servers tools github memory browser" },
  { name: "Updates", icon: "download", group: "Customize", keys: "update version check install feed preview pre-release skip github releases daily" },
  { name: "About", icon: "info", group: "Customize", keys: "version release notes github" },
];

/** What each theme looks like, for the picker (the colours in styles.css / polish.css). */
const THEMES: { id: S["theme"]; name: string; bg: string; side: string; panel: string; text: string; accent: string }[] = [
  { id: "dark", name: "Dark", bg: "#191919", side: "#1f1f1f", panel: "#2a2a2a", text: "#ececec", accent: "#338ced" },
  { id: "warm", name: "Warm", bg: "#262624", side: "#1f1e1d", panel: "#30302e", text: "#ecebe6", accent: "#d97757" },
  { id: "light", name: "Light", bg: "#faf9f5", side: "#f3f1ea", panel: "#ffffff", text: "#1f1e1c", accent: "#c96442" },
  { id: "midnight", name: "Midnight", bg: "#0f1420", side: "#0b0f18", panel: "#161d2c", text: "#e4e9f4", accent: "#6d8dff" },
  { id: "neural", name: "Neural", bg: "#0b0a0c", side: "#141215", panel: "#1c191b", text: "#f1ece8", accent: "#ff8a4c" },
];

function ThemePicker({ value, onChange }: { value: string; onChange: (v: S["theme"]) => void }) {
  return (
    <div className="theme-cards">
      {THEMES.map((t) => (
        <button key={t.id} className={`theme-card ${t.id === value ? "sel" : ""}`} onClick={() => onChange(t.id)} aria-pressed={t.id === value}>
          <span className="theme-mini" style={{ background: t.bg }}>
            <span className="theme-mini-side" style={{ background: t.side }} />
            <span className="theme-mini-main">
              <span className="theme-mini-line" style={{ background: t.text, width: "70%" }} />
              <span className="theme-mini-line" style={{ background: t.text, width: "45%" }} />
              <span className="theme-mini-box" style={{ background: t.panel }}>
                <span className="theme-mini-dot" style={{ background: t.accent }} />
              </span>
            </span>
          </span>
          <span className="theme-name">{t.name}</span>
        </button>
      ))}
    </div>
  );
}

/** Text size as "Aa" at each size (Ctrl+= / Ctrl+- / Ctrl+0 do the same). */
const SIZES: [number, string][] = [[13, "Smaller"], [14, "Small"], [15, "Default"], [16, "Large"], [17, "Larger"], [18, "Largest"]];
function SizePicker({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="seg size-seg">
      {SIZES.map(([v, l], i) => (
        <button key={v} className={v === value ? "sel" : ""} onClick={() => onChange(v)} title={l} aria-label={l} style={{ fontSize: 11 + i * 1.6 }}>
          Aa
        </button>
      ))}
    </div>
  );
}

/** A heading inside a page, over its rows. */
function Section({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <>
      <div className="set-section">{title}</div>
      {children}
    </>
  );
}

function Seg<T extends string | number | boolean>({ value, options, onChange }: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <div className="seg">
      {options.map(([v, l]) => (
        <button key={String(v)} className={v === value ? "sel" : ""} onClick={() => onChange(v)}>
          {l}
        </button>
      ))}
    </div>
  );
}

function Row({ title, desc, children }: { title: string; desc?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="set-row">
      <div className="set-text">
        <div className="set-title">{title}</div>
        {desc && <div className="set-desc">{desc}</div>}
      </div>
      <div className="set-ctl">{children}</div>
    </div>
  );
}

function Toggle({ on, set }: { on: boolean; set: (v: boolean) => void }) {
  return <button className={`toggle ${on ? "on" : ""}`} onClick={() => set(!on)} role="switch" aria-checked={on} />;
}

const keyName = (e: KeyboardEvent) => {
  if (e.code.startsWith("Key")) return e.code.slice(3);
  if (e.code.startsWith("Digit")) return e.code.slice(5);
  if (e.key === " ") return "Space";
  if (e.key.startsWith("Arrow")) return e.key.slice(5);
  return e.key.length === 1 ? e.key.toUpperCase() : e.key;
};

/** The quick-ask shortcut: shown as keys; Change records the next combination. */
/** Back up every chat to one file, or bring chats back from one. */
function BackupRow() {
  const [busy, setBusy] = useState(false);
  const go = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e: any) {
      toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    }
    setBusy(false);
  };
  return (
    <Row title="Back up and restore" desc="All your chats in one file, to keep or to move to another PC. Restoring adds the chats you don't have; one that differs from yours comes back as a copy, so nothing is overwritten. A folder's proj.pel works too.">
      <div className="backup-btns">
        <button
          className="btn"
          disabled={!bridge || busy}
          onClick={() =>
            go(async () => {
              const r = await bridge!.chats.backup();
              if (r) toast(`Backed up ${r.count} chat${r.count === 1 ? "" : "s"}`);
            })
          }
        >
          Back up…
        </button>
        <button
          className="btn ghost"
          disabled={!bridge || busy}
          onClick={() =>
            go(async () => {
              const r = await bridge!.chats.restore();
              if (!r) return;
              await refreshChats();
              const parts = [r.added && `${r.added} added`, r.copies && `${r.copies} as copies`, r.same && `${r.same} already here`].filter(Boolean);
              toast(parts.length ? `Restored: ${parts.join(", ")}` : "No chats in that file.");
            })
          }
        >
          Restore…
        </button>
      </div>
    </Row>
  );
}

function HotkeyRow() {
  const [k, setK] = useState("");
  const [err, setErr] = useState("");
  const [rec, setRec] = useState(false);
  useEffect(() => {
    bridge?.app
      .info()
      .then((i) => (setK(i.hotkey), setErr(i.hotkeyError)))
      .catch(() => {});
  }, []);
  async function apply(acc: string) {
    if (!bridge) return;
    setErr(await bridge.app.hotkey(acc));
    setK(acc);
  }
  useEffect(() => {
    if (!rec) return;
    const f = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      if (e.key === "Escape") return setRec(false);
      if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;
      const mods = [e.ctrlKey && "Control", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Super"].filter(Boolean) as string[];
      if (!mods.length || (mods.length === 1 && mods[0] === "Shift")) return setErr("Use Ctrl or Alt with a key, so it doesn't clash with typing.");
      setRec(false);
      apply([...mods, keyName(e)].join("+"));
    };
    window.addEventListener("keydown", f, true);
    return () => window.removeEventListener("keydown", f, true);
  }, [rec]);
  return (
    <Row title="Quick ask shortcut" desc={<>Opens a small ask bar over any app; Enter sends the question to Pelica. {err && <span className="err-text">{err}</span>}</>}>
      <div className="hotkey">
        {rec ? (
          <span className="hotkey-rec">Press the keys… (Esc cancels)</span>
        ) : k ? (
          <span className="hotkey-keys">
            {k.split("+").map((p) => (
              <kbd key={p}>{p === "Control" ? "Ctrl" : p}</kbd>
            ))}
          </span>
        ) : (
          <span className="muted">Off</span>
        )}
        <button className="btn ghost small" onClick={() => (setErr(""), setRec(!rec))} disabled={!bridge}>
          {rec ? "Cancel" : "Change"}
        </button>
        {k && !rec && (
          <button className="btn ghost small" onClick={() => apply("")}>
            Off
          </button>
        )}
      </div>
    </Row>
  );
}

function DesktopTab() {
  const s = useApp((x) => x.settings);
  const [login, setLogin] = useState(false);
  useEffect(() => {
    bridge?.app.login().then(setLogin).catch(() => {});
  }, []);
  if (!bridge) return <p className="set-desc">These settings need the Pelica desktop app.</p>;
  return (
    <>
      <HotkeyRow />
      <Row title="Keep running in the tray" desc="Closing the window leaves Pelica (and the loaded model) in the tray, so the quick ask bar answers at once. Quit from the tray icon's menu.">
        <Toggle on={s.closeToTray} set={(v) => setSettings({ closeToTray: v })} />
      </Row>
      <Row title="Start with Windows" desc="Starts Pelica in the tray when you sign in.">
        <Toggle on={login} set={async (v) => setLogin(await bridge!.app.login(v))} />
      </Row>
      <Row title="Notifications" desc="When a reply finishes or a command waits for your OK while you're in another window. The taskbar button shows progress either way.">
        <Toggle on={s.notify} set={(v) => setSettings({ notify: v })} />
      </Row>
      <Row title="Computer tools" desc="Lets models see facts about this PC (Windows, CPU, GPU, memory, disks), copy text for you, and read your clipboard or open a workspace file in its app (both ask first).">
        <Toggle on={s.computerTools} set={(v) => setSettings({ computerTools: v })} />
      </Row>
      <p className="set-note">
        Right-click Pelica's taskbar button for New chat and Quick ask. Drop files anywhere on the window to attach them. <kbd>Ctrl</kbd>+<kbd>K</kbd> opens the command palette.
      </p>
    </>
  );
}

function ApiRows() {
  const s = useApp((x) => x.settings);
  const [st, setSt] = useState<{ on: boolean; port: number; error: string } | null>(null);
  const [portText, setPortText] = useState(String(s.apiPort || 11435));
  useEffect(() => {
    let live = true;
    const t = setTimeout(() => bridge?.engine.api?.().then((v) => live && setSt(v)).catch(() => {}), 300);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [s.apiServer, s.apiPort]);
  const url = `http://127.0.0.1:${s.apiPort || 11435}/v1`;
  return (
    <>
      <Row title="OpenAI-compatible server" desc="Lets other programs on this PC (editors, scripts, chat apps) use the loaded model, as an OpenAI-compatible server. Nothing outside this computer can reach it.">
        <Toggle on={s.apiServer} set={(v) => setSettings({ apiServer: v })} />
      </Row>
      {s.apiServer && (
        <Row title="Address" desc={st?.error ? st.error : "Use it as the base URL; any API key works. One request at a time, and not while Pelica is answering you."}>
          <div className="inline">
            <code className="api-url">{url}</code>
            <button className="btn ghost small" onClick={() => bridge?.sys.clipWrite(url)}>
              Copy
            </button>
            <input
              className="text-input narrow"
              value={portText}
              onChange={(e) => setPortText(e.target.value.replace(/\D/g, ""))}
              onBlur={() => {
                const n = Number(portText);
                if (n >= 1024 && n <= 65535) setSettings({ apiPort: n });
                else setPortText(String(s.apiPort || 11435));
              }}
              title="Port"
            />
          </div>
        </Row>
      )}
    </>
  );
}

const mb = (n: number) => `${(n / 1048576).toFixed(0)} MB`;

const ago = (t: number) => {
  const m = Math.round((Date.now() - t) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} minute${m === 1 ? "" : "s"} ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
};

/** Settings → Updates: the newest version, how and when to check, and where (electron/updates.ts). */
function UpdatesTab() {
  const s = useApp((x) => x.settings);
  const known = useApp((x) => x.update);
  const [feed, setFeed] = useState(s.updateFeed);
  const [res, setRes] = useState<UpdateCheck | null>(known);
  const [busy, setBusy] = useState<"check" | "get" | null>(null);
  const [got, setGot] = useState<{ got: number; total: number } | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => bridge?.update.onProgress((v) => setGot(v)), []);
  if (!bridge) return <p className="set-desc">Updates need the Pelica desktop app.</p>;
  const check = async () => {
    setBusy("check");
    setErr("");
    try {
      const r = await checkNow();
      setRes(r);
      if (r.error) setErr(r.error);
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setBusy(null);
    }
  };
  const install = async () => {
    if (!res?.latest) return;
    setBusy("get");
    setErr("");
    setGot(null);
    try {
      // the app closes, the installer runs and starts Pelica again
      await bridge!.update.install(res.latest);
    } catch (e: any) {
      setErr(String(e?.message || e).replace(/^Error invoking remote method '[^']*': (Error: )?/, ""));
      setBusy(null);
    }
  };
  const v = res?.latest?.version;
  const skipped = !!v && v === s.updateSkip;
  const title = res?.newer && v ? `Pelica ${v}${res.preview ? " (preview)" : ""} is available` : res && !res.error ? `Pelica ${res.current} is the newest version` : `You have Pelica ${VERSION}`;
  const status = err
    ? err
    : busy === "get"
      ? got?.total
        ? `Downloading… ${mb(got.got)} of ${mb(got.total)}`
        : got
          ? `Downloading… ${mb(got.got)}`
          : "Downloading…"
      : res?.newer
        ? `${skipped ? "You skipped this version, so Pelica doesn't mention it at start. " : ""}Install downloads it, checks it, and closes Pelica while it installs; Pelica starts again by itself. Your chats and settings stay.`
        : s.updateLast
          ? `Last checked ${ago(s.updateLast)}.`
          : "Not checked yet.";
  return (
    <>
      <Row title={title} desc={status}>
        <div className="inline">
          {res?.newer && !busy && (
            <button className="btn small ghost" onClick={() => setSettings({ updateSkip: skipped ? "" : v! })}>
              {skipped ? "Don't skip" : "Skip this version"}
            </button>
          )}
          {res?.newer && (
            <button className="btn small primary" disabled={!!busy} onClick={install}>
              {busy === "get" ? "Installing…" : "Install"}
            </button>
          )}
          {busy === "get" ? (
            <button className="btn small ghost" onClick={() => bridge?.update.cancel()}>
              Cancel
            </button>
          ) : (
            <button className="btn small" disabled={!!busy} onClick={check}>
              {busy === "check" ? "Checking…" : "Check now"}
            </button>
          )}
        </div>
      </Row>
      {res?.newer && res.latest?.notes?.length ? <ReleaseNotes releases={[{ version: res.latest.version, date: "", title: res.latest.title, notes: res.latest.notes }]} /> : null}
      <Section title="Options" />
      <Row title="Check for updates" desc="When Pelica asks for a newer version by itself, and says so when one is out. Nothing is downloaded until you choose Install.">
        <Seg value={s.updateCheck} options={[["start", "At start"], ["daily", "Once a day"], ["off", "Never"]]} onChange={(x) => setSettings({ updateCheck: x })} />
      </Row>
      <Row title="Preview versions" desc="Also offer versions marked as previews on GitHub: newer features, tried less. Off: only regular releases.">
        <Toggle on={s.updatePreview} set={(x) => (setSettings({ updatePreview: x }), setRes(null))} />
      </Row>
      {s.updateSkip && !(res?.newer && skipped) && (
        <Row title="Skipped version" desc={`Pelica ${s.updateSkip} isn't announced at start. A newer version is.`}>
          <button className="btn small ghost" onClick={() => setSettings({ updateSkip: "" })}>
            Don't skip
          </button>
        </Row>
      )}
      <Section title="Where updates come from" />
      <Row title="Releases" desc="Every version, with its installer and release notes, is on Pelica's GitHub page.">
        <button className="btn small" onClick={() => bridge?.sys.external("https://github.com/HumandubleOfficial/Pelica/releases")}>
          Open on GitHub
        </button>
      </Row>
      <Row title="Update feed" desc={<>The web address or folder of a <code>latest.json</code>, for builds you host yourself. Empty: Pelica's releases on GitHub.</>}>
        <input
          className="text-input wide"
          value={feed}
          onChange={(e) => setFeed(e.target.value)}
          onBlur={() => {
            if (feed.trim() === s.updateFeed) return;
            setSettings({ updateFeed: feed.trim() });
            setRes(null);
          }}
          placeholder="Pelica's GitHub releases"
        />
      </Row>
    </>
  );
}

function ModelsTab() {
  const s = useApp((x) => x.settings);
  const [dirs, setDirs] = useState(s.modelDirs.join("; "));
  const [auto, setAuto] = useState("");
  useEffect(() => setDirs(s.modelDirs.join("; ")), [s.modelDirs.join(";")]);
  useEffect(() => {
    bridge?.lib
      .hardware()
      .then((h) => setAuto(h.downloadDir))
      .catch(() => {});
  }, [s.downloadDir]);
  return (
    <>
      <Section title="Storage" />
      <StorageRows />
      <Section title="Folders" />
      <Row title="Model folders" desc="Where Pelica looks for models (separate with ;). Empty: C:\models and D:\Models. The download folder is always included.">
        <input className="text-input wide" value={dirs} onChange={(e) => setDirs(e.target.value)} onBlur={() => setSettings({ modelDirs: dirs.split(";").map((x) => x.trim()).filter(Boolean) })} placeholder="C:\models; D:\Models" />
      </Row>
      <Row title="Download folder" desc={s.downloadDir ? "Where the model library saves downloads." : `Where the model library saves downloads. Automatic: the default models folder on the drive with the most free space (now ${auto || "…"}).`}>
        <div className="inline">
          <input className="text-input" value={s.downloadDir} placeholder={auto ? `Automatic (${auto})` : "Automatic"} onChange={(e) => setSettings({ downloadDir: e.target.value })} />
          {s.downloadDir && (
            <button className="btn small ghost" onClick={() => setSettings({ downloadDir: "" })}>
              Automatic
            </button>
          )}
          <button className="btn small" onClick={async () => { const d = await bridge?.lib.pickDir(); if (d) setSettings({ downloadDir: d }); }}>
            Choose…
          </button>
        </div>
      </Row>
      <Section title="Loading" />
      <Row title="Load the last model at start" desc={s.lastModel ? `Off: Pelica starts with no model loaded, and you pick one from the model menu. On: it loads ${s.lastModel.split(/[\\/]/).pop()} as it starts.` : "Off: Pelica starts with no model loaded, and you pick one from the model menu."}>
        <Toggle on={s.loadOnStart} set={(v) => setSettings({ loadOnStart: v })} />
      </Row>
      <Row title="Speed up DeepSeek V4 and Qwen3.8 downloads" desc="Rearrange a DeepSeek V4 or Qwen3.8 file as soon as it finishes downloading, so each expert loads in one read (answers about 20% faster). Other programs can't open the file afterwards; Models → Undo speed-up puts it back.">
        <Toggle on={s.autoRepack} set={(v) => setSettings({ autoRepack: v })} />
      </Row>
      <Section title="Local API" />
      <ApiRows />
      <Section title="Engine" />
      <Row title="Restart the engine" desc="Restarts pelica.exe, after changing folders or rebuilding it.">
        <button className="btn small" onClick={() => bridge?.engine.restart()}>
          Restart engine
        </button>
      </Row>
    </>
  );
}

function DevTab() {
  const [log, setLog] = useState("");
  const [info, setInfo] = useState<any>(null);
  useEffect(() => {
    bridge?.engine.log().then(setLog);
    bridge?.app.info().then(setInfo);
  }, []);
  return (
    <>
      {info && (
        <>
          <Row title="Pelica Desktop">
            <span className="muted mono small">{info.version}</span>
          </Row>
          <Row title="Electron">
            <span className="muted mono small">{info.electron}</span>
          </Row>
          <Row title="Engine">
            <span className="muted mono small set-path">{info.pelica}</span>
          </Row>
          <Row title="Your data">
            <span className="inline">
              <span className="muted mono small set-path">{info.userDir}</span>
              <button className="btn ghost small" onClick={() => bridge?.sys.reveal(info.userDir)}>
                Show
              </button>
            </span>
          </Row>
        </>
      )}
      <div className="set-section set-section-row">
        Engine log
        <button className="btn ghost small" onClick={() => bridge?.engine.log().then(setLog)}>
          Refresh
        </button>
      </div>
      <pre className="log-pre">{log.slice(-12000) || "(empty)"}</pre>
    </>
  );
}

export function SettingsModal() {
  const open = useApp((x) => x.settingsOpen);
  const tab = useApp((x) => x.settingsTab);
  const s = useApp((x) => x.settings);
  const st = useApp((x) => x.status);
  const [q, setQ] = useState("");
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ settingsOpen: false });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
  if (!open) return null;
  const set = (p: Partial<S>) => setSettings(p);
  const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const shown = TABS.filter((t) => words.every((w) => `${t.name} ${t.group} ${t.keys}`.toLowerCase().includes(w)));
  let body: React.ReactNode = null;
  switch (tab) {
    case "General":
      body = (
        <>
          <Row title="What should Pelica call you?">
            <input className="text-input" value={s.userName} placeholder="Your name" onChange={(e) => set({ userName: e.target.value })} />
          </Row>
          <Row title="Instructions for every chat" desc="How you like answers: tone, format, things to always keep in mind.">
            <textarea className="text-input wide" rows={4} value={s.customInstructions} onChange={(e) => set({ customInstructions: e.target.value })} placeholder="e.g. Keep answers short. I use Windows and Python 3.12." />
          </Row>
          <Row title="Send with">
            <Seg value={s.sendOnEnter} options={[[true, "Enter"], [false, "Ctrl+Enter"]]} onChange={(v) => set({ sendOnEnter: v })} />
          </Row>
          <Row title="Reply length limit" desc="Tokens per reply at most.">
            <Seg value={s.maxTokens} options={[[2048, "2K"], [4096, "4K"], [8192, "8K"], [16384, "16K"], [32768, "32K"]]} onChange={(v) => set({ maxTokens: v })} />
          </Row>
          <Section title="Saved prompts" />
          <p className="set-intro">Text you send often. Type / and its name in the message box, or pick it from the + menu or the palette (Ctrl+K).</p>
          <PromptList />
          <Section title="Your chats" />
          <BackupRow />
        </>
      );
      break;
    case "Appearance":
      body = (
        <>
          <div className="set-row set-row-block">
            <div className="set-title">Theme</div>
            <ThemePicker value={s.theme} onChange={(v) => set({ theme: v })} />
          </div>
          <Row title="Background" desc="Live experts: the model's experts as a grid behind the chat, lighting up as the model uses them (mixture-of-experts models). The Neural theme turns it on.">
            <Seg value={s.background} options={[["none", "Plain"], ["experts", "Live experts"]]} onChange={(v) => set({ background: v })} />
          </Row>
          <Row title="Window backdrop" desc="Windows 11's Mica or Acrylic material shows softly through the window.">
            <Seg value={bridge ? s.material : "none"} options={[["none", "Solid"], ["mica", "Mica"], ["acrylic", "Acrylic"]]} onChange={(v) => setMaterial(v)} />
          </Row>
          <Row title="Text size" desc={`${SIZES.find((x) => x[0] === s.fontSize)?.[1] ?? "Default"}. Ctrl+= and Ctrl+- change it anywhere; Ctrl+0 goes back to Default.`}>
            <SizePicker value={s.fontSize} onChange={(v) => set({ fontSize: v })} />
          </Row>
          <Row title="Reply font" desc="Replies in a serif face, as in Claude, or the same sans as the rest of the app.">
            <Seg value={s.replyFont} options={[["serif", "Serif"], ["sans", "Sans"]]} onChange={(v) => set({ replyFont: v })} />
          </Row>
          <Row title="Animations" desc="Effort orbs, the shimmer while working, the expert grid.">
            <Toggle on={s.animations} set={(v) => set({ animations: v })} />
          </Row>
          <Section title="Code appearance" />
          <div className="set-row set-row-block code-preview">
            <Diff before={PREVIEW_A} after={PREVIEW_B} path="greet.ts" />
          </div>
          <Row title="Code font" desc="A monospace font for code, diffs and the terminal.">
            <input className="text-input" value={s.codeFont} placeholder="e.g. JetBrains Mono" onChange={(e) => set({ codeFont: e.target.value })} spellCheck={false} />
          </Row>
        </>
      );
      break;
    case "Capabilities":
      body = (
        <>
          <Row title="Web search" desc="Lets every model search the web and read pages. Models are told to search only when a question needs current or outside facts; greetings and small talk never get tools.">
            <Toggle on={s.web} set={(v) => set({ web: v })} />
          </Row>
          <Row title="Steps per task" desc="How many tool calls a task may take before Pelica pauses. Long tasks (building an app) need many.">
            <Seg value={s.maxSteps} options={[[25, "25"], [60, "60"], [150, "150"], [400, "400"]]} onChange={(v) => set({ maxSteps: v })} />
          </Row>
          <Row title="Thinking effort" desc={st?.efforts?.length ? `This model: ${st.efforts.join(", ")}.` : "The loaded model has no effort levels."}>
            {st?.efforts?.length ? <Seg value={st.efforts.includes(s.effort) ? s.effort : st.effort || st.efforts[0]} options={st.efforts.map((e) => [e, e[0].toUpperCase() + e.slice(1)] as [string, string])} onChange={(v) => set({ effort: v })} /> : <span className="muted">—</span>}
          </Row>
        </>
      );
      break;
    case "Desktop":
      body = <DesktopTab />;
      break;
    case "Permissions":
      body = (
        <>
          <p className="set-intro">Switch any time from the chip under the message box, or with <kbd>Shift</kbd>+<kbd>Tab</kbd> in it.</p>
          <div className="perm-cards">
            {PERM_MODES.map((m) => (
              <button key={m.id} className={`perm-card ${s.permMode === m.id ? "sel" : ""}`} onClick={() => set({ permMode: m.id })}>
                <span className={`perm-ico perm-${m.id}`}>
                  <Icon name={m.id === "plan" ? "list" : m.id === "auto" ? "bolt" : m.id === "ask" ? "lock" : "shield"} size={16} />
                </span>
                <b>{m.label}</b>
                <span className="muted small">{m.desc}</span>
              </button>
            ))}
          </div>
          {s.permMode === "auto" && <div className="warn-text">Bypass permissions runs commands without asking. Use it for folders you can afford to lose: deletes still go to the Recycle Bin, but a command can do anything you can.</div>}
          <Row title="Folders" desc="Each chat has its own folder, picked from the folder chip under the message box. Files and commands stay inside that chat's folder; paths outside it are refused.">
            <span className="muted small">{s.recentFolders.length ? `${s.recentFolders.length} recent` : "none yet"}</span>
          </Row>
          <Row title="Deleting" desc="Files Pelica deletes go to the Recycle Bin, in every mode.">
            <span className="pill">Recycle Bin</span>
          </Row>
          <Row title="Command shell">
            <Seg value={s.shell} options={[["cmd", "cmd.exe"], ["powershell", "PowerShell"]]} onChange={(v) => set({ shell: v })} />
          </Row>
        </>
      );
      break;
    case "Context":
      body = (
        <>
          <Row title="Auto-compact" desc="When the context is nearly full, the conversation is summarized (by the model) and continues from the summary.">
            <Toggle on={s.autocompact} set={(v) => set({ autocompact: v })} />
          </Row>
          <Row title="Compact at" desc="How full the context gets before it is summarized.">
            <Seg value={s.compactAt} options={[[0.6, "60%"], [0.7, "70%"], [0.8, "80%"], [0.9, "90%"]]} onChange={(v) => set({ compactAt: v })} />
          </Row>
          <Row title="Trim old tool output" desc="Search results, pages and files from earlier turns are cut to a short preview, so long tasks fill the context much more slowly.">
            <Toggle on={s.trimToolResults} set={(v) => set({ trimToolResults: v })} />
          </Row>
        </>
      );
      break;
    case "CQS":
      body = (
        <>
          <p className="set-intro">CQS (code quality scanner) checks code after Pelica writes it: syntax, brackets, tags, leftover placeholders, and your project's own compilers. It hands what it finds to a fixer subagent, then checks again. Run it any time with /cqs, or /cqs polish to also tidy code and UI.</p>
          <Row title="Scan after changes" desc="Run CQS when a reply changed files.">
            <Toggle on={s.cqsAuto} set={(v) => set({ cqsAuto: v })} />
          </Row>
          <Row title="Fix what it finds" desc="Send errors to the fixer subagent (otherwise just report them).">
            <Toggle on={s.cqsFix} set={(v) => set({ cqsFix: v })} />
          </Row>
          <Row title="Run checkers without asking" desc="Your project's own checkers (tsc, eslint, python, node --check…) only read the code, so CQS runs them quietly. Cargo still asks (build scripts can run code), and Manual mode asks for everything.">
            <Toggle on={s.cqsQuiet} set={(v) => set({ cqsQuiet: v })} />
          </Row>
          <p className="set-note">Besides syntax and compilers, CQS catches files that are imported or linked but were never written, and has the fixer create them. It runs up to two fix rounds, checking again after each.</p>
        </>
      );
      break;
    case "Connectors":
      body = <ConnectorsTab />;
      break;
    case "Models":
      body = <ModelsTab />;
      break;
    case "Devices":
      body = <DevicesTab />;
      break;
    case "Usage":
      body = <UsageTab />;
      break;
    case "Developer":
      body = <DevTab />;
      break;
    case "Updates":
      body = <UpdatesTab />;
      break;
    case "About":
      body = (
        <div className="about">
          <div className="about-hero">
            <img src="logo.svg" alt="" className="about-logo" />
            <div>
              <div className="about-name">Pelica {VERSION}</div>
              <div className="muted small">{st?.gpu ? `Running on ${st.gpu}` : "Mixture-of-experts models on your own PC"}</div>
            </div>
            <button className="btn small ghost about-gh" onClick={() => bridge?.sys.external("https://github.com/HumandubleOfficial/Pelica")}>
              GitHub
            </button>
          </div>
          <p className="set-intro">
            Pelica runs mixture-of-experts models on your own GPU, with the experts that don't fit in VRAM kept in RAM and long contexts spilling to RAM and disk. Nothing leaves your PC except the web searches and pages you allow, the update check on GitHub (Settings → Updates), and what you ask the online connectors you add.
          </p>
          <p className="set-intro">Pelica is open source under the MIT License.</p>
          <Section title="Release notes" />
          <ReleaseNotes releases={RELEASES.filter((r) => r.version !== "next")} />
        </div>
      );
      break;
  }
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ settingsOpen: false })}>
      <div className="modal settings">
        <div className="set-top">
          <span>Settings</span>
          <button className="ghost-btn" onClick={() => app.set({ settingsOpen: false })} aria-label="Close settings">
            <Icon name="x" size={17} />
          </button>
        </div>
        <div className="set-main">
          <nav className="set-nav">
            <label className="set-search">
              <Icon name="search" size={15} />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && shown[0]) app.set({ settingsTab: shown[0].name });
                  if (e.key === "Escape" && q) (e.stopPropagation(), e.nativeEvent.stopImmediatePropagation(), setQ(""));
                }}
                placeholder="Search"
                spellCheck={false}
              />
            </label>
            <div className="set-nav-list">
              {shown.map((t, i) => (
                <div key={t.name} className="set-nav-item">
                  {t.group !== shown[i - 1]?.group && <div className="set-group">{t.group}</div>}
                  <button className={t.name === tab ? "sel" : ""} onClick={() => app.set({ settingsTab: t.name })}>
                    <Icon name={t.icon} size={16} />
                    {t.name}
                  </button>
                </div>
              ))}
              {!shown.length && <div className="set-none">No settings match.</div>}
            </div>
          </nav>
          <div className="set-body">
            <div className="set-page" key={tab}>
              <div className="set-section set-page-title">{tab}</div>
              {body}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
