// Settings, laid out like the Claude app's: a left nav and rows of
// controls.
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

const PREVIEW_A = 'function greet(name: string) {\n  return "Hello, " + name;\n}\n';
const PREVIEW_B = "function greet(name: string) {\n  return `Hello, ${name}!`;\n}\n";

const TABS: { name: string; icon: string }[] = [
  { name: "General", icon: "user" },
  { name: "Appearance", icon: "palette" },
  { name: "Capabilities", icon: "bolt" },
  { name: "Desktop", icon: "monitor" },
  { name: "Permissions", icon: "shield" },
  { name: "Context", icon: "gauge" },
  { name: "CQS", icon: "scan" },
  { name: "Connectors", icon: "plug" },
  { name: "Models", icon: "library" },
  { name: "Devices", icon: "monitor" },
  { name: "Usage", icon: "spark" },
  { name: "Developer", icon: "terminal" },
  { name: "About", icon: "info" },
];

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
      <p className="set-desc">
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
      <Row title="Local API" desc="Lets other programs on this PC (editors, scripts, chat apps) use the loaded model, as an OpenAI-compatible server. Nothing outside this computer can reach it.">
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

/** About → Updates: the feed, a check, and the download-and-install (electron/updates.ts). */
function UpdateRows() {
  const s = useApp((x) => x.settings);
  const known = useApp((x) => x.update);
  const [feed, setFeed] = useState(s.updateFeed);
  const [res, setRes] = useState<UpdateCheck | null>(known);
  const [busy, setBusy] = useState<"check" | "get" | null>(null);
  const [got, setGot] = useState<{ got: number; total: number } | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => bridge?.update.onProgress((v) => setGot(v)), []);
  const check = async () => {
    setBusy("check");
    setErr("");
    try {
      const r = await bridge!.update.check();
      setRes(r);
      app.set({ update: r.newer ? r : null });
      if (r.error) setErr(r.error);
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
  const status = err
    ? err
    : busy === "get"
      ? got?.total
        ? `Downloading… ${mb(got.got)} of ${mb(got.total)}`
        : got
          ? `Downloading… ${mb(got.got)}`
          : "Downloading…"
      : res?.newer
        ? "Install downloads it, checks it, and closes Pelica while it installs; Pelica starts again by itself. Your chats and settings stay."
        : undefined;
  return (
    <>
      <Row title="Update feed" desc={<>Where Pelica looks for new versions: the web address or folder of the <code>latest.json</code> that comes with each installer. Empty: Pelica never checks.</>}>
        <input
          className="text-input wide"
          value={feed}
          onChange={(e) => setFeed(e.target.value)}
          onBlur={() => {
            if (feed.trim() === s.updateFeed) return;
            setSettings({ updateFeed: feed.trim() });
            setRes(null);
          }}
          placeholder="https://…/latest.json"
        />
      </Row>
      {s.updateFeed && (
        <Row title="Check when Pelica starts" desc="Asks the feed once at start and says when a newer version is out. Nothing is downloaded until you choose Install.">
          <Toggle on={s.updateAuto} set={(v) => setSettings({ updateAuto: v })} />
        </Row>
      )}
      {s.updateFeed && (
        <Row title={res?.newer && res.latest ? `Pelica ${res.latest.version} is available` : res && !res.error ? `Pelica ${res.current} is the newest version` : "Updates"} desc={status}>
          <div className="inline">
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
      )}
      {res?.newer && res.latest?.notes?.length ? <ReleaseNotes releases={[{ version: res.latest.version, date: "", title: res.latest.title, notes: res.latest.notes }]} /> : null}
    </>
  );
}

function ModelsTab() {
  const s = useApp((x) => x.settings);
  const [dirs, setDirs] = useState(s.modelDirs.join("; "));
  return (
    <>
      <Row title="Model folders" desc="Where Pelica looks for .gguf files (separate with ;). Empty: C:\models and D:\Models. Takes effect after restarting the engine.">
        <input className="text-input wide" value={dirs} onChange={(e) => setDirs(e.target.value)} onBlur={() => setSettings({ modelDirs: dirs.split(";").map((x) => x.trim()).filter(Boolean) })} placeholder="C:\models; D:\Models" />
      </Row>
      <Row title="Load the last model at start" desc={s.lastModel ? `Off: Pelica starts with no model loaded, and you pick one from the model menu. On: it loads ${s.lastModel.split(/[\\/]/).pop()} as it starts.` : "Off: Pelica starts with no model loaded, and you pick one from the model menu."}>
        <Toggle on={s.loadOnStart} set={(v) => setSettings({ loadOnStart: v })} />
      </Row>
      <Row title="Download folder" desc="Where the model library saves downloads.">
        <div className="inline">
          <input className="text-input" value={s.downloadDir} onChange={(e) => setSettings({ downloadDir: e.target.value })} />
          <button className="btn small" onClick={async () => { const d = await bridge?.lib.pickDir(); if (d) setSettings({ downloadDir: d }); }}>
            Choose…
          </button>
        </div>
      </Row>
      <Row title="Speed up DeepSeek V4 and Qwen3.8 downloads" desc="Rearrange a DeepSeek V4 or Qwen3.8 file as soon as it finishes downloading, so each expert loads in one read (answers about 20% faster). Other programs can't open the file afterwards; Models → Undo speed-up puts it back.">
        <Toggle on={s.autoRepack} set={(v) => setSettings({ autoRepack: v })} />
      </Row>
      <ApiRows />
      <Row title="Engine" desc="Restart pelica.exe (after changing folders or rebuilding it).">
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
        <div className="muted small">
          Pelica Desktop {info.version} · Electron {info.electron} · engine {info.pelica} · data {info.userDir}
        </div>
      )}
      <div className="set-title" style={{ marginTop: 14 }}>
        Engine log
      </div>
      <pre className="log-pre">{log.slice(-12000) || "(empty)"}</pre>
      <button className="btn small" onClick={() => bridge?.engine.log().then(setLog)}>
        Refresh
      </button>
    </>
  );
}

export function SettingsModal() {
  const open = useApp((x) => x.settingsOpen);
  const tab = useApp((x) => x.settingsTab);
  const s = useApp((x) => x.settings);
  const st = useApp((x) => x.status);
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ settingsOpen: false });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
  if (!open) return null;
  const set = (p: Partial<S>) => setSettings(p);
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
          <div className="set-section">Saved prompts</div>
          <p className="set-desc">Text you send often. Type / and its name in the message box, or pick it from the + menu or the palette (Ctrl+K).</p>
          <PromptList />
          <div className="set-section">Your chats</div>
          <BackupRow />
        </>
      );
      break;
    case "Appearance":
      body = (
        <>
          <Row title="Theme">
            <Seg value={s.theme} options={[["dark", "Dark"], ["warm", "Warm"], ["light", "Light"], ["midnight", "Midnight"], ["neural", "Neural"]]} onChange={(v) => set({ theme: v })} />
          </Row>
          <Row title="Background" desc="Live experts: the model's experts as a grid behind the chat, lighting up as the model uses them (mixture-of-experts models). The Neural theme turns it on.">
            <Seg value={s.background} options={[["none", "Plain"], ["experts", "Live experts"]]} onChange={(v) => set({ background: v })} />
          </Row>
          <Row title="Window backdrop" desc="Windows 11's Mica or Acrylic material shows softly through the window.">
            <Seg value={bridge ? s.material : "none"} options={[["none", "Solid"], ["mica", "Mica"], ["acrylic", "Acrylic"]]} onChange={(v) => setMaterial(v)} />
          </Row>
          <Row title="Text size" desc="Ctrl+= and Ctrl+- change it anywhere; Ctrl+0 goes back to Default.">
            <Seg value={s.fontSize} options={[[13, "Smaller"], [14, "Small"], [15, "Default"], [16, "Large"], [17, "Larger"], [18, "Largest"]]} onChange={(v) => set({ fontSize: v })} />
          </Row>
          <Row title="Reply font" desc="Replies in a serif face, as in Claude, or the same sans as the rest of the app.">
            <Seg value={s.replyFont} options={[["serif", "Serif"], ["sans", "Sans"]]} onChange={(v) => set({ replyFont: v })} />
          </Row>
          <div className="set-section">Code appearance</div>
          <div className="code-preview">
            <Diff before={PREVIEW_A} after={PREVIEW_B} path="greet.ts" />
          </div>
          <Row title="Code font" desc="A monospace font for code, diffs and the terminal.">
            <input className="text-input" value={s.codeFont} placeholder="e.g. JetBrains Mono" onChange={(e) => set({ codeFont: e.target.value })} spellCheck={false} />
          </Row>
          <Row title="Animations" desc="Effort orbs, the shimmer while working, the expert grid.">
            <Toggle on={s.animations} set={(v) => set({ animations: v })} />
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
          <p className="set-desc">What Pelica may do without asking you. Switch any time from the chip under the message box, or with <kbd>Shift</kbd>+<kbd>Tab</kbd> in it.</p>
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
          <Row title="Compact at">
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
          <p className="set-desc">CQS (code quality scanner) checks code after Pelica writes it: syntax, brackets, tags, leftover placeholders, and your project's own compilers. It hands what it finds to a fixer subagent, then checks again. Run it any time with /cqs, or /cqs polish to also tidy code and UI.</p>
          <Row title="Scan after changes" desc="Run CQS when a reply changed files.">
            <Toggle on={s.cqsAuto} set={(v) => set({ cqsAuto: v })} />
          </Row>
          <Row title="Fix what it finds" desc="Send errors to the fixer subagent (otherwise just report them).">
            <Toggle on={s.cqsFix} set={(v) => set({ cqsFix: v })} />
          </Row>
          <Row title="Run checkers without asking" desc="Your project's own checkers (tsc, eslint, python, node --check…) only read the code, so CQS runs them quietly. Cargo still asks (build scripts can run code), and Manual mode asks for everything.">
            <Toggle on={s.cqsQuiet} set={(v) => set({ cqsQuiet: v })} />
          </Row>
          <p className="set-desc">Besides syntax and compilers, CQS catches files that are imported or linked but were never written, and has the fixer create them. It runs up to two fix rounds, checking again after each.</p>
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
    case "About":
      body = (
        <div className="about">
          <p>
            <b>Pelica</b> runs mixture-of-experts models on your own GPU, with the experts that don't fit in VRAM kept in RAM and long contexts spilling to RAM and disk. Nothing leaves your PC except the web searches and pages you allow.
          </p>
          <p className="muted small">{st?.gpu ? `GPU: ${st.gpu}` : ""}</p>
          <div className="set-section">Updates</div>
          <UpdateRows />
          <div className="set-section">Pelica {VERSION}: release notes</div>
          <ReleaseNotes releases={RELEASES.filter((r) => r.version !== "next")} />
        </div>
      );
      break;
  }
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ settingsOpen: false })}>
      <div className="modal settings">
        <nav className="set-nav">
          <div className="set-head">Settings</div>
          {TABS.map((t) => (
            <button key={t.name} className={t.name === tab ? "sel" : ""} onClick={() => app.set({ settingsTab: t.name })}>
              <Icon name={t.icon} size={15} />
              {t.name}
            </button>
          ))}
        </nav>
        <div className="set-body">
          <div className="set-body-head">
            <h2>{tab}</h2>
            <button className="ghost-btn" onClick={() => app.set({ settingsOpen: false })}>
              <Icon name="x" size={16} />
            </button>
          </div>
          {body}
        </div>
      </div>
    </div>
  );
}
