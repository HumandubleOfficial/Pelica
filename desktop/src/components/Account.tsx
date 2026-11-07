// Settings → Account: sign in with Google, and sync your chats, projects,
// memory and saved prompts between your PCs through your own Google Drive
// (electron/google.ts, electron/sync.ts). Signing in uses Pelica's own Google
// client; a build without one asks for yours, set up with the steps shown here.
import { useEffect, useState } from "react";
import { bridge } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";
import { SignInHelp } from "./SignInHelp";

const ipcErr = (e: any) =>
  String(e?.message || e).replace(
    /^Error invoking remote method '[^']+': (Error: )?/,
    "",
  );

const ago = (t: number) => {
  if (!t) return "never";
  const s = Math.round((Date.now() - t) / 1000);
  return s < 60
    ? "just now"
    : s < 3600
      ? `${Math.round(s / 60)} min ago`
      : s < 86400
        ? `${Math.round(s / 3600)} h ago`
        : new Date(t).toLocaleDateString();
};

const size = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const plural = (n: number, one: string) => `${n.toLocaleString()} ${one}${n === 1 ? "" : "s"}`;

/** Refreshes the account's status (the sidebar chip and this page read it). */
export async function loadAccount() {
  if (!bridge) return;
  app.set({ account: await bridge.account.status().catch(() => null) });
}

/** The Google "G", drawn as Google draws it. */
export function GoogleG() {
  return (
    <svg width="16" height="16" viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#EA4335"
        d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.6 5.4 2.7 13.3l7.9 6.1C12.5 13.6 17.8 9.5 24 9.5z"
      />
      <path
        fill="#4285F4"
        d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.4c-.5 2.9-2.2 5.3-4.6 7l7.2 5.6c4.2-3.9 7.1-9.6 7.1-17.1z"
      />
      <path
        fill="#FBBC05"
        d="M10.6 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-7.9-6.1C1 16.6 0 20.2 0 24s1 7.4 2.7 10.7l7.9-6.1z"
      />
      <path
        fill="#34A853"
        d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.2-5.6c-2 1.4-4.7 2.3-8.7 2.3-6.2 0-11.5-4.1-13.4-9.8l-7.9 6.1C6.6 42.6 14.6 48 24 48z"
      />
    </svg>
  );
}

type RowT = (p: { title: string; desc?: React.ReactNode; children: React.ReactNode }) => React.ReactElement;
type ToggleT = (p: { on: boolean; set: (v: boolean) => void }) => React.ReactElement;

/** Who you are: on this PC, signed in or not. */
function ProfileRows({ Row, Toggle, picture }: { Row: RowT; Toggle: ToggleT; picture: boolean }) {
  const name = useApp((s) => s.settings.userName);
  const show = useApp((s) => s.settings.accountPicture);
  const [text, setText] = useState(name);
  useEffect(() => setText(name), [name]);
  return (
    <>
      <div className="set-section">Profile</div>
      <Row title="Your name" desc="What Pelica calls you, here and in its replies.">
        <input
          className="text-input acct-field"
          value={text}
          placeholder="Your name"
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => text.trim() !== name && setSettings({ userName: text.trim() })}
          onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
        />
      </Row>
      {picture && (
        <Row title="Show your Google picture" desc="In the sidebar. Off: the first letter of your name.">
          <Toggle on={show} set={(v) => setSettings({ accountPicture: v })} />
        </Row>
      )}
    </>
  );
}

/** What travels between your PCs, one kind at a time. */
function SyncChoices({ Row, Toggle }: { Row: RowT; Toggle: ToggleT }) {
  const skip = useApp((s) => s.settings.syncSkip);
  const on = useApp((s) => s.settings.syncOn);
  const chats = useApp((s) => s.chats.length);
  const projects = useApp((s) => s.settings.projects.length);
  const memories = useApp((s) => s.settings.memories.length);
  const prompts = useApp((s) => s.settings.prompts.length);
  if (!on) return null;
  const kinds: [string, string, string][] = [
    ["chats", "Chats", plural(chats, "chat")],
    ["projects", "Projects", plural(projects, "project")],
    ["memory", "Memory", plural(memories, "memory").replace("memorys", "memories")],
    ["prompts", "Saved prompts", plural(prompts, "prompt")],
    ["profile", "Name and instructions", "What Pelica calls you, and your custom instructions"],
  ];
  return (
    <>
      <div className="set-section">What syncs</div>
      <p className="set-intro">Something turned off stays as it is on each PC. What's already in your Drive isn't removed.</p>
      {kinds.map(([id, title, desc]) => (
        <Row key={id} title={title} desc={desc}>
          <Toggle on={!skip.includes(id)} set={(v) => setSettings({ syncSkip: v ? skip.filter((x) => x !== id) : [...skip, id] })} />
        </Row>
      ))}
    </>
  );
}

/** What's where, a backup, and emptying Drive. */
function DataRows({ Row, signedIn }: { Row: RowT; signedIn: boolean }) {
  const chats = useApp((s) => s.chats.length);
  const projects = useApp((s) => s.settings.projects.length);
  const memories = useApp((s) => s.settings.memories.length);
  const [drive, setDrive] = useState<{ files: number; bytes: number } | null>(null);
  const [driveErr, setDriveErr] = useState("");
  const [sure, setSure] = useState(false);
  const [wiping, setWiping] = useState(false);
  const look = () =>
    bridge?.account
      .usage()
      .then((u) => (setDrive(u), setDriveErr("")))
      .catch((e) => setDriveErr(ipcErr(e)));
  useEffect(() => {
    if (signedIn) look();
  }, [signedIn]);
  async function wipe() {
    setWiping(true);
    try {
      app.set({ account: await bridge!.account.wipe() });
      setDrive({ files: 0, bytes: 0 });
      toast("Your Google Drive folder is empty, and syncing is off. Everything is still on this PC.");
    } catch (e) {
      toast(ipcErr(e));
    }
    setWiping(false);
    setSure(false);
  }
  return (
    <>
      <div className="set-section">Your data</div>
      <Row title="On this PC" desc={`${plural(chats, "chat")}, ${plural(projects, "project")} and ${memories === 1 ? "1 memory" : `${memories.toLocaleString()} memories`}.`}>
        <button
          className="btn"
          disabled={!bridge}
          onClick={async () => {
            try {
              const r = await bridge!.chats.backup();
              if (r) toast(`Saved a backup of ${plural(r.count, "chat")}.`);
            } catch (e) {
              toast(ipcErr(e));
            }
          }}
        >
          <Icon name="download" size={13} /> Back up chats
        </button>
      </Row>
      {signedIn && (
        <>
          <Row
            title="In your Google Drive"
            desc={
              driveErr ? (
                <span className="danger-text">{driveErr}</span>
              ) : drive ? (
                drive.files ? (
                  `${plural(drive.files, "file")}, ${size(drive.bytes)}, in a hidden folder only Pelica can see.`
                ) : (
                  "Nothing yet."
                )
              ) : (
                "Looking…"
              )
            }
          >
            <button className="btn" onClick={look}>
              <Icon name="reload" size={13} /> Refresh
            </button>
          </Row>
          {sure ? (
            <div className="acct-sure">
              <span>This deletes Pelica's folder from your Google Drive for good, and turns syncing off. Your other PCs keep their copies. Delete it?</span>
              <button className="btn small danger" disabled={wiping} onClick={wipe}>
                {wiping ? "Deleting…" : "Delete from Drive"}
              </button>
              <button className="btn small ghost" disabled={wiping} onClick={() => setSure(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <Row title="Delete your data from Google Drive" desc="Empties Pelica's folder and turns syncing off. Everything stays on this PC.">
              <button className="btn" disabled={!drive?.files} onClick={() => setSure(true)}>
                Delete…
              </button>
            </Row>
          )}
          <Row title="Your Google account" desc="Your password, security, and which apps can use your account are managed by Google. Signing out here also takes back Pelica's access.">
            <button className="btn" onClick={() => bridge?.sys.external("https://myaccount.google.com/")}>
              <Icon name="external" size={13} /> Manage
            </button>
          </Row>
        </>
      )}
    </>
  );
}

export function AccountTab({
  Row,
  Toggle,
}: {
  Row: (p: {
    title: string;
    desc?: React.ReactNode;
    children: React.ReactNode;
  }) => React.ReactElement;
  Toggle: (p: { on: boolean; set: (v: boolean) => void }) => React.ReactElement;
}) {
  const acct = useApp((s) => s.account);
  const syncOn = useApp((s) => s.settings.syncOn);
  const clientId = useApp((s) => s.settings.googleClientId);
  const secret = useApp((s) => s.settings.googleClientSecret);
  const [busy, setBusy] = useState(false);
  // (the setup stays open while you type, until Done)
  const [setup, setSetup] = useState(false);
  const [, tick] = useState(0);
  useEffect(() => {
    loadAccount();
    const t = setInterval(() => (loadAccount(), tick((n) => n + 1)), 3000);
    return () => clearInterval(t);
  }, []);
  const p = acct?.profile;
  const ready = !!clientId.trim() || !!acct?.clientId;
  const own = !!clientId.trim();
  const showSetup = setup || (!!acct && !ready);
  async function signIn() {
    setBusy(true);
    try {
      app.set({ account: await bridge!.account.signIn() });
      toast("Signed in. Your chats are syncing.");
    } catch (e) {
      toast(ipcErr(e));
    }
    setBusy(false);
  }
  return (
    <>
      {p ? (
        <div className="acct-card">
          {p.picture ? (
            <img className="acct-pic" src={p.picture} alt="" />
          ) : (
            <span className="acct-pic avatar">
              {p.name.slice(0, 1).toUpperCase()}
            </span>
          )}
          <div className="acct-who">
            <b>{p.name}</b>
            <span className="muted small">{p.email}</span>
          </div>
          <button
            className="btn ghost"
            onClick={async () => {
              app.set({ account: await bridge!.account.signOut() });
              toast("Signed out. Your chats stay on this PC.");
            }}
          >
            Sign out
          </button>
        </div>
      ) : (
        <div className="acct-card signed-out">
          <div className="acct-who">
            <b>Sign in to Pelica</b>
            <span className="muted small">
              Your account is your Google account. Signing in syncs your chats,
              projects, memory and saved prompts between your PCs, through a
              hidden folder in your own Google Drive. Pelica sees only that
              folder, not your files.
            </span>
          </div>
          <button
            className="btn google-btn"
            disabled={!bridge || busy || !ready}
            onClick={signIn}
            title={ready ? "" : "Set up Google sign-in below first"}
          >
            <GoogleG />{" "}
            {busy ? "Waiting for your browser…" : "Sign in with Google"}
          </button>
          {busy && <SignInHelp />}
        </div>
      )}
      <ProfileRows Row={Row} Toggle={Toggle} picture={!!p?.picture} />
      {p && (
        <>
          <div className="set-section">Sync</div>
          <Row
            title="Sync between your PCs"
            desc="Chats, projects, memory, saved prompts, your name and instructions. Models, folders and connectors stay on each PC. A chat changed on two PCs keeps both versions; one deleted on another PC goes to this PC's Recycle Bin."
          >
            <Toggle on={syncOn} set={(v) => setSettings({ syncOn: v })} />
          </Row>
          <Row
            title="Last synced"
            desc={
              acct?.sync.error ? (
                <span className="danger-text">{acct.sync.error}</span>
              ) : acct?.sync.busy ? (
                "Syncing…"
              ) : (
                `${ago(acct?.sync.lastSync || 0)}. It syncs a few seconds after a change and every 3 minutes.`
              )
            }
          >
            <button
              className="btn"
              disabled={!syncOn || acct?.sync.busy}
              onClick={async () => {
                try {
                  app.set({ account: await bridge!.account.syncNow() });
                } catch (e) {
                  toast(ipcErr(e));
                }
              }}
            >
              <Icon name="reload" size={13} /> Sync now
            </button>
          </Row>
          <SyncChoices Row={Row} Toggle={Toggle} />
        </>
      )}
      <DataRows Row={Row} signedIn={!!p} />
      {acct?.builtIn && !own && !setup ? (
        <div className="row-end">
          <button className="btn small ghost" onClick={() => setSetup(true)}>
            Use your own Google client…
          </button>
        </div>
      ) : (
        <div className="set-section set-section-row">
          Google sign-in
          {ready && !showSetup && (
            <button className="btn small ghost" onClick={() => setSetup(true)}>
              Change
            </button>
          )}
        </div>
      )}
      {acct?.builtIn && !own && !setup ? null : !showSetup ? (
        <p className="set-intro">
          {own
            ? `Set up with your own Google client (${clientId.slice(0, 12)}…).`
            : "Signing in uses Pelica's own Google client."}
        </p>
      ) : (
        <>
          <p className="set-intro">
            Pelica signs in with a Google client you own (free, once, about 5
            minutes):
          </p>
          <ol className="acct-steps">
            <li>
              Open{" "}
              <a
                href="https://console.cloud.google.com/projectcreate"
                onClick={(e) => (
                  e.preventDefault(),
                  bridge?.sys.external(
                    "https://console.cloud.google.com/projectcreate",
                  )
                )}
              >
                Google Cloud Console
              </a>{" "}
              and create a project (any name, e.g. Pelica).
            </li>
            <li>
              In APIs &amp; Services → Library, turn on the{" "}
              <b>Google Drive API</b>.
            </li>
            <li>
              In APIs &amp; Services → OAuth consent screen (Google Auth
              Platform), choose External, fill in the app name and your email,
              and add yourself under Test users (Audience). Add the scope{" "}
              <code>.../auth/drive.appdata</code> under Data access.
            </li>
            <li>
              In Clients (Credentials), create an OAuth client of type{" "}
              <b>Desktop app</b>, and copy its client ID and client secret here.
            </li>
          </ol>
          <Row title="Client ID" desc="Ends in .apps.googleusercontent.com">
            <input
              className="text-input acct-field"
              value={clientId}
              spellCheck={false}
              placeholder="1234…apps.googleusercontent.com"
              onChange={(e) => (
                setSetup(true),
                setSettings({ googleClientId: e.target.value.trim() })
              )}
            />
          </Row>
          <Row
            title="Client secret"
            desc="Google gives desktop apps one too; it stays on this PC."
          >
            <input
              className="text-input acct-field"
              type="password"
              value={secret}
              spellCheck={false}
              placeholder="GOCSPX-…"
              onChange={(e) => (
                setSetup(true),
                setSettings({ googleClientSecret: e.target.value.trim() })
              )}
            />
          </Row>
          {(ready || acct?.builtIn) && (
            <div className="row-end">
              {acct?.builtIn && own && (
                <button
                  className="btn small ghost"
                  onClick={() => (
                    setSettings({ googleClientId: "", googleClientSecret: "" }),
                    setSetup(false)
                  )}
                >
                  Use Pelica's client
                </button>
              )}
              <button className="btn small" onClick={() => setSetup(false)}>
                Done
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}
