// Settings → Account: sign in with Google, and sync your chats, projects,
// memory and saved prompts between your PCs through your own Google Drive
// (electron/google.ts, electron/sync.ts). Signing in uses Pelica's own Google
// client; a build without one asks for yours, set up with the steps shown here.
import { useEffect, useState } from "react";
import { bridge } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

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
        </div>
      )}
      {p && (
        <>
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
        </>
      )}
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
