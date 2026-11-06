// The first start: sign up or log in with Google (Settings → Account does the
// same later). Shown once, after What's new, to someone not signed in, when
// this build can sign in at all.
import { useEffect, useState } from "react";
import { bridge } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { GoogleG } from "./Account";
import { Logo } from "./Icons";
import { SignInHelp } from "./SignInHelp";

/** At start, once the account's status is known. */
export function checkSignUp() {
  const s = app.get();
  if (!bridge || s.settings.accountAsked) return;
  if (s.account?.profile) return setSettings({ accountAsked: true });
  if (!s.account?.clientId) return;
  app.set({ signUp: true });
}

export function SignUp() {
  const show = useApp((s) => s.signUp && !s.whatsNew && !s.settingsOpen);
  const [busy, setBusy] = useState(false);
  const close = () => {
    setSettings({ accountAsked: true });
    app.set({ signUp: false });
  };
  useEffect(() => {
    if (!show) return;
    const f = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [show]);
  if (!show) return null;
  async function go() {
    setBusy(true);
    try {
      app.set({ account: await bridge!.account.signIn() });
      close();
      toast("Signed in. Your chats are syncing.");
    } catch (e: any) {
      toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    }
    setBusy(false);
  }
  return (
    <div className="modal-back signup-back">
      <div className="modal signup" role="dialog" aria-label="Sign up or log in">
        <Logo size={52} />
        <h2>Sign up or log in</h2>
        <p className="signup-lead">
          Use your Google account to keep your chats, projects and memory on all your PCs.
        </p>
        <button className="btn google-btn signup-google" disabled={busy} onClick={go}>
          <GoogleG /> {busy ? "Waiting for your browser…" : "Continue with Google"}
        </button>
        {busy && <SignInHelp />}
        <p className="muted small signup-note">
          New to Pelica? This makes your account. Signed in before? Your chats come back.
        </p>
        <button className="signup-skip" onClick={close}>
          Not now
        </button>
        <p className="muted small signup-fine">
          Pelica still runs on this PC. Your account only syncs through a hidden folder in your own Google Drive.
          You can sign in later in Settings → Account.
        </p>
      </div>
    </div>
  );
}
