// Shown while a sign-in waits on the browser. Some browsers can't finish
// Google's page (Firefox can stop at a 400 error), so the same sign-in can go
// on in another one, or be called off instead of waiting out its 5 minutes.
import { bridge } from "../lib/bridge";
import { toast } from "../state";

export function SignInHelp() {
  if (!bridge) return null;
  return (
    <div className="signin-help">
      <span className="muted small">Page not working? Finish in another browser.</span>
      <span className="signin-help-btns">
        <button className="btn small" onClick={() => bridge!.signin.edge().catch((e: any) => toast(e.message))}>
          Open in Edge
        </button>
        <button className="btn small" onClick={() => bridge!.signin.copy().then(() => toast("Link copied. Paste it into another browser."))}>
          Copy link
        </button>
        <button className="btn small ghost" onClick={() => bridge!.signin.cancel()}>
          Cancel
        </button>
      </span>
    </div>
  );
}
