// Sign-in for online connectors (MCP authorization, OAuth 2.1): a server that
// answers 401 names where to sign in (RFC 9728 protected resource metadata,
// then RFC 8414 authorization server metadata). Pelica registers itself there
// (RFC 7591 dynamic client registration), opens the sign-in page in the
// browser with PKCE, takes the answer on a one-time address on this computer
// (127.0.0.1), and keeps the tokens encrypted with Windows' own encryption
// (safeStorage), refreshing them when they run out.
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { safeStorage } from "electron";

interface Token {
  access_token: string;
  refresh_token?: string;
  /** ms since 1970 */
  expires_at?: number;
  client_id: string;
  client_secret?: string;
  token_endpoint: string;
  resource: string;
  scope?: string;
}

let file = "";
let tokens: Record<string, Token> = {};

/** Where the tokens are kept (the user data folder). */
export function init(userDir: string) {
  file = path.join(userDir, "connector-sign-ins.bin");
  try {
    tokens = JSON.parse(safeStorage.decryptString(fs.readFileSync(file)));
  } catch {
    tokens = {};
  }
}

function save() {
  if (!safeStorage.isEncryptionAvailable()) throw new Error("Windows can't encrypt the sign-in here, so Pelica won't keep it.");
  fs.writeFileSync(file, safeStorage.encryptString(JSON.stringify(tokens)));
}

/** The connector's address as a key (no fragment, no trailing slash). */
const keyOf = (url: string) => {
  const u = new URL(url);
  u.hash = "";
  return u.href.replace(/\/$/, "");
};

export const signedIn = (url: string) => !!tokens[keyOf(url)];

export function signOut(url: string) {
  delete tokens[keyOf(url)];
  save();
}

const b64url = (b: Buffer) => b.toString("base64url");

async function getJson(url: string): Promise<any | null> {
  try {
    const r = await fetch(url, { headers: { Accept: "application/json", "MCP-Protocol-Version": "2025-06-18" }, signal: AbortSignal.timeout(15000) });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

/** `Bearer realm="x", resource_metadata="…", scope="a b"` → its parameters. */
export function parseChallenge(h: string | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!h) return out;
  for (const m of h.matchAll(/([\w-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]+))/g)) out[m[1].toLowerCase()] = m[2] !== undefined ? m[2].replace(/\\(.)/g, "$1") : m[3];
  return out;
}

/** The well-known address for `name` on `base`, path inserted after it (RFC 8414 / 9728). */
function wellKnown(base: string, name: string) {
  const u = new URL(base);
  const p = u.pathname.replace(/\/$/, "");
  return [`${u.origin}/.well-known/${name}${p}`, ...(p ? [`${u.origin}/.well-known/${name}`] : [])];
}

async function discover(url: string, challenge: Record<string, string>) {
  // 1. the resource: where its sign-in server is
  let prm: any = null;
  if (challenge.resource_metadata) prm = await getJson(challenge.resource_metadata);
  for (const u of prm ? [] : wellKnown(url, "oauth-protected-resource")) if ((prm = await getJson(u))) break;
  const issuer: string = prm?.authorization_servers?.[0] || new URL(url).origin;
  const resource: string = prm?.resource || keyOf(url);
  // 2. the sign-in server's endpoints
  let as: any = null;
  const iu = new URL(issuer);
  const ip = iu.pathname.replace(/\/$/, "");
  const tries = ip
    ? [`${iu.origin}/.well-known/oauth-authorization-server${ip}`, `${iu.origin}/.well-known/openid-configuration${ip}`, `${iu.origin}${ip}/.well-known/openid-configuration`]
    : [`${iu.origin}/.well-known/oauth-authorization-server`, `${iu.origin}/.well-known/openid-configuration`];
  for (const u of tries) if ((as = await getJson(u))) break;
  // (servers from before discovery: the default paths)
  as ||= { authorization_endpoint: `${iu.origin}/authorize`, token_endpoint: `${iu.origin}/token`, registration_endpoint: `${iu.origin}/register` };
  if (!as.authorization_endpoint || !as.token_endpoint) throw new Error("The server's sign-in settings are incomplete.");
  const scope = challenge.scope || (Array.isArray(prm?.scopes_supported) ? prm.scopes_supported.join(" ") : "");
  return { as, resource, scope };
}

async function post(url: string, body: Record<string, string>, json = false): Promise<any> {
  const r = await fetch(url, {
    method: "POST",
    headers: json ? { "Content-Type": "application/json", Accept: "application/json" } : { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
    body: json ? JSON.stringify(body) : new URLSearchParams(Object.entries(body).filter(([, v]) => v !== undefined && v !== "")).toString(),
    signal: AbortSignal.timeout(20000),
  });
  const text = await r.text();
  let j: any = null;
  try {
    j = JSON.parse(text);
  } catch {}
  if (!r.ok || !j) throw new Error(j?.error_description || j?.error || `HTTP ${r.status}`);
  return j;
}

const PAGE = (title: string, body: string) =>
  `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>${title}</title><body style="font:16px system-ui;background:#262624;color:#ecebe6;display:grid;place-items:center;height:100vh;margin:0"><div style="text-align:center"><h2 style="font-weight:500">${title}</h2><p style="color:#a09d94">${body}</p></div>`;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

let pending: { cancel: () => void } | null = null;

/** Gives up on the sign-in under way. */
export const cancelSignIn = () => pending?.cancel();

/** Signs in to the connector at `url`: opens the browser and waits (5 minutes at most). */
export async function signIn(url: string, wwwAuthenticate: string | undefined, open: (u: string) => void): Promise<void> {
  pending?.cancel();
  const challenge = parseChallenge(wwwAuthenticate);
  const { as, resource, scope } = await discover(url, challenge);
  // the one-time address the sign-in comes back to
  const srv = http.createServer();
  await new Promise<void>((ok, bad) => {
    srv.once("error", bad);
    srv.listen(0, "127.0.0.1", () => ok());
  });
  const redirect = `http://127.0.0.1:${(srv.address() as any).port}/callback`;
  try {
    if (!as.registration_endpoint) throw new Error("This server doesn't let apps register themselves, so Pelica can't sign in to it. Add a token under Headers instead (Authorization: Bearer …).");
    const client = await post(
      as.registration_endpoint,
      { client_name: "Pelica", client_uri: "https://github.com/HumandubleOfficial/Pelica", redirect_uris: [redirect], grant_types: ["authorization_code", "refresh_token"], response_types: ["code"], token_endpoint_auth_method: "none", ...(scope ? { scope } : {}) } as any,
      true,
    );
    const verifier = b64url(crypto.randomBytes(32));
    const state = b64url(crypto.randomBytes(16));
    const auth = new URL(as.authorization_endpoint);
    const q: Record<string, string> = { response_type: "code", client_id: client.client_id, redirect_uri: redirect, code_challenge: b64url(crypto.createHash("sha256").update(verifier).digest()), code_challenge_method: "S256", state, resource };
    if (scope) q.scope = scope;
    for (const [k, v] of Object.entries(q)) auth.searchParams.set(k, v);
    const code = await new Promise<string>((ok, bad) => {
      const t = setTimeout(() => bad(new Error("The sign-in took too long. Try again.")), 5 * 60_000);
      pending = {
        cancel: () => {
          clearTimeout(t);
          bad(new Error("Sign-in cancelled."));
        },
      };
      srv.on("request", (req, res) => {
        const u = new URL(req.url || "/", redirect);
        if (u.pathname !== "/callback") {
          res.writeHead(404).end();
          return;
        }
        const err = u.searchParams.get("error");
        const got = u.searchParams.get("code");
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        if (u.searchParams.get("state") !== state) {
          res.end(PAGE("That sign-in wasn't Pelica's", "Go back to Pelica and try again."));
          return;
        }
        clearTimeout(t);
        if (err || !got) {
          res.end(PAGE("Not signed in", esc(u.searchParams.get("error_description") || err || "No answer came back.")));
          bad(new Error(u.searchParams.get("error_description") || (err === "access_denied" ? "Sign-in was declined." : err || "No answer came back.")));
          return;
        }
        res.end(PAGE("Signed in", "You can close this tab and go back to Pelica."));
        ok(got);
      });
      open(auth.href);
    });
    const tok = await post(as.token_endpoint, { grant_type: "authorization_code", code, redirect_uri: redirect, client_id: client.client_id, code_verifier: verifier, resource, ...(client.client_secret ? { client_secret: client.client_secret } : {}) });
    if (!tok.access_token) throw new Error("The server sent no token.");
    tokens[keyOf(url)] = {
      access_token: tok.access_token,
      refresh_token: tok.refresh_token,
      expires_at: tok.expires_in ? Date.now() + Number(tok.expires_in) * 1000 : undefined,
      client_id: client.client_id,
      client_secret: client.client_secret,
      token_endpoint: as.token_endpoint,
      resource,
      scope: tok.scope || scope || undefined,
    };
    save();
  } finally {
    pending = null;
    srv.close();
    srv.closeAllConnections?.();
  }
}

// a refresh under way, per connector (two calls at once share it: a server
// that hands out a new refresh token each time would refuse the second)
const refreshing = new Map<string, Promise<string | null>>();

/** The connector's access token (refreshed first when it's run out), or null. */
export function tokenFor(url: string): Promise<string | null> {
  const k = keyOf(url);
  if (!refreshing.has(k)) refreshing.set(k, current(k).finally(() => refreshing.delete(k)));
  return refreshing.get(k)!;
}

async function current(k: string): Promise<string | null> {
  const t = tokens[k];
  if (!t) return null;
  if (!t.expires_at || t.expires_at - 60_000 > Date.now()) return t.access_token;
  if (!t.refresh_token) return t.access_token;
  try {
    const r = await post(t.token_endpoint, { grant_type: "refresh_token", refresh_token: t.refresh_token, client_id: t.client_id, resource: t.resource, ...(t.client_secret ? { client_secret: t.client_secret } : {}) });
    tokens[k] = { ...t, access_token: r.access_token, refresh_token: r.refresh_token || t.refresh_token, expires_at: r.expires_in ? Date.now() + Number(r.expires_in) * 1000 : undefined };
    save();
    return r.access_token;
  } catch {
    // (refused: sign in again)
    delete tokens[k];
    save();
    return null;
  }
}

/** A token the server turned away: forget it, so the connector asks to sign in. */
export function rejected(url: string) {
  if (!tokens[keyOf(url)]) return;
  delete tokens[keyOf(url)];
  save();
}
