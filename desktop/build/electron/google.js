"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.profile = exports.UPLOAD = exports.DRIVE = void 0;
exports.init = init;
exports.signIn = signIn;
exports.signOut = signOut;
exports.token = token;
exports.api = api;
// Your Pelica account: sign in with Google (OAuth for installed apps: the
// browser, a one-time address on this computer, PKCE). The account keeps your
// name, email and picture, and a token for the one Drive folder Pelica may
// use (drive.appdata: a hidden folder only Pelica sees, not your files),
// which electron/sync.ts syncs through. Tokens are encrypted with Windows'
// own encryption (safeStorage). The OAuth client is the user's (Settings →
// Account), a "Desktop app" client from Google Cloud Console.
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_http_1 = __importDefault(require("node:http"));
const node_path_1 = __importDefault(require("node:path"));
const electron_1 = require("electron");
// (tests point every Google address at a pretend server)
const BASE = process.env.PELICA_GOOGLE_BASE || "";
const AUTH = BASE ? `${BASE}/auth` : "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN = BASE ? `${BASE}/token` : "https://oauth2.googleapis.com/token";
const REVOKE = BASE ? `${BASE}/revoke` : "https://oauth2.googleapis.com/revoke";
const USERINFO = BASE ? `${BASE}/userinfo` : "https://openidconnect.googleapis.com/v1/userinfo";
exports.DRIVE = BASE ? `${BASE}/drive/v3` : "https://www.googleapis.com/drive/v3";
exports.UPLOAD = BASE ? `${BASE}/upload/drive/v3` : "https://www.googleapis.com/upload/drive/v3";
const SCOPES = "openid email profile https://www.googleapis.com/auth/drive.appdata";
let file = "";
let acct = null;
let onChange = () => { };
function init(userDir, changed) {
    file = node_path_1.default.join(userDir, "account.bin");
    onChange = changed;
    try {
        acct = JSON.parse(electron_1.safeStorage.decryptString(node_fs_1.default.readFileSync(file)));
    }
    catch {
        acct = null;
    }
}
function save() {
    if (!acct) {
        node_fs_1.default.rmSync(file, { force: true });
        return;
    }
    if (!electron_1.safeStorage.isEncryptionAvailable())
        throw new Error("Windows can't encrypt the sign-in here, so Pelica won't keep it.");
    node_fs_1.default.writeFileSync(file, electron_1.safeStorage.encryptString(JSON.stringify(acct)));
}
const profile = () => (acct ? { name: acct.name, email: acct.email, picture: acct.picture } : null);
exports.profile = profile;
const b64url = (b) => b.toString("base64url");
async function post(url, body) {
    const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
        body: new URLSearchParams(Object.entries(body).filter((e) => !!e[1])).toString(),
        signal: AbortSignal.timeout(20000),
    });
    const j = await r.json().catch(() => null);
    if (!r.ok || !j)
        throw Object.assign(new Error(j?.error_description || j?.error || `HTTP ${r.status}`), { code: j?.error });
    return j;
}
async function pictureOf(url) {
    if (!url)
        return "";
    try {
        const r = await fetch(url.replace(/=s\d+-c$/, "=s96-c"), { signal: AbortSignal.timeout(10000) });
        if (!r.ok)
            return "";
        return `data:${r.headers.get("content-type") || "image/jpeg"};base64,${Buffer.from(await r.arrayBuffer()).toString("base64")}`;
    }
    catch {
        return "";
    }
}
const PAGE = (title, body) => `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>${title}</title><body style="font:16px system-ui;background:#262624;color:#ecebe6;display:grid;place-items:center;height:100vh;margin:0"><div style="text-align:center"><h2 style="font-weight:500">${title}</h2><p style="color:#a09d94">${body}</p></div>`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
let pending = null;
/** Signs in with Google: opens the browser and waits (5 minutes at most). */
async function signIn(clientId, clientSecret, open) {
    if (!clientId.trim())
        throw new Error("Add your Google client ID first (Settings → Account → Google sign-in).");
    pending?.();
    const srv = node_http_1.default.createServer();
    await new Promise((ok, bad) => {
        srv.once("error", bad);
        srv.listen(0, "127.0.0.1", () => ok());
    });
    const redirect = `http://127.0.0.1:${srv.address().port}`;
    try {
        const verifier = b64url(node_crypto_1.default.randomBytes(32));
        const state = b64url(node_crypto_1.default.randomBytes(16));
        const u = new URL(AUTH);
        const q = {
            client_id: clientId.trim(),
            redirect_uri: redirect,
            response_type: "code",
            scope: SCOPES,
            code_challenge: b64url(node_crypto_1.default.createHash("sha256").update(verifier).digest()),
            code_challenge_method: "S256",
            state,
            // (a refresh token, so the sign-in lasts)
            access_type: "offline",
            prompt: "consent select_account",
        };
        for (const [k, v] of Object.entries(q))
            u.searchParams.set(k, v);
        const code = await new Promise((ok, bad) => {
            const t = setTimeout(() => bad(new Error("The sign-in took too long. Try again.")), 5 * 60_000);
            pending = () => {
                clearTimeout(t);
                bad(new Error("Sign-in cancelled."));
            };
            srv.on("request", (req, res) => {
                const r = new URL(req.url || "/", redirect);
                if (r.pathname !== "/" || (!r.searchParams.has("code") && !r.searchParams.has("error"))) {
                    res.writeHead(404).end();
                    return;
                }
                res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
                if (r.searchParams.get("state") !== state) {
                    res.end(PAGE("That sign-in wasn't Pelica's", "Go back to Pelica and try again."));
                    return;
                }
                clearTimeout(t);
                const err = r.searchParams.get("error");
                if (err) {
                    res.end(PAGE("Not signed in", esc(err === "access_denied" ? "You declined the sign-in." : err)));
                    bad(new Error(err === "access_denied" ? "Sign-in was declined." : `Google said: ${err}`));
                    return;
                }
                res.end(PAGE("Signed in to Pelica", "You can close this tab and go back to Pelica."));
                ok(r.searchParams.get("code"));
            });
            open(u.href);
        });
        const tok = await post(TOKEN, { code, client_id: clientId.trim(), client_secret: clientSecret.trim() || undefined, redirect_uri: redirect, grant_type: "authorization_code", code_verifier: verifier });
        if (!String(tok.scope || SCOPES).includes("drive.appdata"))
            throw new Error("Pelica needs the Drive app folder to sync. Sign in again and allow it.");
        const who = await fetch(USERINFO, { headers: { Authorization: `Bearer ${tok.access_token}` }, signal: AbortSignal.timeout(15000) }).then((r) => r.json());
        acct = {
            access_token: tok.access_token,
            refresh_token: tok.refresh_token,
            expires_at: Date.now() + Number(tok.expires_in || 3600) * 1000,
            client_id: clientId.trim(),
            client_secret: clientSecret.trim() || undefined,
            name: who.name || who.given_name || who.email || "You",
            email: who.email || "",
            picture: await pictureOf(who.picture),
        };
        save();
        onChange();
        return (0, exports.profile)();
    }
    finally {
        pending = null;
        srv.close();
        srv.closeAllConnections?.();
    }
}
/** Signs out: Google forgets Pelica's token too. */
async function signOut() {
    const a = acct;
    acct = null;
    save();
    onChange();
    if (a)
        await post(REVOKE, { token: a.refresh_token || a.access_token }).catch(() => { });
}
let refreshing = null;
/** A current access token (refreshed when it has run out). */
function token() {
    if (!acct)
        return Promise.reject(new Error("not signed in"));
    if (acct.expires_at - 60_000 > Date.now())
        return Promise.resolve(acct.access_token);
    return (refreshing ||= refresh().finally(() => (refreshing = null)));
}
async function refresh() {
    const a = acct;
    if (!a?.refresh_token)
        throw new Error("The Google sign-in ran out. Sign in again (Settings → Account).");
    try {
        const r = await post(TOKEN, { client_id: a.client_id, client_secret: a.client_secret, refresh_token: a.refresh_token, grant_type: "refresh_token" });
        acct = { ...a, access_token: r.access_token, refresh_token: r.refresh_token || a.refresh_token, expires_at: Date.now() + Number(r.expires_in || 3600) * 1000 };
        save();
        return acct.access_token;
    }
    catch (e) {
        // (revoked or expired for good: signed out)
        if (e.code === "invalid_grant") {
            acct = null;
            save();
            onChange();
            throw new Error("Google ended the sign-in. Sign in again (Settings → Account).");
        }
        throw e;
    }
}
/** A Google API call with the account's token (refreshed once on a 401). */
async function api(url, init = {}) {
    const go = async () => fetch(url, { ...init, headers: { ...init.headers, Authorization: `Bearer ${await token()}` }, signal: AbortSignal.timeout(60000) });
    let r = await go();
    if (r.status === 401 && acct) {
        acct.expires_at = 0;
        r = await go();
    }
    return r;
}
