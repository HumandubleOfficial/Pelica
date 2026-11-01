"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.newKey = void 0;
exports.addresses = addresses;
exports.status = status;
exports.push = push;
exports.sync = sync;
// Pelica on your phone: a small web server on this PC's network (off until
// Settings → Phone turns it on). The phone opens the address in the QR code,
// which carries a pairing key; every request after that needs the key. The
// page (dist/phone.html) mirrors the open chat: the window pushes what it
// shows here (`push`), the phone long-polls for it, and what the phone does
// (send, open a chat, answer an approval, stop) goes to the window (`onCmd`).
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_http_1 = __importDefault(require("node:http"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
let server = null;
let port = 0;
let key = "";
let error = "";
let page = "";
let snap = { rev: 0 };
let waiters = [];
const seen = new Map();
let onCmd = () => { };
/** This PC's IPv4 addresses: the home network's first, Tailscale's apart; virtual adapters left out. */
function addresses() {
    const lan = [];
    let tailscale = null;
    for (const [name, list] of Object.entries(node_os_1.default.networkInterfaces())) {
        for (const a of list || []) {
            if (a.family !== "IPv4" || a.internal)
                continue;
            const [x, y] = a.address.split(".").map(Number);
            // (a self-assigned address: an adapter that isn't connected, Tailscale's too when it's off)
            if (x === 169 && y === 254)
                continue;
            // Tailscale's addresses are 100.64.0.0/10
            if (x === 100 && y >= 64 && y < 128) {
                tailscale = a.address;
                continue;
            }
            if (/vEthernet|VirtualBox|VMware|WSL|Hyper-V|Loopback|Bluetooth|vpn|tailscale/i.test(name))
                continue;
            const rank = x === 192 && y === 168 ? 0 : x === 10 ? 1 : x === 172 && y >= 16 && y < 32 ? 2 : 3;
            lan.push({ ip: a.address, rank });
        }
    }
    return { lan: lan.sort((a, b) => a.rank - b.rank).map((a) => a.ip), tailscale };
}
function status() {
    const now = Date.now();
    for (const [ip, t] of seen)
        if (now - t > 60_000)
            seen.delete(ip);
    return { on: !!server && !error, port, error, ...addresses(), phones: seen.size };
}
/** What the window shows, for the phone. */
function push(s) {
    snap = { ...s, rev: (snap.rev || 0) + 1 };
    const w = waiters;
    waiters = [];
    for (const f of w)
        f();
}
function keyOk(req, url) {
    const got = String(req.headers.authorization || "").replace(/^Bearer\s+/i, "") || url.searchParams.get("k") || "";
    const a = Buffer.from(got);
    const b = Buffer.from(key);
    return a.length === b.length && node_crypto_1.default.timingSafeEqual(a, b);
}
function json(res, code, body) {
    res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    res.end(JSON.stringify(body));
}
const CMDS = new Set(["send", "open", "new", "approve", "stop"]);
function handle(req, res) {
    const url = new URL(req.url || "/", "http://phone");
    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
        res.writeHead(200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        });
        res.end(page || "<p>Pelica's phone page is missing.</p>");
        return;
    }
    if (!url.pathname.startsWith("/api/"))
        return json(res, 404, { error: "not found" });
    if (!keyOk(req, url))
        return json(res, 401, { error: "Pair again: scan the QR code in Pelica → Settings → Phone." });
    seen.set(req.socket.remoteAddress || "?", Date.now());
    if (req.method === "GET" && url.pathname === "/api/state") {
        const rev = Number(url.searchParams.get("rev") || -1);
        if (rev !== snap.rev)
            return json(res, 200, snap);
        // nothing new yet: answer when the window pushes, or after 25 s
        let done = false;
        const reply = () => {
            if (done)
                return;
            done = true;
            clearTimeout(t);
            json(res, 200, snap);
        };
        const t = setTimeout(reply, 25_000);
        waiters.push(reply);
        res.on("close", () => {
            done = true;
            clearTimeout(t);
            waiters = waiters.filter((f) => f !== reply);
        });
        return;
    }
    if (req.method === "POST" && url.pathname === "/api/cmd") {
        let body = "";
        req.setEncoding("utf8");
        req.on("data", (d) => {
            body += d;
            if (body.length > 512_000)
                req.destroy();
        });
        req.on("end", () => {
            let c;
            try {
                c = JSON.parse(body);
            }
            catch {
                return json(res, 400, { error: "bad request" });
            }
            if (!c || !CMDS.has(c.type))
                return json(res, 400, { error: "bad request" });
            onCmd({ type: c.type, id: typeof c.id === "string" ? c.id : undefined, text: typeof c.text === "string" ? c.text : undefined, answer: c.answer === "allow" || c.answer === "always" ? c.answer : "deny" });
            json(res, 200, { ok: true });
        });
        return;
    }
    json(res, 404, { error: "not found" });
}
/** Starts (or restarts, or stops with want 0) the server. */
function sync(want, pairingKey, uiDir, cmd) {
    onCmd = cmd;
    key = pairingKey;
    try {
        page = node_fs_1.default.readFileSync(node_path_1.default.join(uiDir, "phone.html"), "utf8");
    }
    catch {
        page = "";
    }
    if (want === port && (server || !want))
        return;
    server?.close();
    server?.closeAllConnections?.();
    server = null;
    for (const f of waiters)
        f();
    waiters = [];
    port = want;
    error = "";
    if (!want)
        return;
    const srv = node_http_1.default.createServer(handle);
    srv.on("error", (e) => {
        error = e.code === "EADDRINUSE" ? `Port ${want} is in use by another program. Pick another.` : e.message;
        if (server === srv)
            server = null;
    });
    // (all addresses: the phone reaches it over Wi-Fi, or Tailscale; tests
    // bind this computer only, so Windows doesn't ask about the firewall)
    srv.listen(want, process.env.PELICA_PHONE_BIND || "0.0.0.0");
    server = srv;
}
/** A new pairing key: phones paired before need the new QR code. */
const newKey = () => node_crypto_1.default.randomBytes(18).toString("base64url");
exports.newKey = newKey;
