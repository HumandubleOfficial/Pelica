"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.init = init;
exports.before = before;
exports.after = after;
exports.info = info;
exports.undo = undo;
exports.redo = redo;
// Restore points: before Pelica's file tools change a file, it keeps a copy
// here, so a reply's changes can be undone (and put back). One folder per
// reply, under restore/ in Pelica's data: index.json and the copies. Files a
// command changed aren't covered: nothing here sees them. A file Pelica
// created goes to the Recycle Bin on undo, never straight to deletion.
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const electron_1 = require("electron");
const workspace_1 = require("./workspace");
/** Bigger than this isn't copied (it can't be undone, and the page says so). */
const MAX_FILE = 50e6;
const MAX_FOLDER = 200e6;
/** Restore points older than this go, then the oldest until they fit. */
const KEEP_DAYS = 30;
const MAX_TOTAL = 2e9;
let base = "";
let allowed = (r) => r;
function init(userDir, checkRoot) {
    base = node_path_1.default.join(userDir, "restore");
    allowed = checkRoot;
    setTimeout(prune, 20000);
}
const okId = (turn) => !!turn && /^[\w-]{1,64}$/.test(turn);
const dirOf = (turn) => node_path_1.default.join(base, turn);
const copyOf = (turn, n) => node_path_1.default.join(dirOf(turn), String(n));
function load(turn) {
    try {
        return JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(dirOf(turn), "index.json"), "utf8"));
    }
    catch {
        return null;
    }
}
function save(turn, p) {
    node_fs_1.default.mkdirSync(dirOf(turn), { recursive: true });
    node_fs_1.default.writeFileSync(node_path_1.default.join(dirOf(turn), "index.json"), JSON.stringify(p));
}
function hash(p) {
    let st;
    try {
        st = node_fs_1.default.statSync(p);
    }
    catch {
        return "";
    }
    if (st.isDirectory())
        return "folder";
    if (st.size > MAX_FILE)
        return `big:${st.size}:${st.mtimeMs}`;
    return node_crypto_1.default.createHash("sha1").update(node_fs_1.default.readFileSync(p)).digest("hex");
}
/** The size of a folder, stopping once it's past `limit`. */
function sizeOf(p, limit) {
    const st = node_fs_1.default.statSync(p);
    if (!st.isDirectory())
        return st.size;
    let n = 0;
    for (const e of node_fs_1.default.readdirSync(p)) {
        n += sizeOf(node_path_1.default.join(p, e), limit - n);
        if (n > limit)
            break;
    }
    return n;
}
const copy = (from, to) => {
    node_fs_1.default.mkdirSync(node_path_1.default.dirname(to), { recursive: true });
    node_fs_1.default.cpSync(from, to, { recursive: true, force: true, dereference: true });
};
/** Before a file tool changes `rel`: keeps it, the first time in this reply. */
function before(turn, root, name) {
    if (!okId(turn))
        return;
    try {
        const rel = node_path_1.default.normalize(name);
        const p = load(turn) || { created: Date.now(), undone: false, entries: [] };
        if (p.entries.some((e) => e.root === root && e.rel === rel))
            return;
        const file = (0, workspace_1.inside)(root, rel);
        const e = { root, rel, existed: node_fs_1.default.existsSync(file) };
        if (e.existed) {
            const dir = node_fs_1.default.statSync(file).isDirectory();
            if (sizeOf(file, MAX_FOLDER) > (dir ? MAX_FOLDER : MAX_FILE))
                e.cant = true;
            else
                copy(file, copyOf(turn, p.entries.length));
        }
        p.entries.push(e);
        save(turn, p);
    }
    catch {
        // (a restore point that can't be made must not stop the tool)
    }
}
/** After it changed: what Pelica left there. */
function after(turn, root, name) {
    if (!okId(turn))
        return;
    try {
        const rel = node_path_1.default.normalize(name);
        const p = load(turn);
        const e = p?.entries.find((x) => x.root === root && x.rel === rel);
        if (!p || !e)
            return;
        e.expect = hash((0, workspace_1.inside)(root, rel));
        p.undone = false;
        save(turn, p);
    }
    catch { }
}
const changedOnes = (p) => p.entries.filter((e) => e.expect !== undefined);
function info(turns) {
    const points = turns.filter(okId).map(load).filter((p) => !!p);
    const all = points.flatMap(changedOnes);
    return {
        files: [...new Set(all.filter((e) => !e.cant).map((e) => e.rel))],
        cant: [...new Set(all.filter((e) => e.cant).map((e) => e.rel))],
        undone: points.length > 0 && points.every((p) => p.undone),
    };
}
function clashes(turns) {
    const out = new Set();
    // (a file two of these replies changed: only the newest says what it is now)
    const seen = new Set();
    for (const t of [...turns].reverse()) {
        const p = load(t);
        for (const e of p ? changedOnes(p) : []) {
            const key = `${e.root}|${e.rel}`;
            if (e.cant || seen.has(key))
                continue;
            seen.add(key);
            try {
                if (hash((0, workspace_1.inside)(allowed(e.root), e.rel)) !== e.expect)
                    out.add(e.rel);
            }
            catch { }
        }
    }
    return [...out];
}
/** Puts a reply's files back as they were (the newest reply first). */
async function undo(list, force = false) {
    const turns = list.filter(okId);
    const r = { done: [], changed: force ? [] : clashes(turns), failed: [] };
    if (r.changed.length)
        return r;
    for (const t of [...turns].reverse()) {
        const p = load(t);
        if (!p || p.undone)
            continue;
        for (let n = p.entries.length - 1; n >= 0; n--) {
            const e = p.entries[n];
            if (e.expect === undefined || e.cant)
                continue;
            try {
                const file = (0, workspace_1.inside)(allowed(e.root), e.rel);
                const redo = `${copyOf(t, n)}.redo`;
                node_fs_1.default.rmSync(redo, { recursive: true, force: true });
                e.redo = node_fs_1.default.existsSync(file);
                if (e.redo)
                    copy(file, redo);
                if (e.existed)
                    copy(copyOf(t, n), file);
                else if (e.redo)
                    await electron_1.shell.trashItem(file);
                e.expect = hash(file);
                r.done.push(e.rel);
            }
            catch (err) {
                r.failed.push(`${e.rel}: ${err?.message || err}`);
            }
        }
        p.undone = true;
        save(t, p);
    }
    return r;
}
/** Puts back what undo took away. */
async function redo(list, force = false) {
    const turns = list.filter(okId);
    const r = { done: [], changed: force ? [] : clashes(turns), failed: [] };
    if (r.changed.length)
        return r;
    for (const t of turns) {
        const p = load(t);
        if (!p || !p.undone)
            continue;
        for (let n = 0; n < p.entries.length; n++) {
            const e = p.entries[n];
            if (e.expect === undefined || e.cant)
                continue;
            try {
                const file = (0, workspace_1.inside)(allowed(e.root), e.rel);
                if (e.redo)
                    copy(`${copyOf(t, n)}.redo`, file);
                else if (node_fs_1.default.existsSync(file))
                    await electron_1.shell.trashItem(file);
                e.expect = hash(file);
                r.done.push(e.rel);
            }
            catch (err) {
                r.failed.push(`${e.rel}: ${err?.message || err}`);
            }
        }
        p.undone = false;
        save(t, p);
    }
    return r;
}
/** Old restore points go: past KEEP_DAYS, then the oldest over MAX_TOTAL. */
function prune() {
    try {
        const all = node_fs_1.default
            .readdirSync(base)
            .map((id) => ({ id, p: load(id) }))
            .map((x) => ({ id: x.id, created: x.p?.created || 0, size: sizeOf(dirOf(x.id), Infinity) }))
            .sort((a, b) => b.created - a.created);
        let total = 0;
        for (const x of all) {
            total += x.size;
            if (Date.now() - x.created > KEEP_DAYS * 864e5 || total > MAX_TOTAL)
                node_fs_1.default.rmSync(dirOf(x.id), { recursive: true, force: true });
        }
    }
    catch { }
}
