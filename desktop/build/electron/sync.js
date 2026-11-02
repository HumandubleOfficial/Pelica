"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.status = exports.SYNCED = void 0;
exports.init = init;
exports.reset = reset;
exports.soon = soon;
exports.noteDelete = noteDelete;
exports.noteProfile = noteProfile;
exports.syncNow = syncNow;
// Sync between your PCs through your Google Drive's app folder (hidden, only
// Pelica sees it): a file per chat ("chat-<id>.json"), "profile.json" (your
// name, instructions, projects, memory, saved prompts) and "deleted.json"
// (chats deleted on any PC). The newer side wins; a chat changed on two PCs
// between syncs keeps the older version as a copy, so nothing is lost. Chats
// deleted elsewhere go to this PC's Recycle Bin.
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const electron_1 = require("electron");
const google = __importStar(require("./google"));
/** The settings that travel with the account (the rest belong to this PC). */
exports.SYNCED = ["userName", "customInstructions", "projects", "memoryOn", "memories", "prompts"];
let stateFile = "";
let chatsDir = "";
let st = { synced: {}, deleted: {}, profileAt: 0, lastSync: 0 };
let hooks;
let busy = false;
let again = false;
let error = "";
let timer = null;
function init(userDir, chats, h) {
    stateFile = node_path_1.default.join(userDir, "sync-state.json");
    chatsDir = chats;
    hooks = h;
    try {
        st = { ...st, ...JSON.parse(node_fs_1.default.readFileSync(stateFile, "utf8")) };
    }
    catch { }
}
const saveState = () => node_fs_1.default.writeFileSync(stateFile, JSON.stringify(st));
const status = () => ({ busy, lastSync: st.lastSync, error });
exports.status = status;
/** A fresh account: what was synced before (with another account) starts over. */
function reset() {
    st = { synced: {}, deleted: {}, profileAt: st.profileAt ? Date.now() : 0, lastSync: 0 };
    error = "";
    saveState();
}
/** Syncs a few seconds after a change (several changes, one sync). */
function soon(ms = 4000) {
    if (!hooks?.on() || !google.profile())
        return;
    if (timer)
        clearTimeout(timer);
    timer = setTimeout(() => syncNow().catch(() => { }), ms);
}
function noteDelete(id) {
    st.deleted[id] = Date.now();
    delete st.synced[id];
    saveState();
    soon();
}
function noteProfile() {
    st.profileAt = Date.now();
    saveState();
    soon();
}
async function ok(r) {
    if (r.ok)
        return r;
    const j = await r.json().catch(() => null);
    throw new Error(`Google Drive: ${j?.error?.message || `HTTP ${r.status}`}`);
}
async function listAll() {
    const out = [];
    let page = "";
    do {
        const u = `${google.DRIVE}/files?spaces=appDataFolder&pageSize=1000&fields=nextPageToken,files(id,name,appProperties)${page ? `&pageToken=${encodeURIComponent(page)}` : ""}`;
        const j = await (await ok(await google.api(u))).json();
        for (const f of j.files || [])
            out.push({ id: f.id, name: f.name, updated: Number(f.appProperties?.updated || 0) });
        page = j.nextPageToken || "";
    } while (page);
    return out;
}
async function download(id) {
    return (await ok(await google.api(`${google.DRIVE}/files/${id}?alt=media`))).json();
}
/** Writes a file to the app folder (a new one, or over `id`). */
async function upload(name, data, updated, id) {
    const b = "pelica" + Math.random().toString(36).slice(2);
    const meta = id ? { appProperties: { updated: String(updated) } } : { name, parents: ["appDataFolder"], appProperties: { updated: String(updated) } };
    const body = `--${b}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n${JSON.stringify(meta)}\r\n--${b}\r\nContent-Type: application/json\r\n\r\n${JSON.stringify(data)}\r\n--${b}--`;
    const r = await ok(await google.api(`${google.UPLOAD}/files${id ? `/${id}` : ""}?uploadType=multipart&fields=id`, { method: id ? "PATCH" : "POST", headers: { "Content-Type": `multipart/related; boundary=${b}` }, body }));
    return (await r.json()).id;
}
const remove = async (id) => {
    const r = await google.api(`${google.DRIVE}/files/${id}`, { method: "DELETE" });
    if (!r.ok && r.status !== 404)
        await ok(r);
};
// ---------- this PC's chats ----------
const chatFile = (id) => node_path_1.default.join(chatsDir, id + ".json");
function localChats() {
    const out = new Map();
    node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
    for (const f of node_fs_1.default.readdirSync(chatsDir)) {
        if (!f.endsWith(".json"))
            continue;
        try {
            const c = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(chatsDir, f), "utf8"));
            if (c?.id === f.slice(0, -5))
                out.set(c.id, Number(c.updated) || 0);
        }
        catch { }
    }
    return out;
}
function writeChat(c) {
    const p = chatFile(c.id);
    node_fs_1.default.writeFileSync(p + ".tmp", JSON.stringify(c));
    node_fs_1.default.renameSync(p + ".tmp", p);
}
const readChat = (id) => JSON.parse(node_fs_1.default.readFileSync(chatFile(id), "utf8"));
/** The other side of a chat changed on two PCs, kept beside it. */
function keepCopy(c) {
    const id = `${c.id}-v${Date.now().toString(36)}`;
    writeChat({ ...c, id, title: `${c.title || "Chat"} (other version)` });
    return id;
}
// ---------- the profile ----------
function pickProfile(s) {
    const out = {};
    for (const k of exports.SYNCED)
        if (s[k] !== undefined)
            out[k] = s[k];
    return out;
}
/** The first sync on this PC: lists are joined, so nothing here is lost. */
function joinProfile(here, there) {
    const out = { ...here, ...there };
    const join = (k, key) => {
        const a = Array.isArray(there[k]) ? there[k] : [];
        const seen = new Set(a.map(key));
        out[k] = [...a, ...(Array.isArray(here[k]) ? here[k] : []).filter((x) => !seen.has(key(x)))];
    };
    join("projects", (p) => p.id);
    join("memories", (m) => m.id);
    join("prompts", (p) => p.name);
    if (!there.customInstructions && here.customInstructions)
        out.customInstructions = here.customInstructions;
    if (!there.userName && here.userName)
        out.userName = here.userName;
    return out;
}
// ---------- a sync ----------
async function syncNow() {
    if (!google.profile())
        throw new Error("Sign in first (Settings → Account).");
    if (busy) {
        again = true;
        return (0, exports.status)();
    }
    busy = true;
    error = "";
    const changed = new Set();
    let settingsChanged = false;
    try {
        const first = !st.lastSync;
        const remote = await listAll();
        const chats = new Map();
        let profileFile;
        let deletedFile;
        for (const f of remote) {
            const m = f.name.match(/^chat-([\w-]+)\.json$/);
            if (m)
                chats.set(m[1], f);
            else if (f.name === "profile.json")
                profileFile = f;
            else if (f.name === "deleted.json")
                deletedFile = f;
        }
        // deletions, from every PC
        const remoteDeleted = deletedFile ? (await download(deletedFile.id).catch(() => ({}))).ids || {} : {};
        const deleted = { ...remoteDeleted };
        for (const [id, t] of Object.entries(st.deleted))
            deleted[id] = Math.max(t, deleted[id] || 0);
        const local = localChats();
        for (const [id, when] of Object.entries(deleted)) {
            const here = local.get(id);
            if (here !== undefined) {
                // (changed after it was deleted elsewhere: it stays)
                if (here > when) {
                    delete deleted[id];
                    continue;
                }
                await electron_1.shell.trashItem(chatFile(id)).catch(() => { });
                local.delete(id);
                changed.add(id);
            }
            const r = chats.get(id);
            if (r && r.updated <= when) {
                await remove(r.id);
                chats.delete(id);
            }
            delete st.synced[id];
        }
        // chats here: up, down, or both kept
        for (const [id, updated] of local) {
            const r = chats.get(id);
            const base = st.synced[id];
            if (!r) {
                await upload(`chat-${id}.json`, readChat(id), updated);
            }
            else if (r.updated === updated) {
                // (the same on both sides)
            }
            else if (updated > r.updated) {
                if (base !== undefined && r.updated > base)
                    changed.add(keepCopy(await download(r.id)));
                await upload(`chat-${id}.json`, readChat(id), updated, r.id);
            }
            else {
                if (base !== undefined && updated > base)
                    changed.add(keepCopy(readChat(id)));
                writeChat(await download(r.id));
                changed.add(id);
            }
            st.synced[id] = Math.max(updated, r?.updated || 0);
            chats.delete(id);
        }
        // chats only on Drive: down
        for (const [id, r] of chats) {
            writeChat(await download(r.id));
            st.synced[id] = r.updated;
            changed.add(id);
        }
        // copies kept just now go up next time (soon)
        // the profile
        const here = pickProfile(hooks.settings());
        if (profileFile && (first || profileFile.updated > st.profileAt)) {
            const there = await download(profileFile.id);
            const next = first ? joinProfile(here, there) : there;
            hooks.apply(next);
            settingsChanged = true;
            st.profileAt = first ? Date.now() : profileFile.updated;
            if (first)
                await upload("profile.json", next, st.profileAt, profileFile.id);
        }
        else if (!profileFile || st.profileAt > profileFile.updated) {
            st.profileAt ||= Date.now();
            await upload("profile.json", here, st.profileAt, profileFile?.id);
        }
        // deleted.json: what's new, and nothing older than 90 days
        const cutoff = Date.now() - 90 * 86400e3;
        const keep = Object.fromEntries(Object.entries(deleted).filter(([, t]) => t > cutoff));
        if (JSON.stringify(keep) !== JSON.stringify(remoteDeleted))
            await upload("deleted.json", { ids: keep }, Date.now(), deletedFile?.id);
        st.deleted = {};
        st.lastSync = Date.now();
        saveState();
    }
    catch (e) {
        error = String(e?.message || e);
    }
    finally {
        busy = false;
    }
    if (changed.size || settingsChanged)
        hooks.changed({ chats: [...changed], settings: settingsChanged });
    // (copies kept, or a change while this ran: once more)
    if (again || [...changed].some((id) => /-v[0-9a-z]+$/.test(id))) {
        again = false;
        soon(1500);
    }
    return (0, exports.status)();
}
