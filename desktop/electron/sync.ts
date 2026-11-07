// Sync between your PCs through your Google Drive's app folder (hidden, only
// Pelica sees it): a file per chat ("chat-<id>.json"), "profile.json" (your
// name, instructions, projects, memory, saved prompts) and "deleted.json"
// (chats deleted on any PC). The newer side wins; a chat changed on two PCs
// between syncs keeps the older version as a copy, so nothing is lost. Chats
// deleted elsewhere go to this PC's Recycle Bin.
import fs from "node:fs";
import path from "node:path";
import { shell } from "electron";
import * as google from "./google";

/** The settings that travel with the account (the rest belong to this PC). */
export const SYNCED = ["userName", "customInstructions", "projects", "memoryOn", "memories", "prompts"] as const;

/** What Settings → Account can keep off the sync (settings.syncSkip), besides "chats". */
const GROUPS: Record<string, readonly string[]> = { profile: ["userName", "customInstructions"], projects: ["projects"], memory: ["memoryOn", "memories"], prompts: ["prompts"] };

interface State {
  /** chat id → its `updated` when it last matched Drive */
  synced: Record<string, number>;
  /** chats deleted here, not yet written to Drive: id → when */
  deleted: Record<string, number>;
  /** when the synced settings last changed here, and the profile on Drive then */
  profileAt: number;
  lastSync: number;
  /** something was put back on the sync: the profile's lists join again, as on a first sync */
  rejoin?: boolean;
}

export interface SyncStatus {
  busy: boolean;
  lastSync: number;
  error: string;
}

type Hooks = {
  settings: () => Record<string, any>;
  /** settings from another PC */
  apply: (patch: Record<string, any>) => void;
  /** chats changed by a sync (and whether settings did) */
  changed: (c: { chats: string[]; settings: boolean }) => void;
  on: () => boolean;
};

let stateFile = "";
let chatsDir = "";
let st: State = { synced: {}, deleted: {}, profileAt: 0, lastSync: 0 };
let hooks: Hooks;
let busy = false;
let again = false;
let error = "";
let timer: ReturnType<typeof setTimeout> | null = null;

export function init(userDir: string, chats: string, h: Hooks) {
  stateFile = path.join(userDir, "sync-state.json");
  chatsDir = chats;
  hooks = h;
  try {
    st = { ...st, ...JSON.parse(fs.readFileSync(stateFile, "utf8")) };
  } catch {}
}

const saveState = () => fs.writeFileSync(stateFile, JSON.stringify(st));

export const status = (): SyncStatus => ({ busy, lastSync: st.lastSync, error });

/** A fresh account: what was synced before (with another account) starts over. */
export function reset() {
  st = { synced: {}, deleted: {}, profileAt: st.profileAt ? Date.now() : 0, lastSync: 0 };
  error = "";
  saveState();
}

/** Syncs a few seconds after a change (several changes, one sync). */
export function soon(ms = 4000) {
  if (!hooks?.on() || !google.profile()) return;
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => syncNow().catch(() => {}), ms);
}

export function noteDelete(id: string) {
  st.deleted[id] = Date.now();
  delete st.synced[id];
  saveState();
  soon();
}

/** Settings → Account changed what's kept off the sync. */
export function skipChanged(was: string[], now: string[]) {
  if (was.some((g) => g in GROUPS && !now.includes(g))) {
    st.rejoin = true;
    saveState();
  }
  soon(500);
}

export function noteProfile() {
  st.profileAt = Date.now();
  saveState();
  soon();
}

// ---------- Drive ----------

interface Remote {
  id: string;
  name: string;
  updated: number;
  size: number;
}

async function ok(r: Response) {
  if (r.ok) return r;
  const j: any = await r.json().catch(() => null);
  throw new Error(`Google Drive: ${j?.error?.message || `HTTP ${r.status}`}`);
}

async function listAll(): Promise<Remote[]> {
  const out: Remote[] = [];
  let page = "";
  do {
    const u = `${google.DRIVE}/files?spaces=appDataFolder&pageSize=1000&fields=nextPageToken,files(id,name,size,appProperties)${page ? `&pageToken=${encodeURIComponent(page)}` : ""}`;
    const j: any = await (await ok(await google.api(u))).json();
    for (const f of j.files || []) out.push({ id: f.id, name: f.name, updated: Number(f.appProperties?.updated || 0), size: Number(f.size || 0) });
    page = j.nextPageToken || "";
  } while (page);
  return out;
}

async function download(id: string): Promise<any> {
  return (await ok(await google.api(`${google.DRIVE}/files/${id}?alt=media`))).json();
}

/** Writes a file to the app folder (a new one, or over `id`). */
async function upload(name: string, data: unknown, updated: number, id?: string): Promise<string> {
  const b = "pelica" + Math.random().toString(36).slice(2);
  const meta = id ? { appProperties: { updated: String(updated) } } : { name, parents: ["appDataFolder"], appProperties: { updated: String(updated) } };
  const body = `--${b}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n${JSON.stringify(meta)}\r\n--${b}\r\nContent-Type: application/json\r\n\r\n${JSON.stringify(data)}\r\n--${b}--`;
  const r = await ok(await google.api(`${google.UPLOAD}/files${id ? `/${id}` : ""}?uploadType=multipart&fields=id`, { method: id ? "PATCH" : "POST", headers: { "Content-Type": `multipart/related; boundary=${b}` }, body }));
  return ((await r.json()) as any).id;
}

const remove = async (id: string) => {
  const r = await google.api(`${google.DRIVE}/files/${id}`, { method: "DELETE" });
  if (!r.ok && r.status !== 404) await ok(r);
};

/** What's in Pelica's folder on Drive. */
export async function usage() {
  const all = await listAll();
  return { files: all.length, bytes: all.reduce((n, f) => n + f.size, 0) };
}

/** Empties Pelica's folder on Drive (this PC keeps everything). */
export async function wipe() {
  if (busy) throw new Error("A sync is running. Try again in a moment.");
  busy = true;
  if (timer) clearTimeout(timer);
  try {
    for (const f of await listAll()) await remove(f.id);
    reset();
  } finally {
    busy = false;
  }
}

// ---------- this PC's chats ----------

const chatFile = (id: string) => path.join(chatsDir, id + ".json");

function localChats(): Map<string, number> {
  const out = new Map<string, number>();
  fs.mkdirSync(chatsDir, { recursive: true });
  for (const f of fs.readdirSync(chatsDir)) {
    if (!f.endsWith(".json")) continue;
    try {
      const c = JSON.parse(fs.readFileSync(path.join(chatsDir, f), "utf8"));
      if (c?.id === f.slice(0, -5)) out.set(c.id, Number(c.updated) || 0);
    } catch {}
  }
  return out;
}

function writeChat(c: any) {
  const p = chatFile(c.id);
  fs.writeFileSync(p + ".tmp", JSON.stringify(c));
  fs.renameSync(p + ".tmp", p);
}

const readChat = (id: string) => JSON.parse(fs.readFileSync(chatFile(id), "utf8"));

/** The other side of a chat changed on two PCs, kept beside it. */
function keepCopy(c: any): string {
  const id = `${c.id}-v${Date.now().toString(36)}`;
  writeChat({ ...c, id, title: `${c.title || "Chat"} (other version)` });
  return id;
}

// ---------- the profile ----------

function pickProfile(s: Record<string, any>) {
  const out: Record<string, any> = {};
  for (const k of SYNCED) if (s[k] !== undefined) out[k] = s[k];
  return out;
}

/** The first sync on this PC: lists are joined, so nothing here is lost. */
function joinProfile(here: Record<string, any>, there: Record<string, any>) {
  const out = { ...here, ...there };
  const join = (k: string, key: (x: any) => string) => {
    const a: any[] = Array.isArray(there[k]) ? there[k] : [];
    const seen = new Set(a.map(key));
    out[k] = [...a, ...(Array.isArray(here[k]) ? here[k] : []).filter((x) => !seen.has(key(x)))];
  };
  join("projects", (p) => p.id);
  join("memories", (m) => m.id);
  join("prompts", (p) => p.name);
  if (!there.customInstructions && here.customInstructions) out.customInstructions = here.customInstructions;
  if (!there.userName && here.userName) out.userName = here.userName;
  return out;
}

// ---------- a sync ----------

export async function syncNow(): Promise<SyncStatus> {
  if (!google.profile()) throw new Error("Sign in first (Settings → Account).");
  if (busy) {
    again = true;
    return status();
  }
  busy = true;
  error = "";
  const changed = new Set<string>();
  let settingsChanged = false;
  try {
    const first = !st.lastSync;
    const remote = await listAll();
    const chats = new Map<string, Remote>();
    let profileFile: Remote | undefined;
    let deletedFile: Remote | undefined;
    for (const f of remote) {
      const m = f.name.match(/^chat-([\w-]+)\.json$/);
      if (m) chats.set(m[1], f);
      else if (f.name === "profile.json") profileFile = f;
      else if (f.name === "deleted.json") deletedFile = f;
    }

    const skip = new Set<string>(hooks.settings().syncSkip || []);
    if (!skip.has("chats")) {
    // deletions, from every PC
    const remoteDeleted: Record<string, number> = deletedFile ? (await download(deletedFile.id).catch(() => ({}))).ids || {} : {};
    const deleted: Record<string, number> = { ...remoteDeleted };
    for (const [id, t] of Object.entries(st.deleted)) deleted[id] = Math.max(t, deleted[id] || 0);
    const local = localChats();
    for (const [id, when] of Object.entries(deleted)) {
      const here = local.get(id);
      if (here !== undefined) {
        // (changed after it was deleted elsewhere: it stays)
        if (here > when) {
          delete deleted[id];
          continue;
        }
        await shell.trashItem(chatFile(id)).catch(() => {});
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
      } else if (r.updated === updated) {
        // (the same on both sides)
      } else if (updated > r.updated) {
        if (base !== undefined && r.updated > base) changed.add(keepCopy(await download(r.id)));
        await upload(`chat-${id}.json`, readChat(id), updated, r.id);
      } else {
        if (base !== undefined && updated > base) changed.add(keepCopy(readChat(id)));
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

    // deleted.json: what's new, and nothing older than 90 days
    const cutoff = Date.now() - 90 * 86400e3;
    const keep = Object.fromEntries(Object.entries(deleted).filter(([, t]) => t > cutoff));
    if (JSON.stringify(keep) !== JSON.stringify(remoteDeleted)) await upload("deleted.json", { ids: keep }, Date.now(), deletedFile?.id);
    st.deleted = {};
    }

    // the profile, without what's kept off the sync: that's neither sent nor
    // taken here, and what's on Drive for the other PCs stays as it is
    const held = new Set(Object.entries(GROUPS).flatMap(([g, keys]) => (skip.has(g) ? keys : [])));
    const shared = (o: Record<string, any>) => Object.fromEntries(Object.entries(o).filter(([k]) => !held.has(k)));
    const here = shared(pickProfile(hooks.settings()));
    const join = first || !!st.rejoin;
    if (profileFile && (join || profileFile.updated > st.profileAt)) {
      const there = await download(profileFile.id);
      const next = shared(join ? joinProfile(here, shared(there)) : there);
      hooks.apply(next);
      settingsChanged = true;
      st.profileAt = join ? Date.now() : profileFile.updated;
      if (join) await upload("profile.json", { ...there, ...next }, st.profileAt, profileFile.id);
    } else if (!profileFile || st.profileAt > profileFile.updated) {
      st.profileAt ||= Date.now();
      const there = held.size && profileFile ? await download(profileFile.id).catch(() => ({})) : {};
      await upload("profile.json", { ...there, ...here }, st.profileAt, profileFile?.id);
    }
    st.rejoin = false;
    st.lastSync = Date.now();
    saveState();
  } catch (e: any) {
    error = String(e?.message || e);
  } finally {
    busy = false;
  }
  if (changed.size || settingsChanged) hooks.changed({ chats: [...changed], settings: settingsChanged });
  // (copies kept, or a change while this ran: once more)
  if (again || [...changed].some((id) => /-v[0-9a-z]+$/.test(id))) {
    again = false;
    soon(1500);
  }
  return status();
}
