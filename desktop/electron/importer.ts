// Chats from another app's data export: ChatGPT (Settings → Data controls →
// Export) and Claude (Settings → Privacy → Export data). Both send a zip with
// a conversations.json; either the zip or that file can be picked. Each
// conversation becomes a Pelica chat with a fixed id from its id there, so
// importing the same export again skips what's already here.
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

type From = "chatgpt" | "claude";
type Out = { id: string; role: "user" | "assistant"; content: string; ts: number; model?: string };

/** The conversations.json of an export (a zip is unpacked with Windows' tar). */
async function readExport(file: string): Promise<any> {
  let text: string;
  if (/\.zip$/i.test(file)) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pelica-import-"));
    try {
      await new Promise<void>((ok, bad) =>
        execFile(TAR, ["-xf", file, "-C", dir, "conversations.json"], { windowsHide: true }, (e) => (e ? bad(new Error("That zip has no conversations.json. Pick the zip from ChatGPT's or Claude's data export.")) : ok())),
      );
      text = fs.readFileSync(path.join(dir, "conversations.json"), "utf8");
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  } else text = fs.readFileSync(file, "utf8");
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("That file isn't a ChatGPT or Claude export.");
  }
}

// Windows' own tar (bsdtar, which reads zips), never another tar on the PATH
const TAR = path.join(process.env.SystemRoot || "C:/Windows", "System32", "tar.exe");

const safeId = (s: string) => String(s).replace(/[^\w-]/g, "").slice(0, 64);
const ms = (t: any) => (typeof t === "number" ? (t < 1e12 ? t * 1000 : t) : Date.parse(t) || Date.now());
// ChatGPT's citation marks (private-use characters, or 【4†source】)
const clean = (s: string) =>
  s
    .replace(/[ \t]*\ue200[^\ue201]*\ue201/g, "")
    .replace(/[ \t]*【\d+(?::\d+)?†[^】]*】/g, "")
    .trim();

function chatgpt(c: any): { id: string; title: string; created: number; updated: number; messages: Out[] } | null {
  const map = c.mapping || {};
  // the branch that was showing: from the current node up to the root
  const chain: any[] = [];
  let node = map[c.current_node];
  const seen = new Set<string>();
  while (node && !seen.has(node.id)) {
    seen.add(node.id);
    chain.push(node);
    node = map[node.parent];
  }
  chain.reverse();
  const out: Out[] = [];
  for (const n of chain) {
    const m = n.message;
    const role = m?.author?.role;
    if (role !== "user" && role !== "assistant") continue;
    if (m.metadata?.is_visually_hidden_from_conversation) continue;
    const ct = m.content?.content_type;
    if (ct !== "text" && ct !== "multimodal_text") continue;
    const parts = (m.content.parts || []).map((p: any) => (typeof p === "string" ? p : p?.content_type?.includes("image") ? "[image]" : "")).filter(Boolean);
    const text = clean(parts.join("\n\n"));
    if (!text) continue;
    const last = out[out.length - 1];
    // (one reply split over several messages, around tool calls)
    if (last && last.role === "assistant" && role === "assistant") last.content += "\n\n" + text;
    else out.push({ id: "", role, content: text, ts: m.create_time ? ms(m.create_time) : last?.ts ?? ms(c.create_time), model: role === "assistant" ? m.metadata?.model_slug : undefined });
  }
  const id = safeId(c.conversation_id || c.id || "");
  if (!id || !out.length) return null;
  return { id, title: c.title || "Imported chat", created: ms(c.create_time), updated: ms(c.update_time || c.create_time), messages: out };
}

function claude(c: any): { id: string; title: string; created: number; updated: number; messages: Out[] } | null {
  const out: Out[] = [];
  for (const m of c.chat_messages || []) {
    const role = m.sender === "human" ? "user" : m.sender === "assistant" ? "assistant" : null;
    if (!role) continue;
    const blocks = Array.isArray(m.content) ? m.content.filter((b: any) => b?.type === "text" && b.text).map((b: any) => b.text) : [];
    let text = (blocks.length ? blocks.join("\n\n") : m.text || "").trim();
    for (const a of m.attachments || []) {
      const body = String(a.extracted_content || "").slice(0, 50_000);
      // (Pelica's own attachment block: it shows as a file chip)
      if (body) text += `\n\n<attachment name="${String(a.file_name || "file").replace(/"/g, "'")}">\n${body}\n</attachment>`;
      else text += `\n\n[Attached: ${a.file_name || "file"}]`;
    }
    for (const f of m.files || []) if (f?.file_name) text += `\n\n[Attached: ${f.file_name}]`;
    text = text.trim();
    if (!text) continue;
    out.push({ id: "", role, content: text, ts: ms(m.created_at || c.created_at) });
  }
  const id = safeId(c.uuid || "");
  if (!id || !out.length) return null;
  return { id, title: c.name || "Imported chat", created: ms(c.created_at), updated: ms(c.updated_at || c.created_at), messages: out };
}

/** Writes the export's conversations into the chats folder; already imported ones are skipped. */
export async function importChats(file: string, chatsDir: string): Promise<{ from: From; added: number; same: number; empty: number }> {
  const data = await readExport(file);
  const list: any[] = Array.isArray(data) ? data : Array.isArray(data?.conversations) ? data.conversations : [];
  const from: From | null = list.some((c) => c?.mapping) ? "chatgpt" : list.some((c) => Array.isArray(c?.chat_messages)) ? "claude" : null;
  if (!from) throw new Error("That file isn't a ChatGPT or Claude export.");
  fs.mkdirSync(chatsDir, { recursive: true });
  const r = { from, added: 0, same: 0, empty: 0 };
  for (const c of list) {
    let x: ReturnType<typeof chatgpt> = null;
    try {
      x = from === "chatgpt" ? chatgpt(c) : claude(c);
    } catch {}
    if (!x) {
      r.empty++;
      continue;
    }
    const id = `${from === "chatgpt" ? "gpt" : "cl"}-${x.id}`;
    const p = path.join(chatsDir, id + ".json");
    if (fs.existsSync(p)) {
      r.same++;
      continue;
    }
    const messages = x.messages.map((m, i) => ({ ...m, id: `${id}-${i}` }));
    fs.writeFileSync(p, JSON.stringify({ id, title: x.title, created: x.created, updated: x.updated, folder: null, imported: { from, id: x.id }, messages }));
    r.added++;
  }
  return r;
}
