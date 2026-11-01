"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.importChats = importChats;
// Chats from another app's data export: ChatGPT (Settings → Data controls →
// Export) and Claude (Settings → Privacy → Export data). Both send a zip with
// a conversations.json; either the zip or that file can be picked. Each
// conversation becomes a Pelica chat with a fixed id from its id there, so
// importing the same export again skips what's already here.
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
/** The conversations.json of an export (a zip is unpacked with Windows' tar). */
async function readExport(file) {
    let text;
    if (/\.zip$/i.test(file)) {
        const dir = node_fs_1.default.mkdtempSync(node_path_1.default.join(node_os_1.default.tmpdir(), "pelica-import-"));
        try {
            await new Promise((ok, bad) => (0, node_child_process_1.execFile)(TAR, ["-xf", file, "-C", dir, "conversations.json"], { windowsHide: true }, (e) => (e ? bad(new Error("That zip has no conversations.json. Pick the zip from ChatGPT's or Claude's data export.")) : ok())));
            text = node_fs_1.default.readFileSync(node_path_1.default.join(dir, "conversations.json"), "utf8");
        }
        finally {
            node_fs_1.default.rmSync(dir, { recursive: true, force: true });
        }
    }
    else
        text = node_fs_1.default.readFileSync(file, "utf8");
    try {
        return JSON.parse(text);
    }
    catch {
        throw new Error("That file isn't a ChatGPT or Claude export.");
    }
}
// Windows' own tar (bsdtar, which reads zips), never another tar on the PATH
const TAR = node_path_1.default.join(process.env.SystemRoot || "C:/Windows", "System32", "tar.exe");
const safeId = (s) => String(s).replace(/[^\w-]/g, "").slice(0, 64);
const ms = (t) => (typeof t === "number" ? (t < 1e12 ? t * 1000 : t) : Date.parse(t) || Date.now());
// ChatGPT's citation marks (private-use characters, or 【4†source】)
const clean = (s) => s
    .replace(/[ \t]*\ue200[^\ue201]*\ue201/g, "")
    .replace(/[ \t]*【\d+(?::\d+)?†[^】]*】/g, "")
    .trim();
function chatgpt(c) {
    const map = c.mapping || {};
    // the branch that was showing: from the current node up to the root
    const chain = [];
    let node = map[c.current_node];
    const seen = new Set();
    while (node && !seen.has(node.id)) {
        seen.add(node.id);
        chain.push(node);
        node = map[node.parent];
    }
    chain.reverse();
    const out = [];
    for (const n of chain) {
        const m = n.message;
        const role = m?.author?.role;
        if (role !== "user" && role !== "assistant")
            continue;
        if (m.metadata?.is_visually_hidden_from_conversation)
            continue;
        const ct = m.content?.content_type;
        if (ct !== "text" && ct !== "multimodal_text")
            continue;
        const parts = (m.content.parts || []).map((p) => (typeof p === "string" ? p : p?.content_type?.includes("image") ? "[image]" : "")).filter(Boolean);
        const text = clean(parts.join("\n\n"));
        if (!text)
            continue;
        const last = out[out.length - 1];
        // (one reply split over several messages, around tool calls)
        if (last && last.role === "assistant" && role === "assistant")
            last.content += "\n\n" + text;
        else
            out.push({ id: "", role, content: text, ts: m.create_time ? ms(m.create_time) : last?.ts ?? ms(c.create_time), model: role === "assistant" ? m.metadata?.model_slug : undefined });
    }
    const id = safeId(c.conversation_id || c.id || "");
    if (!id || !out.length)
        return null;
    return { id, title: c.title || "Imported chat", created: ms(c.create_time), updated: ms(c.update_time || c.create_time), messages: out };
}
function claude(c) {
    const out = [];
    for (const m of c.chat_messages || []) {
        const role = m.sender === "human" ? "user" : m.sender === "assistant" ? "assistant" : null;
        if (!role)
            continue;
        const blocks = Array.isArray(m.content) ? m.content.filter((b) => b?.type === "text" && b.text).map((b) => b.text) : [];
        let text = (blocks.length ? blocks.join("\n\n") : m.text || "").trim();
        for (const a of m.attachments || []) {
            const body = String(a.extracted_content || "").slice(0, 50_000);
            // (Pelica's own attachment block: it shows as a file chip)
            if (body)
                text += `\n\n<attachment name="${String(a.file_name || "file").replace(/"/g, "'")}">\n${body}\n</attachment>`;
            else
                text += `\n\n[Attached: ${a.file_name || "file"}]`;
        }
        for (const f of m.files || [])
            if (f?.file_name)
                text += `\n\n[Attached: ${f.file_name}]`;
        text = text.trim();
        if (!text)
            continue;
        out.push({ id: "", role, content: text, ts: ms(m.created_at || c.created_at) });
    }
    const id = safeId(c.uuid || "");
    if (!id || !out.length)
        return null;
    return { id, title: c.name || "Imported chat", created: ms(c.created_at), updated: ms(c.updated_at || c.created_at), messages: out };
}
/** Writes the export's conversations into the chats folder; already imported ones are skipped. */
async function importChats(file, chatsDir) {
    const data = await readExport(file);
    const list = Array.isArray(data) ? data : Array.isArray(data?.conversations) ? data.conversations : [];
    const from = list.some((c) => c?.mapping) ? "chatgpt" : list.some((c) => Array.isArray(c?.chat_messages)) ? "claude" : null;
    if (!from)
        throw new Error("That file isn't a ChatGPT or Claude export.");
    node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
    const r = { from, added: 0, same: 0, empty: 0 };
    for (const c of list) {
        let x = null;
        try {
            x = from === "chatgpt" ? chatgpt(c) : claude(c);
        }
        catch { }
        if (!x) {
            r.empty++;
            continue;
        }
        const id = `${from === "chatgpt" ? "gpt" : "cl"}-${x.id}`;
        const p = node_path_1.default.join(chatsDir, id + ".json");
        if (node_fs_1.default.existsSync(p)) {
            r.same++;
            continue;
        }
        const messages = x.messages.map((m, i) => ({ ...m, id: `${id}-${i}` }));
        node_fs_1.default.writeFileSync(p, JSON.stringify({ id, title: x.title, created: x.created, updated: x.updated, folder: null, imported: { from, id: x.id }, messages }));
        r.added++;
    }
    return r;
}
