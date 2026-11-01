// Reading across saved chats: search inside messages (the sidebar) and usage
// totals (Settings → Usage). Plain functions over parsed chat JSON, with no
// Node imports, so the UI runs the same code on its in-memory chats when it
// has no Electron bridge.

export interface SearchHit {
  id: string;
  title: string;
  folder: string | null;
  updated: number;
  /** messages in the chat that match */
  hits: number;
  /** the first match with some words around it */
  before: string;
  match: string;
  after: string;
}

export interface ModelUsage {
  model: string;
  replies: number;
  generated: number;
  decodeS: number;
  prompt: number;
  prefillS: number;
  /** median seconds to the first token (replies that recorded it) */
  firstS: number | null;
  /** the fastest moment seen (tok/s) */
  peak: number;
}

export interface Usage {
  chats: number;
  /** your messages */
  messages: number;
  replies: number;
  generated: number;
  prompt: number;
  decodeS: number;
  prefillS: number;
  thinkS: number;
  /** the last 30 days, oldest first: replies and tokens written each day */
  days: { day: string; replies: number; tokens: number }[];
  models: ModelUsage[];
  /** when the oldest chat started */
  since: number | null;
}

const ATTACH = /\n*<attachment name="[^"]*"( kind="image")?>\n[\s\S]*?\n<\/attachment>/g;

/** Chats whose messages (yours and Pelica's replies, not tool output or
 *  attachments) contain `q`, newest first, at most `limit`. */
export function searchChats(chats: any[], q: string, limit = 30): SearchHit[] {
  const needle = String(q || "").trim().toLowerCase();
  if (needle.length < 2) return [];
  const out: SearchHit[] = [];
  for (const c of chats) {
    if (!c || !Array.isArray(c.messages)) continue;
    let hits = 0;
    let first: { text: string; at: number } | null = null;
    for (const m of [...(c.archive || []), ...c.messages]) {
      if ((m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string" || m.summary) continue;
      const text = m.content.replace(ATTACH, "");
      const at = text.toLowerCase().indexOf(needle);
      if (at < 0) continue;
      hits++;
      if (!first) first = { text, at };
    }
    if (!first) continue;
    const { text, at } = first;
    const flat = (s: string) => s.replace(/\s+/g, " ");
    let before = flat(text.slice(Math.max(0, at - 40), at));
    if (at > 40) before = "…" + before.replace(/^\S*\s/, "");
    let after = flat(text.slice(at + needle.length, at + needle.length + 80));
    if (at + needle.length + 80 < text.length) after = after.replace(/\s\S*$/, "") + "…";
    out.push({ id: c.id, title: c.title || "Untitled", folder: c.folder || null, updated: c.updated || 0, hits, before, match: flat(text.slice(at, at + needle.length)), after });
  }
  return out.sort((a, b) => b.updated - a.updated).slice(0, limit);
}

const dayKey = (t: number) => {
  const d = new Date(t);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

/** Totals over every reply that recorded its stats, by model and by day. */
export function chatUsage(chats: any[], now = Date.now()): Usage {
  const u: Usage = { chats: 0, messages: 0, replies: 0, generated: 0, prompt: 0, decodeS: 0, prefillS: 0, thinkS: 0, days: [], models: [], since: null };
  const days = new Map<string, { replies: number; tokens: number }>();
  for (let i = 29; i >= 0; i--) days.set(dayKey(now - i * 86400000), { replies: 0, tokens: 0 });
  const by = new Map<string, ModelUsage & { firsts: number[] }>();
  for (const c of chats) {
    if (!c || !Array.isArray(c.messages)) continue;
    u.chats++;
    if (c.created && (u.since === null || c.created < u.since)) u.since = c.created;
    for (const m of [...(c.archive || []), ...c.messages]) {
      if (m.role === "user" && !m.summary) u.messages++;
      const s = m.role === "assistant" ? m.stats : null;
      if (!s || !(s.generated >= 0)) continue;
      u.replies++;
      u.generated += s.generated || 0;
      u.prompt += s.prompt || 0;
      u.decodeS += s.decode_s || 0;
      u.prefillS += s.prefill_s || 0;
      u.thinkS += s.think_s || 0;
      const d = days.get(dayKey(m.ts || c.updated || 0));
      if (d) (d.replies++, (d.tokens += s.generated || 0));
      const name = String(m.model || "Unknown model").replace(/\.gguf$/i, "");
      let r = by.get(name);
      if (!r) by.set(name, (r = { model: name, replies: 0, generated: 0, decodeS: 0, prompt: 0, prefillS: 0, firstS: null, peak: 0, firsts: [] }));
      r.replies++;
      r.generated += s.generated || 0;
      r.decodeS += s.decode_s || 0;
      r.prompt += s.prompt || 0;
      r.prefillS += s.prefill_s || 0;
      if (typeof s.first_s === "number") r.firsts.push(s.first_s);
      r.peak = Math.max(r.peak, s.peak || 0, s.decode_s > 0.5 ? s.generated / s.decode_s : 0);
    }
  }
  u.days = [...days].map(([day, v]) => ({ day, ...v }));
  u.models = [...by.values()]
    .map(({ firsts, ...r }) => {
      const f = firsts.sort((a, b) => a - b);
      return { ...r, firstS: f.length ? f[Math.floor(f.length / 2)] : null };
    })
    .sort((a, b) => b.generated - a.generated);
  return u;
}
