// A chat as one Markdown document: to save next to your work, or to paste
// somewhere else. Replies and your messages in full; tool calls as a line
// each (with a command's output folded away), thinking left out.
import { app, toast, type Chat, type Msg } from "../state";
import { bridge } from "./bridge";

const when = (t: number) => new Date(t).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

/** A message as you typed it, attachments shown as their names. */
function userText(s: string): string {
  const names: string[] = [];
  const body = s.replace(/\n*<attachment name="([^"]*)"[^>]*>[\s\S]*?<\/attachment>/g, (_m, n: string) => (names.push(n), "")).trim();
  return names.length ? `${body}\n\n*Attached: ${names.map((n) => `\`${n}\``).join(", ")}*` : body;
}

function toolLine(m: Extract<Msg, { role: "tool" }>): string {
  const u = m.ui;
  let line = `> 🔧 **${u.title}**${u.detail ? ` — ${u.detail.replace(/\s+/g, " ").slice(0, 160)}` : ""}`;
  if (u.kind === "command" && u.output) line += `\n\n<details><summary>output</summary>\n\n\`\`\`\n${u.output.slice(-4000)}\n\`\`\`\n\n</details>`;
  return line;
}

export function chatMarkdown(c: Chat): string {
  const out: string[] = [`# ${c.title || "Untitled"}`, "", `*${when(c.created)}${c.folder ? ` · ${c.folder}` : ""} · exported from Pelica*`, ""];
  const all = [...(c.archive || []), ...c.messages];
  for (const m of all) {
    if (m.role === "user") {
      if (m.summary) continue;
      out.push("---", "", `**You** · ${when(m.ts)}`, "", userText(m.content), "");
    } else if (m.role === "assistant") {
      if (!m.content.trim() || m.ack) continue;
      out.push(`**Pelica**${m.model ? ` · ${m.model}` : ""}`, "", m.content.trim(), "");
    } else if (m.role === "tool") {
      out.push(toolLine(m), "");
    } else if (m.kind === "compact" || m.kind === "error") {
      out.push(`*${m.kind === "compact" ? "Compacted" : "Stopped"}: ${m.text}*`, "");
    }
  }
  return out.join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
}

async function load(id?: string): Promise<Chat | null> {
  const cur = app.get().chat;
  if (!id || cur?.id === id) return cur && (cur.messages.length || cur.archive?.length) ? cur : null;
  return bridge ? await bridge.chats.load(id).catch(() => null) : null;
}

/** The chat (the open one by default) on the clipboard, as Markdown. */
export async function copyChat(id?: string) {
  const c = await load(id);
  if (!c) return toast("Nothing to copy yet.");
  const md = chatMarkdown(c);
  if (bridge) await bridge.sys.clipWrite(md);
  else await navigator.clipboard.writeText(md);
  toast(`Copied “${c.title}” as Markdown.`);
}

/** The chat saved as a .md file where you choose (the workspace folder first). */
export async function exportChat(id?: string) {
  const c = await load(id);
  if (!c) return toast("Nothing to export yet.");
  const md = chatMarkdown(c);
  if (!bridge) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    a.download = `${c.title || "chat"}.md`;
    a.click();
    return;
  }
  try {
    const p = await bridge.sys.saveAs(c.title, md, c.folder || app.get().settings.recentFolders[0] || undefined);
    if (p) toast(`Saved to ${p}`);
  } catch (e: any) {
    toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
  }
}
