// Pelica on your phone: while Settings → Phone is on, what the window shows
// (the open chat, the reply being written, an approval, the chat list) goes to
// the phone server in main (electron/phone.ts), and what the phone does comes
// back here and runs as if typed in the window.
import { send, stop } from "../agent/agent";
import { app, openChat, startNewChat, type Msg, type ToolUI } from "../state";
import { bridge, type PhoneCmd } from "./bridge";

const VERB: Record<ToolUI["kind"], string> = {
  search: "Searched the web",
  page: "Read a page",
  read: "Read",
  write: "Wrote",
  edit: "Edited",
  list: "Listed",
  grep: "Searched files",
  command: "Ran",
  todo: "Updated the to-do list",
  browser: "Opened in the browser",
  mcp: "Used",
  trash: "Moved to the Recycle Bin",
  error: "Failed",
  denied: "Not allowed",
  cqs: "Checked the code",
  system: "Looked at this computer",
  clipboard: "Used the clipboard",
  open: "Opened",
  memory: "Remembered",
};

function line(m: Msg) {
  switch (m.role) {
    case "user":
      return { id: m.id, role: "user", text: m.content, photos: m.images?.length || 0 };
    case "assistant":
      return m.content.trim() ? { id: m.id, role: "assistant", text: m.content, model: m.model } : null;
    case "tool": {
      const verb = m.ui.kind === "memory" && m.ui.detail && m.ui.detail !== "Remembered" ? m.ui.detail : VERB[m.ui.kind] || m.name;
      const pics = m.images?.length ? ` · ${m.images.length} picture${m.images.length === 1 ? "" : "s"}` : "";
      return { id: m.id, role: "tool", text: `${verb}${m.ui.kind === "todo" || m.ui.kind === "system" ? "" : " " + (m.ui.path || m.ui.title || "")}`.trim() + pics };
    }
    case "event":
      return { id: m.id, role: "event", text: m.text, kind: m.kind };
  }
}

function snapshot() {
  const s = app.get();
  const c = s.chat;
  const d = s.draft && s.draft.chatId === c?.id ? s.draft : null;
  return {
    model: s.status?.state === "ready" ? s.status.model || "" : "",
    state: s.status?.state || "error",
    running: s.running,
    // a reply being written in another chat than the open one
    elsewhere: s.running && s.liveId && s.liveId !== c?.id ? s.chats.find((x) => x.id === s.liveId)?.title || "another chat" : null,
    chats: s.chats.slice(0, 60).map((x) => ({ id: x.id, title: x.title, updated: x.updated })),
    chat: c ? { id: c.id, title: c.title, messages: c.messages.slice(-80).map(line).filter(Boolean) } : null,
    draft: d ? { text: d.content, phase: d.phase, tool: d.toolName || "" } : null,
    approval: s.approval ? { id: s.approval.id, title: s.approval.title, command: s.approval.command, args: s.approval.args || "", why: s.approval.why || "", chatTitle: s.approval.chatTitle || "" } : null,
    toast: s.toast,
  };
}

async function onCmd(c: PhoneCmd) {
  const s = app.get();
  switch (c.type) {
    case "open":
      if (c.id && c.id !== s.chat?.id) await openChat(c.id);
      app.set({ view: "chat" });
      break;
    case "new":
      // (a reply being written carries on in its own chat, as in the window)
      startNewChat(null);
      break;
    case "send": {
      const text = (c.text || "").trim();
      if (!text) return;
      if (c.id && c.id !== s.chat?.id && s.chats.some((x) => x.id === c.id)) await openChat(c.id);
      app.set({ view: "chat" });
      await send(text);
      break;
    }
    case "approve": {
      const a = s.approval;
      if (a && a.id === c.id) a.resolve(c.answer);
      break;
    }
    case "stop":
      stop();
      break;
  }
}

let off: (() => void) | null = null;

/** Mirrors the window to the phone server while Settings → Phone is on. */
export function startPhone() {
  if (!bridge) return;
  bridge.phone.onCmd((c) => {
    if (app.get().settings.phoneOn) onCmd(c).catch(() => {});
  });
  let timer: ReturnType<typeof setTimeout> | null = null;
  let last: unknown[] = [];
  const watch = () => {
    const s = app.get();
    if (!s.settings.phoneOn) {
      off?.();
      off = null;
      return;
    }
    if (off) return;
    const push = () => {
      timer = null;
      bridge!.phone.push(snapshot()).catch(() => {});
    };
    push();
    off = app.sub(() => {
      const t = app.get();
      // (only what the phone shows; a reply streaming pushes at most ~7 times a second)
      const now = [t.chat, t.chats, t.draft, t.approval, t.running, t.status?.state, t.status?.model, t.toast, t.liveId];
      if (now.every((v, i) => v === last[i])) return;
      last = now;
      if (!timer) timer = setTimeout(push, 140);
    });
  };
  watch();
  app.sub(watch);
}
