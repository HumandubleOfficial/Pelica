// Pelica and Windows. While the model works, the taskbar button fills (the
// prompt pass) or pulses, and the tray icon says what it is doing; when a
// reply finishes, or a command waits for approval, while you're in another
// window, a notification says so. The tray, the jump list, notifications and
// the quick-ask bar send their actions here.
import { send } from "../agent/agent";
import { app, lastFinished, newChat, openChat, persist, startNewChat, toast, uid, type AppState } from "../state";
import { bridge, type AppAction } from "./bridge";

const PHASE: Record<string, string> = { waiting: "starting", prefill: "reading", thinking: "thinking", writing: "writing", tool: "using a tool", compacting: "compacting", cqs: "checking code" };

function busyOf(s: AppState): { label: string; progress: number } | null {
  if (!s.running) return null;
  const d = s.draft;
  if (!d) return { label: "working", progress: -1 };
  if (d.phase === "prefill" && d.prefill?.total) return { label: `reading ${d.prefill.total.toLocaleString()} tokens`, progress: Math.round((d.prefill.done / d.prefill.total) * 50) / 50 };
  const what = d.phase === "tool" && d.toolName ? `using ${d.toolName}` : PHASE[d.phase] || "working";
  return { label: d.tps && d.phase !== "tool" ? `${what} · ${Math.round(d.tps)} tok/s` : what, progress: -1 };
}

/** A reply's text, plain, for a notification. */
function gist(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, "[code]")
    .replace(/[#*_`>|]/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 200);
}

function onAction(a: AppAction) {
  const s = app.get();
  switch (a.kind) {
    case "new-chat":
      startNewChat();
      setTimeout(() => window.dispatchEvent(new Event("pelica:focus")), 50);
      break;
    case "ask": {
      // from the quick bar, already answered there: the exchange becomes a chat to continue
      if (a.reply !== undefined) {
        if (s.running) return toast("Wait for the reply to finish, or stop it first.");
        const t = Date.now();
        const c = newChat(null);
        c.title = a.text.replace(/\n\n<attachment[\s\S]*$/, "").replace(/\s+/g, " ").slice(0, 48) || "Quick ask";
        c.messages = [
          { id: uid(), role: "user", content: a.text, ts: t },
          { id: uid(), role: "assistant", content: a.reply, ts: t },
        ];
        app.set({ chat: c, view: "chat", context: null });
        persist(c);
        setTimeout(() => window.dispatchEvent(new Event("pelica:focus")), 50);
        break;
      }
      if (s.running) {
        toast("Pelica is busy: your question is in the message box for when it's done.");
        window.dispatchEvent(new CustomEvent("pelica:compose", { detail: a.text }));
        break;
      }
      // a fresh chat, unless the open one is still empty
      if (s.chat && (s.chat.messages.length || s.chat.archive?.length)) startNewChat();
      else app.set({ view: "chat" });
      send(a.text);
      break;
    }
    case "open-chat":
      openChat(a.id);
      break;
    case "view":
      app.set({ view: a.view });
      break;
    case "settings":
      app.set({ settingsOpen: true });
      break;
  }
}

export function startDesktop() {
  if (!bridge) return;
  const b = bridge;
  b.app.onAction(onAction);

  let sent = "";
  let pending: ReturnType<typeof setTimeout> | null = null;
  let wasRunning = false;
  let startedAt = 0;
  let approvalId = "";
  app.sub(() => {
    const s = app.get();
    // the taskbar and tray, at most a few times a second
    const busy = busyOf(s);
    const key = JSON.stringify(busy);
    if (key !== sent && !pending) {
      pending = setTimeout(() => {
        pending = null;
        const now = busyOf(app.get());
        sent = JSON.stringify(now);
        b.app.busy(now).catch(() => {});
      }, busy ? 300 : 0);
    }
    if (!s.settings.notify) {
      wasRunning = s.running;
      return;
    }
    // a command waiting for approval
    if (s.approval && s.approval.id !== approvalId) {
      approvalId = s.approval.id;
      b.app.notify({ title: "Pelica needs your OK", body: `${s.approval.title} ${s.approval.command}`, chatId: s.chat?.id }).catch(() => {});
    }
    // a finished reply (one that took a moment: quick ones you've seen anyway)
    if (s.running && !wasRunning) startedAt = Date.now();
    const done = lastFinished();
    if (!s.running && wasRunning && done && Date.now() - startedAt > 4000) {
      const msgs = done.messages;
      const last = msgs[msgs.length - 1];
      const reply = [...msgs].reverse().find((m) => m.role === "assistant" && m.content);
      const body = last?.role === "event" && last.kind === "error" ? `Stopped: ${last.text}` : reply && reply.role === "assistant" ? gist(reply.content) || "Done." : "Done.";
      b.app.notify({ title: done.title || "Pelica", body, chatId: done.id }).catch(() => {});
    }
    wasRunning = s.running;
  });
}
