// The conversation: messages, tool cards, events (compaction, CQS), and the
// reply being written, live (with its speed as it goes).
import { Fragment, memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { run, send } from "../agent/agent";
import { app, persist, toast, uid, updateChat, useApp, type Draft, type Msg, type Stats } from "../state";
import { Icon, Orbs, TodoMark } from "./Icons";
import { AttachChips, splitAttachments } from "./Attachments";
import { Markdown } from "./Markdown";
import { ToolGroup } from "./ToolCard";
import { SpeedGraph, TweenNum } from "./Speed";
import { FindBar } from "./FindBar";
import { editPrompt } from "./Prompts";

/** Keeps a scroll box at its bottom as it grows, until the user scrolls up;
 *  scrolling back down to the bottom picks it up again. Content growing never
 *  moves scrollTop up, so any upward move is the user's. */
function follow(el: HTMLElement, stick: { current: boolean }, last: { current: number }) {
  if (el.scrollHeight - el.scrollTop - el.clientHeight < 24) stick.current = true;
  else if (el.scrollTop < last.current - 1) stick.current = false;
  last.current = el.scrollTop;
}

function Thinking({ text, seconds, live }: { text: string; seconds?: number; live?: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const last = useRef(0);
  useEffect(() => {
    if (open) stick.current = true;
  }, [open]);
  useLayoutEffect(() => {
    const el = ref.current;
    if (live && open && el && stick.current) {
      el.scrollTop = el.scrollHeight;
      last.current = el.scrollTop;
    }
  }, [text, live, open]);
  return (
    <div className={`thinking ${live ? "live" : ""}`}>
      <button className="thinking-head" onClick={() => setOpen(!open)}>
        <span>{live ? "Thinking…" : `Thought${seconds ? ` for ${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s` : ""}`}</span>
        <Icon name={open ? "down" : "chevron"} size={12} />
      </button>
      {open && (
        <div
          className="thinking-body"
          ref={ref}
          onScroll={(e) => follow(e.currentTarget, stick, last)}
          onWheel={(e) => {
            if (e.deltaY < 0) stick.current = false;
          }}
        >
          <Markdown text={text} streaming={live} quiet className="md-think" />
        </div>
      )}
    </div>
  );
}

const secs = (x: number) => (x < 10 ? x.toFixed(1) : Math.round(x).toString()) + "s";
const count = (n: number) => n.toLocaleString();

/** Where a reply's time went: the GPU's layers, attention, the experts (in
 *  VRAM, on the CPU, waiting for the drive). */
function SpeedPanel({ s, close }: { s: Stats; close: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  // above the reply, or below it when the chat would cut it off at the top
  const [below, setBelow] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    let p = el.parentElement;
    while (p && !/auto|scroll/.test(getComputedStyle(p).overflowY)) p = p.parentElement;
    if (el.getBoundingClientRect().top < (p ? p.getBoundingClientRect().top : 0) + 4) setBelow(true);
  }, []);
  useEffect(() => {
    const off = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && close();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("mousedown", off);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", off);
      document.removeEventListener("keydown", esc);
    };
  }, [close]);
  const p = s.parts;
  const rows: [string, number, string][] = p
    ? [
        ["GPU layers", p.gpu_s, "sp-gpu"],
        ["Attention", p.attention_s, "sp-att"],
        ["Experts in VRAM", p.experts_vram_s, "sp-vram"],
        ["Experts on the CPU", p.experts_cpu_s, "sp-cpu"],
        ["Waiting for the drive", p.drive_s, "sp-drive"],
        ["Other", p.other_s, "sp-other"],
      ]
    : [];
  const total = rows.reduce((a, r) => a + r[1], 0);
  const shown = rows.filter((r) => total > 0 && r[1] / total >= 0.005);
  return (
    <div className={`speed-panel${below ? " below" : ""}`} ref={ref} onMouseDown={(e) => e.stopPropagation()}>
      <div className="sp-head">
        <b>{s.tps.toFixed(1)} tok/s</b>
        {s.peak ? <span className="muted">{` · fastest ${s.peak.toFixed(1)}`}</span> : null}
      </div>
      <SpeedGraph values={s.speed} w={260} h={36} />
      <div className="sp-line">
        Prompt {count(s.prompt)} tokens{s.prefill_s > 0.05 ? ` in ${secs(s.prefill_s)} (${Math.round(s.prompt / Math.max(s.prefill_s, 1e-3))} tok/s)` : ""} · reply {count(s.generated)} tokens in {secs(s.decode_s)}
      </div>
      {p && total > 0 ? (
        <>
          <div className="sp-bar">
            {shown.map((r) => (
              <span key={r[0]} className={r[2]} style={{ width: `${(r[1] / total) * 100}%` }} title={`${r[0]}: ${secs(r[1])}`} />
            ))}
          </div>
          <div className="sp-legend">
            {shown.map((r) => (
              <div key={r[0]}>
                <i className={r[2]} />
                <span>{r[0]}</span>
                <span className="muted">{secs(r[1])}</span>
              </div>
            ))}
          </div>
          <div className="sp-line muted">
            Experts: {count(p.in_vram)} in VRAM, {count(p.on_cpu)} on the CPU
            {p.drive_reads > 0 ? `; ${count(p.drive_reads)} read from the drive (${p.drive_gb.toFixed(1)} GB)` : ""}.
            {p.passes > 0 && s.generated > p.passes * 1.05 ? ` Draft-ahead: ${(s.generated / p.passes).toFixed(2)} tokens per step.` : ""}
          </div>
        </>
      ) : (
        <div className="sp-line muted">This model doesn't report where its time goes yet.</div>
      )}
    </div>
  );
}

function StatsLine({ m }: { m: Extract<Msg, { role: "assistant" }> }) {
  const s = m.stats;
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  if (!s) return null;
  return (
    <div className={`stats-line${open ? " open" : ""}`}>
      {open && <SpeedPanel s={s} close={close} />}
      <span className="stats-speed clickable" onMouseDown={(e) => e.stopPropagation()} onClick={() => setOpen(!open)} title={s.peak ? `average ${s.tps.toFixed(1)} tok/s, fastest ${s.peak.toFixed(1)}: click for where the time went` : "generation speed: click for where the time went"}>
        <SpeedGraph values={s.speed} w={48} h={14} />
        {s.tps.toFixed(1)} tok/s
      </span>
      {s.first_s !== undefined && s.first_s > 0 && <span title="time to the first token">first token {s.first_s.toFixed(1)}s</span>}
      <span>{s.generated} tokens</span>
      {s.prefill_s > 0.05 && <span title="reading the prompt">{s.prefill_s.toFixed(1)}s prompt</span>}
      {s.ctx > 0 && (
        <span title="context used">
          {Math.round((s.ctx_used / s.ctx) * 100)}% of {Math.round(s.ctx / 1024)}K
        </span>
      )}
      {m.model && <span>{m.model}</span>}
    </div>
  );
}

function CqsCard({ text, detail }: { text: string; detail?: string }) {
  const [open, setOpen] = useState(false);
  let d: any = {};
  try {
    d = JSON.parse(detail || "{}");
  } catch {}
  const busy = d.state === "scanning" || d.state === "fixing";
  return (
    <div className={`cqs-card ${d.state}`}>
      <div className="cqs-head" onClick={() => setOpen(!open)}>
        <span className="cqs-badge">{busy ? <span className="spinner" /> : <Icon name="scan" size={14} />}CQS</span>
        <span className="cqs-text">{text.replace(/^CQS:\s*/, "")}</span>
        <Icon name={open ? "down" : "chevron"} size={12} />
      </div>
      {open && (
        <div className="cqs-body">
          {d.files && <div className="muted small">Files: {d.files.join(", ")}</div>}
          {d.ran?.length > 0 && <div className="muted small">Checked with: {d.ran.join(", ")}</div>}
          {d.skipped?.length > 0 && <div className="muted small">Skipped (declined): {d.skipped.join(", ")}</div>}
          {d.issues?.length > 0 && (
            <ul className="cqs-issues">
              {d.issues.map((i: any, k: number) => (
                <li key={k} className={i.severity}>
                  <span className="mono">
                    {i.file}
                    {i.line ? `:${i.line}` : ""}
                  </span>{" "}
                  {i.message}
                </li>
              ))}
            </ul>
          )}
          {d.steps?.length > 0 && (
            <div className="cqs-steps">
              {d.steps.map((s: string, k: number) => (
                <div key={k}>· {s}</div>
              ))}
            </div>
          )}
          {d.summary && <Markdown text={d.summary} />}
        </div>
      )}
    </div>
  );
}

// Messages are immutable (a change makes a new object), so rows re-render
// only when theirs changed, not on every frame of the reply being written.
const EventRow = memo(function EventRow({ m }: { m: Extract<Msg, { role: "event" }> }) {
  const [open, setOpen] = useState(false);
  if (m.kind === "cqs") return <CqsCard text={m.text} detail={m.detail} />;
  if (m.kind === "compact")
    return (
      <div className="compact-row">
        <button onClick={() => setOpen(!open)}>
          <Icon name="layers" size={13} /> {m.text} <Icon name={open ? "down" : "chevron"} size={12} />
        </button>
        {open && m.detail && (
          <div className="compact-summary">
            <Markdown text={m.detail} />
          </div>
        )}
      </div>
    );
  return (
    <div className={`event-row ${m.kind}`}>
      <Icon name={m.kind === "error" ? "x" : "info"} size={14} />
      <span>{m.text}</span>
      {m.kind === "error" && (
        <button className="link-btn" onClick={() => run()}>
          Try again
        </button>
      )}
      {m.kind === "info" && /send "continue"/i.test(m.text) && (
        <button className="link-btn" onClick={() => send("Continue.")}>
          Continue
        </button>
      )}
    </div>
  );
});

const UserMsg = memo(function UserMsg({ m }: { m: Extract<Msg, { role: "user" }> }) {
  if (m.summary) return null;
  const { text, files } = splitAttachments(m.content, m.images);
  return (
    <div className="user-row">
      <AttachChips files={files} />
      {text && <div className="user-bubble">{text}</div>}
      <div className="msg-actions">
        <button className="ghost-btn" title="Copy" onClick={() => navigator.clipboard.writeText(m.content)}>
          <Icon name="copy" size={13} />
        </button>
        <button
          className="ghost-btn"
          title="Edit and resend"
          onClick={() => {
            if (app.get().running) return toast("Wait for the reply to finish, or stop it first.");
            updateChat((c) => ({
              ...c,
              messages: c.messages.slice(
                0,
                c.messages.findIndex((x) => x.id === m.id),
              ),
            }));
            window.dispatchEvent(new CustomEvent("pelica:compose", { detail: { text, files } }));
          }}
        >
          <Icon name="edit" size={13} />
        </button>
        <button className="ghost-btn" title="Save as a prompt" onClick={() => editPrompt({ index: null, name: text.split("\n")[0].slice(0, 40), text })}>
          <Icon name="prompt" size={13} />
        </button>
      </div>
    </div>
  );
});

type ToolMsg = Extract<Msg, { role: "tool" }>;

/** Drops the last reply (and its tool calls) and answers the last message again. */
function retryLast() {
  const c = app.get().chat;
  if (!c) return;
  if (app.get().running) return toast("Wait for the reply to finish, or stop it first.");
  const i = c.messages.map((m) => m.role).lastIndexOf("user");
  if (i < 0) return;
  updateChat((x) => ({ ...x, messages: x.messages.slice(0, i + 1) }));
  run();
}

/** A new chat with this one's messages up to the reply, to try another direction. */
async function branchFrom(id: string) {
  const c = app.get().chat;
  if (!c) return;
  if (app.get().running) return toast("Wait for the reply to finish, or stop it first.");
  const i = c.messages.findIndex((m) => m.id === id);
  const now = Date.now();
  const b = { ...c, id: uid(), title: `${c.title} (branch)`, created: now, updated: now, pinned: false, messages: c.messages.slice(0, i + 1) };
  await persist(b);
  app.set({ chat: b, view: "chat", context: null });
  toast("Branched into a new chat");
}

/** Reads the reply out with Windows' voices (again to stop). */
function ReadAloud({ text }: { text: string }) {
  const [on, setOn] = useState(false);
  useEffect(() => () => void (on && speechSynthesis.cancel()), [on]);
  const go = () => {
    if (on) return (speechSynthesis.cancel(), setOn(false));
    const plain = text
      .replace(/```[\s\S]*?```/g, " (code) ")
      .replace(/[*_#>`]/g, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
    const u = new SpeechSynthesisUtterance(plain);
    u.onend = u.onerror = () => setOn(false);
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
    setOn(true);
  };
  return (
    <button className={`ghost-btn ${on ? "on" : ""}`} title={on ? "Stop reading" : "Read aloud"} onClick={go}>
      <Icon name={on ? "stop" : "speaker"} size={14} />
    </button>
  );
}

/** "Just now", "5 minutes ago": kept fresh. */
function Ago({ ts }: { ts: number }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 30000);
    return () => clearInterval(t);
  }, []);
  const s = (Date.now() - ts) / 1000;
  const n = (v: number, u: string) => `${Math.floor(v)} ${u}${Math.floor(v) === 1 ? "" : "s"} ago`;
  const text = s < 60 ? "Just now" : s < 3600 ? n(s / 60, "minute") : s < 86400 ? n(s / 3600, "hour") : new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return (
    <span className="msg-ago" title={new Date(ts).toLocaleString()}>
      {text}
    </span>
  );
}

/** Tool calls in a row go in one box (a reply that only calls tools doesn't break the row). */
function groupTools(msgs: Msg[]): (Msg | ToolMsg[])[] {
  const out: (Msg | ToolMsg[])[] = [];
  for (const m of msgs) {
    if (m.role === "tool") {
      const last = out[out.length - 1];
      if (Array.isArray(last)) last.push(m);
      else out.push([m]);
    } else if (m.role === "assistant" && !m.content && !m.reasoning) continue;
    else out.push(m);
  }
  return out;
}

/** Consecutive assistant and tool messages: one turn of Pelica's. */
const Turn = memo(
  function Turn({ msgs, last }: { msgs: Msg[]; last: boolean }) {
    const text = msgs
      .filter((m) => m.role === "assistant")
      .map((m) => (m as any).content)
      .filter(Boolean)
      .join("\n\n");
    const final = [...msgs].reverse().find((m) => m.role === "assistant") as Extract<Msg, { role: "assistant" }> | undefined;
    const liveNow = useApp((s) => s.running && s.liveId === s.chat?.id) && last;
    if (msgs.length === 1 && final?.ack) return null;
    const groups = groupTools(msgs);
    return (
      <div className="turn">
        <div className="turn-avatar">
          <Orbs live={liveNow} />
        </div>
        <div className="turn-body">
          {groups.map((g, gi) =>
            Array.isArray(g) ? (
              <ToolGroup key={g[0].id} items={g.map((m) => m.ui)} live={liveNow && gi === groups.length - 1} />
            ) : g.role === "assistant" ? (
              <div key={g.id}>
                {g.reasoning && <Thinking text={g.reasoning} seconds={g.stats?.think_s} />}
                {g.content && <Markdown text={g.content} />}
              </div>
            ) : null,
          )}
          {final && !last && null}
          {final && !liveNow && (
            <div className={`turn-foot ${last ? "last" : ""}`}>
              <div className="msg-actions">
                <button className="ghost-btn" title="Copy" onClick={() => navigator.clipboard.writeText(text)}>
                  <Icon name="copy" size={14} />
                </button>
                <button className="ghost-btn" title="Branch from here" onClick={() => branchFrom(final.id)}>
                  <Icon name="branch" size={14} />
                </button>
                <ReadAloud text={text} />
                {last && (
                  <button className="ghost-btn" title="Retry: answer the last message again" onClick={retryLast}>
                    <Icon name="reload" size={14} />
                  </button>
                )}
                <Ago ts={final.ts} />
              </div>
              <StatsLine m={final} />
            </div>
          )}
        </div>
      </div>
    );
  },
  (a, b) => a.last === b.last && a.msgs.length === b.msgs.length && a.msgs.every((m, i) => m === b.msgs[i]),
);

function PhaseLine({ d }: { d: Draft }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((x) => x + 1), 500);
    return () => clearInterval(t);
  }, []);
  const secs = (Date.now() - d.started) / 1000;
  let label = "";
  if (d.phase === "waiting") label = "Starting…";
  else if (d.phase === "prefill") label = d.prefill && d.prefill.total > 600 ? `Reading ${d.prefill.total.toLocaleString()} tokens… ${Math.round((d.prefill.done / d.prefill.total) * 100)}%` : "Reading…";
  // (the thinking box says "Thinking…" once its text starts)
  else if (d.phase === "thinking") label = d.reasoning ? "" : "Thinking…";
  else if (d.phase === "tool") label = d.toolName ? `Working: ${d.toolName.replace(/_/g, " ")}…` : "Working…";
  else if (d.phase === "compacting") label = "Compacting the conversation…";
  else if (d.phase === "cqs") label = "CQS is checking the code…";
  else label = "";
  return (
    <div className="phase-line">
      {label && <span className="shimmer">{label}</span>}
      {(d.phase === "writing" || d.phase === "thinking" || d.phase === "compacting" || d.phase === "cqs") && d.tps > 0 && (
        <span className="tps-live" title="Speed: the graph is moment to moment, the number the average so far">
          <SpeedGraph values={d.speed} live />
          <TweenNum v={d.tps} /> tok/s
        </span>
      )}
      {secs > 2 && <span className="muted small">{Math.round(secs)}s</span>}
    </div>
  );
}

/** The reply being written: the only part that changes every frame. */
function LiveDraft({ chatId, cont, scroller, stick }: { chatId: string; cont: boolean; scroller: React.RefObject<HTMLDivElement | null>; stick: React.RefObject<boolean> }) {
  const d = useApp((s) => s.draft);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });
  return d && d.chatId === chatId ? <DraftView d={d} cont={cont} /> : null;
}

/** `cont`: it carries on the turn above (after tool calls), whose orbs are
 *  already moving, so it shows none of its own. */
function DraftView({ d, cont }: { d: Draft; cont: boolean }) {
  return (
    <div className={`turn draft ${cont ? "cont" : ""}`}>
      <div className="turn-avatar">{!cont && <Orbs live />}</div>
      <div className="turn-body">
        {d.reasoning && <Thinking text={d.reasoning} live={d.phase === "thinking"} />}
        {d.content && d.phase !== "compacting" && <Markdown text={d.content} streaming />}
        {d.phase === "compacting" && d.content && <div className="muted small compact-live">{d.content.slice(-400)}</div>}
        <PhaseLine d={d} />
      </div>
    </div>
  );
}

/** The plan above the message box while it has steps left: just the steps. */
function Todos() {
  const todos = useApp((s) => s.chat?.todos);
  if (!todos?.length || todos.every((t) => t.status === "completed")) return null;
  return (
    <div className="todo-panel">
      <ul className="todos">
        {todos.map((t, i) => (
          <li key={i} className={t.status}>
            <TodoMark status={t.status} />
            {t.content}
          </li>
        ))}
      </ul>
    </div>
  );
}

const dayOf = (t: number) => new Date(t).toDateString();

/** "Today", "Yesterday", the weekday this past week, else the date. */
function dayLabel(t: number) {
  const d = new Date(t);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return "Today";
  const y = new Date(now);
  y.setDate(y.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return "Yesterday";
  if (now.getTime() - t < 6 * 86400000) return d.toLocaleDateString(undefined, { weekday: "long" });
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: d.getFullYear() !== now.getFullYear() ? "numeric" : undefined });
}

/** Select text in the chat and a Quote button shows above it: the words go
 *  into the message box as a quote, to reply to that part. */
function QuoteButton({ col }: { col: React.RefObject<HTMLDivElement | null> }) {
  const [at, setAt] = useState<{ x: number; y: number; below: boolean; text: string } | null>(null);
  useEffect(() => {
    const check = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim() || "";
      if (!sel || !text || !sel.rangeCount || !col.current) return setAt(null);
      const r = sel.getRangeAt(0);
      if (!col.current.contains(r.commonAncestorContainer)) return setAt(null);
      const rect = r.getBoundingClientRect();
      if (!rect.width && !rect.height) return setAt(null);
      const below = rect.top < 70;
      setAt({ x: Math.min(window.innerWidth - 50, Math.max(50, rect.left + rect.width / 2)), y: below ? rect.bottom : rect.top, below, text });
    };
    const later = () => setTimeout(check, 0);
    const gone = () => !window.getSelection()?.toString() && setAt(null);
    const hide = () => setAt(null);
    document.addEventListener("mouseup", later);
    document.addEventListener("keyup", later);
    document.addEventListener("selectionchange", gone);
    // (it would stay where the words were)
    document.addEventListener("scroll", hide, true);
    return () => {
      document.removeEventListener("mouseup", later);
      document.removeEventListener("keyup", later);
      document.removeEventListener("selectionchange", gone);
      document.removeEventListener("scroll", hide, true);
    };
  }, []);
  if (!at) return null;
  return (
    <button
      className={`quote-btn ${at.below ? "below" : ""}`}
      style={{ left: at.x, top: at.y }}
      onMouseDown={(e) => e.preventDefault()}
      onClick={() => {
        const q = at.text
          .split("\n")
          .map((l) => (l.trim() ? "> " + l : ">"))
          .join("\n");
        window.dispatchEvent(new CustomEvent("pelica:insert", { detail: q + "\n\n" }));
        window.getSelection()?.removeAllRanges();
        setAt(null);
      }}
    >
      <Icon name="chat" size={13} /> Quote
    </button>
  );
}

export function ChatView() {
  const chat = useApp((s) => s.chat);
  const [showArchive, setShowArchive] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const col = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const last = useRef(0);
  // scrolled well up from the latest message: the jump button shows
  const [away, setAway] = useState(false);
  const live = useApp((s) => s.running && s.liveId === s.chat?.id);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) {
      el.scrollTop = el.scrollHeight;
      last.current = el.scrollTop;
    }
  });
  // a new chat starts at its bottom
  useEffect(() => {
    stick.current = true;
  }, [chat?.id]);
  // Alt+↑ / Alt+↓: to your previous / next message
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || (e.key !== "ArrowUp" && e.key !== "ArrowDown")) return;
      const el = scroller.current;
      if (!el || !col.current || app.get().view !== "chat") return;
      e.preventDefault();
      const top = el.getBoundingClientRect().top + 16;
      const pos = [...col.current.querySelectorAll<HTMLElement>(".user-row")].map((r) => r.getBoundingClientRect().top - top);
      const to = e.key === "ArrowUp" ? [...pos].reverse().find((p) => p < -4) : pos.find((p) => p > 4);
      if (to === undefined) {
        if (e.key === "ArrowDown") (stick.current = true), el.scrollTo({ top: el.scrollHeight, behavior: app.get().settings.animations ? "smooth" : "auto" });
        return;
      }
      stick.current = false;
      el.scrollTo({ top: el.scrollTop + to, behavior: app.get().settings.animations ? "smooth" : "auto" });
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  if (!chat) return null;
  const rows: { key: string; node: React.ReactNode | Msg[]; ts: number }[] = [];
  const list = [...(showArchive ? (chat.archive || []).map((m) => ({ ...m, id: "a" + m.id })) : []), ...chat.messages];
  let turn: Msg[] = [];
  // the last turn is the one after the last message of yours, even with a
  // note (length limit, CQS) under it: it gets Retry and the live orbs
  let lastTurn = -1;
  const flush = () => {
    if (turn.length) (lastTurn = rows.length), rows.push({ key: turn[0].id, node: turn, ts: turn[0].ts });
    turn = [];
  };
  for (const m of list) {
    if (m.role === "assistant" || m.role === "tool") {
      turn.push(m);
      continue;
    }
    flush();
    if (m.role === "user") (lastTurn = -1), rows.push({ key: m.id, node: <UserMsg m={m} />, ts: m.ts });
    else rows.push({ key: m.id, node: <EventRow m={m} />, ts: m.ts });
  }
  // (the last message is Pelica's: a reply being written carries that turn on)
  const turn0 = !!list.length && (list[list.length - 1].role === "assistant" || list[list.length - 1].role === "tool");
  flush();
  for (let i = 0; i < rows.length; i++) if (Array.isArray(rows[i].node)) rows[i].node = <Turn msgs={rows[i].node as Msg[]} last={i === lastTurn} />;
  // a chat that spans more than one day gets a line where each day starts
  const days = rows.length > 1 && !!rows[0].ts && dayOf(rows[0].ts) !== dayOf(rows[rows.length - 1].ts || Date.now());

  return (
    <div
      className="chat-scroll"
      ref={scroller}
      onScroll={(e) => {
        const el = e.currentTarget;
        follow(el, stick, last);
        const far = el.scrollHeight - el.scrollTop - el.clientHeight > 400;
        if (far !== away) setAway(far);
      }}
      onWheel={(e) => {
        // scrolling up inside the thinking box shouldn't let go of the chat
        if (e.deltaY < 0 && !(e.target as HTMLElement).closest?.(".thinking-body")) stick.current = false;
      }}
    >
      <FindBar root={col} />
      <QuoteButton col={col} />
      <div className="chat-col" ref={col}>
        {chat.archive?.length ? (
          <button className="archive-btn" onClick={() => setShowArchive(!showArchive)}>
            {showArchive ? "Hide" : "Show"} {chat.archive.length} messages from before compaction
          </button>
        ) : null}
        {rows.map((r, i) => (
          <Fragment key={r.key}>
            {days && r.ts && (i === 0 || dayOf(rows[i - 1].ts) !== dayOf(r.ts)) && (
              <div className="day-sep">
                <span>{dayLabel(r.ts)}</span>
              </div>
            )}
            <div>{r.node as React.ReactNode}</div>
          </Fragment>
        ))}
        <LiveDraft chatId={chat.id} cont={turn0} scroller={scroller} stick={stick} />
        <div style={{ height: 24 }} />
      </div>
      {away && (
        <div className="jump-wrap">
          <button
            className="jump-latest"
            title="Jump to the latest message"
            onClick={() => {
              const el = scroller.current;
              if (!el) return;
              stick.current = true;
              el.scrollTo({ top: el.scrollHeight, behavior: app.get().settings.animations ? "smooth" : "auto" });
            }}
          >
            <Icon name="down" size={15} />
            {live && <span>Pelica is writing</span>}
          </button>
        </div>
      )}
      <Todos />
    </div>
  );
}
