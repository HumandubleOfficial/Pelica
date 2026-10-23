// The conversation: messages, tool cards, events (compaction, CQS), and the
// reply being written, live (with its speed as it goes).
import { memo, useEffect, useLayoutEffect, useRef, useState } from "react";
import { run } from "../agent/agent";
import { app, updateChat, useApp, type Draft, type Msg } from "../state";
import { EffortOrb } from "./Effort";
import { Icon, Logo } from "./Icons";
import { AttachChips, splitAttachments } from "./Attachments";
import { Markdown } from "./Markdown";
import { ToolCard } from "./ToolCard";

function Thinking({ text, seconds, live, effort }: { text: string; seconds?: number; live?: boolean; effort?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (live && open && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, live, open]);
  return (
    <div className={`thinking ${live ? "live" : ""}`}>
      <button className="thinking-head" onClick={() => setOpen(!open)}>
        <EffortOrb effort={effort || "medium"} size={16} active={!!live} />
        <span>{live ? "Thinking…" : `Thought${seconds ? ` for ${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s` : ""}`}</span>
        <Icon name={open ? "down" : "chevron"} size={12} />
      </button>
      {open && (
        <div className="thinking-body" ref={ref}>
          {text}
        </div>
      )}
    </div>
  );
}

function StatsLine({ m }: { m: Extract<Msg, { role: "assistant" }> }) {
  const s = m.stats;
  if (!s) return null;
  return (
    <div className="stats-line">
      <span title="generation speed">{s.tps.toFixed(1)} tok/s</span>
      <span>{s.generated} tokens</span>
      {s.prefill_s > 0.05 && <span title="reading the prompt">{s.prefill_s.toFixed(1)}s prompt</span>}
      <span title="context used">
        {Math.round((s.ctx_used / s.ctx) * 100)}% of {Math.round(s.ctx / 1024)}K
      </span>
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
            if (app.get().running) return;
            updateChat((c) => ({ ...c, messages: c.messages.slice(0, c.messages.findIndex((x) => x.id === m.id)) }));
            window.dispatchEvent(new CustomEvent("pelica:compose", { detail: { text, files } }));
          }}
        >
          <Icon name="edit" size={13} />
        </button>
      </div>
    </div>
  );
});

/** Consecutive assistant and tool messages: one turn of Pelica's. */
const Turn = memo(
  function Turn({ msgs, last }: { msgs: Msg[]; last: boolean }) {
  const text = msgs
    .filter((m) => m.role === "assistant")
    .map((m) => (m as any).content)
    .filter(Boolean)
    .join("\n\n");
  const final = [...msgs].reverse().find((m) => m.role === "assistant") as Extract<Msg, { role: "assistant" }> | undefined;
  if (msgs.length === 1 && final?.ack) return null;
  return (
    <div className="turn">
      <div className="turn-avatar">
        <Logo size={22} />
      </div>
      <div className="turn-body">
        {msgs.map((m) =>
          m.role === "assistant" ? (
            <div key={m.id}>
              {m.reasoning && <Thinking text={m.reasoning} seconds={m.stats?.think_s} effort={m.effort} />}
              {m.content && <Markdown text={m.content} />}
            </div>
          ) : m.role === "tool" ? (
            <ToolCard key={m.id} ui={m.ui} />
          ) : null,
        )}
        {final && !last && null}
        {final && (
          <div className="turn-foot">
            <div className="msg-actions">
              <button className="ghost-btn" title="Copy" onClick={() => navigator.clipboard.writeText(text)}>
                <Icon name="copy" size={13} />
              </button>
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
  const effort = useApp((s) => (s.status?.efforts?.includes(s.settings.effort) ? s.settings.effort : s.status?.effort));
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((x) => x + 1), 500);
    return () => clearInterval(t);
  }, []);
  const secs = (Date.now() - d.started) / 1000;
  let label = "";
  if (d.phase === "waiting") label = "Starting…";
  else if (d.phase === "prefill") label = d.prefill && d.prefill.total > 600 ? `Reading ${d.prefill.total.toLocaleString()} tokens… ${Math.round((d.prefill.done / d.prefill.total) * 100)}%` : "Reading…";
  else if (d.phase === "thinking") label = "Thinking…";
  else if (d.phase === "tool") label = d.toolName ? `Working: ${d.toolName.replace(/_/g, " ")}…` : "Working…";
  else if (d.phase === "compacting") label = "Compacting the conversation…";
  else if (d.phase === "cqs") label = "CQS is checking the code…";
  else label = "";
  return (
    <div className="phase-line">
      {label && (
        <>
          <EffortOrb effort={d.phase === "thinking" ? effort : "low"} size={16} />
          <span className="shimmer">{label}</span>
        </>
      )}
      {(d.phase === "writing" || d.phase === "thinking" || d.phase === "compacting" || d.phase === "cqs") && d.tps > 0 && <span className="tps-live">{d.tps.toFixed(1)} tok/s</span>}
      {secs > 2 && <span className="muted small">{Math.round(secs)}s</span>}
    </div>
  );
}

/** The reply being written: the only part that changes every frame. */
function LiveDraft({ chatId, scroller, stick }: { chatId: string; scroller: React.RefObject<HTMLDivElement | null>; stick: React.RefObject<boolean> }) {
  const d = useApp((s) => s.draft);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });
  return d && d.chatId === chatId ? <DraftView d={d} /> : null;
}

function DraftView({ d }: { d: Draft }) {
  const effort = useApp((s) => (s.status?.efforts?.includes(s.settings.effort) ? s.settings.effort : s.status?.effort));
  return (
    <div className="turn draft">
      <div className="turn-avatar">
        <Logo size={22} />
      </div>
      <div className="turn-body">
        {d.reasoning && <Thinking text={d.reasoning} live={d.phase === "thinking"} effort={effort} />}
        {d.content && d.phase !== "compacting" && <Markdown text={d.content} streaming />}
        {d.phase === "compacting" && d.content && <div className="muted small compact-live">{d.content.slice(-400)}</div>}
        <PhaseLine d={d} />
      </div>
    </div>
  );
}

function Todos() {
  const todos = useApp((s) => s.chat?.todos);
  const [open, setOpen] = useState(true);
  if (!todos?.length || todos.every((t) => t.status === "completed")) return null;
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <div className="todo-panel">
      <button className="todo-panel-head" onClick={() => setOpen(!open)}>
        <Icon name="list" size={14} /> Plan <span className="muted">{done}/{todos.length}</span>
        <span className="todo-progress">
          <span style={{ width: `${(done / todos.length) * 100}%` }} />
        </span>
        <Icon name={open ? "down" : "chevron"} size={12} />
      </button>
      {open && (
        <ul className="todos">
          {todos.map((t, i) => (
            <li key={i} className={t.status}>
              <span className="todo-box">{t.status === "completed" ? <Icon name="check" size={11} stroke={2.6} /> : t.status === "in_progress" ? <span className="todo-dot" /> : null}</span>
              {t.content}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ChatView() {
  const chat = useApp((s) => s.chat);
  const [showArchive, setShowArchive] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  if (!chat) return null;
  const rows: { key: string; node: React.ReactNode }[] = [];
  const list = [...(showArchive ? (chat.archive || []).map((m) => ({ ...m, id: "a" + m.id })) : []), ...chat.messages];
  let turn: Msg[] = [];
  const flush = (last: boolean) => {
    if (turn.length) rows.push({ key: turn[0].id, node: <Turn msgs={turn} last={last} /> });
    turn = [];
  };
  for (const m of list) {
    if (m.role === "assistant" || m.role === "tool") {
      turn.push(m);
      continue;
    }
    flush(false);
    if (m.role === "user") rows.push({ key: m.id, node: <UserMsg m={m} /> });
    else rows.push({ key: m.id, node: <EventRow m={m} /> });
  }
  flush(true);

  return (
    <div
      className="chat-scroll"
      ref={scroller}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      <div className="chat-col">
        {chat.archive?.length ? (
          <button className="archive-btn" onClick={() => setShowArchive(!showArchive)}>
            {showArchive ? "Hide" : "Show"} {chat.archive.length} messages from before compaction
          </button>
        ) : null}
        {rows.map((r) => (
          <div key={r.key}>{r.node}</div>
        ))}
        <LiveDraft chatId={chat.id} scroller={scroller} stick={stick} />
        <div style={{ height: 24 }} />
      </div>
      <Todos />
    </div>
  );
}
