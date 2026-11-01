// The quick-ask bar: a small window the global shortcut opens over whatever
// you're doing. Type and press Enter: the loaded model answers right in the
// bar (no tools, a short answer), streaming. "Continue in Pelica" opens the
// question and answer as a chat in the main window; Ctrl+Enter sends the
// question there straight away (for work that needs files, commands or the
// web). The clipboard chip adds what you copied (read only when clicked).
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { bridge } from "../lib/bridge";
import { engine, type Status } from "../lib/engine";
import { Icon, Logo } from "./Icons";
import { Markdown } from "./Markdown";
import { attachmentBlock } from "./Attachments";

type Phase = "idle" | "reading" | "thinking" | "writing" | "done" | "error";

export function QuickAsk() {
  const [text, setText] = useState("");
  const [clip, setClip] = useState<string | null>(null);
  const [st, setSt] = useState<Status | null>(null);
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [err, setErr] = useState("");
  const [tps, setTps] = useState(0);
  const card = useRef<HTMLDivElement>(null);
  const ta = useRef<HTMLTextAreaElement>(null);
  const ctl = useRef<AbortController | null>(null);
  const lastQA = useRef<{ q: string; a: string } | null>(null);

  const refresh = () => engine.status().then(setSt).catch(() => setSt(null));
  useEffect(() => {
    refresh();
    ta.current?.focus();
    return bridge?.quick.onShown(() => {
      refresh();
      ta.current?.focus();
      ta.current?.select();
    });
  }, []);
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(160, el.scrollHeight) + "px";
  }, [text]);
  // the window follows the card's height
  useLayoutEffect(() => {
    if (!card.current || !bridge) return;
    const ro = new ResizeObserver(() => bridge!.quick.resize(Math.ceil(card.current!.getBoundingClientRect().height) + 40));
    ro.observe(card.current);
    return () => ro.disconnect();
  }, []);

  const question = () => {
    const t = text.trim();
    return (t || "Look at this:") + (clip ? attachmentBlock("Clipboard", clip) : "");
  };
  const busy = phase === "reading" || phase === "thinking" || phase === "writing";

  async function ask() {
    if ((!text.trim() && !clip) || busy) return;
    if (st?.state !== "ready") return toMain();
    const q = question();
    setAsked(text.trim() || "About the clipboard");
    setAnswer("");
    setErr("");
    setTps(0);
    setPhase("reading");
    setText("");
    setClip(null);
    const c = new AbortController();
    ctl.current = c;
    let out = "";
    try {
      const effort = st.efforts?.includes("off") ? "off" : st.efforts?.includes("low") ? "low" : undefined;
      await engine.chat(
        {
          messages: [
            { role: "system", content: `You are Pelica, answering in a small quick-ask bar on the user's Windows PC. Today is ${new Date().toDateString()}. Answer directly and briefly (a few sentences or a short list; code only if asked). Answer only what was asked, with nothing after the answer.` },
            { role: "user", content: q },
          ],
          tools: [],
          effort,
          max_tokens: 1200,
        },
        (e) => {
          if (e.t === "think") setPhase("thinking");
          else if (e.t === "text") {
            out += e.d;
            setAnswer(out);
            setPhase("writing");
          } else if (e.t === "stats") setTps(e.tps);
        },
        c.signal,
      );
      setPhase("done");
      // (kept for "Continue in Pelica")
      lastQA.current = { q, a: out };
    } catch (e: any) {
      if (c.signal.aborted) {
        setPhase(out ? "done" : "idle");
        lastQA.current = { q, a: out };
        return;
      }
      const m = String(e?.message || e);
      setErr(/busy/i.test(m) ? "Pelica is busy with another reply. Send this to the main window instead, or try again in a moment." : m);
      setPhase("error");
      lastQA.current = { q, a: "" };
    } finally {
      ctl.current = null;
    }
  }
  function stop() {
    ctl.current?.abort();
    engine.stop().catch(() => {});
  }

  /** The question (and the answer so far, if any) as a chat in the main window. */
  function toMain() {
    const qa = lastQA.current;
    if (text.trim() || clip || !qa) {
      if (!text.trim() && !clip) return;
      bridge?.quick.submit(question());
      setText("");
      setClip(null);
      return;
    }
    bridge?.quick.submit(qa.q, qa.a || undefined);
  }

  function reset() {
    setAsked("");
    setAnswer("");
    setPhase("idle");
    setErr("");
    lastQA.current = null;
    ta.current?.focus();
  }

  const ready = st?.state === "ready";
  const shown = phase !== "idle";
  return (
    <div className="quick-wrap">
      <div className={`quick ${shown ? "has-answer" : ""}`} ref={card}>
        <div className="quick-row">
          <span className={`quick-logo ${busy ? "busy" : ""}`}>
            <Logo size={26} />
          </span>
          <textarea
            ref={ta}
            rows={1}
            value={text}
            placeholder={shown ? "Ask something else…" : "Ask Pelica anything…"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                if (busy) stop();
                else bridge?.quick.hide();
              } else if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (e.ctrlKey) toMain();
                else ask();
              }
            }}
          />
          {busy ? (
            <button className="send stop" onClick={stop} title="Stop (Esc)">
              <Icon name="stop" size={14} stroke={2.4} />
            </button>
          ) : (
            <button className="send" disabled={!text.trim() && !clip} onClick={ask} title="Ask (Enter)">
              <Icon name="send" size={16} stroke={2.2} />
            </button>
          )}
        </div>
        {clip !== null && <div className="quick-clip mono">{clip.slice(0, 220) || "(the clipboard has no text)"}</div>}
        {shown && (
          <div className="quick-answer">
            <div className="quick-q">{asked}</div>
            {phase === "error" ? (
              <div className="err-text small">{err}</div>
            ) : answer ? (
              <div className="quick-md">
                <Markdown text={answer} streaming={phase === "writing"} />
              </div>
            ) : (
              <div className="quick-wait">
                <span className="shimmer">{phase === "thinking" ? "Thinking…" : "Reading…"}</span>
              </div>
            )}
            {!busy && (
              <div className="quick-actions">
                <button className="chip" onClick={toMain} title="Open this as a chat in Pelica, where it can use files, commands and the web">
                  <Icon name="chat" size={13} /> Continue in Pelica
                </button>
                {answer && (
                  <button className="chip" onClick={() => navigator.clipboard.writeText(answer)}>
                    <Icon name="copy" size={13} /> Copy
                  </button>
                )}
                <button className="chip" onClick={reset}>
                  <Icon name="x" size={13} /> Clear
                </button>
                <span className="grow" />
                {tps > 0 && <span className="muted small">{tps.toFixed(0)} tok/s</span>}
              </div>
            )}
          </div>
        )}
        <div className="quick-bar">
          <button
            className={`chip ${clip !== null ? "on" : ""}`}
            onClick={async () => {
              if (clip !== null) return setClip(null);
              setClip((await bridge?.quick.clip()) ?? "");
              ta.current?.focus();
            }}
            title="Add the text you copied"
          >
            <Icon name="clipboard" size={13} /> {clip !== null ? `Clipboard · ${clip.length.toLocaleString()} chars` : "With clipboard"}
          </button>
          <span className="grow" />
          <span className="quick-status">
            <span className={`status-dot ${st?.state || "empty"}`} />
            {ready ? st!.model : st?.state === "loading" ? `Loading ${st.model || ""}…` : "No model loaded"}
          </span>
          <span className="muted small">
            <kbd>Enter</kbd> ask · <kbd>Ctrl+Enter</kbd> in Pelica · <kbd>Esc</kbd> {busy ? "stop" : "close"}
          </span>
        </div>
      </div>
    </div>
  );
}
