// Voice mode: talk with Pelica hands-free. It listens with Windows' speech
// recognizer (the mic button's), sends what you said when you pause, reads
// the reply aloud a sentence at a time as it's written, then listens again.
// Space (or Interrupt) stops it talking and listens; Esc (or End) leaves.
import { useEffect } from "react";
import { send, stop as stopReply } from "../agent/agent";
import { bridge } from "../lib/bridge";
import { app, toast, useApp } from "../state";
import { Icon, Orbs } from "./Icons";

type Phase = "starting" | "listening" | "thinking" | "speaking";

let offDictate: (() => void) | null = null;
let ticker: ReturnType<typeof setInterval> | null = null;
/** what has been handed to the speech synthesizer of the reply being written */
let spoken = "";
let queued = 0;

const phase = (p: Phase, heard?: string) => {
  const v = app.get().voice;
  if (v) app.set({ voice: { phase: p, heard: heard ?? v.heard } });
};

/** A reply as it should sound: code, links and Markdown marks left out. An
 *  unfinished code block at the end isn't read until it's finished. */
export function speakable(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, " (code) ")
    .replace(/```[\s\S]*$/, "")
    .replace(/<[^>]+>/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\[\d+(?:,\s*\d+)*\]/g, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\|.*\|\s*$/gm, "")
    .replace(/[*_#>`~]/g, "");
}

function say(text: string) {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t || !app.get().voice) return;
  const u = new SpeechSynthesisUtterance(t);
  queued++;
  u.onend = u.onerror = () => {
    queued = Math.max(0, queued - 1);
    if (!queued && !app.get().running && app.get().voice?.phase === "speaking") listen();
  };
  phase("speaking");
  speechSynthesis.speak(u);
}

/** Reads the finished sentences of the reply being written that aren't read yet. */
function follow(final = false) {
  const st = app.get();
  const live = st.draft && st.draft.chatId === st.chat?.id ? st.draft.content : "";
  let text = speakable(live);
  if (final) {
    const last = [...(st.chat?.messages || [])].reverse().find((m) => m.role === "assistant");
    text = speakable(last && last.role === "assistant" ? last.content : "");
  }
  // (a new step of the reply, after a tool call: start over)
  if (!text.startsWith(spoken)) spoken = "";
  const rest = text.slice(spoken.length);
  const end = final ? rest.length : Math.max(rest.search(/[.!?:;]\s(?=[^.!?:;]*$)/) + 1, rest.lastIndexOf("\n\n"));
  if (end > 0) {
    say(rest.slice(0, end));
    spoken += rest.slice(0, end);
  }
}

async function heard(text: string) {
  const t = text.trim();
  if (!t || app.get().voice?.phase !== "listening") return;
  bridge?.media.dictateStop();
  phase("thinking", t);
  spoken = "";
  ticker = setInterval(() => follow(), 250);
  try {
    await send(t);
  } finally {
    if (ticker) clearInterval(ticker);
    ticker = null;
  }
  if (!app.get().voice) return;
  follow(true);
  if (!queued) listen();
}

// (tests drive a turn without a microphone: window.__pelica.voiceHeard("…"))
if (typeof window !== "undefined") (window as any).__voiceHeard = heard;

function listen() {
  if (!app.get().voice || !bridge) return;
  phase("starting");
  bridge.media.dictateStart();
}

export function startVoice() {
  const st = app.get();
  if (!bridge) return;
  if (st.voice) return;
  if (st.status?.state !== "ready") return toast(st.status?.state === "loading" ? "The model is still loading." : "Load a model first (model menu under the message box).");
  if (st.running) return toast("Wait for the reply to finish, or stop it first.");
  app.set({ voice: { phase: "starting", heard: "" } });
  offDictate = bridge.media.onDictate((e) => {
    const v = app.get().voice;
    if (!v) return;
    if (e.kind === "ready") phase("listening");
    else if (e.kind === "text") heard(e.text);
    else if (e.kind === "error") {
      toast(e.error);
      endVoice();
    }
  });
  listen();
}

/** Stops it talking (and the reply being written) and listens again. */
export function interruptVoice() {
  if (!app.get().voice) return;
  speechSynthesis.cancel();
  queued = 0;
  if (app.get().running) stopReply();
  if (app.get().voice?.phase !== "listening") listen();
}

export function endVoice() {
  if (!app.get().voice) return;
  app.set({ voice: null });
  offDictate?.();
  offDictate = null;
  if (ticker) clearInterval(ticker);
  ticker = null;
  speechSynthesis.cancel();
  queued = 0;
  bridge?.media.dictateStop();
}

const WORDS: Record<Phase, string> = {
  starting: "Starting the microphone…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

/** The bar over the message box while voice mode is on. */
export function VoiceBar() {
  const v = useApp((s) => s.voice);
  useEffect(() => {
    if (!v) return;
    const f = (e: KeyboardEvent) => {
      if (e.key === "Escape") endVoice();
      else if (e.key === " " && !(e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement)) {
        e.preventDefault();
        interruptVoice();
      }
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [!!v]);
  if (!v) return null;
  return (
    <div className={`voice-bar ${v.phase}`} role="status" aria-live="polite">
      <div className="voice-orb">
        <Orbs live={v.phase === "thinking" || v.phase === "speaking"} />
        {v.phase === "listening" && <span className="voice-ring" />}
      </div>
      <div className="voice-words">
        <b>{WORDS[v.phase]}</b>
        <span className="muted small">{v.heard ? `“${v.heard}”` : v.phase === "listening" ? "Say something, then pause." : ""}</span>
      </div>
      {(v.phase === "thinking" || v.phase === "speaking") && (
        <button className="btn small" onClick={interruptVoice} title="Stop talking and listen (Space)">
          Interrupt
        </button>
      )}
      <button className="ghost-btn" onClick={endVoice} title="End voice mode (Esc)">
        <Icon name="x" size={15} />
      </button>
    </div>
  );
}

/** The composer's button that starts voice mode. */
export function VoiceButton() {
  const on = useApp((s) => !!s.voice);
  if (!bridge) return null;
  return (
    <button className={`chip icon-only voice-btn ${on ? "on" : ""}`} onClick={() => (on ? endVoice() : startVoice())} title={on ? "End voice mode" : "Voice mode: talk with Pelica, and hear its answers"}>
      <Icon name="voice" size={15} />
    </button>
  );
}
