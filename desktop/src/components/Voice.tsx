// Voice mode: talk with Pelica hands-free. It listens with Windows' speech
// recognizer (the mic button's), sends what you said when you pause, reads
// the reply aloud a sentence at a time as it's written, then listens again.
// It opens full-window (the words as they're read, the voice to use) and
// folds down to a bar over the message box. Space (or Interrupt) stops it
// talking and listens; Esc leaves. With "Talk over Pelica" on, the microphone
// stays open while it talks, so just talking stops it.
import { useEffect, useState } from "react";
import { send, stop as stopReply } from "../agent/agent";
import { bridge } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { Icon, Orbs } from "./Icons";
import { WinBtns } from "./Sidebar";
import { Select } from "./Select";

type Phase = "starting" | "listening" | "thinking" | "speaking";

let offDictate: (() => void) | null = null;
let ticker: ReturnType<typeof setInterval> | null = null;
/** what has been handed to the speech synthesizer of the reply being written */
let spoken = "";
let queued = 0;
/** the microphone is open (a recognizer is running) */
let micOn = false;
/** which turn is live: a turn cut short leaves its reply unread */
let turn = 0;

const phase = (p: Phase, heard?: string) => {
  const v = app.get().voice;
  if (v) app.set({ voice: { ...v, phase: p, heard: heard ?? v.heard } });
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

/** The chosen voice and speed, for anything Pelica reads aloud. */
export function tuneVoice(u: SpeechSynthesisUtterance) {
  const { voiceName, voiceRate } = app.get().settings;
  const v = voiceName ? speechSynthesis.getVoices().find((x) => x.name === voiceName) : undefined;
  if (v) u.voice = v;
  u.rate = voiceRate;
}

/** The voices installed on this PC (they can arrive a moment after start). */
export function useVoices() {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(() => speechSynthesis.getVoices());
  useEffect(() => {
    const f = () => setVoices(speechSynthesis.getVoices());
    speechSynthesis.addEventListener("voiceschanged", f);
    f();
    return () => speechSynthesis.removeEventListener("voiceschanged", f);
  }, []);
  return voices;
}

/** Plays a line in the voice just chosen (not during a voice-mode reply). */
export function sampleVoice() {
  if (app.get().voice?.phase === "speaking") return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance("This is how I'll sound.");
  tuneVoice(u);
  speechSynthesis.speak(u);
}

function say(text: string) {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t || !app.get().voice) return;
  const u = new SpeechSynthesisUtterance(t);
  tuneVoice(u);
  const mine = turn;
  queued++;
  u.onstart = () => {
    const v = app.get().voice;
    if (v && mine === turn) app.set({ voice: { ...v, line: t } });
  };
  u.onend = u.onerror = () => {
    if (mine !== turn) return;
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

/** Stops what's being read and written, without touching the microphone. */
function cut() {
  turn++;
  speechSynthesis.cancel();
  queued = 0;
  if (ticker) clearInterval(ticker);
  ticker = null;
  if (app.get().running) stopReply();
}

async function heard(text: string) {
  const t = text.trim();
  const v = app.get().voice;
  if (!t || !v) return;
  const open = app.get().settings.voiceTalkOver;
  if (v.phase !== "listening" && !(open && (v.phase === "thinking" || v.phase === "speaking"))) return;
  if (v.phase !== "listening") {
    if (ownWords(t)) return;
    cut();
    // (the reply being stopped has to finish stopping before the next starts)
    for (let i = 0; i < 60 && app.get().running; i++) await new Promise((r) => setTimeout(r, 50));
  }
  if (!open) {
    bridge?.media.dictateStop();
    micOn = false;
  }
  const mine = ++turn;
  phase("thinking", t);
  const now = app.get().voice;
  if (now) app.set({ voice: { ...now, line: "" } });
  spoken = "";
  ticker = setInterval(() => follow(), 250);
  try {
    await send(t);
  } finally {
    if (mine === turn) {
      if (ticker) clearInterval(ticker);
      ticker = null;
    }
  }
  if (!app.get().voice || mine !== turn) return;
  follow(true);
  if (!queued) listen();
}

// (tests drive a turn without a microphone)
if (typeof window !== "undefined") {
  (window as any).__voiceHeard = heard;
  (window as any).__voiceSaying = saying;
  (window as any).__voiceOwn = ownWords;
}

/** Is this the speakers being picked up: mostly words Pelica just said? */
function ownWords(text: string) {
  const said = new Set(
    spoken
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s']/gu, " ")
      .split(/\s+/)
      .slice(-80),
  );
  const words = text.toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, " ").split(/\s+/).filter(Boolean);
  return !!words.length && words.filter((w) => said.has(w)).length / words.length >= 0.6;
}

/** Words on the way in while Pelica talks: stop talking at once, the rest of
 *  the sentence arrives as "text" when it's finished. */
function saying(text: string) {
  const v = app.get().voice;
  if (!v || !app.get().settings.voiceTalkOver || v.phase !== "speaking") return;
  if (text.trim().split(/\s+/).length < 2 || ownWords(text)) return;
  cut();
  phase("listening", "");
}

function level(n: number) {
  document.querySelector<HTMLElement>(".voice-stage")?.style.setProperty("--lvl", String(Math.min(1, n / 60)));
}

function listen() {
  if (!app.get().voice || !bridge) return;
  if (micOn) return phase("listening");
  phase("starting");
  bridge.media.dictateStart();
}

export function startVoice() {
  const st = app.get();
  if (!bridge) return;
  if (st.voice) return;
  if (st.status?.state !== "ready") return toast(st.status?.state === "loading" ? "The model is still loading." : "Load a model first (model menu under the message box).");
  if (st.running) return toast("Wait for the reply to finish, or stop it first.");
  app.set({ voice: { phase: "starting", heard: "", line: "", full: true } });
  offDictate = bridge.media.onDictate((e) => {
    const v = app.get().voice;
    if (!v) return;
    if (e.kind === "ready") {
      micOn = true;
      if (v.phase === "starting") phase("listening");
    } else if (e.kind === "text") heard(e.text);
    else if (e.kind === "saying") saying(e.text);
    else if (e.kind === "level") level(e.level);
    else if (e.kind === "end") micOn = false;
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
  cut();
  if (app.get().voice?.phase !== "listening") listen();
}

export function endVoice() {
  if (!app.get().voice) return;
  app.set({ voice: null });
  offDictate?.();
  offDictate = null;
  turn++;
  if (ticker) clearInterval(ticker);
  ticker = null;
  speechSynthesis.cancel();
  queued = 0;
  micOn = false;
  bridge?.media.dictateStop();
}

const fold = (full: boolean) => {
  const v = app.get().voice;
  if (v) app.set({ voice: { ...v, full } });
};

const WORDS: Record<Phase, string> = {
  starting: "Starting the microphone…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

function useVoiceKeys(on: boolean) {
  useEffect(() => {
    if (!on) return;
    const f = (e: KeyboardEvent) => {
      if (e.key === "Escape") endVoice();
      else if (e.key === " " && !(e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement || (e.target as HTMLElement).closest?.(".select, .menu-at"))) {
        e.preventDefault();
        interruptVoice();
      }
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [on]);
}

const pickVoice = (patch: { voiceName?: string; voiceRate?: number }) => {
  setSettings(patch);
  setTimeout(sampleVoice, 0);
};

/** Which voice, from the ones installed; a choice plays a sample. */
export function VoiceSelect() {
  const voices = useVoices();
  const name = useApp((s) => s.settings.voiceName);
  return (
    <Select label="Voice" value={name} onChange={(voiceName) => pickVoice({ voiceName })} options={[{ value: "", label: "Windows' default" }, ...voices.map((v) => ({ value: v.name, label: v.name.replace(/^Microsoft /, "").replace(/ - .*$/, "") }))]} />
  );
}

export function SpeedSelect() {
  const rate = useApp((s) => s.settings.voiceRate);
  return (
    <Select
      label="Speed"
      value={rate}
      onChange={(voiceRate) => pickVoice({ voiceRate })}
      options={[
        { value: 0.85, label: "Slower" },
        { value: 1, label: "Normal" },
        { value: 1.2, label: "Faster" },
        { value: 1.4, label: "Fastest" },
      ]}
    />
  );
}

/** The voice, speed and talk-over choices along the full view's foot. */
function VoiceChoices() {
  const over = useApp((s) => s.settings.voiceTalkOver);
  return (
    <div className="voice-choices">
      <label className="voice-choice">
        <span>Voice</span>
        <VoiceSelect />
      </label>
      <label className="voice-choice">
        <span>Speed</span>
        <SpeedSelect />
      </label>
      <label className="voice-choice voice-over" title="The microphone stays open while Pelica talks, so you can just start talking. Use headphones, or it can hear itself.">
        <span>Talk over Pelica (use headphones)</span>
        <button role="switch" aria-checked={over} className={`toggle ${over ? "on" : ""}`} onClick={() => setSettings({ voiceTalkOver: !over })} />
      </label>
    </div>
  );
}

/** Voice mode with the whole window to itself. */
export function VoiceFull() {
  const v = useApp((s) => s.voice);
  const full = !!v?.full;
  useVoiceKeys(full);
  if (!v || !full) return null;
  const busy = v.phase === "thinking" || v.phase === "speaking";
  return (
    <div className={`voice-full ${v.phase}`} role="dialog" aria-label="Voice mode">
      <div className="voice-full-top">
        <button className="ghost-btn nodrag" title="Show the chat" onClick={() => fold(false)}>
          <Icon name="chevronDown" size={17} />
        </button>
        <WinBtns />
      </div>
      <div className="voice-stage">
        <div className="voice-disc">
          <i />
          <i />
          <i />
          <b />
        </div>
        <div className="voice-state" role="status" aria-live="polite" key={v.phase}>
          {WORDS[v.phase]}
        </div>
        <div className="voice-said">
          {v.phase === "speaking" && v.line ? (
            <p className="voice-line" key={v.line}>
              {v.line}
            </p>
          ) : v.heard ? (
            <p className="voice-heard" key={v.heard}>“{v.heard}”</p>
          ) : (
            <p className="voice-hint">{v.phase === "listening" ? "Say something, then pause." : ""}</p>
          )}
        </div>
      </div>
      <div className="voice-full-foot">
        <VoiceChoices />
        <div className="voice-full-btns">
          <button className="btn" disabled={!busy} onClick={interruptVoice} title="Stop talking and listen (Space)">
            Interrupt
          </button>
          <button className="btn voice-end" onClick={endVoice} title="End voice mode (Esc)">
            End
          </button>
        </div>
      </div>
    </div>
  );
}

/** The bar over the message box, with voice mode folded down. */
export function VoiceBar() {
  const v = useApp((s) => s.voice);
  useVoiceKeys(!!v && !v.full);
  if (!v || v.full) return null;
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
      <button className="ghost-btn" onClick={() => fold(true)} title="Full view">
        <Icon name="maximize" size={15} />
      </button>
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
