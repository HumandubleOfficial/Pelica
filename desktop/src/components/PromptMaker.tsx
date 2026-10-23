// The prompt maker: you say roughly what you want, the loaded model asks a
// few questions about it, and turns your answers into a tuned prompt you can
// edit, send, copy, or keep as your instructions for every chat.
import { useEffect, useRef, useState } from "react";
import { engine } from "../lib/engine";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

interface Question {
  q: string;
  options: string[];
}

type Step = "goal" | "asking" | "answer" | "writing" | "done";

const FALLBACK: Question[] = [
  { q: "Who is the answer for?", options: ["Just me", "Beginners", "Experts", "Customers"] },
  { q: "What should the result look like?", options: ["Short answer", "Step-by-step list", "Full document", "Code"] },
  { q: "What tone fits?", options: ["Neutral", "Friendly", "Formal", "Playful"] },
  { q: "Anything it must include or avoid?", options: [] },
];

const ASK = `You help people write better prompts for AI assistants. The user gives a rough goal. Ask the 3 to 5 questions whose answers would most improve the prompt: missing context, audience, output format, length, constraints, examples, success criteria. Skip what the goal already answers. Each question gets 2 to 4 short likely answers (or none when it needs free text).
Reply with JSON only, no other text: {"questions":[{"q":"…","options":["…","…"]}]}`;

const WRITE = `You write prompts for AI assistants. From the user's goal and their answers, write one clear, complete prompt they can paste into an assistant. Use what makes prompts work: say the task and its purpose up front, give the context and audience, state the output format and length, list constraints, say what a great result looks like, and ask the assistant to ask for anything essential that's missing. Use short sections or bullets when it helps. Write it in the second person to the assistant. Reply with the prompt only: no preface, no quotes around it, no explanation.`;

/** The first {...} in a reply, parsed (models sometimes wrap JSON in prose or fences). */
function parseQuestions(s: string): Question[] | null {
  const a = s.indexOf("{");
  const b = s.lastIndexOf("}");
  if (a < 0 || b <= a) return null;
  try {
    const v = JSON.parse(s.slice(a, b + 1));
    const list = (v.questions || []).filter((x: any) => x && typeof x.q === "string" && x.q.trim()).slice(0, 5);
    return list.length ? list.map((x: any) => ({ q: x.q.trim(), options: (Array.isArray(x.options) ? x.options : []).filter((o: any) => typeof o === "string" && o.trim()).slice(0, 4) })) : null;
  } catch {
    return null;
  }
}

function quickEffort(): string | undefined {
  const e = app.get().status?.efforts;
  return e?.includes("off") ? "off" : e?.includes("low") ? "low" : undefined;
}

export function PromptMaker() {
  const open = useApp((s) => s.promptMaker);
  const st = useApp((s) => s.status);
  const running = useApp((s) => s.running);
  const [step, setStep] = useState<Step>("goal");
  const [goal, setGoal] = useState("");
  const [qs, setQs] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<string[]>([]);
  const [out, setOut] = useState("");
  const [tweak, setTweak] = useState("");
  const [err, setErr] = useState("");
  const ctl = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!open) return;
    const f = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [open]);
  if (!open) return null;

  function close() {
    ctl.current?.abort();
    app.set({ promptMaker: false });
  }
  function reset() {
    ctl.current?.abort();
    setStep("goal");
    setQs([]);
    setAnswers([]);
    setOut("");
    setTweak("");
    setErr("");
  }

  async function complete(system: string, user: string, onText?: (s: string) => void, max = 900): Promise<string> {
    const c = new AbortController();
    ctl.current = c;
    let text = "";
    await engine.chat({ messages: [{ role: "system", content: system }, { role: "user", content: user }], tools: [], effort: quickEffort(), max_tokens: max }, (e) => {
      if (e.t === "text") {
        text += e.d;
        onText?.(text);
      }
    }, c.signal);
    return text;
  }

  const ready = st?.state === "ready" && !running;
  const why = st?.state !== "ready" ? "Load a model first: the prompt maker uses the model you have loaded." : running ? "Pelica is replying in a chat: the prompt maker can start when it's done." : "";

  async function ask() {
    if (!goal.trim() || !ready) return;
    setErr("");
    setStep("asking");
    try {
      const r = await complete(ASK, `Goal: ${goal.trim()}`);
      const got = parseQuestions(r) || FALLBACK;
      setQs(got);
      setAnswers(got.map(() => ""));
      setStep("answer");
    } catch (e: any) {
      if (ctl.current?.signal.aborted) return;
      setErr(String(e?.message || e));
      setStep("goal");
    }
  }

  async function write(extra = "") {
    if (!ready) return;
    setErr("");
    setStep("writing");
    const qa = qs.map((q, i) => `- ${q.q} ${answers[i]?.trim() || "(no answer: use your judgement)"}`).join("\n");
    const user = `Goal: ${goal.trim()}\n\nAnswers:\n${qa}${extra ? `\n\nThe prompt so far:\n${out}\n\nChange it like this: ${extra}` : ""}`;
    setOut("");
    try {
      const r = await complete(WRITE, user, setOut, 1600);
      setOut(r.trim().replace(/^```\w*\n?|\n?```$/g, ""));
      setStep("done");
      setTweak("");
    } catch (e: any) {
      if (ctl.current?.signal.aborted) return;
      setErr(String(e?.message || e));
      setStep(qs.length ? "answer" : "goal");
    }
  }

  function use() {
    window.dispatchEvent(new CustomEvent("pelica:compose", { detail: out }));
    close();
  }
  function keep() {
    const cur = app.get().settings.customInstructions.trim();
    setSettings({ customInstructions: cur ? `${cur}\n\n${out}` : out });
    toast("Added to your instructions for every chat (Settings → General).");
  }

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal prompt-maker">
        <div className="pm-head">
          <span className="pm-badge">
            <Icon name="wand" size={16} />
          </span>
          <div>
            <h2>Prompt maker</h2>
            <div className="muted small">{step === "goal" ? "Say roughly what you want. A few questions follow." : step === "answer" || step === "asking" ? "Answer what you can: skip the rest." : "Edit it, then use it."}</div>
          </div>
          <span className="grow" />
          <div className="pm-steps">
            {["Goal", "Questions", "Prompt"].map((s, i) => (
              <span key={s} className={(step === "goal" ? 0 : step === "asking" || step === "answer" ? 1 : 2) >= i ? "on" : ""}>
                {s}
              </span>
            ))}
          </div>
          <button className="ghost-btn" onClick={close} title="Close (Esc)">
            <Icon name="x" size={16} />
          </button>
        </div>

        {step === "goal" && (
          <div className="pm-body">
            <textarea
              className="text-input wide pm-goal"
              rows={4}
              autoFocus
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || !e.shiftKey) && (e.preventDefault(), ask())}
              placeholder="e.g. a prompt that reviews my Python code for bugs and explains them simply"
            />
            <div className="pm-examples">
              {["Write a cover letter for a job", "Explain a topic like I'm new to it", "Plan a small web app", "Summarize long documents"].map((x) => (
                <button key={x} className="pill-btn" onClick={() => setGoal(x)}>
                  {x}
                </button>
              ))}
            </div>
          </div>
        )}

        {step === "asking" && (
          <div className="pm-body pm-wait">
            <span className="spinner" /> {st?.model} is thinking of questions…
          </div>
        )}

        {step === "answer" && (
          <div className="pm-body pm-qs">
            {qs.map((q, i) => (
              <div key={i} className="pm-q">
                <div className="pm-q-text">
                  <span className="pm-num">{i + 1}</span>
                  {q.q}
                </div>
                {q.options.length > 0 && (
                  <div className="pm-opts">
                    {q.options.map((o) => (
                      <button key={o} className={`pill-btn ${answers[i] === o ? "sel" : ""}`} onClick={() => setAnswers((a) => a.map((x, j) => (j === i ? (x === o ? "" : o) : x)))}>
                        {o}
                      </button>
                    ))}
                  </div>
                )}
                <input className="text-input wide" value={q.options.includes(answers[i]) ? "" : answers[i]} placeholder={q.options.length ? "Or your own answer" : "Your answer"} onChange={(e) => setAnswers((a) => a.map((x, j) => (j === i ? e.target.value : x)))} />
              </div>
            ))}
          </div>
        )}

        {(step === "writing" || step === "done") && (
          <div className="pm-body">
            <textarea className="text-input wide pm-out" value={out} onChange={(e) => setOut(e.target.value)} readOnly={step === "writing"} spellCheck={false} />
            {step === "done" && (
              <div className="inline pm-tweak">
                <input className="text-input wide" value={tweak} onChange={(e) => setTweak(e.target.value)} onKeyDown={(e) => e.key === "Enter" && tweak.trim() && write(tweak.trim())} placeholder="Tune it: e.g. shorter, add an example, more formal…" />
                <button className="btn small" disabled={!tweak.trim() || !ready} onClick={() => write(tweak.trim())}>
                  <Icon name="sparkle" size={13} /> Tune
                </button>
              </div>
            )}
          </div>
        )}

        {(err || why) && <div className="err-text pm-err">{err || why}</div>}

        <div className="pm-foot">
          {step !== "goal" && (
            <button className="btn ghost small" onClick={reset}>
              Start over
            </button>
          )}
          <span className="grow" />
          {step === "goal" && (
            <button className="btn primary" disabled={!goal.trim() || !ready} onClick={ask}>
              Next <Icon name="chevron" size={13} />
            </button>
          )}
          {step === "answer" && (
            <>
              <button className="btn ghost small" onClick={() => write()} disabled={!ready}>
                Skip questions
              </button>
              <button className="btn primary" onClick={() => write()} disabled={!ready}>
                <Icon name="wand" size={14} /> Make my prompt
              </button>
            </>
          )}
          {(step === "asking" || step === "writing") && (
            <button className="btn ghost small" onClick={() => (ctl.current?.abort(), setStep(step === "asking" ? "goal" : "done"))}>
              Stop
            </button>
          )}
          {step === "done" && (
            <>
              <button className="btn ghost small" onClick={keep} title="Add it to Settings → General → Instructions for every chat">
                Save as instructions
              </button>
              <button className="btn small" onClick={() => (navigator.clipboard.writeText(out), toast("Prompt copied"))}>
                <Icon name="copy" size={13} /> Copy
              </button>
              <button className="btn primary" onClick={use}>
                <Icon name="send" size={13} /> Use in chat
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
