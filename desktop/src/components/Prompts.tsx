// Saved prompts: text you send often, kept under a name. Typed as /name in the
// message box, picked from the + menu or the palette, saved from one of your
// messages (its bookmark button) or from the box, managed in Settings →
// General.
import { useEffect, useState } from "react";
import { app, setSettings, toast, useApp, type SavedPrompt } from "../state";
import { Icon } from "./Icons";

/** A prompt's name as typed after "/" (lower case, dashes). */
export const slug = (name: string) =>
  name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

/** Puts a prompt in the message box (after what's already typed). */
export function insertPrompt(text: string) {
  if (app.get().view !== "chat") app.set({ view: "chat" });
  setTimeout(() => window.dispatchEvent(new CustomEvent("pelica:insert", { detail: text })), 30);
}

/** Opens the editor on a new prompt (with this text) or a saved one. */
export function editPrompt(p: { index: number | null; name?: string; text?: string }) {
  app.set({ promptEdit: { index: p.index, name: p.name || "", text: p.text || "" } });
}

export function PromptEditor() {
  const ed = useApp((s) => s.promptEdit);
  const prompts = useApp((s) => s.settings.prompts);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  useEffect(() => {
    if (!ed) return;
    setName(ed.name);
    setText(ed.text);
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ promptEdit: null });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [ed]);
  if (!ed) return null;
  const s = slug(name);
  const clash = s && prompts.some((p, i) => i !== ed.index && slug(p.name) === s);
  const ok = !!s && !!text.trim() && !clash;
  const save = () => {
    if (!ok) return;
    const p: SavedPrompt = { name: name.trim(), text: text.trim() };
    const list = ed.index === null ? [...prompts, p] : prompts.map((x, i) => (i === ed.index ? p : x));
    setSettings({ prompts: list });
    app.set({ promptEdit: null });
    toast(`Saved: type /${s} to use it`);
  };
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ promptEdit: null })}>
      <div className="modal prompt-edit" role="dialog" aria-label="Saved prompt">
        <div className="whatsnew-head">
          <h2>{ed.index === null ? "Save a prompt" : "Edit prompt"}</h2>
          <button className="ghost-btn" title="Close" onClick={() => app.set({ promptEdit: null })}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="prompt-edit-body">
          <label className="field">
            <span>Name</span>
            <input className="text-input" autoFocus value={name} placeholder="e.g. Review my code" onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && save()} />
            <small className={clash ? "err-text" : "muted"}>{clash ? `/${s} is taken by another prompt.` : s ? `Type /${s} in the message box to use it.` : " "}</small>
          </label>
          <label className="field">
            <span>Prompt</span>
            <textarea className="text-input" rows={8} value={text} placeholder="The text that goes in the message box" onChange={(e) => setText(e.target.value)} />
          </label>
        </div>
        <div className="whatsnew-foot prompt-edit-foot">
          {ed.index !== null && (
            <button
              className="btn ghost danger-text"
              onClick={() => {
                const gone = prompts[ed.index!];
                setSettings({ prompts: prompts.filter((_, i) => i !== ed.index) });
                app.set({ promptEdit: null });
                toast(`Deleted “${gone.name}”`, { label: "Undo", run: () => setSettings({ prompts: [...app.get().settings.prompts, gone] }) });
              }}
            >
              Delete
            </button>
          )}
          <span className="grow" />
          <button className="btn ghost" onClick={() => app.set({ promptEdit: null })}>
            Cancel
          </button>
          <button className="btn" disabled={!ok} onClick={save}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

/** Settings → General: the saved prompts, with edit and new. */
export function PromptList() {
  const prompts = useApp((s) => s.settings.prompts);
  return (
    <div className="prompt-list">
      {prompts.map((p, i) => (
        <div key={i} className="prompt-item">
          <span className="mono prompt-slug">/{slug(p.name)}</span>
          <span className="prompt-text" title={p.text}>
            {p.text.split("\n")[0]}
          </span>
          <button className="ghost-btn" title="Edit" onClick={() => editPrompt({ index: i, name: p.name, text: p.text })}>
            <Icon name="edit" size={13} />
          </button>
        </div>
      ))}
      {!prompts.length && <div className="muted small">None yet. Save one from the bookmark under any of your messages, or here.</div>}
      <button className="btn ghost small-btn" onClick={() => editPrompt({ index: null })}>
        <Icon name="plus" size={13} /> New prompt
      </button>
    </div>
  );
}
