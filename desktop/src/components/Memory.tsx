// Settings → Memory: what Pelica remembers about the user across chats. The
// model adds to it with the remember tool (tools.ts) and every chat's system
// prompt lists it (agent.ts); here the user reads, edits, adds and removes.
import { useState } from "react";
import { setSettings, uid, useApp, type Memory } from "../state";
import { Icon } from "./Icons";

export function MemoryList() {
  const list = useApp((s) => s.settings.memories);
  const [edit, setEdit] = useState<{ id: string; text: string } | null>(null);
  const [add, setAdd] = useState("");
  const save = (m: Memory[]) => setSettings({ memories: m });
  const commit = () => {
    if (!edit) return;
    const t = edit.text.trim().replace(/\s+/g, " ");
    save(t ? list.map((m) => (m.id === edit.id ? { ...m, text: t } : m)) : list.filter((m) => m.id !== edit.id));
    setEdit(null);
  };
  const addOne = () => {
    const t = add.trim().replace(/\s+/g, " ");
    if (!t) return;
    if (!list.some((m) => m.text.toLowerCase() === t.toLowerCase())) save([...list, { id: uid(), text: t, created: Date.now(), by: "you" }]);
    setAdd("");
  };
  return (
    <div className="prompt-list memory-list">
      {list.map((m) =>
        edit?.id === m.id ? (
          <div key={m.id} className="prompt-item">
            <input className="text-input memory-edit" autoFocus value={edit.text} onChange={(e) => setEdit({ id: m.id, text: e.target.value })} onBlur={commit} onKeyDown={(e) => (e.key === "Enter" ? commit() : e.key === "Escape" && setEdit(null))} />
          </div>
        ) : (
          <div key={m.id} className="prompt-item">
            <span className="prompt-text memory-text" title={`Saved by ${m.by === "you" ? "you" : "Pelica"}, ${new Date(m.created).toLocaleDateString()}`} onDoubleClick={() => setEdit({ id: m.id, text: m.text })}>
              {m.text}
            </span>
            <span className="muted small memory-by">{m.by === "you" ? "You" : "Pelica"}</span>
            <button className="ghost-btn" title="Edit" onClick={() => setEdit({ id: m.id, text: m.text })}>
              <Icon name="edit" size={13} />
            </button>
            <button className="ghost-btn" title="Forget" onClick={() => save(list.filter((x) => x.id !== m.id))}>
              <Icon name="x" size={13} />
            </button>
          </div>
        ),
      )}
      {!list.length && (
        <div className="empty-state">
          <span className="empty-icon">
            <Icon name="brain" size={20} />
          </span>
          <b>Nothing remembered yet</b>
          <span className="muted small">Say “remember that…” in any chat, or add something here. Pelica sees these in every chat.</span>
        </div>
      )}
      <div className="memory-add">
        <input className="text-input" value={add} placeholder="Add something to remember, e.g. “I live in Lisbon”" onChange={(e) => setAdd(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addOne()} />
        <button className="btn small" disabled={!add.trim()} onClick={addOne}>
          Add
        </button>
      </div>
    </div>
  );
}
