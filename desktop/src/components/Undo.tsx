// Under a reply that changed files: undo those changes, or put them back.
// Files changed since aren't touched without asking.
import { useEffect, useState } from "react";
import { bridge, type RestoreResult } from "../lib/bridge";
import { app, toast, uid, updateChat } from "../state";

const count = (n: number) => `${n} ${n === 1 ? "file" : "files"}`;
const names = (l: string[]) => (l.length > 3 ? `${l.slice(0, 3).join(", ")} and ${l.length - 3} more` : l.join(", "));

export function UndoChanges({ turns, commands }: { turns: string[]; commands: boolean }) {
  const [info, setInfo] = useState<{ files: string[]; cant: string[]; undone: boolean } | null>(null);
  const [clash, setClash] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const key = turns.join();
  const refresh = () => bridge?.restore.info(turns).then(setInfo).catch(() => {});
  useEffect(() => void refresh(), [key]);
  if (!bridge || !info || !(info.files.length + info.cant.length)) return null;
  const putBack = info.undone;
  if (!info.files.length) return <div className="undo-row muted small">{names(info.cant)} too big to undo</div>;

  async function act(force = false) {
    if (app.get().running) return toast("Wait for the reply to finish, or stop it first.");
    setBusy(true);
    try {
      const r: RestoreResult = await (putBack ? bridge!.restore.redo(turns, force) : bridge!.restore.undo(turns, force));
      if (r.changed.length) return setClash(r.changed);
      setClash(null);
      if (r.done.length) {
        const text = `${putBack ? "Put back" : "Undid"} Pelica's changes to ${count(r.done.length)}`;
        const note = `[The user ${putBack ? "put back" : "undid"} your changes to: ${r.done.join(", ")}. Read a file again before relying on what you remember of it.]\n\n`;
        // (undo, put back, undo: one line that says where it ended up)
        updateChat((c) => {
          const end = c.messages[c.messages.length - 1];
          const keep = end?.role === "event" && end.forModel ? c.messages.slice(0, -1) : c.messages;
          return { ...c, messages: [...keep, { id: uid(), role: "event", kind: "info", text, forModel: note, ts: Date.now() }] };
        });
      }
      if (r.failed.length) toast(`Couldn't restore ${r.failed[0]}`);
      await refresh();
    } catch (e: any) {
      toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    } finally {
      setBusy(false);
    }
  }

  if (clash)
    return (
      <div className="undo-row clash">
        <span>{names(clash)} changed after this reply. Your newer changes would be replaced, though Put back brings them back.</span>
        <button className="btn small" disabled={busy} onClick={() => act(true)}>
          {putBack ? "Put back anyway" : "Undo anyway"}
        </button>
        <button className="btn small" onClick={() => setClash(null)}>
          Cancel
        </button>
      </div>
    );
  return (
    <div className="undo-row">
      <span className="muted small" title={info.files.join("\n")}>
        {putBack ? `Undone: ${count(info.files.length)} back as they were` : `Changed ${count(info.files.length)}`}
        {commands ? ". Commands it ran aren't undone" : ""}
        {info.cant.length ? `. ${names(info.cant)} too big to undo` : ""}
      </span>
      <button className="btn small" disabled={busy} onClick={() => act()}>
        {busy ? "Working…" : putBack ? "Put back" : "Undo these changes"}
      </button>
    </div>
  );
}
