// Scheduled tasks: prompts that run on their own (once, hourly, daily, on
// weekdays, weekly), each run in a chat of its own. See agent/tasks.ts.
import { useEffect, useState } from "react";
import { blankTask, DAYS, describe, dueAt, removeTask, runTask, saveTask, type Repeat, type Task } from "../agent/tasks";
import { bridge } from "../lib/bridge";
import { app, openChat, useApp } from "../state";
import { Icon } from "./Icons";

const when = (t: number) => {
  const d = new Date(t);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : d.toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
};

function Editor({ task, done }: { task: Task; done: () => void }) {
  const [t, setT] = useState(task);
  const set = (p: Partial<Task>) => setT({ ...t, ...p });
  const ok = t.prompt.trim().length > 0 && (t.repeat !== "once" || !!t.date);
  return (
    <div className="task-edit">
      <label>
        Name
        <input className="text-input wide" value={t.name} placeholder="Morning news summary" onChange={(e) => set({ name: e.target.value })} />
      </label>
      <label>
        What should Pelica do?
        <textarea className="text-input wide" rows={5} value={t.prompt} placeholder="Search the web for today's top tech news and write a short summary with links." onChange={(e) => set({ prompt: e.target.value })} />
      </label>
      <div className="task-when">
        <select className="text-input" value={t.repeat} onChange={(e) => set({ repeat: e.target.value as Repeat })}>
          <option value="once">Once</option>
          <option value="hourly">Every hour</option>
          <option value="daily">Every day</option>
          <option value="weekdays">Weekdays</option>
          <option value="weekly">Every week</option>
        </select>
        {t.repeat === "once" && <input className="text-input" type="date" value={t.date || ""} onChange={(e) => set({ date: e.target.value })} />}
        {t.repeat === "weekly" && (
          <select className="text-input" value={t.day} onChange={(e) => set({ day: Number(e.target.value) })}>
            {DAYS.map((d, i) => (
              <option key={d} value={i}>
                {d}
              </option>
            ))}
          </select>
        )}
        <input className="text-input" type="time" value={t.time} onChange={(e) => set({ time: e.target.value || "09:00" })} />
      </div>
      <div className="task-folder">
        <Icon name="folder" size={14} />
        <span className={t.folder ? "" : "muted"}>{t.folder || "No folder (chat only)"}</span>
        {bridge && (
          <button
            className="btn ghost small"
            onClick={async () => {
              const f = await bridge!.folder.pick();
              if (f) set({ folder: f });
            }}
          >
            Choose…
          </button>
        )}
        {t.folder && (
          <button className="btn ghost small" onClick={() => set({ folder: null })}>
            None
          </button>
        )}
      </div>
      <div className="task-actions">
        <button className="btn ghost" onClick={done}>
          Cancel
        </button>
        <button
          className="btn primary"
          disabled={!ok}
          onClick={() => {
            // (a changed schedule counts from now, not from the last run)
            const moved = t.repeat !== task.repeat || t.time !== task.time || t.day !== task.day || t.date !== task.date;
            saveTask({ ...t, name: t.name.trim() || t.prompt.trim().slice(0, 40), created: moved ? Date.now() : t.created, last: moved ? undefined : t.last, enabled: moved ? true : t.enabled });
            done();
          }}
        >
          Save
        </button>
      </div>
    </div>
  );
}

export function TasksModal() {
  const open = useApp((s) => s.tasksOpen);
  const list = useApp((s) => (s.settings.tasks || []) as Task[]);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const [edit, setEdit] = useState<Task | null>(null);
  const [, tick] = useState(0);
  const close = () => (setEdit(null), app.set({ tasksOpen: false }));
  useEffect(() => {
    if (!open) return;
    const f = (e: KeyboardEvent) => e.key === "Escape" && (edit ? setEdit(null) : close());
    window.addEventListener("keydown", f);
    const t = setInterval(() => tick((x) => x + 1), 30_000);
    return () => (window.removeEventListener("keydown", f), clearInterval(t));
  }, [open, edit]);
  if (!open) return null;
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal tasks" role="dialog" aria-label="Scheduled tasks">
        <div className="whatsnew-head">
          <h2>{edit ? (list.some((x) => x.id === edit.id) ? "Edit task" : "New task") : "Scheduled tasks"}</h2>
          <button className="ghost-btn" title="Close" onClick={close}>
            <Icon name="x" size={16} />
          </button>
        </div>
        {edit ? (
          <Editor task={edit} done={() => setEdit(null)} />
        ) : (
          <div className="tasks-body">
            <p className="set-desc">Pelica runs each task on the loaded model in a chat of its own and lets you know when it's done. It has to be running (the tray is enough); a time it missed runs when it next starts. To run something once in the background right now, type /background and the prompt.</p>
            {list.length === 0 && <div className="muted tasks-empty">No tasks yet.</div>}
            {list.map((t) => {
              const next = t.enabled ? dueAt(t) : null;
              return (
                <div key={t.id} className={`task-row${t.enabled ? "" : " off"}`}>
                  <div className="task-main" onClick={() => setEdit(t)}>
                    <div className="task-name">{t.name || "Untitled task"}</div>
                    <div className="muted small">
                      {describe(t)}
                      {next ? ` · next ${when(next)}` : t.repeat === "once" && t.last ? " · done" : !t.enabled ? " · paused" : ""}
                    </div>
                  </div>
                  {t.lastChat && (
                    <button className="btn ghost small" title={t.last ? `Ran ${when(t.last)}` : ""} onClick={() => (close(), app.set({ view: "chat" }), openChat(t.lastChat!))}>
                      Last result
                    </button>
                  )}
                  <button className="btn ghost small" onClick={() => runTask(t, true)}>
                    Run now
                  </button>
                  <button className="btn ghost small" onClick={() => saveTask({ ...t, enabled: !t.enabled, ...(t.enabled ? {} : { created: Date.now(), last: undefined }) })}>
                    {t.enabled ? "Pause" : "Resume"}
                  </button>
                  <button className="ghost-btn" title="Delete" onClick={() => removeTask(t.id)}>
                    <Icon name="trash" size={14} />
                  </button>
                </div>
              );
            })}
            <div className="task-actions">
              <button className="btn primary" onClick={() => setEdit(blankTask(folder))}>
                New task
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
