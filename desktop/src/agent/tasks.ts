// Background and scheduled tasks: a prompt that runs in a chat of its own
// while you do something else (/background), now or on a schedule (Scheduled
// tasks: once, hourly, daily, on weekdays, weekly). The answer is left in that
// chat, and a notification says when it's done. Tasks live in settings; a
// time missed while Pelica was closed runs when it next starts.
import { app, newChat, setSettings, toast, uid, type Chat } from "../state";
import { run } from "./agent";

export type Repeat = "once" | "hourly" | "daily" | "weekdays" | "weekly";

export interface Task {
  id: string;
  name: string;
  prompt: string;
  folder: string | null;
  repeat: Repeat;
  /** "HH:MM" (hourly: only the minutes count) */
  time: string;
  /** weekly: 0 Sunday … 6 Saturday */
  day: number;
  /** once: the day it runs, "YYYY-MM-DD" */
  date?: string;
  enabled: boolean;
  created: number;
  /** when it last ran, and the chat it ran in */
  last?: number;
  lastChat?: string;
}

export const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/** The first time at or after `from` the task is due (null: never again). */
export function nextRun(t: Task, from: number): number | null {
  const [hh, mm] = t.time.split(":").map((x) => Number(x) || 0);
  if (t.repeat === "once") {
    const [y, mo, d] = (t.date || "").split("-").map(Number);
    if (!y) return null;
    const at = new Date(y, mo - 1, d, hh, mm).getTime();
    return t.last ? null : at;
  }
  const c = new Date(from);
  c.setSeconds(0, 0);
  if (t.repeat === "hourly") {
    c.setMinutes(mm);
    if (c.getTime() < from) c.setHours(c.getHours() + 1);
    return c.getTime();
  }
  c.setHours(hh, mm);
  for (let i = 0; i < 8; i++) {
    const ok = c.getTime() >= from && (t.repeat === "daily" || (t.repeat === "weekdays" && c.getDay() >= 1 && c.getDay() <= 5) || (t.repeat === "weekly" && c.getDay() === t.day));
    if (ok) return c.getTime();
    c.setDate(c.getDate() + 1);
  }
  return null;
}

/** When the task is next due: after its last run (or since it was made). */
export function dueAt(t: Task): number | null {
  return nextRun(t, (t.last ?? t.created) + 1);
}

export function describe(t: Task): string {
  const at = (() => {
    const [hh, mm] = t.time.split(":").map((x) => Number(x) || 0);
    return new Date(2000, 0, 1, hh, mm).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  })();
  switch (t.repeat) {
    case "once":
      return `Once, ${t.date ? new Date(t.date + "T00:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }) : "?"} at ${at}`;
    case "hourly":
      return `Every hour at :${t.time.split(":")[1] || "00"}`;
    case "daily":
      return `Every day at ${at}`;
    case "weekdays":
      return `Weekdays at ${at}`;
    case "weekly":
      return `Every ${DAYS[t.day]} at ${at}`;
  }
}

const tasks = (): Task[] => (app.get().settings.tasks || []) as Task[];

export function saveTask(t: Task) {
  const list = tasks();
  setSettings({ tasks: list.some((x) => x.id === t.id) ? list.map((x) => (x.id === t.id ? t : x)) : [...list, t] });
}

export function removeTask(id: string) {
  setSettings({ tasks: tasks().filter((x) => x.id !== id) });
}

export function blankTask(folder: string | null): Task {
  const d = new Date(Date.now() + 3600_000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return { id: uid(), name: "", prompt: "", folder, repeat: "daily", time: `${pad(d.getHours())}:00`, day: d.getDay(), date: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`, enabled: true, created: Date.now() };
}

/** Runs `prompt` in a new chat without opening it. Null when Pelica is busy
 *  (a reply running, or no model loaded); else the chat's id. */
export function startBackground(title: string, prompt: string, folder: string | null): string | null {
  const st = app.get();
  if (st.running || st.status?.state !== "ready") return null;
  const c: Chat = { ...newChat(folder), title, messages: [{ id: uid(), role: "user", content: prompt, ts: Date.now() }] };
  void run(c);
  return c.id;
}

/** Runs a task now (the button, or when it's due). */
export function runTask(t: Task, manual = false): boolean {
  const id = startBackground(t.name || "Scheduled task", t.prompt, t.folder);
  if (!id) {
    if (manual) toast(app.get().running ? "Pelica is replying: the task runs when it's done." : "Load a model first: tasks run on the loaded model.");
    return false;
  }
  saveTask({ ...t, last: Date.now(), lastChat: id, enabled: t.repeat === "once" ? false : t.enabled });
  return true;
}

let timer = 0;
/** Checks every half minute for tasks that are due, and runs the first one
 *  (one at a time; a task that's due while Pelica is busy runs after). */
export function startScheduler() {
  if (timer) return;
  const tick = () => {
    const now = Date.now();
    const due = tasks()
      .filter((t) => t.enabled && t.prompt.trim())
      .map((t) => ({ t, at: dueAt(t) }))
      .filter((x): x is { t: Task; at: number } => x.at !== null && x.at <= now)
      .sort((a, b) => a.at - b.at)[0];
    if (due) runTask(due.t);
  };
  timer = window.setInterval(tick, 30_000);
  setTimeout(tick, 5000);
}
