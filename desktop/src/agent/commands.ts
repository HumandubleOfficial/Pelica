// Slash commands typed in the message box.
import { bridge } from "../lib/bridge";
import { copyChat, exportChat } from "../lib/exportChat";
import { app, PERM_MODES, setChatFolder, setSettings, startNewChat, toast, updateChat } from "../state";
import { compactNow, forgetNotes, pushEvent, refreshContext, run, send } from "./agent";
import { runCQS } from "./cqs";

export interface Command {
  name: string;
  args?: string;
  help: string;
  run(arg: string): void | Promise<void>;
}

export async function pickFolder() {
  if (!bridge) return toast("Folders need the Pelica desktop app.");
  const p = await bridge.folder.pick();
  if (!p) return;
  setChatFolder(p);
  forgetNotes(p);
  toast(`Working in ${p}`);
}

export const COMMANDS: Command[] = [
  { name: "help", help: "List the commands", run: () => pushEvent("info", "Commands: " + COMMANDS.map((c) => `/${c.name}${c.args ? " " + c.args : ""}`).join(", ")) },
  { name: "compact", args: "[focus]", help: "Summarize the conversation to free up context", run: (a) => compactNow(a) },
  {
    name: "autocompact",
    args: "[on|off|percent]",
    help: "Compact automatically when the context fills up",
    run: (a) => {
      const s = app.get().settings;
      const v = a.trim().toLowerCase();
      if (/^\d+%?$/.test(v)) setSettings({ autocompact: true, compactAt: Math.min(95, Math.max(30, parseInt(v))) / 100 });
      else setSettings({ autocompact: v ? v === "on" : !s.autocompact });
      const n = app.get().settings;
      toast(`Auto-compact ${n.autocompact ? `on at ${Math.round(n.compactAt * 100)}% of the context` : "off"}`);
    },
  },
  { name: "context", help: "Show what fills the context", run: async () => { await refreshContext(); window.dispatchEvent(new Event("pelica:context")); } },
  { name: "clear", help: "Start a new chat", run: () => startNewChat() },
  { name: "new", help: "Start a new chat", run: () => startNewChat() },
  { name: "folder", help: "Choose the folder Pelica works in", run: () => pickFolder() },
  { name: "cqs", args: "[polish] [files]", help: "Scan code for problems and fix them (polish: also tidy code and UI)", run: (a) => { const parts = a.split(/\s+/).filter(Boolean); const polish = parts[0] === "polish"; return runCQS(polish ? parts.slice(1) : parts, polish ? "polish" : "check", true); } },
  {
    name: "effort",
    args: "<level>",
    help: "Set the reasoning effort",
    run: (a) => {
      const ef = app.get().status?.efforts || [];
      if (!ef.length) return toast("This model has no effort levels.");
      if (!ef.includes(a.trim())) return toast(`Levels for this model: ${ef.join(", ")}`);
      setSettings({ effort: a.trim() });
      toast(`Effort: ${a.trim()}`);
    },
  },
  { name: "web", args: "[on|off]", help: "Turn web search on or off", run: (a) => { setSettings({ web: a ? a.trim() === "on" : !app.get().settings.web }); toast(`Web search ${app.get().settings.web ? "on" : "off"}`); } },
  { name: "brain", help: "Inspect the model's insides", run: () => app.set({ view: "brain" }) },
  { name: "library", help: "Browse and download models", run: () => app.set({ view: "library" }) },
  { name: "model", help: "Choose a model", run: () => app.set({ view: "library" }) },
  { name: "settings", help: "Open settings", run: () => app.set({ settingsOpen: true }) },
  { name: "theme", args: "<dark|warm|light|midnight|neural>", help: "Change the theme", run: (a) => { const t = a.trim() as any; if (["dark", "warm", "light", "midnight", "neural"].includes(t)) setSettings({ theme: t }); else toast("Themes: dark, warm, light, midnight, neural"); } },
  {
    name: "retry",
    help: "Answer the last message again",
    run: () => {
      const c = app.get().chat;
      if (!c) return;
      const i = c.messages.map((m) => m.role).lastIndexOf("user");
      if (i < 0) return;
      updateChat((x) => ({ ...x, messages: x.messages.slice(0, i + 1) }));
      run();
    },
  },
  { name: "continue", help: "Keep going where it stopped", run: () => send("Continue.") },
  { name: "init", help: "Write PELICA.md: notes about this project for future chats", run: () => send("Look through this project (list_files, read the key files) and write PELICA.md in its root: what the project is, how it's organized, how to build, run and test it, and conventions to follow. Keep it short and useful for working on the project later.") },
  { name: "export", help: "Save this chat as a Markdown file", run: () => exportChat() },
  { name: "prompt", help: "Open the prompt maker", run: () => app.set({ promptMaker: true }) },
  { name: "compare", help: "Compare the loaded model with flagship models", run: () => app.set({ compare: "" }) },
  {
    name: "mode",
    args: "[ask|edits|plan|auto]",
    help: "Switch the permission mode",
    run: (a) => {
      const m = PERM_MODES.find((x) => x.id === a.trim().toLowerCase() || x.label.toLowerCase() === a.trim().toLowerCase());
      if (!m) return pushEvent("info", "Modes: " + PERM_MODES.map((x) => `${x.id} (${x.desc})`).join(", "));
      setSettings({ permMode: m.id });
      toast(`${m.label}: ${m.desc}`);
    },
  },
  { name: "copy", help: "Copy this whole chat as Markdown", run: () => copyChat() },
];

/** Runs `text` if it's a command; false if it isn't one. */
export async function command(text: string): Promise<boolean> {
  const m = text.trim().match(/^\/(\w+)\s*([\s\S]*)$/);
  if (!m) return false;
  const c = COMMANDS.find((x) => x.name === m[1].toLowerCase());
  if (!c) {
    toast(`No command /${m[1]}. Type /help.`);
    return true;
  }
  await c.run(m[2] || "");
  return true;
}
