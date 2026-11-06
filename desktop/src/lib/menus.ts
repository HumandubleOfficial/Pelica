// What Settings → Menus can take off Pelica's menus. A hidden item keeps
// working where it has another way in (a shortcut, the palette): it only
// leaves the menu. Ids are "<menu>.<item>", kept in settings.menuHide.
import { useApp } from "../state";
import type { MenuItem } from "../components/Menu";

export type MenuEntry = { id: string; label: string; icon: string; desc?: string; key?: string; sub?: boolean; danger?: boolean };
export type MenuDef = {
  id: string;
  title: string;
  /** its name on the tab that picks it */
  short: string;
  icon: string;
  desc: string;
  items: MenuEntry[];
  /** the items a line follows */
  seps?: string[];
  /** the row that can't be hidden */
  kept?: MenuEntry;
  /** its rows have no icons on the real thing */
  plain?: boolean;
};

export const MENUS: MenuDef[] = [
  {
    id: "plus",
    title: "The + menu",
    short: "+ menu",
    icon: "plus",
    desc: "Next to the message box.",
    seps: ["web"],
    items: [
      { id: "files", label: "Add files or photos", icon: "clip", desc: "or drop or paste them" },
      { id: "folder", label: "Choose a folder", icon: "folder", desc: "Let Pelica work in it" },
      { id: "web", label: "Web search", icon: "globe", desc: "Searches when a question needs it" },
      { id: "prompts", label: "Saved prompts", icon: "prompt", desc: "or type / and the name" },
      { id: "maker", label: "Prompt maker", icon: "wand", desc: "Answer a few questions, get a tuned prompt" },
      { id: "commands", label: "Commands", icon: "command", desc: "/compact, /cqs, /export…" },
      { id: "compare", label: "Compare models", icon: "compare", desc: "With the flagship models" },
    ],
  },
  {
    id: "chat",
    title: "A chat's menu",
    short: "Chats",
    icon: "chat",
    desc: "Right-click a chat, or use the … next to it or the chat's name at the top. Delete always stays.",
    plain: true,
    seps: ["select", "export"],
    kept: { id: "delete", label: "Delete", icon: "trash", danger: true },
    items: [
      { id: "open", label: "Open in", icon: "external", sub: true },
      { id: "pin", label: "Pin", icon: "pin", key: "P" },
      { id: "rename", label: "Rename", icon: "edit", key: "R" },
      { id: "fork", label: "Fork", icon: "branch", key: "F" },
      { id: "select", label: "Select chats", icon: "check", key: "S" },
      { id: "tags", label: "Tags", icon: "circle", sub: true },
      { id: "project", label: "Move to project", icon: "stack", sub: true },
      { id: "folder", label: "Move to folder", icon: "folder", sub: true },
      { id: "font", label: "Reply font", icon: "file", sub: true },
      { id: "export", label: "Export", icon: "download", sub: true },
    ],
  },
  {
    id: "view",
    title: "View options",
    short: "View",
    icon: "more",
    desc: "The … at the top right of a chat.",
    seps: ["export"],
    items: [
      { id: "compact", label: "Compact session", icon: "layers" },
      { id: "plan", label: "Plan", icon: "list" },
      { id: "copy", label: "Copy as Markdown", icon: "copy" },
      { id: "export", label: "Export chat", icon: "download" },
      { id: "awake", label: "Keep computer awake", icon: "monitor", desc: "Only for this session" },
    ],
  },
  {
    id: "side",
    title: "Sidebar",
    short: "Sidebar",
    icon: "sidebar",
    desc: "The top of the sidebar. New chat always stays.",
    kept: { id: "new", label: "New chat", icon: "plus" },
    items: [
      { id: "search", label: "Search", icon: "search" },
      { id: "projects", label: "Projects", icon: "stack" },
      { id: "models", label: "Models", icon: "library" },
      { id: "tags", label: "Tag filter", icon: "circle" },
    ],
  },
  {
    id: "reply",
    title: "Under a reply",
    short: "Replies",
    icon: "copy",
    desc: "The buttons under each of Pelica's replies.",
    items: [
      { id: "copy", label: "Copy", icon: "copy" },
      { id: "branch", label: "Branch from here", icon: "branch" },
      { id: "read", label: "Read aloud", icon: "speaker" },
      { id: "retry", label: "Retry", icon: "reload" },
    ],
  },
  {
    id: "user",
    title: "Your menu",
    short: "You",
    icon: "user-circle",
    desc: "Your name at the bottom of the sidebar.",
    items: [
      { id: "account", label: "Account", icon: "user-circle" },
      { id: "new", label: "What's new", icon: "sparkle" },
      { id: "keys", label: "Keyboard shortcuts", icon: "keyboard" },
      { id: "about", label: "About Pelica", icon: "info" },
    ],
  },
];

/** Asks whether an item is on its menu: `const shows = useShown(); shows("plus.web")`. */
export function useShown() {
  const hide = useApp((s) => s.settings.menuHide);
  return (id: string) => !hide.includes(id);
}

/** A menu's items without the hidden ones, and without the separators that
 *  would leave at the top, at the bottom or twice in a row. */
export function shownItems(items: MenuItem[], hide: string[]): MenuItem[] {
  const out: MenuItem[] = [];
  for (const i of items) {
    if (i !== "sep" && "id" in i && i.id && hide.includes(i.id)) continue;
    if (i === "sep" && (!out.length || out[out.length - 1] === "sep")) continue;
    out.push(i);
  }
  if (out[out.length - 1] === "sep") out.pop();
  return out;
}
