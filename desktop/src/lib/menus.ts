// What Settings → Menus can take off Pelica's menus. A hidden item keeps
// working where it has another way in (a shortcut, the palette): it only
// leaves the menu. Ids are "<menu>.<item>", kept in settings.menuHide.
import { useApp } from "../state";
import type { MenuItem } from "../components/Menu";

export const MENUS: { id: string; title: string; desc: string; items: [string, string][] }[] = [
  {
    id: "plus",
    title: "The + menu",
    desc: "Next to the message box.",
    items: [
      ["files", "Add files or photos"],
      ["folder", "Choose a folder"],
      ["web", "Web search"],
      ["prompts", "Saved prompts"],
      ["maker", "Prompt maker"],
      ["commands", "Commands"],
      ["compare", "Compare models"],
    ],
  },
  {
    id: "chat",
    title: "A chat's menu",
    desc: "Right-click a chat, or use the … next to it or the chat's name at the top. Delete always stays.",
    items: [
      ["open", "Open in"],
      ["pin", "Pin"],
      ["rename", "Rename"],
      ["fork", "Fork"],
      ["select", "Select chats"],
      ["tags", "Tags"],
      ["project", "Move to project"],
      ["folder", "Move to folder"],
      ["font", "Reply font"],
      ["export", "Export"],
    ],
  },
  {
    id: "view",
    title: "View options",
    desc: "The … at the top right of a chat.",
    items: [
      ["compact", "Compact session"],
      ["plan", "Plan"],
      ["copy", "Copy as Markdown"],
      ["export", "Export chat"],
      ["awake", "Keep computer awake"],
    ],
  },
  {
    id: "side",
    title: "Sidebar",
    desc: "The top of the sidebar. New chat always stays.",
    items: [
      ["search", "Search"],
      ["projects", "Projects"],
      ["models", "Models"],
      ["tags", "Tag filter"],
    ],
  },
  {
    id: "reply",
    title: "Under a reply",
    desc: "The buttons under each of Pelica's replies.",
    items: [
      ["copy", "Copy"],
      ["branch", "Branch from here"],
      ["read", "Read aloud"],
      ["retry", "Retry"],
    ],
  },
  {
    id: "user",
    title: "Your menu",
    desc: "Your name at the bottom of the sidebar.",
    items: [
      ["account", "Account"],
      ["new", "What's new"],
      ["keys", "Keyboard shortcuts"],
      ["about", "About Pelica"],
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
