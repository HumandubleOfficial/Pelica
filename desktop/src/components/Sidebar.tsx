// The left sidebar (as in the Claude app): new chat, search, sections,
// pinned and recent chats, the user chip with settings.
import { Fragment, useEffect, useRef, useState } from "react";
import { app, deleteChat, deleteChats, forkChat, openChat, openProject, patchChat, searchMessages, setChatFolder, setSettings, startNewChat, toggleSidebar, updateChat, useApp } from "../state";
import type { SearchHit } from "../../electron/chatdata";
import { bridge } from "../lib/bridge";
import { copyChat, exportChat } from "../lib/exportChat";
import { compactNow } from "../agent/agent";
import { DOCKS } from "./Dock";
import { Icon, Logo } from "./Icons";
import { Menu, type MenuItem } from "./Menu";
import { useShown } from "../lib/menus";
import { SideUpdate } from "./Updates";

const base = (p: string) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

const TAG_COLORS: [string, string][] = [["Orange", "#d97757"], ["Blue", "#5b9bd5"], ["Green", "#6aa84f"], ["Pink", "#c27ba0"], ["Gold", "#b8923a"], ["Purple", "#7e7ad1"], ["Teal", "#3fa796"], ["Red", "#c96a6a"]];
/** A tag's colour: the one picked for it, else one that comes from its name. */
const tagColor = (t: string, own = app.get().settings.tagColors) => {
  if (own[t]) return own[t];
  let h = 0;
  for (const c of t) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return TAG_COLORS[h % TAG_COLORS.length][1];
};
/** Renames a tag on every chat (to a name in use: they become one tag). */
async function renameTag(from: string, to: string) {
  for (const c of app.get().chats.filter((x) => x.tags?.includes(from))) await patchChat(c.id, (x) => ({ ...x, tags: [...new Set((x.tags || []).map((t) => (t === from ? to : t)))] }));
  const { [from]: color, ...rest } = app.get().settings.tagColors;
  setSettings({ tagColors: color ? { ...rest, [to]: color } : rest });
  if (app.get().sideTag === from) app.set({ sideTag: to });
}

/** What the sidebar's date headings call a chat last changed at `ts`. */
function whenLabel(ts: number) {
  const day = new Date();
  day.setHours(0, 0, 0, 0);
  const back = (n: number) => day.getTime() - n * 864e5;
  return ts >= back(0) ? "Today" : ts >= back(1) ? "Yesterday" : ts >= back(6) ? "Last 7 days" : ts >= back(29) ? "Last 30 days" : "Older";
}

const CHAT_DRAG = "application/x-pelica-chat";
const dragsChat = (e: React.DragEvent) => e.dataTransfer.types.includes(CHAT_DRAG);

/** Up and Down move through the chat rows. */
function rowKeys(e: React.KeyboardEvent) {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  const t = e.target as HTMLElement;
  if (!t.classList.contains("chat-row-main")) return;
  const rows = [...e.currentTarget.querySelectorAll<HTMLElement>(".chat-row-main")];
  const next = rows[rows.indexOf(t) + (e.key === "ArrowDown" ? 1 : -1)];
  if (next) {
    e.preventDefault();
    next.focus();
  }
}

/** Puts a chat in front of a pinned one, pinning it if it wasn't. */
function pinBefore(id: string, before: string) {
  const order = app.get().chats.filter((c) => c.pinned && c.id !== id);
  const mine = app.get().settings.pinOrder;
  const rank = (x: string) => (mine.includes(x) ? mine.indexOf(x) : mine.length);
  const ids = order.map((c) => c.id).sort((a, b) => rank(a) - rank(b));
  ids.splice(ids.indexOf(before), 0, id);
  setSettings({ pinOrder: ids });
  if (!app.get().chats.find((c) => c.id === id)?.pinned) patchChat(id, (c) => ({ ...c, pinned: true }));
}
const allTags = () => [...new Set(app.get().chats.flatMap((c) => c.tags || []))].sort((a, b) => a.localeCompare(b));
const setTag = (id: string, tag: string, on: boolean) => patchChat(id, (c) => ({ ...c, tags: on ? [...new Set([...(c.tags || []), tag])] : (c.tags || []).filter((t) => t !== tag) }));

/** Ticks a chat; with Shift, every chat from the last one ticked to this one. */
let lastPicked: string | null = null;
function togglePick(id: string, range: boolean) {
  const p = app.get().picked || [];
  if (range && lastPicked) {
    const order = [...document.querySelectorAll<HTMLElement>(".chat-row[data-id]")].map((e) => e.dataset.id!);
    const a = order.indexOf(lastPicked);
    const b = order.indexOf(id);
    if (a >= 0 && b >= 0) return app.set({ picked: [...new Set([...p, ...order.slice(Math.min(a, b), Math.max(a, b) + 1)])] });
  }
  lastPicked = id;
  const next = p.includes(id) ? p.filter((x) => x !== id) : [...p, id];
  app.set({ picked: next.length ? next : null });
}

/** What a chat's menu offers (the sidebar's "…" and right-click, and the
 *  title bar's chevron), as in Claude's. */
function chatMenu(id: string, o: { pinned: boolean; folder: string | null; project?: string; rename: () => void; title?: boolean; tags?: string[]; newTag?: () => void }): MenuItem[] {
  const recent = app.get().settings.recentFolders;
  const projects = app.get().settings.projects;
  const font = app.get().settings.replyFont;
  const items: MenuItem[] = [];
  if (o.folder && bridge) {
    items.push({ id: "chat.open", label: "Open in", sub: [
      { label: "File Explorer", icon: "folder", on: () => bridge!.sys.open(o.folder!, "").catch((e: any) => app.set({ toast: e.message })) },
      ...(o.title ? [{ label: "Terminal", icon: "terminal", on: () => app.set({ dock: "terminal" }) }] : []),
    ] }, "sep");
  }
  items.push(
    { id: "chat.pin", label: o.pinned ? "Unpin" : "Pin", key: "P", on: () => patchChat(id, (c) => ({ ...c, pinned: !o.pinned })) },
    { id: "chat.rename", label: "Rename", key: "R", on: o.rename },
    { id: "chat.fork", label: "Fork", key: "F", on: () => forkChat(id) },
    ...(o.title ? [] : [{ id: "chat.select", label: "Select chats", key: "S", on: () => ((lastPicked = id), app.set({ picked: [id] })) }]),
    "sep",
  );
  const tags = allTags();
  const mine = o.tags || [];
  if (tags.length || o.newTag)
    items.push({ id: "chat.tags", label: "Tags", sub: [
      ...tags.map((t) => ({ label: t, checked: mine.includes(t), on: () => setTag(id, t, !mine.includes(t)) })),
      ...(tags.length && o.newTag ? (["sep"] as MenuItem[]) : []),
      ...(o.newTag ? [{ label: "New tag…", icon: "plus", on: o.newTag }] : []),
    ] });
  items.push({ id: "chat.project", label: "Move to project", sub: [
    ...projects.map((p) => ({ label: p.name, icon: "stack", checked: p.id === o.project, on: () => patchChat(id, (c) => ({ ...c, project: p.id })) })),
    ...(projects.length ? (["sep"] as MenuItem[]) : []),
    { label: "New project…", icon: "plus", on: () => { const p = { id: Date.now().toString(36), name: "New project", instructions: "", files: [], folder: o.folder, created: Date.now() }; setSettings({ projects: [...projects, p] }); patchChat(id, (c) => ({ ...c, project: p.id })).then(() => openProject(p.id)); } },
    { label: "No project", icon: "x", disabled: !o.project, on: () => patchChat(id, (c) => ({ ...c, project: undefined })) },
  ] });
  if (!o.title) {
    items.push({ id: "chat.folder", label: "Move to folder", sub: [
      ...recent.map((p) => ({ label: base(p), icon: "folder", checked: p === o.folder, on: () => (app.get().chat?.id === id ? setChatFolder(p) : patchChat(id, (c) => ({ ...c, folder: p }))) })),
      ...(recent.length ? (["sep"] as MenuItem[]) : []),
      { label: "No folder", icon: "x", disabled: !o.folder, on: () => patchChat(id, (c) => ({ ...c, folder: null })) },
    ] });
  } else {
    items.push({ id: "chat.font", label: "Reply font", sub: [
      { label: "Serif", checked: font === "serif", on: () => setSettings({ replyFont: "serif" }) },
      { label: "Sans serif", checked: font === "sans", on: () => setSettings({ replyFont: "sans" }) },
    ] });
  }
  items.push(
    { id: "chat.export", label: "Export", sub: [
      { label: "Copy as Markdown", icon: "copy", on: () => copyChat(id) },
      { label: "Save as Markdown…", icon: "download", on: () => exportChat(id) },
    ] },
    "sep",
    { label: "Delete", key: "D", danger: true, on: () => deleteChat(id) },
  );
  return items;
}

/** Renames a chat in place: the title turns into a box with its text selected. */
function RenameBox({ title, done, className, hint }: { title: string; done: (t: string | null) => void; className: string; hint?: string }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => ref.current?.select(), []);
  return <input ref={ref} className={className} defaultValue={title} placeholder={hint} spellCheck={false} onBlur={(e) => done(e.target.value)} onKeyDown={(e) => (e.key === "Enter" ? e.currentTarget.blur() : e.key === "Escape" && done(null))} />;
}

function ChatRow({ id, title, pinned, active, folder, project, tag = true, tags = [] }: { id: string; title: string; pinned: boolean; active: boolean; folder: string | null; project?: string; tag?: boolean; tags?: string[] }) {
  const live = useApp((s) => s.liveId === id);
  const selecting = useApp((s) => !!s.picked);
  const ticked = useApp((s) => !!s.picked?.includes(id));
  const [tagging, setTagging] = useState(false);
  const [dropHere, setDropHere] = useState(false);
  const own = useApp((s) => s.settings.tagColors);
  const pname = useApp((s) => (project ? s.settings.projects.find((p) => p.id === project)?.name : undefined));
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const [editing, setEditing] = useState(false);
  const more = useRef<HTMLButtonElement>(null);
  function rename(t: string | null) {
    setEditing(false);
    if (t === null || !t.trim() || t.trim() === title) return;
    patchChat(id, (c) => ({ ...c, title: t.trim() }));
  }
  function addTag(t: string | null) {
    setTagging(false);
    if (t?.trim()) setTag(id, t.trim(), true);
  }
  function click(e: React.MouseEvent) {
    if (selecting || e.ctrlKey || e.metaKey || e.shiftKey) togglePick(id, e.shiftKey);
    else openChat(id);
  }
  const openMenu = () => {
    const r = more.current!.getBoundingClientRect();
    setMenu(menu ? null : { x: r.left, y: r.bottom + 4 });
  };
  return (
    <div
      data-id={id}
      className={`chat-row ${active ? "active" : ""} ${live ? "live" : ""} ${menu ? "menu-open" : ""} ${ticked ? "ticked" : ""} ${dropHere ? "drop-before" : ""}`}
      onContextMenu={(e) => (e.preventDefault(), !selecting && setMenu({ x: e.clientX, y: e.clientY }))}
      draggable={!selecting && !editing && !tagging}
      onDragStart={(e) => {
        e.dataTransfer.setData(CHAT_DRAG, id);
        e.dataTransfer.effectAllowed = "move";
      }}
      onDragOver={(e) => {
        if (!pinned || !dragsChat(e)) return;
        e.preventDefault();
        setDropHere(true);
      }}
      onDragLeave={() => setDropHere(false)}
      onDrop={(e) => {
        setDropHere(false);
        const from = e.dataTransfer.getData(CHAT_DRAG);
        if (pinned && from && from !== id) pinBefore(from, id);
      }}
    >
      {tagging ? (
        <RenameBox className="chat-row-edit" title="" hint="Tag name" done={addTag} />
      ) : editing ? (
        <RenameBox className="chat-row-edit" title={title} done={rename} />
      ) : (
        <button className="chat-row-main" onClick={click} onDoubleClick={() => !selecting && setEditing(true)} title={folder ? `${title}\n${folder}` : title} aria-pressed={selecting ? ticked : undefined}>
          {selecting && (
            <span className="pick-box">
              <Icon name="check" size={11} stroke={3} />
            </span>
          )}
          {live && <span className="live-dot" title="Pelica is replying here" />}
          <span className="chat-row-title">{title}</span>
          {tags.length > 0 && (
            <span className="chat-row-tags" title={tags.join(", ")}>
              {tags.slice(0, 3).map((t) => (
                <i key={t} style={{ background: tagColor(t, own) }} />
              ))}
            </span>
          )}
          {pname ? <span className="chat-row-folder chat-row-project">{pname}</span> : folder && tag && <span className="chat-row-folder">{folder.split(/[\\/]/).filter(Boolean).pop()}</span>}
        </button>
      )}
      {!selecting && (
        <button ref={more} className={`chat-row-more ${menu ? "open menu-open" : ""}`} title="More" onClick={openMenu}>
          <Icon name="more" size={16} />
        </button>
      )}
      {menu && <Menu at={menu} ignore={more} close={() => setMenu(null)} items={chatMenu(id, { pinned, folder, project, tags, rename: () => setEditing(true), newTag: () => setTagging(true) })} />}
    </div>
  );
}

/** A chat found by its messages: its title, and the first match in context.
 *  Opening it starts Find in chat on the same words. */
function MessageHit({ h, q }: { h: SearchHit; q: string }) {
  return (
    <button
      className="msg-hit"
      title={h.hits > 1 ? `${h.hits} messages match` : undefined}
      onClick={async () => {
        await openChat(h.id);
        if (app.get().compact) app.set({ drawer: false });
        setTimeout(() => window.dispatchEvent(new CustomEvent("pelica:find", { detail: q })), 60);
      }}
    >
      <span className="msg-hit-title">{h.title}</span>
      <span className="msg-hit-snip">
        {h.before}
        <mark>{h.match}</mark>
        {h.after}
      </span>
    </button>
  );
}

/** With chats ticked: what can be done to all of them. */
function PickBar() {
  const picked = useApp((s) => s.picked);
  const chats = useApp((s) => s.chats);
  const [sure, setSure] = useState(false);
  const [tagMenu, setTagMenu] = useState<{ x: number; y: number } | null>(null);
  const [naming, setNaming] = useState(false);
  const tagBtn = useRef<HTMLButtonElement>(null);
  const n = picked?.length || 0;
  useEffect(() => setSure(false), [n]);
  useEffect(() => {
    if (!picked) return;
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ picked: null });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [!picked]);
  if (!picked) return null;
  const mine = chats.filter((c) => picked.includes(c.id));
  const allPinned = mine.length > 0 && mine.every((c) => c.pinned);
  const each = async (f: (id: string) => unknown) => {
    for (const id of picked) await f(id);
  };
  const tags = allTags();
  const toggleTag = (t: string) => {
    const on = !mine.every((c) => c.tags?.includes(t));
    each((id) => setTag(id, t, on));
  };
  return (
    <div className="pick-bar">
      {naming ? (
        <RenameBox
          className="chat-row-edit pick-name"
          title=""
          hint="Tag name"
          done={(t) => {
            setNaming(false);
            if (t?.trim()) each((id) => setTag(id, t.trim(), true));
          }}
        />
      ) : (
        <div className="pick-head">
          <b>{n} selected</b>
          <button className="pick-link" onClick={() => app.set({ picked: [...document.querySelectorAll<HTMLElement>(".chat-row[data-id]")].map((e) => e.dataset.id!) })}>
            Select all
          </button>
          <button className="ghost-btn" title="Done (Esc)" onClick={() => app.set({ picked: null })}>
            <Icon name="x" size={14} />
          </button>
        </div>
      )}
      <div className="pick-btns">
        <button className="btn small" onClick={() => each((id) => patchChat(id, (c) => ({ ...c, pinned: !allPinned })))}>
          {allPinned ? "Unpin" : "Pin"}
        </button>
        <button
          ref={tagBtn}
          className="btn small"
          onClick={() => {
            const r = tagBtn.current!.getBoundingClientRect();
            setTagMenu(tagMenu ? null : { x: r.left, y: r.top - 8 });
          }}
        >
          Tag
        </button>
        <button
          className={`btn small pick-delete ${sure ? "sure" : ""}`}
          onClick={() => {
            if (!sure) return setSure(true);
            deleteChats(picked);
            app.set({ picked: null });
          }}
        >
          {sure ? `Delete ${n}?` : "Delete"}
        </button>
      </div>
      {tagMenu && (
        <Menu
          at={tagMenu}
          ignore={tagBtn}
          close={() => setTagMenu(null)}
          items={[
            ...tags.map((t) => ({ label: t, checked: mine.length > 0 && mine.every((c) => c.tags?.includes(t)), on: () => toggleTag(t) })),
            ...(tags.length ? (["sep"] as MenuItem[]) : []),
            { label: "New tag…", icon: "plus", on: () => setNaming(true) },
          ]}
        />
      )}
    </div>
  );
}

/** The tags in use, to narrow the list to one. */
function TagFilter() {
  const chats = useApp((s) => s.chats);
  const on = useApp((s) => s.sideTag);
  const own = useApp((s) => s.settings.tagColors);
  const [menu, setMenu] = useState<{ x: number; y: number; tag: string } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [over, setOver] = useState<string | null>(null);
  const tags = [...new Set(chats.flatMap((c) => c.tags || []))].sort((a, b) => a.localeCompare(b));
  useEffect(() => {
    if (on && !tags.includes(on)) app.set({ sideTag: null });
  }, [tags.join("\n"), on]);
  if (!tags.length) return null;
  return (
    <div className="side-tags">
      {tags.map((t) =>
        renaming === t ? (
          <RenameBox
            key={t}
            className="side-tag-edit"
            title={t}
            done={(n) => {
              setRenaming(null);
              if (n?.trim() && n.trim() !== t) renameTag(t, n.trim());
            }}
          />
        ) : (
          <button
            key={t}
            className={`side-tag ${t === on ? "on" : ""} ${over === t ? "drop" : ""}`}
            style={{ "--tag": tagColor(t, own) } as React.CSSProperties}
            title="Click to show only these chats. Right-click to rename or recolour. Drop a chat here to tag it."
            onClick={() => app.set({ sideTag: t === on ? null : t })}
            onContextMenu={(e) => (e.preventDefault(), setMenu({ x: e.clientX, y: e.clientY, tag: t }))}
            onDragOver={(e) => {
              if (!dragsChat(e)) return;
              e.preventDefault();
              setOver(t);
            }}
            onDragLeave={() => setOver(null)}
            onDrop={(e) => {
              setOver(null);
              const id = e.dataTransfer.getData(CHAT_DRAG);
              if (id) setTag(id, t, true);
            }}
          >
            <i />
            {t}
          </button>
        ),
      )}
      {menu && (
        <Menu
          at={menu}
          close={() => setMenu(null)}
          items={[
            { label: "Rename…", on: () => setRenaming(menu.tag) },
            {
              label: "Colour",
              sub: TAG_COLORS.map(([name, c]) => ({
                label: name,
                checked: tagColor(menu.tag, own) === c,
                right: <i className="tag-swatch" style={{ background: c }} />,
                on: () => setSettings({ tagColors: { ...own, [menu.tag]: c } }),
              })),
            },
            "sep",
            {
              label: `Remove “${menu.tag}” from all chats`,
              danger: true,
              on: async () => {
                for (const c of app.get().chats.filter((x) => x.tags?.includes(menu.tag))) await setTag(c.id, menu.tag, false);
              },
            },
          ]}
        />
      )}
    </div>
  );
}

/** A section's name in the chat list: click it to fold the section. */
function SideHead({ id, label }: { id: string; label: string }) {
  const folded = useApp((s) => s.settings.sideFolded);
  const shut = folded.includes(id);
  return (
    <div className={`side-group-head top ${shut ? "folded" : ""}`}>
      <button className="side-group-name" aria-expanded={!shut} onClick={() => setSettings({ sideFolded: shut ? folded.filter((x) => x !== id) : [...folded, id] })}>
        {label}
        <span className="sp-caret" />
      </button>
    </div>
  );
}

export function Sidebar() {
  const allChats = useApp((s) => s.chats);
  const sideTag = useApp((s) => s.sideTag);
  const selecting = useApp((s) => !!s.picked);
  const chats = sideTag ? allChats.filter((c) => c.tags?.includes(sideTag)) : allChats;
  const cur = useApp((s) => s.chat?.id);
  const view = useApp((s) => s.view);
  const brainMenu = useApp((s) => s.settings.brainMenu);
  const shows = useShown();
  const acct = useApp((s) => s.account?.profile);
  const syncOn = useApp((s) => s.settings.syncOn);
  const name = useApp((s) => s.settings.userName);
  const search = useApp((s) => s.search);
  const [searching, setSearching] = useState(false);
  const [userMenu, setUserMenu] = useState<{ x: number; y: number } | null>(null);
  const q = search.trim().toLowerCase();
  const shown = q ? chats.filter((c) => c.title.toLowerCase().includes(q)) : chats;
  // chats with the words in their messages (those already listed by title left out)
  const [hits, setHits] = useState<SearchHit[]>([]);
  useEffect(() => {
    if (q.length < 2) return setHits([]);
    let stale = false;
    const t = setTimeout(() => searchMessages(q).then((h) => stale || setHits(h)), 200);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [q]);
  const msgHits = hits.filter((h) => !shown.some((c) => c.id === h.id));
  const pinOrder = useApp((s) => s.settings.pinOrder);
  const pinRank = (id: string) => (pinOrder.includes(id) ? pinOrder.indexOf(id) : pinOrder.length);
  const pinned = shown.filter((c) => c.pinned).sort((a, b) => pinRank(a.id) - pinRank(b.id));
  const recent = shown.filter((c) => !c.pinned);
  const folded = useApp((s) => s.settings.sideFolded);
  const shut = (id: string) => !q && folded.includes(id);
  // each folder is a group named after it, newest first; chats with no folder come after
  const groups = new Map<string, typeof recent>();
  const loose: typeof recent = [];
  for (const c of recent) {
    if (!c.folder || q) loose.push(c);
    else {
      const k = c.folder.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
      groups.set(k, [...(groups.get(k) || []), c]);
    }
  }
  const fname = base;
  const nav = (label: string, icon: string, onClick: () => void, active = false) => (
    <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}>
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </button>
  );
  return (
    <aside className="sidebar">
      <SideResize />
      <div className="side-top">
        <div className="brand">
          <Logo size={22} />
          <span>Pelica</span>
        </div>
        <button className="ghost-btn" title="Hide sidebar" onClick={() => toggleSidebar(false)}>
          <Icon name="sidebar" size={16} />
        </button>
      </div>
      <nav className="side-nav">
        {nav("New chat", "plus", () => startNewChat())}
        {searching ? (
          <div className="side-search">
            <Icon name="search" size={15} />
            <input autoFocus value={search} placeholder="Search chats" onChange={(e) => app.set({ search: e.target.value })} onBlur={() => !search && setSearching(false)} onKeyDown={(e) => e.key === "Escape" && (app.set({ search: "" }), setSearching(false))} />
          </div>
        ) : (
          shows("side.search") && nav("Search", "search", () => setSearching(true))
        )}
        {shows("side.projects") && nav("Projects", "stack", () => app.set({ view: "projects" }), view === "projects" || view === "project")}
        {shows("side.models") && nav("Models", "library", () => app.set({ view: "library" }), view === "library")}
        {brainMenu && nav("Brain", "brain", () => app.set({ view: "brain" }), view === "brain")}
      </nav>
      {shows("side.tags") && <TagFilter />}
      <div className={`side-lists ${selecting ? "selecting" : ""}`} onKeyDown={rowKeys}>
        {pinned.length > 0 && !q && <SideHead id="@pinned" label="Pinned" />}
        {!shut("@pinned") && pinned.map((c) => (
          <ChatRow key={c.id} id={c.id} title={c.title} pinned active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} tags={c.tags} />
        ))}
        {[...groups].map(([k, list]) => {
          const shut = folded.includes(k);
          const path = list[0].folder!;
          return (
            <div key={k} className={`side-group ${shut ? "folded" : ""}`}>
              <div className="side-group-head">
                <button className="side-group-name" title={path} onClick={() => setSettings({ sideFolded: shut ? folded.filter((x) => x !== k) : [...folded, k] })} aria-expanded={!shut}>
                  {fname(path)}
                  <span className="sp-caret" />
                </button>
                <button className="side-group-new" title={`New chat in ${fname(path)}`} onClick={() => startNewChat(path)}>
                  <Icon name="plus" size={13} />
                </button>
              </div>
              {!shut && list.map((c) => <ChatRow key={c.id} id={c.id} title={c.title} pinned={false} active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} tag={false} tags={c.tags} />)}
            </div>
          );
        })}
        {q && loose.length > 0 && <div className="side-label">Results</div>}
        {loose.map((c, i) => (
          <Fragment key={c.id}>
            {!q && (i === 0 || whenLabel(loose[i - 1].updated) !== whenLabel(c.updated)) && <SideHead id={`@${whenLabel(c.updated)}`} label={whenLabel(c.updated)} />}
            {!shut(`@${whenLabel(c.updated)}`) && <ChatRow id={c.id} title={c.title} pinned={false} active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} tags={c.tags} />}
          </Fragment>
        ))}
        {msgHits.length > 0 && <div className="side-label">In messages</div>}
        {msgHits.map((h) => (
          <MessageHit key={h.id} h={h} q={search.trim()} />
        ))}
        {q && !shown.length && !msgHits.length && <div className="side-empty">No chats mention “{search.trim()}”.</div>}
        {!chats.length && <div className="side-empty">{sideTag ? "No chats have that tag." : "Your chats will show up here."}</div>}
      </div>
      <PickBar />
      <SideUpdate />
      <div className="side-foot">
        <button
          className={`user-chip ${userMenu ? "open" : ""}`}
          aria-haspopup="menu"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            setUserMenu(userMenu ? null : { x: r.left, y: r.top - 150 });
          }}
        >
          {acct?.picture ? <img className="avatar" src={acct.picture} alt="" /> : <span className="avatar">{(name || acct?.name || "You").slice(0, 1).toUpperCase()}</span>}
          <span className="user-name">
            {name || acct?.name || "You"}
            <span className="user-plan"> · {acct ? (syncOn ? "Synced" : "Not syncing") : "Local"}</span>
          </span>
          <Icon name="chevronDown" size={13} />
        </button>
        <button className="ghost-btn" title="Settings" onClick={() => app.set({ settingsOpen: true })}>
          <Icon name="settings" size={16} />
        </button>
      </div>
      {userMenu && (
        <Menu
          at={userMenu}
          close={() => setUserMenu(null)}
          items={[
            { id: "user.account", label: "Account", icon: "user-circle", on: () => app.set({ settingsOpen: true, settingsTab: "Account" }) },
            { id: "user.new", label: "What's new", icon: "sparkle", on: () => app.set({ whatsNew: "all" }) },
            { id: "user.keys", label: "Keyboard shortcuts", icon: "keyboard", on: () => app.set({ keys: true }) },
            { id: "user.about", label: "About Pelica", icon: "info", on: () => app.set({ settingsOpen: true, settingsTab: "About" }) },
          ]}
        />
      )}
    </aside>
  );
}

/** The sidebar's right edge: drag it to make the sidebar wider or narrower
 *  (200–480px, saved); double-click it to go back to the usual width. */
function SideResize() {
  const down = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = app.get().settings.sideWidth;
    // (on <body> while dragging, over the saved width on <html>)
    const root = document.body;
    let w = w0;
    document.body.classList.add("resizing-side");
    const move = (m: PointerEvent) => {
      w = Math.round(Math.max(200, Math.min(480, w0 + m.clientX - x0)));
      root.style.setProperty("--left-w", `${w}px`);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      document.body.classList.remove("resizing-side");
      setSettings({ sideWidth: w });
      root.style.removeProperty("--left-w");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };
  return <div className="side-resize" role="separator" aria-orientation="vertical" aria-label="Resize the sidebar" title="Drag to resize · double-click to reset" onPointerDown={down} onDoubleClick={() => setSettings({ sideWidth: 272 })} />;
}

/** The title bar's "…": things to do with this chat, and keeping the PC awake. */
function ViewOptions() {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const awake = useApp((s) => s.awake);
  const running = useApp((s) => s.running);
  const has = useApp((s) => !!s.chat && s.chat.messages.length > 0);
  const panel = useApp((s) => s.settings.sidePanel);
  const hidden = useApp((s) => s.settings.panelHidden);
  const keepAwake = async () => {
    if (!bridge) return app.set({ toast: "Keeping the computer awake needs the Pelica desktop app." });
    app.set({ awake: await bridge.sys.awake(!awake) });
  };
  const planShown = panel && !hidden.includes("plan");
  const items: MenuItem[] = [
    { id: "view.compact", label: "Compact session", icon: "layers", disabled: !has || running, on: () => compactNow() },
    { id: "view.plan", label: "Plan", icon: "list", checked: planShown, on: () => (app.set({ dock: null }), setSettings({ sidePanel: true, panelHidden: planShown ? [...hidden, "plan"] : hidden.filter((x) => x !== "plan") })) },
    { id: "view.copy", label: "Copy as Markdown", icon: "copy", disabled: !has, on: () => copyChat() },
    { id: "view.export", label: "Export chat…", icon: "download", disabled: !has, on: () => exportChat() },
    "sep",
    { id: "view.awake", label: "Keep computer awake", desc: "Only for this session", stay: true, right: <span className={`toggle ${awake ? "on" : ""}`} role="switch" aria-checked={awake} />, on: keepAwake },
  ];
  return (
    <div className="pop-wrap">
      <button ref={btn} className={`ghost-btn ${open ? "menu-open" : ""}`} title="View options" onClick={() => setOpen(!open)}>
        <Icon name="more" size={16} />
      </button>
      {open && <Menu className="tb-pop" ignore={btn} close={() => setOpen(false)} items={items} />}
    </div>
  );
}

/** The chat's name in the title bar, with a chevron for its menu; Rename
 *  turns it into a box in place. */
function ChatTitle() {
  const chat = useApp((s) => s.chat);
  const project = useApp((s) => s.settings.projects.find((p) => p.id === s.chat?.project));
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  if (!chat) return <div className="title-text" />;
  const saved = chat.messages.length > 0 || !!chat.archive?.length;
  if (!saved) return <div className="title-text" />;
  const rename = (t: string | null) => {
    setEditing(false);
    if (t !== null && t.trim() && t.trim() !== chat.title) updateChat((c) => ({ ...c, title: t.trim() }));
  };
  return (
    <div className="title-text chat-title">
      {project && (
        <>
          <button className="title-project nodrag" onClick={() => openProject(project.id)} title="Open the project">
            {project.name}
          </button>
          <span className="title-slash">/</span>
        </>
      )}
      <div className="pop-wrap title-wrap nodrag" ref={wrap}>
        {editing ? (
          <RenameBox className="title-edit" title={chat.title} done={rename} />
        ) : (
          <button className={`title-btn ${open ? "open menu-open" : ""}`} onClick={() => setOpen(!open)} onDoubleClick={() => (setOpen(false), setEditing(true))} title="Chat menu">
            <span className="title-name">{chat.title}</span>
            <Icon name="chevronDown" size={14} stroke={2.4} />
          </button>
        )}
        {open && !editing && <Menu className="title-pop" ignore={wrap} close={() => setOpen(false)} items={chatMenu(chat.id, { tags: chat.tags, pinned: !!chat.pinned, folder: chat.folder, project: chat.project, rename: () => setEditing(true), title: true })} />}
      </div>
    </div>
  );
}

export function TitleBar() {
  const [max, setMax] = useState(false);
  const sidebar = useApp((s) => s.sidebar);
  const compact = useApp((s) => s.compact);
  const title = useApp((s) => (s.view === "library" ? "Models" : s.view === "brain" ? "Brain" : s.view === "projects" ? "Projects" : s.view === "project" ? s.settings.projects.find((p) => p.id === s.projectId)?.name || "Project" : s.chat?.title || ""));
  const panelable = useApp((s) => s.view === "chat" && !!s.chat && (s.chat.messages.length > 0 || !!s.chat.archive?.length));
  const panel = useApp((s) => s.settings.sidePanel);
  const isChat = useApp((s) => s.view === "chat");
  const dock = useApp((s) => s.dock);
  useEffect(() => {
    if (!bridge) return;
    bridge.win.isMaximized().then(setMax);
    return bridge.win.onState((s) => setMax(s.maximized));
  }, []);
  return (
    <div className="titlebar">
      {(!sidebar || compact) && (
        <button className="ghost-btn nodrag" title="Show sidebar" onClick={() => toggleSidebar(true)}>
          <Icon name="sidebar" size={16} />
        </button>
      )}
      {compact && (
        <button className="ghost-btn nodrag tb-new" title="New chat" onClick={() => startNewChat()}>
          <Icon name="plus" size={16} />
        </button>
      )}
      {isChat ? <ChatTitle /> : <div className="title-text">{title}</div>}
      {isChat && (
        <div className="tb-tools nodrag">
          {panelable && (
            <button className={`ghost-btn rail-toggle ${panel && !dock ? "on" : ""}`} title={panel ? "Hide session rail" : "Show session rail"} onClick={() => (dock ? (app.set({ dock: null }), setSettings({ sidePanel: true })) : setSettings({ sidePanel: !panel }))}>
              <Icon name="rail" size={16} />
            </button>
          )}
          {panelable && <span className="tb-sep" aria-hidden="true" />}
          {DOCKS.map((d) => (
            <button key={d.id} className={`ghost-btn dock-btn ${dock === d.id ? "on" : ""}`} title={d.label} onClick={() => app.set({ dock: dock === d.id ? null : d.id })}>
              <Icon name={d.icon} size={16} />
            </button>
          ))}
          <ViewOptions />
        </div>
      )}
      <WinBtns max={max} />
    </div>
  );
}

/** Minimize, maximize, close (also on views that cover the title bar). */
export function WinBtns({ max: known }: { max?: boolean }) {
  const [own, setOwn] = useState(false);
  useEffect(() => {
    if (known !== undefined || !bridge) return;
    bridge.win.isMaximized().then(setOwn).catch(() => {});
    return bridge.win.onState((st) => setOwn(st.maximized));
  }, [known === undefined]);
  const max = known ?? own;
  if (!bridge) return null;
  return (
    <div className="win-btns nodrag">
      <button onClick={() => bridge!.win.minimize()} title="Minimize">
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M0 5.5h10" stroke="currentColor" /></svg>
      </button>
      <button onClick={() => bridge!.win.maximize()} title={max ? "Restore" : "Maximize"}>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" aria-hidden="true">
          {max ? <path d="M2.5 2.5V1.5a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H7.5M1.5 2.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z" /> : <rect x=".5" y=".5" width="9" height="9" rx="1" />}
        </svg>
      </button>
      <button className="close" onClick={() => bridge!.win.close()} title="Close">
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M.5.5l9 9M9.5.5l-9 9" stroke="currentColor" strokeLinecap="round" /></svg>
      </button>
    </div>
  );
}
