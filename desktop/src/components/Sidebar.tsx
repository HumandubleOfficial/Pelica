// The left sidebar (as in the Claude app): new chat, search, sections,
// pinned and recent chats, the user chip with settings.
import { useEffect, useRef, useState } from "react";
import { app, deleteChat, openChat, persist, setSettings, startNewChat, useApp } from "../state";
import { bridge } from "../lib/bridge";
import { copyChat, exportChat } from "../lib/exportChat";
import { Icon, Logo } from "./Icons";

function ChatRow({ id, title, pinned, active, folder }: { id: string; title: string; pinned: boolean; active: boolean; folder: string | null }) {
  const live = useApp((s) => s.liveId === id);
  const [menu, setMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setMenu(false);
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, []);
  async function pin() {
    setMenu(false);
    const c = bridge ? await bridge.chats.load(id).catch(() => null) : app.get().chat?.id === id ? app.get().chat : null;
    if (!c) return;
    const n = { ...c, pinned: !pinned };
    if (app.get().chat?.id === id) app.set({ chat: n });
    persist(n);
  }
  async function rename(t: string) {
    setEditing(false);
    if (!t.trim() || t === title) return;
    const c = bridge ? await bridge.chats.load(id).catch(() => null) : app.get().chat;
    if (!c) return;
    const n = { ...c, title: t.trim() };
    if (app.get().chat?.id === id) app.set({ chat: n });
    persist(n);
  }
  return (
    <div className={`chat-row ${active ? "active" : ""} ${live ? "live" : ""}`} ref={ref}>
      {editing ? (
        <input className="chat-row-edit" autoFocus defaultValue={title} onBlur={(e) => rename(e.target.value)} onKeyDown={(e) => (e.key === "Enter" ? rename(e.currentTarget.value) : e.key === "Escape" && setEditing(false))} />
      ) : (
        <button className="chat-row-main" onClick={() => openChat(id)} onDoubleClick={() => setEditing(true)} title={folder ? `${title}\n${folder}` : title}>
          {live && <span className="live-dot" title="Pelica is replying here" />}
          <span className="chat-row-title">{title}</span>
          {folder && <span className="chat-row-folder">{folder.split(/[\\/]/).filter(Boolean).pop()}</span>}
        </button>
      )}
      <button className="chat-row-more" onClick={() => setMenu(!menu)}>
        <Icon name="more" size={14} />
      </button>
      {menu && (
        <div className="popover row-menu">
          <button onClick={pin}>
            <Icon name="pin" size={13} /> {pinned ? "Unpin" : "Pin"}
          </button>
          <button onClick={() => (setMenu(false), setEditing(true))}>
            <Icon name="edit" size={13} /> Rename
          </button>
          <button onClick={() => (setMenu(false), exportChat(id))}>
            <Icon name="download" size={13} /> Export as Markdown…
          </button>
          <button onClick={() => (setMenu(false), copyChat(id))}>
            <Icon name="copy" size={13} /> Copy as Markdown
          </button>
          <button className="danger" onClick={() => (setMenu(false), deleteChat(id))}>
            <Icon name="trash" size={13} /> Delete
          </button>
        </div>
      )}
    </div>
  );
}

export function Sidebar() {
  const chats = useApp((s) => s.chats);
  const cur = useApp((s) => s.chat?.id);
  const view = useApp((s) => s.view);
  const name = useApp((s) => s.settings.userName);
  const search = useApp((s) => s.search);
  const [searching, setSearching] = useState(false);
  const q = search.trim().toLowerCase();
  const shown = q ? chats.filter((c) => c.title.toLowerCase().includes(q)) : chats;
  const pinned = shown.filter((c) => c.pinned);
  const recent = shown.filter((c) => !c.pinned);
  const nav = (label: string, icon: string, onClick: () => void, active = false) => (
    <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}>
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </button>
  );
  return (
    <aside className="sidebar">
      <div className="side-top">
        <div className="brand">
          <Logo size={22} />
          <span>Pelica</span>
        </div>
        <button className="ghost-btn" title="Hide sidebar" onClick={() => app.set({ sidebar: false })}>
          <Icon name="sidebar" size={16} />
        </button>
      </div>
      <nav className="side-nav">
        {searching ? (
          <div className="side-search">
            <Icon name="search" size={15} />
            <input autoFocus value={search} placeholder="Search chats" onChange={(e) => app.set({ search: e.target.value })} onBlur={() => !search && setSearching(false)} onKeyDown={(e) => e.key === "Escape" && (app.set({ search: "" }), setSearching(false))} />
          </div>
        ) : (
          nav("Search", "search", () => setSearching(true))
        )}
        {nav("New chat", "plus", startNewChat)}
        {nav("Models", "library", () => app.set({ view: "library" }), view === "library")}
        {nav("Brain", "brain", () => app.set({ view: "brain" }), view === "brain")}
      </nav>
      <div className="side-lists">
        {pinned.length > 0 && <div className="side-label">Pinned</div>}
        {pinned.map((c) => (
          <ChatRow key={c.id} id={c.id} title={c.title} pinned active={c.id === cur && view === "chat"} folder={c.folder} />
        ))}
        {recent.length > 0 && <div className="side-label">Recents</div>}
        {recent.map((c) => (
          <ChatRow key={c.id} id={c.id} title={c.title} pinned={false} active={c.id === cur && view === "chat"} folder={c.folder} />
        ))}
        {!chats.length && <div className="side-empty">Your chats will show up here.</div>}
      </div>
      <button className="user-chip" onClick={() => app.set({ settingsOpen: true, settingsTab: "General" })}>
        <span className="avatar">{(name || "You").slice(0, 1).toUpperCase()}</span>
        <span className="user-name">
          {name || "You"}
          <small>Local · private</small>
        </span>
        <Icon name="settings" size={15} />
      </button>
    </aside>
  );
}

export function TitleBar() {
  const [max, setMax] = useState(false);
  const sidebar = useApp((s) => s.sidebar);
  const title = useApp((s) => (s.view === "library" ? "Models" : s.view === "brain" ? "Brain" : s.chat?.title || ""));
  const panelable = useApp((s) => s.view === "chat" && !!s.chat && (s.chat.messages.length > 0 || !!s.chat.archive?.length));
  const panel = useApp((s) => s.settings.sidePanel);
  useEffect(() => {
    if (!bridge) return;
    bridge.win.isMaximized().then(setMax);
    return bridge.win.onState((s) => setMax(s.maximized));
  }, []);
  return (
    <div className="titlebar">
      {!sidebar && (
        <button className="ghost-btn nodrag" title="Show sidebar" onClick={() => app.set({ sidebar: true })}>
          <Icon name="sidebar" size={16} />
        </button>
      )}
      <div className="title-text">{title !== "New chat" ? title : ""}</div>
      {panelable && (
        <button className={`ghost-btn nodrag panel-toggle ${panel ? "on" : ""}`} title={panel ? "Hide the side panel" : "Show the side panel (environment, context, connectors, outputs)"} onClick={() => setSettings({ sidePanel: !panel })}>
          <Icon name="panel" size={16} />
        </button>
      )}
      {bridge && (
        <div className="win-btns nodrag">
          <button onClick={() => bridge!.win.minimize()} title="Minimize">
            <Icon name="minimize" size={14} />
          </button>
          <button onClick={() => bridge!.win.maximize()} title={max ? "Restore" : "Maximize"}>
            <Icon name={max ? "restore" : "maximize"} size={12} />
          </button>
          <button className="close" onClick={() => bridge!.win.close()} title="Close">
            <Icon name="x" size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
