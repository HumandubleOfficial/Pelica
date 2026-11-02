// The left sidebar (as in the Claude app): new chat, search, sections,
// pinned and recent chats, the user chip with settings.
import { useEffect, useRef, useState } from "react";
import { app, deleteChat, forkChat, openChat, openProject, patchChat, searchMessages, setChatFolder, setSettings, startNewChat, toggleSidebar, updateChat, useApp } from "../state";
import type { SearchHit } from "../../electron/chatdata";
import { bridge } from "../lib/bridge";
import { copyChat, exportChat } from "../lib/exportChat";
import { compactNow } from "../agent/agent";
import { DOCKS } from "./Dock";
import { Icon, Logo } from "./Icons";
import { Menu, type MenuItem } from "./Menu";

const base = (p: string) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

/** What a chat's menu offers (the sidebar's "…" and right-click, and the
 *  title bar's chevron), as in Claude's. */
function chatMenu(id: string, o: { pinned: boolean; folder: string | null; project?: string; rename: () => void; title?: boolean }): MenuItem[] {
  const recent = app.get().settings.recentFolders;
  const projects = app.get().settings.projects;
  const font = app.get().settings.replyFont;
  const items: MenuItem[] = [];
  if (o.folder && bridge) {
    items.push({ label: "Open in", sub: [
      { label: "File Explorer", icon: "folder", on: () => bridge!.sys.open(o.folder!, "").catch((e: any) => app.set({ toast: e.message })) },
      ...(o.title ? [{ label: "Terminal", icon: "terminal", on: () => app.set({ dock: "terminal" }) }] : []),
    ] }, "sep");
  }
  items.push(
    { label: o.pinned ? "Unpin" : "Pin", key: "P", on: () => patchChat(id, (c) => ({ ...c, pinned: !o.pinned })) },
    { label: "Rename", key: "R", on: o.rename },
    { label: "Fork", key: "F", on: () => forkChat(id) },
    "sep",
  );
  items.push({ label: "Move to project", sub: [
    ...projects.map((p) => ({ label: p.name, icon: "layers", checked: p.id === o.project, on: () => patchChat(id, (c) => ({ ...c, project: p.id })) })),
    ...(projects.length ? (["sep"] as MenuItem[]) : []),
    { label: "New project…", icon: "plus", on: () => { const p = { id: Date.now().toString(36), name: "New project", instructions: "", files: [], folder: o.folder, created: Date.now() }; setSettings({ projects: [...projects, p] }); patchChat(id, (c) => ({ ...c, project: p.id })).then(() => openProject(p.id)); } },
    { label: "No project", icon: "x", disabled: !o.project, on: () => patchChat(id, (c) => ({ ...c, project: undefined })) },
  ] });
  if (!o.title) {
    items.push({ label: "Move to folder", sub: [
      ...recent.map((p) => ({ label: base(p), icon: "folder", checked: p === o.folder, on: () => (app.get().chat?.id === id ? setChatFolder(p) : patchChat(id, (c) => ({ ...c, folder: p }))) })),
      ...(recent.length ? (["sep"] as MenuItem[]) : []),
      { label: "No folder", icon: "x", disabled: !o.folder, on: () => patchChat(id, (c) => ({ ...c, folder: null })) },
    ] });
  } else {
    items.push({ label: "Reply font", sub: [
      { label: "Serif", checked: font === "serif", on: () => setSettings({ replyFont: "serif" }) },
      { label: "Sans serif", checked: font === "sans", on: () => setSettings({ replyFont: "sans" }) },
    ] });
  }
  items.push(
    { label: "Export", sub: [
      { label: "Copy as Markdown", icon: "copy", on: () => copyChat(id) },
      { label: "Save as Markdown…", icon: "download", on: () => exportChat(id) },
    ] },
    "sep",
    { label: "Delete", key: "D", danger: true, on: () => deleteChat(id) },
  );
  return items;
}

/** Renames a chat in place: the title turns into a box with its text selected. */
function RenameBox({ title, done, className }: { title: string; done: (t: string | null) => void; className: string }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => ref.current?.select(), []);
  return <input ref={ref} className={className} defaultValue={title} spellCheck={false} onBlur={(e) => done(e.target.value)} onKeyDown={(e) => (e.key === "Enter" ? e.currentTarget.blur() : e.key === "Escape" && done(null))} />;
}

function ChatRow({ id, title, pinned, active, folder, project, tag = true }: { id: string; title: string; pinned: boolean; active: boolean; folder: string | null; project?: string; tag?: boolean }) {
  const live = useApp((s) => s.liveId === id);
  const pname = useApp((s) => (project ? s.settings.projects.find((p) => p.id === project)?.name : undefined));
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const [editing, setEditing] = useState(false);
  const more = useRef<HTMLButtonElement>(null);
  function rename(t: string | null) {
    setEditing(false);
    if (t === null || !t.trim() || t.trim() === title) return;
    patchChat(id, (c) => ({ ...c, title: t.trim() }));
  }
  const openMenu = () => {
    const r = more.current!.getBoundingClientRect();
    setMenu(menu ? null : { x: r.left, y: r.bottom + 4 });
  };
  return (
    <div className={`chat-row ${active ? "active" : ""} ${live ? "live" : ""} ${menu ? "menu-open" : ""}`} onContextMenu={(e) => (e.preventDefault(), setMenu({ x: e.clientX, y: e.clientY }))}>
      {editing ? (
        <RenameBox className="chat-row-edit" title={title} done={rename} />
      ) : (
        <button className="chat-row-main" onClick={() => openChat(id)} onDoubleClick={() => setEditing(true)} title={folder ? `${title}\n${folder}` : title}>
          {live && <span className="live-dot" title="Pelica is replying here" />}
          <span className="chat-row-title">{title}</span>
          {pname ? <span className="chat-row-folder chat-row-project">{pname}</span> : folder && tag && <span className="chat-row-folder">{folder.split(/[\\/]/).filter(Boolean).pop()}</span>}
        </button>
      )}
      <button ref={more} className={`chat-row-more ${menu ? "open menu-open" : ""}`} title="More" onClick={openMenu}>
        <Icon name="more" size={16} />
      </button>
      {menu && <Menu at={menu} ignore={more} close={() => setMenu(null)} items={chatMenu(id, { pinned, folder, project, rename: () => setEditing(true) })} />}
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

export function Sidebar() {
  const chats = useApp((s) => s.chats);
  const cur = useApp((s) => s.chat?.id);
  const view = useApp((s) => s.view);
  const brainMenu = useApp((s) => s.settings.brainMenu);
  const acct = useApp((s) => s.account?.profile);
  const syncOn = useApp((s) => s.settings.syncOn);
  const name = useApp((s) => s.settings.userName);
  const search = useApp((s) => s.search);
  const [searching, setSearching] = useState(false);
  const q = search.trim().toLowerCase();
  const shown = q ? chats.filter((c) => c.title.toLowerCase().includes(q)) : chats;
  // chats with the words in their messages (those already listed by title left out)
  const [hits, setHits] = useState<SearchHit[]>([]);
  useEffect(() => {
    if (q.length < 2) return setHits([]);
    const t = setTimeout(() => searchMessages(q).then(setHits), 200);
    return () => clearTimeout(t);
  }, [q]);
  const msgHits = hits.filter((h) => !shown.some((c) => c.id === h.id));
  const pinned = shown.filter((c) => c.pinned);
  const recent = shown.filter((c) => !c.pinned);
  const folded = useApp((s) => s.settings.sideFolded);
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
        {searching ? (
          <div className="side-search">
            <Icon name="search" size={15} />
            <input autoFocus value={search} placeholder="Search chats" onChange={(e) => app.set({ search: e.target.value })} onBlur={() => !search && setSearching(false)} onKeyDown={(e) => e.key === "Escape" && (app.set({ search: "" }), setSearching(false))} />
          </div>
        ) : (
          nav("Search", "search", () => setSearching(true))
        )}
        {nav("New chat", "plus", () => startNewChat())}
        {nav("Projects", "layers", () => app.set({ view: "projects" }), view === "projects" || view === "project")}
        {nav("Models", "library", () => app.set({ view: "library" }), view === "library")}
        {brainMenu && nav("Brain", "brain", () => app.set({ view: "brain" }), view === "brain")}
      </nav>
      <div className="side-lists">
        {pinned.length > 0 && <div className="side-label">Pinned</div>}
        {pinned.map((c) => (
          <ChatRow key={c.id} id={c.id} title={c.title} pinned active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} />
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
              {!shut && list.map((c) => <ChatRow key={c.id} id={c.id} title={c.title} pinned={false} active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} tag={false} />)}
            </div>
          );
        })}
        {loose.length > 0 && <div className="side-label">{q ? "Results" : groups.size ? "Chats" : "Recents"}</div>}
        {loose.map((c) => (
          <ChatRow key={c.id} id={c.id} title={c.title} pinned={false} active={c.id === cur && view === "chat"} folder={c.folder} project={c.project} />
        ))}
        {msgHits.length > 0 && <div className="side-label">In messages</div>}
        {msgHits.map((h) => (
          <MessageHit key={h.id} h={h} q={search.trim()} />
        ))}
        {q && !shown.length && !msgHits.length && <div className="side-empty">No chats mention “{search.trim()}”.</div>}
        {!chats.length && <div className="side-empty">Your chats will show up here.</div>}
      </div>
      <button className="user-chip" onClick={() => app.set({ settingsOpen: true, settingsTab: acct ? "Account" : "General" })}>
        {acct?.picture ? <img className="avatar" src={acct.picture} alt="" /> : <span className="avatar">{(name || acct?.name || "You").slice(0, 1).toUpperCase()}</span>}
        <span className="user-name">
          {name || acct?.name || "You"}
          <small>{acct ? (syncOn ? "Synced with Google" : "Signed in · not syncing") : "Local · private"}</small>
        </span>
        <Icon name="settings" size={15} />
      </button>
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
    { label: "Compact session", icon: "layers", disabled: !has || running, on: () => compactNow() },
    { label: "Plan", icon: "list", checked: planShown, on: () => (app.set({ dock: null }), setSettings({ sidePanel: true, panelHidden: planShown ? [...hidden, "plan"] : hidden.filter((x) => x !== "plan") })) },
    { label: "Copy as Markdown", icon: "copy", disabled: !has, on: () => copyChat() },
    { label: "Export chat…", icon: "download", disabled: !has, on: () => exportChat() },
    "sep",
    { label: "Keep computer awake", desc: "Only for this session", stay: true, right: <span className={`toggle ${awake ? "on" : ""}`} role="switch" aria-checked={awake} />, on: keepAwake },
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
        {open && !editing && <Menu className="title-pop" ignore={wrap} close={() => setOpen(false)} items={chatMenu(chat.id, { pinned: !!chat.pinned, folder: chat.folder, project: chat.project, rename: () => setEditing(true), title: true })} />}
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
