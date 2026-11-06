// Settings → Menus: pick a menu, and what it shows changes in a live picture of
// it. The list and the picture are the same thing: pointing at a row lights up
// the other side, and clicking either one hides or shows the item.
import { useState, type ReactNode } from "react";
import { MENUS, type MenuDef, type MenuEntry } from "../lib/menus";
import { setSettings, useApp } from "../state";
import { Icon } from "./Icons";

type Live = { shown: (id: string) => boolean; flip: (id: string) => void; hot: string; setHot: (id: string) => void };

/** Closes up when its item is hidden. */
function Fold({ off, children }: { off: boolean; children: ReactNode }) {
  return (
    <div className={`mm-fold ${off ? "off" : ""}`}>
      <div>{children}</div>
    </div>
  );
}

function Row({ i, live, icons }: { i: MenuEntry; live: Live; icons: boolean }) {
  const on = live.shown(i.id);
  return (
    <Fold off={!on}>
      <button
        className={`mi ${live.hot === i.id ? "open" : ""}`}
        tabIndex={on ? 0 : -1}
        title={`Hide ${i.label}`}
        onClick={() => live.flip(i.id)}
        onPointerEnter={() => live.setHot(i.id)}
        onPointerLeave={() => live.setHot("")}
      >
        {icons && <Icon name={i.icon} size={15} />}
        <span className="mi-text">
          <span className="mi-label">{i.label}</span>
          {i.desc && <span className="mi-desc">{i.desc}</span>}
        </span>
        {i.key && <span className="mi-key">{i.key}</span>}
        {i.sub && <Icon name="chevron" size={13} className="mi-more" />}
      </button>
    </Fold>
  );
}

/** A row that always stays, so it can't be clicked away. */
function Kept({ i, icons }: { i: MenuEntry; icons: boolean }) {
  return (
    <div className={`mi mm-kept ${i.danger ? "danger" : ""}`} title="This one always stays">
      {icons && <Icon name={i.icon} size={15} />}
      <span className="mi-text">
        <span className="mi-label">{i.label}</span>
      </span>
    </div>
  );
}

/** The menu itself, with the same lines between its groups the real one draws. */
function MenuBox({ m, live }: { m: MenuDef; live: Live }) {
  const icons = !m.plain;
  let since = false;
  const rows: ReactNode[] = [];
  m.items.forEach((i, n) => {
    rows.push(<Row key={i.id} i={i} live={live} icons={icons} />);
    if (live.shown(i.id)) since = true;
    if (!m.seps?.includes(i.id)) return;
    const after = m.items.slice(n + 1).some((x) => live.shown(x.id)) || !!m.kept;
    const line = since && after;
    if (line) since = false;
    rows.push(
      <Fold key={`${i.id}-line`} off={!line}>
        <div className="menu-sep" />
      </Fold>,
    );
  });
  const none = !m.kept && !m.items.some((i) => live.shown(i.id));
  return (
    <div className="popover menu mm-menu">
      {rows}
      {m.kept && <Kept i={m.kept} icons={icons} />}
      {none && <div className="mm-none">Everything here is hidden</div>}
    </div>
  );
}

function SidePicture({ m, live }: { m: MenuDef; live: Live }) {
  const row = (i: MenuEntry) => (
    <Fold key={i.id} off={!live.shown(i.id)}>
      {i.id === "tags" ? (
        <button className={`mm-tags ${live.hot === i.id ? "hot" : ""}`} title={`Hide ${i.label}`} onClick={() => live.flip(i.id)} onPointerEnter={() => live.setHot(i.id)} onPointerLeave={() => live.setHot("")}>
          <i style={{ background: "#8a7fd6" }} /> Food <i style={{ background: "#d6655a" }} /> Home <i style={{ background: "#5fae6c" }} /> Travel
        </button>
      ) : (
        <button className={`mm-nav ${live.hot === i.id ? "hot" : ""}`} title={`Hide ${i.label}`} onClick={() => live.flip(i.id)} onPointerEnter={() => live.setHot(i.id)} onPointerLeave={() => live.setHot("")}>
          <Icon name={i.icon} size={15} /> {i.label}
        </button>
      )}
    </Fold>
  );
  return (
    <div className="mm-side">
      <div className="mm-brand">
        <img src="logo.svg" alt="" /> Pelica
      </div>
      <div className="mm-nav mm-kept" title="This one always stays">
        <Icon name={m.kept!.icon} size={15} /> {m.kept!.label}
      </div>
      {m.items.map(row)}
      <div className="mm-lines">
        <i style={{ width: "72%" }} />
        <i style={{ width: "54%" }} />
      </div>
    </div>
  );
}

function ReplyPicture({ m, live }: { m: MenuDef; live: Live }) {
  return (
    <div className="mm-reply">
      <div className="mm-lines">
        <i style={{ width: "94%" }} />
        <i style={{ width: "100%" }} />
        <i style={{ width: "62%" }} />
      </div>
      <div className="mm-btns">
        {m.items.map((i) => {
          const on = live.shown(i.id);
          return (
            <button
              key={i.id}
              className={`mm-btn ${on ? "" : "off"} ${live.hot === i.id ? "hot" : ""}`}
              tabIndex={on ? 0 : -1}
              title={`Hide ${i.label}`}
              onClick={() => live.flip(i.id)}
              onPointerEnter={() => live.setHot(i.id)}
              onPointerLeave={() => live.setHot("")}
            >
              <Icon name={i.icon} size={14} />
            </button>
          );
        })}
        <span className="muted small">2 min ago</span>
      </div>
    </div>
  );
}

/** Where the menu opens, drawn around it. */
function Picture({ m, live }: { m: MenuDef; live: Live }) {
  if (m.id === "side") return <SidePicture m={m} live={live} />;
  if (m.id === "reply") return <ReplyPicture m={m} live={live} />;
  const box = <MenuBox m={m} live={live} />;
  if (m.id === "plus")
    return (
      <div className="mm-up">
        {box}
        <div className="mm-composer">
          <span className="mm-round on">
            <Icon name="plus" size={15} />
          </span>
          <span className="muted">Write a message</span>
          <span className="mm-round send">
            <Icon name="send" size={14} />
          </span>
        </div>
      </div>
    );
  if (m.id === "user")
    return (
      <div className="mm-up">
        {box}
        <div className="mm-user">
          <span className="mm-avatar">Y</span> You <span className="muted">· Local</span>
        </div>
      </div>
    );
  if (m.id === "view")
    return (
      <div className="mm-down right">
        <span className="mm-round on">
          <Icon name="more" size={15} />
        </span>
        {box}
      </div>
    );
  return (
    <div className="mm-down">
      <div className="mm-chat">
        <span>Sourdough questions</span>
        <Icon name="more" size={15} />
      </div>
      {box}
    </div>
  );
}

export function MenusTab() {
  const hide = useApp((x) => x.settings.menuHide);
  const [pick, setPick] = useState(MENUS[0].id);
  const [hot, setHot] = useState("");
  const m = MENUS.find((x) => x.id === pick)!;
  const full = (id: string) => `${m.id}.${id}`;
  const shown = (id: string) => !hide.includes(full(id));
  const flip = (id: string) => {
    const f = full(id);
    setSettings({ menuHide: hide.includes(f) ? hide.filter((x) => x !== f) : [...hide, f] });
  };
  const live: Live = { shown, flip, hot, setHot };
  const hiddenHere = m.items.filter((i) => !shown(i.id));
  return (
    <div className="mm">
      <div className="mm-head">
        <p className="set-intro">Click anything in the picture to take it off the menu. Nothing gets turned off: it only stops showing there.</p>
        <button className="btn small" disabled={!hide.length} onClick={() => setSettings({ menuHide: [] })}>
          Show everything
        </button>
      </div>
      <div className="seg mm-tabs">
        {MENUS.map((x) => {
          const off = x.items.filter((i) => hide.includes(`${x.id}.${i.id}`)).length;
          return (
            <button key={x.id} className={x.id === pick ? "sel" : ""} onClick={() => (setPick(x.id), setHot(""))}>
              <Icon name={x.icon} size={16} />
              <span className="mm-tab-name">{x.short}</span>
              <span className={`mm-tab-count ${off ? "some" : ""}`}>{off ? `${off} hidden` : "All shown"}</span>
            </button>
          );
        })}
      </div>
      <div className="mm-split">
        <div className="mm-left">
          <div className="mm-title">
            <b>{m.title}</b>
            {hiddenHere.length > 0 && (
              <button className="link-btn small" onClick={() => setSettings({ menuHide: hide.filter((x) => !x.startsWith(`${m.id}.`)) })}>
                Show all
              </button>
            )}
          </div>
          <p className="mm-where">{m.desc}</p>
          <div className="mm-list">
            {m.items.map((i) => {
              const on = shown(i.id);
              return (
                <button
                  key={i.id}
                  className={`mm-row ${on ? "" : "off"} ${hot === i.id ? "hot" : ""}`}
                  role="switch"
                  aria-checked={on}
                  onClick={() => flip(i.id)}
                  onPointerEnter={() => setHot(i.id)}
                  onPointerLeave={() => setHot("")}
                >
                  <Icon name={i.icon} size={15} />
                  <span className="mm-row-name">{i.label}</span>
                  <span className={`toggle ${on ? "on" : ""}`} />
                </button>
              );
            })}
          </div>
        </div>
        <div className="mm-stage">
          <Picture m={m} live={live} />
        </div>
      </div>
    </div>
  );
}
