// The keyboard shortcuts sheet (Ctrl+/, /keys, the palette): every key the
// app answers to, in one place.
import { useEffect, useState } from "react";
import { bridge } from "../lib/bridge";
import { app, useApp } from "../state";
import { Icon } from "./Icons";

const keys = (k: string) =>
  k.split(" / ").map((alt, i) => (
    <span key={i} className="keys-alt">
      {i > 0 && <span className="muted"> or </span>}
      {alt.split("+").map((p, j) => (
        <kbd key={j}>{p}</kbd>
      ))}
    </span>
  ));

export function Shortcuts() {
  const open = useApp((s) => s.keys);
  const enter = useApp((s) => s.settings.sendOnEnter);
  const [hotkey, setHotkey] = useState("");
  useEffect(() => {
    if (!open) return;
    bridge?.app
      .info()
      .then((i) => setHotkey(i.hotkey))
      .catch(() => {});
    const f = (e: KeyboardEvent) => e.key === "Escape" && app.set({ keys: false });
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [open]);
  if (!open) return null;
  const groups: [string, [string, string][]][] = [
    [
      "Anywhere",
      [
        ["Command palette", "Ctrl+K"],
        ["New chat", "Ctrl+N"],
        ["Find in chat", "Ctrl+F"],
        ["Focus the message box", "Ctrl+L"],
        ["Show or hide the sidebar", "Ctrl+B"],
        ["Settings", "Ctrl+,"],
        ["Bigger / smaller text", "Ctrl+= / Ctrl+-"],
        ["Default text size", "Ctrl+0"],
        ["Your previous / next message in the chat", "Alt+↑ / Alt+↓"],
        ["This list", "Ctrl+/"],
        ...(hotkey ? ([["Quick ask, from any app", hotkey.replace(/CommandOrControl|CmdOrCtrl/g, "Ctrl")]] as [string, string][]) : []),
      ],
    ],
    [
      "Message box",
      [
        ["Send", enter ? "Enter" : "Ctrl+Enter"],
        ["New line", enter ? "Shift+Enter" : "Enter"],
        ["Your earlier messages", "↑ / ↓"],
        ["Stop the reply", "Esc"],
        ["Queue a message while Pelica writes", enter ? "Enter" : "Ctrl+Enter"],
        ["Next permission mode", "Shift+Tab"],
        ["Commands", "/"],
        ["Attach a file from the folder", "@"],
        ["A saved prompt", "/name"],
      ],
    ],
    [
      "Find in chat",
      [
        ["Next match", "Enter"],
        ["Previous match", "Shift+Enter"],
        ["Close", "Esc"],
      ],
    ],
  ];
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ keys: false })}>
      <div className="modal keys-sheet" role="dialog" aria-label="Keyboard shortcuts">
        <div className="whatsnew-head">
          <h2>Keyboard shortcuts</h2>
          <button className="ghost-btn" title="Close" onClick={() => app.set({ keys: false })}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="keys-body">
          {groups.map(([g, rows]) => (
            <section key={g}>
              <div className="set-section">{g}</div>
              {rows.map(([what, k]) => (
                <div key={what} className="keys-row">
                  <span>{what}</span>
                  <span className="keys-keys">{keys(k)}</span>
                </div>
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
