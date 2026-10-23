// A shell command from a reply, in a box with a play button: runs in the
// chat's folder when the user presses it, with a small live output.
import { useEffect, useRef, useState } from "react";
import { bridge } from "../lib/bridge";
import { uid, useApp } from "../state";
import { pickFolder } from "../agent/commands";
import { Icon } from "./Icons";

export function RunBox({ command, lang }: { command: string; lang: string }) {
  const folder = useApp((s) => s.chat?.folder ?? null);
  const [state, setState] = useState<"idle" | "running" | "done">("idle");
  const [out, setOut] = useState("");
  const [code, setCode] = useState<number | null>(null);
  const [open, setOpen] = useState(true);
  const id = useRef("");
  const pre = useRef<HTMLPreElement>(null);
  const cmd = command.trim().split("\n").filter((l) => !/^\s*#/.test(l)).map((l) => l.replace(/^\s*[$>]\s+/, "")).join(" && ");

  useEffect(() => {
    if (pre.current) pre.current.scrollTop = pre.current.scrollHeight;
  }, [out]);

  async function play() {
    if (!bridge) return;
    if (!folder) return pickFolder();
    id.current = uid();
    setState("running");
    setOut("");
    setCode(null);
    setOpen(true);
    const off = bridge.cmd.onData((v) => v.id === id.current && setOut((o) => (o + v.data).slice(-30000)));
    try {
      // long-running servers keep going; the box shows their output
      const r = await bridge.cmd.run(id.current, folder, cmd, 24 * 3600 * 1000, false);
      setCode(r.code);
    } finally {
      off();
      setState("done");
    }
  }

  return (
    <div className={`runbox ${state}`}>
      <div className="runbox-head">
        <button className={`play ${state === "running" ? "stop" : ""}`} title={state === "running" ? "Stop" : folder ? `Run in ${folder}` : "Choose a folder to run this in"} onClick={() => (state === "running" ? bridge?.cmd.kill(id.current) : play())} disabled={!bridge}>
          <Icon name={state === "running" ? "stop" : "play"} size={13} stroke={2.2} />
        </button>
        <code className="runbox-cmd">{command.trim()}</code>
        <span className="runbox-lang">{lang}</span>
        <button className="ghost-btn" title="Copy" onClick={() => navigator.clipboard.writeText(command.trim())}>
          <Icon name="copy" size={13} />
        </button>
        {state !== "idle" && (
          <button className="ghost-btn" onClick={() => setOpen(!open)} title={open ? "Hide output" : "Show output"}>
            <Icon name={open ? "down" : "chevron"} size={13} />
          </button>
        )}
      </div>
      {state !== "idle" && open && (
        <div className="runbox-out">
          <pre ref={pre}>{out || (state === "running" ? "Running…" : "(no output)")}</pre>
          <div className="runbox-status">
            {state === "running" ? (
              <>
                <span className="dot-pulse" /> running in {folder}
              </>
            ) : (
              <span className={code === 0 ? "ok" : "bad"}>{code === null ? "stopped" : `exit code ${code}`}</span>
            )}
            {state === "done" && out && (
              <button
                className="link-btn"
                onClick={() => window.dispatchEvent(new CustomEvent("pelica:compose", { detail: `${code !== 0 ? "This command failed" : "I ran"}:\n\`\`\`\n${cmd}\n\`\`\`\nOutput:\n\`\`\`\n${out.slice(-3000)}\n\`\`\`\n${code !== 0 ? "Please fix it." : ""}` }))}
              >
                {code !== 0 ? "Ask Pelica to fix" : "Send output to chat"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
