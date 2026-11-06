// Settings → Devices: other computers lending their RAM to the model through
// `pelica bridge`. The list lives in settings (the engine gets it as
// PELICA_DEVICES at start) and every change is sent to the engine at once.
import { useEffect, useRef, useState } from "react";
import { Fold } from "./Fold";
import { bridge } from "../lib/bridge";
import { engine, type Device } from "../lib/engine";
import { setSettings, useApp } from "../state";
import { Icon } from "./Icons";

const STATE: Record<string, string> = { connected: "Connected", connecting: "Connecting…", waiting: "Waiting for a model", error: "Not reachable" };
const BRIDGE_CMD = "pelica.exe bridge --ram-gb 8";

export function DevicesTab() {
  const list = useApp((x) => x.settings.devices);
  const [live, setLive] = useState<{ active: boolean; devices: Device[] } | null>(null);
  const [addr, setAddr] = useState("");
  const [code, setCode] = useState("");
  const [found, setFound] = useState<{ addr: string; lend_gb: number; name: string }[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [exe, setExe] = useState("");
  const codeRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bridge?.app.info().then((i) => setExe(i.pelica)).catch(() => {});
    let on = true;
    const tick = () => engine.devices().then((v) => on && setLive(v)).catch(() => on && setLive(null));
    tick();
    const t = setInterval(tick, 2000);
    return () => ((on = false), clearInterval(t));
  }, []);

  function save(next: { addr: string; code: string }[]) {
    setSettings({ devices: next });
    engine.setDevices(next).then(setLive).catch(() => {});
  }
  function add() {
    const a = addr.trim();
    if (!a) return;
    save([...list.filter((d) => d.addr !== a), { addr: a, code: code.trim() }]);
    setAddr("");
    setCode("");
  }
  async function scan() {
    setScanning(true);
    try {
      setFound(await engine.scanDevices());
    } catch {
      setFound([]);
    }
    setScanning(false);
  }

  return (
    <>
      <p className="set-desc">
        Another computer on your network can lend the model its RAM. Pelica keeps some of the model's experts there and has that computer work them, so less has to come off the drive for each token. A wired connection works best. For now this helps DeepSeek V4 and Qwen3.8 models, which are too big for this PC's VRAM and RAM. Both computers need the same Pelica version.
      </p>

      {list.length > 0 && (
        <div className="dev-list">
          {list.map((d) => {
            const s = live?.devices.find((x) => x.addr === d.addr);
            const state = s?.state || "waiting";
            return (
              <div key={d.addr} className="dev-row">
                <span className={`dev-dot ${state}`} />
                <div className="dev-main">
                  <div className="dev-name">
                    {s?.name || d.addr}
                    {s?.name && <span className="muted small"> {d.addr}</span>}
                  </div>
                  <div className="muted small">
                    {STATE[state] || state}
                    {state === "connected" && s && (
                      <>
                        {" "}
                        · {(s.used_gb || 0).toFixed(1)} of {(s.lend_gb || 0).toFixed(1)} GB · {s.experts} experts · {s.kernel}, {s.threads} threads
                        {s.runs ? ` · ${(s.avg_ms || 0).toFixed(1)} ms a layer` : ""}
                      </>
                    )}
                  </div>
                  {state === "error" && s?.error && <div className="err-text small">{s.error}</div>}
                </div>
                <button className="ghost-btn" title="Remove" onClick={() => save(list.filter((x) => x.addr !== d.addr))}>
                  <Icon name="x" size={15} />
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="set-section">Add a device</div>
      <div className="dev-add">
        <input className="text-input" value={addr} onChange={(e) => setAddr(e.target.value)} placeholder="Address, e.g. 192.168.1.20" spellCheck={false} onKeyDown={(e) => e.key === "Enter" && codeRef.current?.focus()} />
        <input ref={codeRef} className="text-input dev-code" value={code} onChange={(e) => setCode(e.target.value.replace(/\s/g, "").slice(0, 32))} placeholder="Pairing code" spellCheck={false} onKeyDown={(e) => e.key === "Enter" && add()} />
        <button className="btn small" onClick={add} disabled={!addr.trim()}>
          Add
        </button>
        <button className="btn ghost small" onClick={scan} disabled={scanning}>
          {scanning ? "Looking…" : "Find devices"}
        </button>
      </div>
      {found && (
        <div className="dev-found">
          {found.length === 0 && <span className="muted small">No devices answered. Check that `pelica bridge` is running on the other computer and that its firewall allows it.</span>}
          {found
            .filter((f) => !list.some((d) => d.addr === f.addr))
            .map((f) => (
              <button key={f.addr} className="dev-found-item" onClick={() => (setAddr(f.addr), setFound(null), setTimeout(() => codeRef.current?.focus()))}>
                <Icon name="monitor" size={14} />
                {f.name} <span className="muted small">{f.addr} · {f.lend_gb.toFixed(1)} GB</span>
              </button>
            ))}
        </div>
      )}

      <Fold title="How to set up another computer">
      <ol className="dev-steps">
        <li>
          Copy <code>pelica.exe</code> to the other computer (Windows 10 or 11, 64-bit).{" "}
          {exe && (
            <button className="link-btn" onClick={() => bridge?.sys.reveal(exe)}>
              Show it in Explorer
            </button>
          )}
        </li>
        <li>
          On that computer, open a terminal in that folder and run <code>{BRIDGE_CMD}</code>{" "}
          <button className="link-btn" onClick={() => bridge?.sys.clipWrite(BRIDGE_CMD)}>
            Copy
          </button>
          . Set <code>--ram-gb</code> to what it can spare: leave it at least 4 GB for itself.
        </li>
        <li>When Windows asks, allow Pelica on private networks.</li>
        <li>It shows its address and a 6-digit pairing code. Add them here, or use Find devices.</li>
      </ol>
      <p className="set-desc">
        Ethernet or a direct cable helps much more than Wi-Fi: each token sends a little data to every device and waits for the answer. If a device stops answering, its experts move back to this PC and the reply carries on.
      </p>
      <p className="set-desc">
        Phones and tablets can't lend memory. iOS stops apps that hold gigabytes in the background and has no way to run the bridge, and their Wi-Fi is too slow for the back-and-forth.
      </p>
      </Fold>
    </>
  );
}
