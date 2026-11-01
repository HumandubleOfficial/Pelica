// Settings → Phone: turn on the page for phones, and pair one by scanning the
// QR code (the address on this PC's network, with the pairing key).
import QRCode from "qrcode";
import { useEffect, useState } from "react";
import { bridge, type PhoneStatus } from "../lib/bridge";
import { setSettings, toast, useApp } from "../state";

export function PhoneTab({ Row, Toggle }: { Row: (p: { title: string; desc?: React.ReactNode; children: React.ReactNode }) => React.ReactElement; Toggle: (p: { on: boolean; set: (v: boolean) => void }) => React.ReactElement }) {
  const on = useApp((s) => s.settings.phoneOn);
  const port = useApp((s) => s.settings.phonePort);
  const [st, setSt] = useState<PhoneStatus | null>(null);
  const [via, setVia] = useState<"lan" | "tailscale">("lan");
  const [svg, setSvg] = useState("");
  const [portText, setPortText] = useState(String(port));
  useEffect(() => {
    if (!bridge) return;
    let live = true;
    const tick = () => bridge!.phone.status().then((s) => live && setSt(s)).catch(() => {});
    tick();
    const t = setInterval(tick, 2000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [on, port]);
  const host = via === "tailscale" && st?.tailscale ? st.tailscale : st?.lan[0];
  const url = st?.on && host && st.key ? `http://${host}:${st.port}/?k=${st.key}` : "";
  useEffect(() => {
    if (!url) return setSvg("");
    QRCode.toString(url, { type: "svg", margin: 1, errorCorrectionLevel: "M", color: { dark: "#1f1e1c", light: "#ffffff" } })
      .then(setSvg)
      .catch(() => setSvg(""));
  }, [url]);
  return (
    <>
      <Row title="Pelica on your phone" desc="Chat with Pelica from your phone's browser while this PC runs the model. Your phone and this PC need to be on the same Wi-Fi (or both on Tailscale). Nothing is sent over the internet.">
        <Toggle on={on} set={(v) => setSettings({ phoneOn: v })} />
      </Row>
      {on && (
        <>
          {st?.error ? (
            <div className="set-row">
              <span className="danger-text">{st.error}</span>
            </div>
          ) : !st?.lan.length && !st?.tailscale ? (
            <div className="set-row muted">This PC isn't on a network right now.</div>
          ) : (
            <div className="phone-pair">
              <div className="phone-qr" dangerouslySetInnerHTML={{ __html: svg }} />
              <div className="phone-steps">
                <b>Scan this with your phone's camera</b>
                <ol>
                  <li>Open the link it shows. The page works in any browser.</li>
                  <li>To keep it, use the browser's Add to Home Screen.</li>
                  <li>If Windows asks whether Pelica may use the network, allow it on private networks.</li>
                </ol>
                {st?.tailscale && st.lan.length > 0 && (
                  <div className="seg phone-via">
                    <button className={via === "lan" ? "sel" : ""} onClick={() => setVia("lan")}>
                      Same Wi-Fi
                    </button>
                    <button className={via === "tailscale" ? "sel" : ""} onClick={() => setVia("tailscale")}>
                      Tailscale (anywhere)
                    </button>
                  </div>
                )}
                <div className="muted small">
                  {host}:{st?.port} · {st?.phones ? `${st.phones} phone${st.phones === 1 ? "" : "s"} connected` : "no phone connected"}
                </div>
              </div>
            </div>
          )}
          <Row title="Port" desc="The phone page's port on this PC.">
            <input
              className="text-input narrow"
              value={portText}
              onChange={(e) => setPortText(e.target.value.replace(/\D/g, "").slice(0, 5))}
              onBlur={() => {
                const n = Number(portText);
                if (n >= 1024 && n <= 65535) setSettings({ phonePort: n });
                else setPortText(String(port));
              }}
            />
          </Row>
          <Row title="Paired phones" desc="Anyone with the QR code's link can use Pelica from this network. A new code stops every phone paired before; scan it again on yours.">
            <button
              className="btn"
              onClick={async () => {
                setSt(await bridge!.phone.newKey());
                toast("New code made. Phones paired before need to scan it again.");
              }}
            >
              New code
            </button>
          </Row>
        </>
      )}
    </>
  );
}
