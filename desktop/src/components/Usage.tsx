// Settings → Usage: what your chats add up to (replies, tokens written, time)
// and how fast each model has been, counted from the stats every reply
// keeps. Nothing is tracked beyond what the saved chats already hold.
import { useEffect, useState } from "react";
import type { Usage } from "../../electron/chatdata";
import { loadUsage } from "../state";

const num = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}K` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(Math.round(n)));
const dur = (s: number) => (s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`);
const rate = (tok: number, s: number) => (s > 0.5 ? (tok / s).toFixed(tok / s < 10 ? 2 : 1) : "–");
const dayName = (d: string) => new Date(d + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });

export function UsageTab() {
  const [u, setU] = useState<Usage | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    loadUsage()
      .then(setU)
      .catch((e) => setErr(String(e?.message || e)));
  }, []);
  if (err) return <p className="set-desc">Couldn't read your chats: {err}</p>;
  if (!u) return <p className="set-desc">Adding up your chats…</p>;
  if (!u.replies) return <p className="set-desc">Nothing to show yet: usage is counted from the replies in your chats. Send a message and come back.</p>;
  const top = Math.max(1, ...u.days.map((d) => d.tokens));
  const cards: [string, string, string?][] = [
    ["Chats", num(u.chats)],
    ["Your messages", num(u.messages)],
    ["Replies", num(u.replies), "every step of a reply counts, tool calls too"],
    ["Tokens written", num(u.generated)],
    ["Time writing", dur(u.decodeS + u.prefillS), `${dur(u.prefillS)} reading prompts, ${dur(u.decodeS)} writing`],
    ["Average speed", `${rate(u.generated, u.decodeS)} tok/s`],
  ];
  return (
    <div className="usage">
      <p className="set-desc">
        Counted from the replies saved in your chats{u.since ? `, since ${new Date(u.since).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" })}` : ""}. Deleted chats aren't counted, and nothing else is recorded.
      </p>
      <div className="usage-cards">
        {cards.map(([k, v, tip]) => (
          <div key={k} className="usage-card" title={tip}>
            <span className="usage-v">{v}</span>
            <span className="usage-k">{k}</span>
          </div>
        ))}
      </div>
      <div className="set-section">Tokens written, last 30 days</div>
      <div className="usage-days">
        {u.days.map((d) => (
          <div key={d.day} className="usage-day" title={`${dayName(d.day)}: ${d.replies} replies, ${num(d.tokens)} tokens`}>
            <span className={`usage-bar ${d.tokens ? "" : "zero"}`} style={{ height: `${d.tokens ? Math.max(4, (d.tokens / top) * 100) : 0}%` }} />
          </div>
        ))}
      </div>
      <div className="usage-axis muted small">
        <span>{dayName(u.days[0].day)}</span>
        <span>Today</span>
      </div>
      <div className="set-section">By model</div>
      <div className="usage-table-wrap">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Replies</th>
              <th>Tokens</th>
              <th title="tokens written per second while writing">Speed</th>
              <th title="prompt tokens read per second">Prompt</th>
              <th title="median time to the first token">First token</th>
              <th title="the fastest moment of any reply">Fastest</th>
            </tr>
          </thead>
          <tbody>
            {u.models.map((m) => (
              <tr key={m.model}>
                <td className="usage-model" title={m.model}>
                  {m.model}
                </td>
                <td>{num(m.replies)}</td>
                <td>{num(m.generated)}</td>
                <td>{rate(m.generated, m.decodeS)}</td>
                <td>{rate(m.prompt, m.prefillS)}</td>
                <td>{m.firstS === null ? "–" : `${m.firstS.toFixed(m.firstS < 10 ? 1 : 0)}s`}</td>
                <td>{m.peak ? m.peak.toFixed(1) : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="set-desc">Speeds are tokens a second. Prompt speed counts only the part of each prompt that wasn't already read (Pelica reuses what it read for earlier turns).</p>
    </div>
  );
}
