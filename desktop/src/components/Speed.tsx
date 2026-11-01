// Generation speed, alive: the tok/s number glides to each new reading (and
// brightens a moment when it goes up), and a small graph draws the speed
// moment to moment. Finished replies keep their graph in the stats line.
import { useEffect, useRef, useState } from "react";
import { useApp } from "../state";

/** A number that eases to its new value instead of jumping. */
export function TweenNum({ v, digits = 1, ms = 450 }: { v: number; digits?: number; ms?: number }) {
  const anim = useApp((s) => s.settings.animations);
  const [shown, setShown] = useState(v);
  const [dir, setDir] = useState<"" | "up" | "down">("");
  const from = useRef(v);
  const cur = useRef(v);
  useEffect(() => {
    if (!anim) return void ((cur.current = v), setShown(v));
    from.current = cur.current;
    const d = v - from.current;
    if (Math.abs(d) < 10 ** -digits / 2) return;
    setDir(d > 0 ? "up" : "down");
    const t0 = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      const e = 1 - (1 - k) ** 3;
      cur.current = from.current + d * e;
      setShown(cur.current);
      if (k < 1) raf = requestAnimationFrame(step);
      else setDir("");
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [v, anim]);
  return <span className={`tween ${dir}`}>{shown.toFixed(digits)}</span>;
}

/** Speed through a reply as a line with a soft fill; the last point glows
 *  while `live`. Scaled to its own fastest reading. */
export function SpeedGraph({ values, live = false, w = 64, h = 18, title }: { values?: number[]; live?: boolean; w?: number; h?: number; title?: string }) {
  if (!values || values.length < 2) return null;
  const max = Math.max(...values) * 1.1 || 1;
  const x = (i: number) => (i / (values.length - 1)) * (w - 3) + 1.5;
  const y = (v: number) => h - 2 - (v / max) * (h - 4);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const last = values.length - 1;
  return (
    <svg className={`speed-graph ${live ? "live" : ""}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-label={title || "speed"}>
      {title && <title>{title}</title>}
      <path className="sg-fill" d={`M${x(0)},${h} L${pts.join(" L")} L${x(last)},${h} Z`} />
      <polyline className="sg-line" points={pts.join(" ")} />
      <circle className="sg-dot" cx={x(last)} cy={y(values[last])} r={live ? 2 : 1.5} />
    </svg>
  );
}
