// Line icons (24-unit grid, drawn with the current color).
const P: Record<string, string> = {
  plus: "M12 5v14M5 12h14",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4",
  chat: "M4 5h16v11H8l-4 4z",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  globe: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18",
  brain: "M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a3 3 0 0 0-3-1zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1",
  library: "M4 4h4v16H4zM10 4h4v16h-4zM16 5l4 1-3 14-4-1z",
  settings: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
  send: "M12 19V5M5 12l7-7 7 7",
  stop: "M7 7h10v10H7z",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  check: "M5 12l5 5 9-10",
  x: "M6 6l12 12M18 6L6 18",
  chevron: "M9 6l6 6-6 6",
  down: "M6 9l6 6 6-6",
  play: "M8 5l11 7-11 7z",
  pause: "M8 5v14M16 5v14",
  file: "M14 3H6v18h12V7zM14 3v4h4",
  edit: "M4 20h4L19 9l-4-4L4 16zM14 6l4 4",
  terminal: "M4 5h16v14H4zM7 9l3 3-3 3M12 15h5",
  list: "M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  pin: "M9 4h6l-1 6 3 3H7l3-3zM12 13v7",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  back: "M15 6l-6 6 6 6",
  forward: "M9 6l6 6-6 6",
  reload: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  sidebar: "M4 4h16v16H4zM9 4v16",
  minimize: "M5 12h14",
  maximize: "M5 5h14v14H5z",
  restore: "M8 8h11v11H8zM5 16V5h11",
  cpu: "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4",
  spark: "M12 3v6M12 15v6M3 12h6M15 12h6M5.6 5.6l4.2 4.2M14.2 14.2l4.2 4.2M5.6 18.4l4.2-4.2M14.2 9.8l4.2-4.2",
  download: "M12 4v12M6 11l6 6 6-6M4 20h16",
  shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
  scan: "M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M7 12h10",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  bolt: "M13 3L5 13h6l-1 8 8-10h-6z",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z",
  plug: "M9 3v5M15 3v5M7 8h10v4a5 5 0 0 1-10 0zM12 17v4",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  palette: "M12 3a9 9 0 0 0 0 18c1 0 2-1 1.5-2-.6-1.2.3-2.5 1.5-2.5H18a3 3 0 0 0 3-3c0-5-4-10.5-9-10.5zM7.5 11h.01M10 7h.01M14.5 7h.01",
  gauge: "M12 14l4-5M4 18a9 9 0 1 1 16 0",
  wand: "M4 20L15 9M14 4v2M18 8h2M17 5l1.5-1.5M19 11l1 1",
  clip: "M20 11.5l-8.2 8.2a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4L15 7",
  clipboard: "M9 4h6v3H9zM9 5H6v16h12V5h-3M9 12h6M9 16h4",
  monitor: "M3 4h18v12H3zM8 20h8M12 16v4",
  eye: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  code: "M9 7l-5 5 5 5M15 7l5 5-5 5",
  bell: "M6 9a6 6 0 0 1 12 0c0 6 2 8 2 8H4s2-2 2-8zM10 21h4",
  command: "M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3z",
  keyboard: "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10",
  window: "M3 5h18v14H3zM3 9h18",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z",
  image: "M4 5h16v14H4zM4 16l4.5-4.5 3.5 3.5 2.5-2.5L20 18M15.5 9.5a1 1 0 1 0 0-.01",
  mic: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM5 11a7 7 0 0 0 14 0M12 18v3",
  panel: "M3 5h18v14H3zM15 5v14",
  enter: "M20 5v7a3 3 0 0 1-3 3H5M9 11l-4 4 4 4",
  compare:"M4 20V10M10 20V4M16 20v-7M22 20H2",
  lock: "M6 11h12v10H6zM8.5 11V7.5a3.5 3.5 0 0 1 7 0V11",
};

export function Icon({ name, size = 16, stroke = 1.75, className }: { name: string; size?: number; stroke?: number; className?: string }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={P[name] || P.info} />
    </svg>
  );
}

/** Pelica's mark: a pelican's head and bill in a warm tile. */
export function Logo({ size = 28, mono = false }: { size?: number; mono?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-label="Pelica">
      <defs>
        <linearGradient id="plg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffb26b" />
          <stop offset="1" stopColor="#e5533d" />
        </linearGradient>
      </defs>
      {!mono && <rect width="64" height="64" rx="18" fill="url(#plg)" />}
      <path d="M17 47c0-15 5-27 15-27 6 0 9 4 9.5 8L55 31.5c1 .4 1 1.6 0 2L41.5 36C40 44 33 48 26 51z" fill={mono ? "currentColor" : "#fff"} />
      <path d="M41.5 36c3 5 8.5 4 13.5-2.5" fill="none" stroke={mono ? "currentColor" : "#fff"} strokeWidth="2.6" strokeLinecap="round" opacity=".85" />
      <circle cx="33" cy="27" r="2.3" fill={mono ? "var(--bg)" : "#e5533d"} />
    </svg>
  );
}
