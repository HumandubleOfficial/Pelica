// Pelica's site: scroll reveals, the app window's chat demo, the three-tier
// animation, the speed chart and the latest release from GitHub.
(() => {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s) => document.querySelector(s);

  // reveal sections as they scroll in (and grow the chart's bars)
  const seen = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), seen.unobserve(e.target))),
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );
  document.querySelectorAll(".reveal, #chart").forEach((el) => seen.observe(el));

  // the window flattens as it scrolls into view
  const win = $("#win");
  const tilt = () => {
    if (still || innerWidth < 640) return (win.style.transform = "none");
    const r = win.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, (innerHeight - r.top) / (innerHeight * 0.75)));
    win.style.transform = `rotateX(${(8 * (1 - p)).toFixed(2)}deg)`;
  };
  addEventListener("scroll", tilt, { passive: true });
  addEventListener("resize", tilt);
  tilt();

  // ---- the chat demo -------------------------------------------------------
  const scenes = [
    {
      model: "DeepSeek V4 Flash · 164 GB",
      folder: "~/photos",
      tps: 4.5,
      cps: 55,
      tiers: [100, 78, 64],
      q: "Rename the holiday photos by the date they were taken. Show me the plan first.",
      a: [
        { tool: "list_files ~/photos · 214 photos" },
        { tool: "read_exif · 211 with a date" },
        "Here's the plan: I'll rename each photo to the date and time it was taken, like <b>2026-07-14_0931.jpg</b>, keeping them in order. <b>3 photos</b> have no date, so I'll leave those as they are. Want me to go ahead?",
      ],
    },
    {
      model: "gpt-oss-20b · 12.3 GB",
      folder: "No folder",
      tps: 128,
      cps: 320,
      tiers: [84, 0, 0],
      q: "Why is the sky blue?",
      a: [
        "Sunlight is a mix of every colour. On its way through the air it bumps into tiny gas molecules, which scatter <b>short, blue wavelengths</b> far more than long red ones. That scattered blue reaches your eyes from every direction, so the whole sky glows blue. At sunset the light crosses much more air, the blue is scattered away first, and the <b>reds and oranges</b> are left.",
      ],
    },
    {
      model: "Qwen3.8 Flash Next · 94 GB",
      folder: "No folder",
      tps: 16.5,
      cps: 120,
      tiers: [100, 92, 38],
      q: "Plan 3 days in Lisbon on a budget.",
      a: [
        { tool: "web_search · lisbon budget tips" },
        "Here's a plan that keeps costs low: <b>Day 1</b>, Alfama and the castle, riding tram 28 early to skip the queues. <b>Day 2</b>, Belém by train for the monastery, the tower and pastéis de nata. <b>Day 3</b>, a day trip to Sintra; a 24-hour transport pass covers the train for under €12.",
      ],
    },
  ];
  const q = $("#w-q"),
    a = $("#w-a"),
    speed = $("#w-speed");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const pieces = (html) => html.match(/<[^>]+>|&[a-z]+;|[^<&]/g) || [];
  let running = false;

  async function play(s) {
    $("#w-model").textContent = s.model;
    $("#w-path").textContent = s.folder;
    $("#w-chip").textContent = s.folder;
    ["vram", "ram", "ssd"].forEach((t, i) => ($("#m-" + t).style.width = s.tiers[i] + "%"));
    q.textContent = s.q;
    a.innerHTML = "";
    speed.textContent = "thinking…";
    await sleep(700);
    let tick = setInterval(() => (speed.textContent = (s.tps * (0.92 + Math.random() * 0.14)).toFixed(s.tps < 20 ? 1 : 0) + " tok/s"), 260);
    for (const part of s.a) {
      if (typeof part === "object") {
        const t = document.createElement("div");
        t.className = "tool";
        t.innerHTML = `<i></i>${part.tool}`;
        t.style.marginBottom = "8px";
        t.style.marginRight = "8px";
        a.appendChild(t);
        await sleep(650);
        continue;
      }
      const p = document.createElement("div");
      p.style.marginTop = "6px";
      a.appendChild(p);
      const bits = pieces(part);
      let html = "";
      const caret = '<span class="caret"></span>';
      for (let i = 0; i < bits.length; ) {
        const n = Math.max(1, Math.round(s.cps / 60));
        for (let k = 0; k < n && i < bits.length; k++) html += bits[i++];
        p.innerHTML = html + caret;
        await sleep(1000 / 60);
      }
      p.innerHTML = html;
    }
    clearInterval(tick);
    speed.textContent = s.tps.toFixed(s.tps < 20 ? 1 : 0) + " tok/s";
    await sleep(3200);
  }

  async function loop() {
    if (running) return;
    running = true;
    if (still) {
      const s = scenes[1];
      $("#w-model").textContent = s.model;
      q.textContent = s.q;
      a.innerHTML = s.a[0];
      speed.textContent = s.tps + " tok/s";
      ["vram", "ram", "ssd"].forEach((t, i) => ($("#m-" + t).style.width = s.tiers[i] + "%"));
      return;
    }
    for (let i = 0; ; i = (i + 1) % scenes.length) await play(scenes[i]);
  }
  loop();

  // ---- the three tiers -----------------------------------------------------
  const cv = $("#tiers");
  const g = cv.getContext("2d");
  const W = cv.width,
    H = cv.height,
    COLS = 16,
    PAD = 24,
    CW = (W - PAD * 2) / COLS;
  const TIERS = [
    { name: "GPU", sub: "graphics memory", rows: 2, color: [242, 120, 75] },
    { name: "RAM", sub: "memory", rows: 4, color: [232, 180, 92] },
    { name: "SSD", sub: "your drive", rows: 5, color: [125, 143, 168] },
  ];
  let y = 0;
  TIERS.forEach((t) => {
    t.y = y;
    t.h = 60 + t.rows * 52 + 10;
    t.slots = new Array(t.rows * COLS).fill(null);
    t.share = [0.7, 0.2, 0.1][TIERS.indexOf(t)];
    y += t.h + 10;
  });
  const slotXY = (ti, i) => {
    const t = TIERS[ti];
    return [PAD + (i % COLS) * CW + 6, t.y + 60 + Math.floor(i / COLS) * 52];
  };
  const total = TIERS.reduce((n, t) => n + t.slots.length, 0);
  // popularity: a few favourites and a long tail, drifting over time
  const rank = [...Array(total).keys()].sort(() => Math.random() - 0.5);
  const experts = rank.map((r, id) => ({ id, rank: r, heat: 0, flash: 0, tier: 0, slot: 0, x: 0, y: 0 }));
  const pop = (e) => 1 / Math.pow(e.rank + 3, 1.35);
  // start roughly sorted, with some noise so there's work to do
  [...experts]
    .sort((p, q) => p.rank + Math.random() * 60 - (q.rank + Math.random() * 60))
    .forEach((e, k) => {
      let ti = 0,
        i = k;
      while (i >= TIERS[ti].slots.length) i -= TIERS[ti++].slots.length;
      e.tier = ti;
      e.slot = i;
      TIERS[ti].slots[i] = e;
      [e.x, e.y] = slotXY(ti, i);
    });

  function route() {
    // one token: 8 experts, picked by popularity
    const sum = experts.reduce((n, e) => n + pop(e), 0);
    const picked = new Set();
    while (picked.size < 8) {
      let r = Math.random() * sum;
      for (const e of experts) if ((r -= pop(e)) <= 0) (picked.add(e), (r = Infinity));
    }
    const hits = [0, 0, 0];
    experts.forEach((e) => (e.heat *= 0.94));
    picked.forEach((e) => ((e.heat += 1), (e.flash = 1), hits[e.tier]++));
    TIERS.forEach((t, i) => (t.share = t.share * 0.9 + (hits[i] / 8) * 0.1));
    // promote: a lower tier's hottest expert swaps with the tier above's coldest
    for (let ti = 1; ti < TIERS.length; ti++) {
      const up = TIERS[ti - 1].slots,
        down = TIERS[ti].slots;
      const cold = up.reduce((m, e) => (e.heat < m.heat ? e : m));
      const hot = down.reduce((m, e) => (e.heat > m.heat ? e : m));
      if (hot.heat > cold.heat * 1.25 + 0.15) {
        [hot.tier, cold.tier] = [cold.tier, hot.tier];
        [hot.slot, cold.slot] = [cold.slot, hot.slot];
        TIERS[hot.tier].slots[hot.slot] = hot;
        TIERS[cold.tier].slots[cold.slot] = cold;
      }
    }
    // the favourites drift
    if (Math.random() < 0.08) {
      for (let k = 0; k < 6; k++) {
        const p = experts[(Math.random() * total) | 0],
          q2 = experts[(Math.random() * total) | 0];
        [p.rank, q2.rank] = [q2.rank, p.rank];
      }
    }
  }

  const rr = (x, y2, w, h, r) => {
    g.beginPath();
    g.roundRect(x, y2, w, h, r);
  };
  function draw() {
    g.clearRect(0, 0, W, H);
    TIERS.forEach((t) => {
      const [r, gg, b] = t.color;
      rr(0.5, t.y + 0.5, W - 1, t.h - 1, 16);
      g.fillStyle = `rgba(${r},${gg},${b},0.045)`;
      g.fill();
      g.strokeStyle = "rgba(255,245,230,0.08)";
      g.lineWidth = 1;
      g.stroke();
      g.font = "600 30px Inter, Segoe UI, sans-serif";
      g.fillStyle = `rgb(${r},${gg},${b})`;
      g.fillText(t.name, PAD, t.y + 42);
      const w = g.measureText(t.name).width;
      g.font = "400 27px Inter, Segoe UI, sans-serif";
      g.fillStyle = "#8f897e";
      g.fillText(t.sub, PAD + w + 12, t.y + 42);
      const label = `${Math.round(t.share * 100)}% of reads`;
      g.textAlign = "right";
      g.font = "500 26px JetBrains Mono, Consolas, monospace";
      g.fillStyle = "#c9c3b8";
      g.fillText(label, W - PAD, t.y + 42);
      g.textAlign = "left";
    });
    const cw = CW - 12,
      ch = 40;
    for (const e of experts) {
      const [tx, ty] = slotXY(e.tier, e.slot);
      e.x += (tx - e.x) * 0.12;
      e.y += (ty - e.y) * 0.12;
      const [r, gg, b] = TIERS[e.tier].color;
      const glow = Math.min(1, e.heat / 2.5);
      rr(e.x, e.y, cw, ch, 8);
      g.fillStyle = `rgba(${r},${gg},${b},${0.1 + glow * 0.7})`;
      if (e.flash > 0.05) {
        g.shadowColor = `rgba(${r},${gg},${b},${e.flash})`;
        g.shadowBlur = 22 * e.flash;
      }
      g.fill();
      g.shadowBlur = 0;
      if (e.flash > 0.05) {
        g.strokeStyle = `rgba(255,245,230,${e.flash * 0.9})`;
        g.lineWidth = 2;
        g.stroke();
      }
      e.flash *= 0.9;
    }
  }

  let on = false,
    last = 0;
  function frame(t) {
    if (!on) return;
    if (t - last > 150) (route(), (last = t));
    draw();
    requestAnimationFrame(frame);
  }
  if (still) {
    for (let k = 0; k < 120; k++) route();
    experts.forEach((e) => ([e.x, e.y] = slotXY(e.tier, e.slot)));
    draw();
  } else {
    new IntersectionObserver(([e]) => {
      on = e.isIntersecting;
      if (on) requestAnimationFrame(frame);
    }).observe(cv);
    draw();
  }

  // ---- the speed chart's tooltips ------------------------------------------
  const tip = $("#tip");
  document.querySelectorAll(".bar").forEach((b) => {
    b.addEventListener("mousemove", (ev) => {
      tip.innerHTML = b.dataset.tip;
      tip.style.opacity = 1;
      const x = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 12);
      tip.style.left = x + "px";
      tip.style.top = ev.clientY + 16 + "px";
    });
    b.addEventListener("mouseleave", () => (tip.style.opacity = 0));
  });

  // ---- a decorative QR code ------------------------------------------------
  const qr = $("#qr");
  if (qr) {
    const c = qr.getContext("2d"),
      n = 25,
      s = qr.width / n;
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    c.fillStyle = "#1a1816";
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (rnd() < 0.48) c.fillRect(i * s, j * s, s, s);
    [
      [0, 0],
      [n - 7, 0],
      [0, n - 7],
    ].forEach(([x, y2]) => {
      c.fillStyle = "#f5f1ea";
      c.fillRect((x - 1) * s, (y2 - 1) * s, 9 * s, 9 * s);
      c.fillStyle = "#1a1816";
      c.fillRect(x * s, y2 * s, 7 * s, 7 * s);
      c.fillStyle = "#f5f1ea";
      c.fillRect((x + 1) * s, (y2 + 1) * s, 5 * s, 5 * s);
      c.fillStyle = "#1a1816";
      c.fillRect((x + 2) * s, (y2 + 2) * s, 3 * s, 3 * s);
    });
  }

  // ---- the latest release --------------------------------------------------
  fetch("https://api.github.com/repos/HumandubleOfficial/Pelica/releases/latest")
    .then((r) => (r.ok ? r.json() : null))
    .then((r) => {
      if (!r || !r.tag_name) return;
      $("#ver").textContent = r.tag_name;
      const title = (r.name || "").replace(/^(Pelica\s+)?v?\d+\.\d+(\.\d+)?:?\s*/i, "");
      if (title) $("#ver-text").textContent = title[0].toUpperCase() + title.slice(1);
      const exe = (r.assets || []).find((x) => /\.exe$/i.test(x.name));
      if (exe) $("#size").textContent = `${Math.round(exe.size / 1048576)} MB installer`;
    })
    .catch(() => {});
})();
