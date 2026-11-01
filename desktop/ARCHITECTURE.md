# Pelica Desktop — architecture and API contract

Pelica Desktop is an Electron + TypeScript app (`desktop/`) in front of the
Rust engine (`pelica.exe`). The split:

| Rust (`src/`) | TypeScript (`desktop/`) |
|---|---|
| loading models, inference, chat templates for every engine, tool-call parsing, token counts, engine stats | the window and UI, the agent loop, every tool (web, files, commands, MCP), approvals, compaction, CQS, chats storage |

The Electron main process starts `pelica.exe app --no-open --port <free> --ui <desktop/dist>`
and opens `http://127.0.0.1:<port>/` in a frameless `BrowserWindow`. The
Rust server serves the built UI from `--ui` (same origin as the API, so no
CORS). Without `--ui` it keeps serving the old `app.html`.

## Chat API v2 (Rust)

`POST /api/chat` keeps its v1 behaviour when `v` is absent. With `"v": 2` it is
a stateless, OpenAI-like completion that **stops at a tool call** and lets
the caller run the tool:

Request:
```json
{
  "v": 2,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "text before the call", "reasoning": "optional thinking",
     "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {"query": "..."}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "result text"}
  ],
  "tools": [{"name": "web_search", "description": "...", "parameters": { "type": "object", ... }}],
  "effort": "off|low|medium|high|max|on",
  "max_tokens": 4096
}
```
- The server renders the conversation with the loaded model's own format:
  - gguf MoE (DeepSeek V4 included, which runs on `src/ds4.rs` and reports
    `"engine": "moe"`, `"arch": "deepseek4"`): its Jinja chat template, with `tools` (OpenAI function schemas)
    when the template prints tools, else the tools described in the system
    message in the `<tool_call>{json}</tool_call>` format;
  - gpt-oss: harmony, tools as `functions` in the developer message, calls
    `to=functions.NAME`, results as `functions.NAME to=assistant` messages;
  - GLM (tiered): its template's tool format (or the words fallback).
- `system` content is used verbatim (the TS side owns all instructions,
  grounding text included). No server-side web tools in v2.
- Prefix reuse works across calls (the cache keeps the last rendered ids).

Server-sent events (`data: {...}\n\n`), in order:
| `t` | fields | meaning |
|---|---|---|
| `start` | `prompt`, `reused`, `ctx` | prompt tokens, how many came from the cache, context size |
| `prefill` | `done`, `total` | prompt reading progress |
| `think` | `d` | reasoning text |
| `text` | `d` | answer text (tool-call markup never appears here) |
| `stats` | `tps`, `generated`, `elapsed_s` | live speed, sent at most every 250 ms while decoding |
| `tool_call` | `id`, `name`, `arguments` (object) | the model called a tool; generation has stopped |
| `experts` | first `layers`, `experts`; then `hits`: flat `[layer, expert, picks, …]` | with `"experts": true` in the request (MoE): expert activity since the last event, for the brain map and the background |
| `done` | as v1, plus `finish`: `"stop"`, `"length"`, `"tool_call"`, `"stopped"` | end of the call |
| `error` | `error` | failure |

`POST /api/count` (extend): `{ "parts": {"system": "...", "messages": [...], "tools": [...]} }`
→ `{ "system": n, "messages": n, "tools": n, "total": n }` token counts with the
loaded tokenizer, for the context breakdown.

`POST /api/trace` `{ "text": "..." }` (MoE models): runs the text (at most
384 tokens) through the model's prompt pass and returns where each token
went: `{ "tokens": [str], "layers": L, "experts": E, "routes":
[[[expert, weight], …top-k] per layer] per token }`. The plan kernel writes
the routes into the engine's state buffer as it routes, so the trace is
the real routing, not a re-computation. It clears the prefix cache.

`GET /api/status` (extend): add `"tools": true` once v2 is available,
`"engine": "moe" | "oss" | "tiered"`, and `"agent": bool` once a model is
loaded: whether it was trained for tool calls (its chat template takes
tools; gpt-oss and GLM always). The app warns when it's false and a folder
is chosen, since such models describe tasks instead of doing them.

## TypeScript side

- `desktop/electron/` main + preload: window (frameless, custom title bar),
  spawning `pelica.exe`, folder picker, filesystem and command execution
  (sandboxed to the chosen workspace), links handed to the default browser,
  IPC.
- `desktop/src/` renderer (React + Vite): Claude-desktop-style UI (sidebar:
  Search, New chat, Models, Brain, Pinned, Recents with each chat's folder
  and a live dot on the chat being replied in; centered
  greeting + composer; settings modal with left nav), streaming markdown
  rendered live (no raw `**`), live tok/s, tool cards (searched / read /
  created file / edited file with diff / ran command with output), run boxes
  with a play button and mini output, context ring with a breakdown
  (system, messages, tools, MCP), slash commands (`/compact`,
  `/autocompact`, `/clear`, `/model`, `/context`, `/folder`, `/cqs`, ...).
- Agent loop (`desktop/src/agent/`): calls v2 until `finish != "tool_call"`,
  runs tools, enforces permissions, up to a high step limit for long tasks
  with a todo list tool. It works on its own copy of the chat (`beginLive` /
  `updateLive` / `endLive` in `state.ts`, `liveId` in the store), so the user
  can open other chats or start new ones while a reply is written; the other
  chats can be read and typed in, and send once the engine is free (one
  reply at a time: the engine answers one request).
- No model loads at start: main passes `--no-autoload` (without it the
  engine loads the first .gguf it finds), and adds `--model lastModel` only
  when `settings.loadOnStart` is on (Settings → Models, off by default).
- The Brain's map grows grander with the model's total parameters
  (`grandeur()` in `NeuralMap.tsx`, 0 at 2B to 1 at 500B on a log scale): a
  connection web from ~4B, drifting particles, a denser and brighter cortex,
  a cooler node tint, more signals at once, data rings from ~100B, and a
  nebula background per size class (`.nmap.grand-0..3`).
- The message box is one line: a + menu (files and photos, the chat's
  folder, web search, prompt maker, commands, compare), the text, send and
  the mic; under it the permission mode on the left and the model, effort
  and context ring on the right.
- The side panel, or session rail (`components/SidePanel.tsx`,
  `settings.sidePanel`, hidden under 1080 px), sits right of a chat: Plan
  (`chat.todos`, while there are any), Environment (this computer and GPU,
  the chat's folder, permissions, web search), Context (the `/api/count`
  breakdown plus the auto-compact buffer and Compact session; folded, it
  shows the percentage), Connectors (state and tool counts) and Outputs
  (files the chat wrote or edited, opened beside the chat). Folded sections
  are kept in `settings.panelFolded`. Right-click the rail for its Show menu
  (`settings.panelHidden`, Reset to defaults); drag a card by the dots at its
  top right to move it (`settings.panelOrder`).
- `polish.css` (loaded last) holds Claude's look sampled from its window:
  sidebar #1f1f1f with a #303030 edge, the open chat #404040, the message
  box's lighter rim and soft shadow, glass menus (backdrop blur), fades at
  the top of the chat and above the message box, one family of radii, soft
  hover transitions, and the reply mark (`Orbs`: four warm balls that
  circle and breathe while a reply is written).
- The sidebar groups chats under their folder (named after it, foldable,
  `settings.sideFolded`, + starts a chat there); chats with no folder come
  after under "Chats".
- Tool calls in a row collapse to one summary line ("Ran 2 commands,
  edited ui.tsx ▸", `ToolGroup`), open while one runs or waits and while
  the reply is live. `run_command` takes a `description` shown as its row;
  opened, a command shows `$ command` (shell colors) and its output. Diffs
  mark the changed words of paired -/+ lines. Under the last reply: copy,
  branch (a new chat up to that reply), read aloud (speechSynthesis) and
  when. Replies are serif by default (`settings.replyFont`); code uses
  `settings.codeFont` via `--mono`, with Claude Dark-like token colors.
  Settings › Appearance has a Code appearance preview. Rail cards drag by a
  compact six-dot grip (`Grip` in Icons.tsx): the card follows the pointer
  on a spring (stiffness 2000, damping 86, stepped at 1/240 s so slow frames
  can't overshoot), lifts with a shadow, the others glide aside (0.36 s
  ease-out), and on release it eases into its slot before the order is saved.
  The rail's type and colours are measured from Claude's (transparent cards
  on a #222 line, 14.5px rows, grey icons and legend, only percentages bright,
  a 4px bar with 2px gaps). `window.__pelica.app` exposes the store for
  DevTools.
- Expanded, the rows share one box (`.tool-group`), one line each in
  Claude Code's style: a grey verb, the object, +/− counts and a small
  triangle; a file's row opens to its full path and a diff with one gutter
  ("+ 12"), a new file all green. Plan steps show a dashed circle (to do), a
  turning arc (doing) or a check (done).
- Tooltips (`components/Tooltip.tsx`): every `title` becomes Pelica's own
  label above the element (the title moves to `data-tip`).
- Title bar tools (chat view): show/hide the session rail, then Terminal,
  Changes, Browser and Files, which open the tool pane (`components/Dock.tsx`,
  `app.dock`; the rail hides while it's open), and "…" view options (Compact
  session, Plan, Copy as Markdown, Export chat, Keep computer awake). Terminal
  runs each line through `cmd:run` in the chat's folder (cd is tracked by the
  pane; programs that read stdin don't work). Changes diffs each file from
  its first `before` to its last `after` in the chat. Browser is a
  `<webview>` (`webviewTag`; main strips preloads in `will-attach-webview`,
  allows only http(s), and loads pop-ups in the same pane) on the
  `persist:pelica-browser` partition. Files lists the folder through
  `fs:list` (one level at a time) and opens files in the artifact panel. Keep
  computer awake is `powerSaveBlocker("prevent-app-suspension")` via
  `sys:awake`, until turned off or Pelica quits.
- Menus (`components/Menu.tsx`, Claude's look: solid #303030, #383838 rim
  and lines): rows with an icon, a label, an optional description and a
  letter on the right that runs the row while the menu is open; submenus open
  to the side; `at` places one at a point (right-click), in a portal. Used by
  the mode picker (Manual / Accept edits / Plan / Bypass permissions, keys
  1-4), the chat's title in the title bar (a chevron menu: Open in, Pin,
  Rename in place, Fork, Reply font, Export, Delete), each sidebar chat ("…"
  or right-click: Pin, Rename, Fork, Move to folder, Export, Delete) and the
  view options. `patchChat` / `forkChat` in `state.ts` change any saved chat.
- Icon buttons are as bright as the text at rest; hover only adds a grey box
  (`--hover`), a button that's on keeps one (`--btn-on`) and hovering it
  lightens it (`--btn-on-hover`). Held down, the box lightens at once
  (`--press`, 16%; `--press-on` for a button that's on) and fades back on
  release; a button whose menu is open has `menu-open` and keeps `--press`
  until the menu closes. The message box doesn't change when focused.
- The chat is one centred column: `--col` = min(780px, window − left
  sidebar − session rail), counting the rail's 290px whether it's open or
  not, so toggling the rail only re-centres the column (no line re-wraps);
  the message box and plan panel use the same width. `--col` is worked out
  on `.shell`, so it follows a hidden or dragged sidebar.
- The sidebar is resizable: `settings.sideWidth` (200–480px) sets
  `--left-w` on `<html>`; dragging its right edge (`.side-resize`) sets a
  live `--left-w` on `<body>` (store updates can't snap it back), saved on
  release; double-click resets to 272px.
- The message box is Claude's size: 44px for one line, 12px corners, a
  #404040 rim.
- Folders are per chat (`chat.folder`, picked from the + menu or the side
  panel; `settings.recentFolders` keeps the last 8). The old global
  `settings.folder` only seeds the recent list.
- Permission modes (`settings.permMode`, composer chip, Shift+Tab, `/mode`):
  `ask` (file writes/edits/deletes and commands ask), `edits` (files
  automatic, commands ask; the default), `plan` (read-only tools only and a
  plan-mode system prompt), `auto` (nothing asks). Paths stay inside the
  chat's folder in every mode; deletes go to the Recycle Bin.
- Photos (`components/Attachments.tsx`): pasted, dropped or picked images are
  downscaled to 1568 px JPEG and kept on the user message (`images`). A
  model whose status says `vision: true` (Qwen3.8 with an `mmproj-*.gguf`
  beside it) gets them as `images: [{w, h, rgb}]` on the v2 message: resized
  as Qwen's processor does (sides in multiples of 32, at most 768 x 768) and
  sent as raw RGB in base64 (`lib/pixels.ts`, prepared once per photo before
  a turn), so the engine needs no image decoder. Otherwise the
  `<attachment kind="image">` block carries the text Windows OCR reads
  (`media:ocr`, `electron/media.ts`, `Windows.Media.Ocr` through Windows
  PowerShell), labelled as OCR so the model doesn't claim to see the picture.
- Scheduled tasks (`agent/tasks.ts`, `components/Tasks.tsx`, `settings.tasks`):
  a prompt run in a new chat that isn't opened (`run(chat)`), now
  (`/background`) or on a schedule checked every 30 s; one at a time, a missed
  time runs at the next start. Local API (`main.ts` `syncApi`): a fixed-port
  pass-through to the engine's `/v1/` routes, off unless `settings.apiServer`.
- Voice (`media:dictateStart` / `media:dictateStop`, events `media:dictate`):
  System.Speech dictation on the default microphone in a PowerShell child;
  each recognised phrase is appended to the message box.
- Prompt maker (`components/PromptMaker.tsx`): goal → the loaded model asks
  3–5 questions as JSON → answers → a tuned prompt (editable, tunable again,
  copy, use in chat, or append to the custom instructions).
- Model comparison (`components/Compare.tsx`): a local model next to GPT-6
  Astra, Claude Fable 5.1, Claude Opus 5.5, Kimi K3 and GLM 5.3. Scores are
  Vals AI's independent runs where they exist, else the vendor's own
  (striped, with the setup noted); missing scores show n/a. Update the
  tables there when new numbers are published.
- Connectors (`components/Connectors.tsx`): one-click MCP presets (files,
  memory, sequential thinking, fetch, git, time, SQLite, Playwright, GitHub,
  Brave Search) with the one field each needs, a custom-server form, per
  server on/off, restart (`mcp:restart`), tools and log, and the JSON under
  Advanced. `mcp:launchers` tells which of npx / uvx are installed;
  arguments with spaces are quoted for cmd.exe.
- Web search: `web_search` + `open_page` through Electron's `session.fetch`
  in the main process (DuckDuckGo, Bing, then a hidden rendered window), with
  local/private-address refusal. The system prompt and tool descriptions tell
  the model to search only when the request needs current or external facts
  (never for greetings or small talk).

## Windows integration (main process, `electron/main.ts`)

- One instance (`requestSingleInstanceLock`): a second start (the jump list's
  New chat / Quick ask, a shortcut) reaches the running one as
  `second-instance` with its flags (`--new-chat`, `--quick`, `--hidden`).
- Tray icon (the logo, cached in the user folder as `icon.png`) with Open,
  New chat, Quick ask, Brain, Models, Settings, Quit; its tooltip says what
  the model is doing. "Keep running in the tray" makes closing hide the window.
- The quick-ask bar is a second, transparent, always-on-top window loading
  the same UI at `#quick`; the global shortcut (`settings.hotkey`, default
  `Alt+Shift+Space`) toggles it. Enter answers in the bar itself (v2 chat,
  no tools, a short system prompt, `max_tokens` 1200, streamed Markdown);
  "Continue in Pelica" sends `quick:submit(text, reply)`, which reaches the
  main window as `app:action {kind:"ask", text, reply}` and becomes a new
  chat holding that exchange. Ctrl+Enter sends only the text, and the main
  window starts a new chat with it.
- `fs:raw` reads one workspace file as is (4 MB cap) for `@`-mentions in the
  message box (`fs:find "*"` lists the candidates, cached 15 s per folder).
- `sys:saveAs(name, text, dir)` shows a save dialog (the chat's workspace
  folder first, else Documents) and writes the exported chat; nothing is
  written without the dialog. `src/lib/exportChat.ts` builds the Markdown:
  messages in full, attachments as names, tool calls as one line each,
  thinking left out.
- Model downloads (`electron/library.ts`): `lib:pause` stops one and keeps
  `<file>.part` plus a `<file>.part.json` note (repo, file, size), so
  `lib:downloads(dir)` lists it as paused after a restart; `lib:resume`
  continues with an HTTP Range request (a run waits for the previous one to
  close its file); `lib:cancel` sends both files to the Recycle Bin.
- Speeding up DeepSeek V4 files (main.ts `startRepack`): `lib:repack(path,
  undo)` asks with a Windows dialog, unloads the model if it's loaded, then
  runs `pelica.exe repack <path> [--undo]` in the background. Its stderr
  lines "(N%, about Ns left)" become `lib:repack` events (`lib:repacks`
  lists them after the page reopens); `lib:repackStop` kills it, which is
  safe (the journal finishes the layer next time); quitting kills it too. A
  download that finishes while `settings.autoRepack` is on is repacked
  without asking (the setting is the consent). The Models page reads
  `repack` from `/api/models`.
- Workspace roots only grow through the folder picker: `settings:set`
  ignores `roots`, and every `fs:*` call checks its root against them.
- The renderer reports progress with `app:busy` (`setProgressBar`: the prompt
  pass as a fraction, generation as indeterminate) and asks for
  notifications with `app:notify`, shown only when the window isn't focused
  (with `flashFrame`); clicking one opens that chat.
- `backgroundMaterial` (`settings.material`: none / mica / acrylic) with a
  transparent window background; the CSS lets it through the sidebar and
  main area (`:root[data-material]`), `nativeTheme` follows the app theme.
- Start with Windows uses `setLoginItemSettings` with `--hidden` (off unless
  the user turns it on).
- `sys:*` IPC for the computer tools: `system_info` (no user name or paths),
  clipboard read and write, `open_file` (`shell.openPath`, inside the
  workspace). Reading the clipboard and opening files go through the same
  approval dialog as commands.

## Installer (`desktop/installer/`, `npm run installer`)

No downloads: everything comes from `node_modules` and Windows itself.

- `make.mjs` stages `release/stage/Pelica`: Electron's files with
  `electron.exe` renamed to `Pelica.exe`, `resources/app` (package.json,
  `build/electron`, `dist`) and `resources/pelica.exe` (the engine).
- `icons.cjs` runs in Electron: an offscreen window draws `public/logo.svg`
  at 16–256 px into `pelica.ico` (PNG entries) and `logo.png`.
- `buildtool.cs`, compiled with the .NET Framework 4 `csc.exe` that ships
  with Windows (C# 5: no `$""`, `?.` or `=>` members), stamps the icon and
  version info into both exes (`UpdateResource`) and zips the payload.
- `setup.cs` is the installer (WinForms, custom-painted, per-monitor DPI),
  with `payload.zip` embedded; built with `/define:UNINSTALL` it is the
  uninstaller. It installs per user to `%LOCALAPPDATA%\Programs\Pelica`,
  writes `uninstall.dat` (the files it put there, so uninstall and upgrades
  touch nothing else), shortcuts with AppUserModelID `Pelica.Desktop`, and
  `HKCU\...\Uninstall\Pelica`, and adds `<dir>\resources` (the engine, so
  `pelica` works in a terminal) to the user PATH in `HKCU\Environment`
  (`UserPath`: only that entry is added or removed, the rest stays as
  stored; then `WM_SETTINGCHANGE`). A running Pelica from that folder is
  closed first. Flags: `/S`, `/RELAUNCH` (with `/S`: start Pelica afterwards), `/D=<dir>`, `/NODESKTOP`, `/NOSHORTCUTS`,
  `/NOREG`, `/NOPATH`, `/NOLAUNCH`, `/DATA` (uninstall: user data to the Recycle Bin),
  `/SHOT=<dir>` (renders every page to PNG, for checking the UI).
- `make.mjs` signs the three exes and the installer when a certificate is
  set (`PELICA_SIGN_PFX` + `PELICA_SIGN_PASSWORD`, or
  `PELICA_SIGN_THUMBPRINT`; PowerShell's `Set-AuthenticodeSignature`,
  SHA-256, timestamped), and writes `release/latest.json` (version, file
  name, sha256, notes): the update feed.
- Updates (`electron/updates.ts`, Settings → About → Updates, `components/Updates.tsx`
  for the check at start): `settings.updateFeed` is a web address or path of a
  `latest.json` (empty: never checked); `updateAuto` checks it 15 s after
  start. Install downloads into `<userData>/updates`, requires the feed's
  sha256 and, when the running Pelica.exe is signed, a valid signature by the
  same subject, then starts the installer with `/S /RELAUNCH /D=<this
  folder>` and quits. Only the packaged app updates itself.
- The packaged app takes `--user-data-dir=<dir>` for a separate profile
  (settings, chats, its own single-instance lock), which is how it is
  tested beside an installed copy.

## System prompt and agent behaviour (agent/agent.ts, agent/tools.ts)

- `systemPrompt` is built in sections: identity (Pelica desktop app, local
  model name, date), workspace (folder, Windows cmd.exe, what the current
  permission mode means via `MODE_NOTE`), "you build things, you don't hand
  out instructions", a numbered work method (look, plan with todo_write,
  build, run to check, report), a guide to each available tool, and accuracy
  rules (only claim what a tool result shows; never invent files, APIs,
  versions or links). Plan mode and no-folder chats get their own section
  (no folder: write complete files in the reply, not steps).
- `handBack()` in `loop()` drops and redoes a reply (up to twice per run)
  that tells the user to do the work by hand (`MANUAL` regex plus a code
  block or numbered list) or ends by promising a step without a tool call
  (`PROMISE`). Only when write_file is available.
- Tool descriptions say when and how to use each tool (read_file's
  number prefix, edit_file's exact-match rule, cmd.exe syntax, background
  servers).
- DeepSeek V4 (engine, src/app.rs): status lists efforts off/low/high/max;
  `with_effort` prepends the official encoder's high/max reasoning-effort
  text to the system message in thinking mode (low adds nothing).

## Repository safety check (electron/repocheck.ts, agent/tools.ts)

- `reposIn(text)` finds GitHub repos in a URL or command: github.com links,
  `git@github.com:`, raw.githubusercontent / codeload, `github:o/r`,
  `gh repo clone`, `degit`. It returns at most 4 repos.
- For each repo, `checkRepo` makes these GitHub API calls: repo metadata
  (archived, fork, stars, age), the owner's account age, the git tree
  (executables, big scripts), and raw files that run code on install or open
  (package.json install hooks, setup.py, .vscode/tasks.json `folderOpen`).
  It looks for downloaders, `iex`, base64 and webhooks in those files.
- The verdict is ok, caution or risky. A 404 or 451 counts as risky (deleted
  or taken down). A rate limit gives "couldn't check". Results are cached
  for 30 minutes (`web:repos` IPC).
- open_page puts the note above the page text. run_command checks commands
  that fetch code (`FETCHES`: git clone, npm/pip install from git, curl,
  degit…) and adds the note to the result.
- A risky verdict asks for approval even in Bypass permissions mode
  (`askApproval(..., force=true)`).
- The tool card shows each verdict (`ToolUI.safety`, collapsible, open when
  risky).

## proj.pel (electron/main.ts, state.ts restoreFolder)

- `chats:save` queues the chat for `<folder>/proj.pel` when it has a folder
  the user picked (`roots`). Writes are flushed 1.5 s later and on quit.
- The file is JSON `{ pelica: "project", version: 1, updated, chats }` and
  is written to `proj.pel.tmp` first, then renamed.
- A chat moved to another folder leaves the old folder's file. Deleting a
  chat (`chats:delete`) leaves its entry, by design.
- `setChatFolder` calls `folder:restore`. Chats in `proj.pel` with no file
  in `chats/` are written back with `folder` set to where the folder is
  now. If the open chat is empty, the newest restored chat opens, and a
  toast says how many came back.
- workspace.ts skips `PROJ_FILE` in list_files, search_files and findFiles.

## Devices (components/Devices.tsx, src/bridge.rs)

- Settings → Devices lists the PCs lending RAM through `pelica bridge`
  (`settings.devices: {addr, code}[]`).
- `startEngine` passes the list as `PELICA_DEVICES`. Every change is also
  POSTed to `/api/devices`, so it applies without a restart.
- The page polls `GET /api/devices` every 2 s for each device's state,
  memory, expert count and latency.
- Find devices calls `GET /api/devices/scan`, a UDP broadcast on port 8651.
- Only the DeepSeek V4 engine uses devices so far.

## Thinking and reply text (Markdown.tsx, polish.css)

- The thinking box renders Markdown (`<Markdown quiet className="md-think">`:
  no run boxes, sans font, same size as the box).
- Replies use Claude-like list spacing and markers (disc → circle → square,
  markers at 72% of the text colour), smaller headings, and a plain
  blockquote.
- Buttons' grey hover box appears and disappears at once: background-color
  is left out of every transition list.

## Selection and auto-scroll (polish.css, ChatView.tsx)

- Menus, popovers, the palette, title bar, sidebar, composer footer, buttons
  and the thinking header have `user-select: none` (text fields inside them
  stay selectable). Message text stays selectable.
- `follow()` keeps a scroll box pinned to its bottom while it grows, until
  the user scrolls up (any upward scrollTop move, or an upward wheel); getting
  back within 24 px of the bottom pins it again. Used by the chat and by the
  live thinking box separately: scrolling up inside the thinking box doesn't
  unpin the chat, and vice versa.

## Compact mode (window under 820 px)

- `COMPACT` media query in state.ts; App keeps `compact` in the store and
  adds `.shell.compact`. The window can shrink to 460 × 420 (main.ts).
- The sidebar becomes a drawer (`drawer` in the store, separate from the
  saved `sidebar` choice): fixed over the chat with a scrim; the scrim,
  Escape, picking a chat or page, or New chat close it. `toggleSidebar()`
  (Ctrl+B, palette, title bar, sidebar button) acts on the drawer in compact
  mode and on the column otherwise.
- Title bar: sidebar button + New chat on the left, the title centred, the
  dock icons and "…" on the right (dock icons hide under 560 px). The rail
  button and its divider hide under 1080 px.
- The chat column and message box fill the width with 18 / 14 px margins.

## Reply length (engine app_v2 chat_moe, agent.ts)

- `max_tokens` (Settings → Max tokens) limits the answer only; thinking may
  use the rest of the context window. Counting thinking against it ended
  DeepSeek V4 High/Max replies (8K tokens ≈ 45 min at 3 tok/s) before they
  answered.
- A reply that runs out of room sends `{t: "cut", while: "thinking" |
  "answer"}`; the agent adds a note saying which limit was hit and how to get
  past it.

## Queue, message search, usage (0.6)

- **Queue** (`state.queued`, Composer `queueMsg` / `QueuedChip`): Enter while
  this chat's reply runs stores the message (attachments and photos
  included; a second one is appended). `run()` in agent.ts sends it after the
  loop, only if the reply ended by itself (not stopped, no error) and the same
  chat is still open; otherwise the chip stays with Send now / Edit / remove.
  Slash commands still can't run while a reply runs.
- **Search inside messages**: `electron/chatdata.ts` (pure functions, no Node
  imports, so the UI runs the same code on its in-memory chats without the
  bridge). `searchChats` looks at user and assistant text (attachments,
  tool output and compaction summaries left out) and returns the first match
  with context; IPC `chats:search`. The sidebar shows hits not already
  listed by title under "In messages"; opening one dispatches
  `pelica:find` (FindBar opens on that query).
- **Usage**: `chatUsage` adds up every assistant message's `stats` by model
  (median `first_s`, peak) and by day for the last 30 days; IPC
  `chats:usage`, Settings → Usage (components/Usage.tsx).
- **Drafts**: Composer keeps unsent text and files per chat id in a module map
  (this session only); ↑ / ↓ in an empty box walk the chat's user messages.
- Shortcuts sheet: components/Shortcuts.tsx (`state.keys`, Ctrl+/, `/keys`).
- `window.__pelica` exposes `app`, `persist`, `openChat` and `refreshChats`, so
  the engine-served UI (.claude/launch.json "pelica-ui") can be tested with
  seeded in-memory chats.

## Saved prompts, quotes, Undo, backups (0.7)

- **Saved prompts**: `settings.prompts` ({name, text}); components/Prompts.tsx
  (`slug`, `insertPrompt`, `PromptEditor` on `state.promptEdit`, `PromptList`
  in Settings → General). The "/" menu lists commands first, then prompts
  whose slug matches (picking one puts its text in the box; it never runs).
  `pelica:insert` appends text to the box (prompts, quotes); `pelica:compose`
  still replaces it.
- **Quote**: ChatView `QuoteButton` watches mouseup/keyup for a selection
  inside the chat column and inserts it as `> ` lines.
- **Undo**: `toast(msg, action)` (`state.toastAction`, 7 s with an action).
  `deleteChat` keeps the chat it trashed and Undo saves it again.
- **Backup / restore**: IPC `chats:backup` (save dialog, `{pelica:"backup",
  version:1, created, chats}`) and `chats:restore` (accepts a backup or a
  proj.pel; ids must match `[w-]+`; a missing chat is written, an equal one or
  an equal earlier copy is skipped, a different one is written as
  `<id>-r<time>` titled "(restored)").
- Day lines: ChatView rows carry `ts`; when the first and last differ by
  day, a `.day-sep` goes before each row that starts a new day.
- Keys: Alt+↑/↓ (ChatView, to `.user-row`s above/below the top of the view),
  Ctrl+= / Ctrl+- / Ctrl+0 (App `textSize`, sizes 13–18).

## Terminal chat (0.8: src/cli.rs, cli_tools.rs, term.rs)

- `pelica` / `pelica chat` (also `pelica -c`, `--mode`, ...) runs `cli::run`.
  It is a port of agent/agent.ts and tools.ts in Rust: the same tool specs
  and descriptions, system prompt sections (with a terminal "Style"),
  hand-back nudges, repeated-call guard, compaction (archive + summary +
  ack, the app's format) and `wire()` trimming.
- Engine: `main.ts` writes `userData/engine.json` (`{port, pid, app}`) once
  its engine answers and removes it in `stopEngine`. The CLI uses that port
  when `/api/status` answers there; otherwise `app::serve_background` starts
  the engine on a free port inside the CLI's process, with stderr sent to
  `%TEMP%\pelica-cli.log` (`term::quiet_stderr`).
- Chats are written to `%APPDATA%\Pelica\chats\<id>.json` (whole to `.tmp`,
  then renamed) after every message, `folder` = the working folder.
- Console (`term.rs`): VT colours unless `NO_COLOR` or not a tty. Ctrl+C is
  handled while busy (`BUSY`) or reading (`READING`): it sets `INTERRUPT`,
  and while reading it also types an Enter into the console input
  (`WriteConsoleInputW`), because Rust's stdin retries `ReadConsoleW` after
  a Ctrl+C abort and would otherwise keep waiting. Questions flush the input
  buffer first, so keys typed during a reply can't answer them.
- A tool call is written out of sight (the engine's gate), so after 1.5 s
  without text the reply line is ended and a "Working… Ns · N tokens"
  status line shows until text or the call arrives.
- Tested in a real console through ConPTY (a scratch harness that runs the
  CLI in a hidden pseudo console, types, and sends 0x03 for Ctrl+C).
