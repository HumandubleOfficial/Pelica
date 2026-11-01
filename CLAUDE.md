# Pelica: working rules

## Every update gets a version and release notes

Each piece of work that changes Pelica (the engine or the desktop app) gets
an entry in `desktop/src/changelog.json` before it ships:

- Add an entry at the top: `{ "version": "next", "title": "...", "notes": [...] }`.
  Add `"bump": "minor"` for a bigger release (new features); leave it out
  for a patch (fixes, small changes).
- Notes are for the user, not for developers: one line per change, plain
  words, what they will notice ("Prompts are read about twice as fast"). If a
  speed changed, say by how much.
- If an entry is already `"next"` (unreleased), add to its notes instead of
  making a second one.

`npm run installer` (in `desktop/`) runs `installer/bump.mjs` first. It turns
`"next"` into the next version number with today's date, and writes that
version into `desktop/package.json` and `Cargo.toml`. The installer refuses
to build when the version has no notes. The app shows the new notes once,
in a What's new window, the first time it starts on that version. Settings →
About, the palette ("What's new") and `/changelog` show every release.
