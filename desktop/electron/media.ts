// Photos and voice, with what Windows already has: OCR (Windows.Media.Ocr)
// reads the text in a photo for models that can't see images, and dictation
// (System.Speech) turns the microphone into text for the message box. Both
// run in Windows PowerShell, so nothing is downloaded.
import { spawn, ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const PS = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
const encoded = (script: string) => Buffer.from(script, "utf16le").toString("base64");
const quote = (s: string) => `'${s.replace(/'/g, "''")}'`;

const OCR = `
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation\`1' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime] | Out-Null
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $eng) { throw 'No OCR language is installed in Windows (Settings > Time & language > Language).' }
$res = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
$stream.Dispose()
foreach ($l in $res.Lines) { [Console]::Out.WriteLine($l.Text) }
`;

/** The text Windows reads in an image (a data: URL), line by line. */
export async function ocr(dataUrl: string): Promise<string> {
  const m = /^data:image\/(png|jpe?g|bmp|gif);base64,(.+)$/i.exec(String(dataUrl));
  if (!m) throw new Error("Not an image Windows can read (PNG, JPEG, BMP or GIF).");
  const file = path.join(os.tmpdir(), `pelica-ocr-${process.pid}-${Date.now()}.${m[1].toLowerCase().replace("jpeg", "jpg")}`);
  fs.writeFileSync(file, Buffer.from(m[2], "base64"));
  try {
    return await new Promise<string>((resolve, reject) => {
      const p = spawn(PS, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded(`$path = ${quote(file)}\n${OCR}`)], { windowsHide: true });
      let out = "";
      let err = "";
      p.stdout.on("data", (d) => (out += d));
      p.stderr.on("data", (d) => (err += d));
      const t = setTimeout(() => p.kill(), 60000);
      p.on("error", reject);
      p.on("close", (code) => {
        clearTimeout(t);
        if (code === 0) resolve(out.replace(/\r/g, "").trim());
        else reject(new Error(cleanErr(err) || "Windows OCR failed."));
      });
    });
  } finally {
    fs.rm(file, { force: true }, () => {});
  }
}

const DICTATE = `
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$all = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
if (-not $all -or $all.Count -eq 0) { [Console]::Out.WriteLine('NONE'); exit 3 }
$ui = [Globalization.CultureInfo]::CurrentUICulture
$ri = ($all | Where-Object { $_.Culture.Name -eq $ui.Name } | Select-Object -First 1)
if (-not $ri) { $ri = ($all | Where-Object { $_.Culture.TwoLetterISOLanguageName -eq $ui.TwoLetterISOLanguageName } | Select-Object -First 1) }
if (-not $ri) { $ri = $all[0] }
$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine($ri)
$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$r.SetInputToDefaultAudioDevice()
[Console]::Out.WriteLine('READY ' + $ri.Culture.Name); [Console]::Out.Flush()
while ($true) {
  $res = $r.Recognize([TimeSpan]::FromSeconds(20))
  if ($res -and $res.Text) { [Console]::Out.WriteLine('TEXT ' + $res.Text); [Console]::Out.Flush() }
}
`;

export type DictateEvent = { kind: "ready"; lang: string } | { kind: "text"; text: string } | { kind: "error"; error: string } | { kind: "end" };

let rec: ChildProcess | null = null;

/** Starts listening on the default microphone; each phrase arrives as a "text" event. */
export function dictateStart(emit: (e: DictateEvent) => void) {
  dictateStop();
  const p = spawn(PS, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded(DICTATE)], { windowsHide: true });
  rec = p;
  let buf = "";
  let err = "";
  let none = false;
  p.stdout!.on("data", (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).replace(/\r$/, "");
      buf = buf.slice(i + 1);
      if (line === "NONE") none = true;
      else if (line.startsWith("READY")) emit({ kind: "ready", lang: line.slice(6) });
      else if (line.startsWith("TEXT ")) emit({ kind: "text", text: line.slice(5) });
    }
  });
  p.stderr!.on("data", (d) => (err += d));
  p.on("error", (e) => emit({ kind: "error", error: e.message }));
  p.on("close", (code) => {
    if (rec === p) rec = null;
    if (none) emit({ kind: "error", error: "Windows has no speech recognizer installed. Add one in Settings > Time & language > Speech (or Language > your language > Speech recognition)." });
    else if (code && code !== 0 && !p.killed) {
      const e = cleanErr(err);
      emit({ kind: "error", error: /audio|device|microphone/i.test(e) ? "No microphone found (or Windows blocked it: Settings > Privacy > Microphone)." : e || "Dictation stopped." });
    }
    emit({ kind: "end" });
  });
}

export function dictateStop() {
  if (rec && !rec.killed) rec.kill();
  rec = null;
}

/** PowerShell's error text without its position and category lines. */
function cleanErr(s: string): string {
  const lines = s.replace(/\r/g, "").split("\n").map((l) => l.trim()).filter((l) => l && !/^(At line|At char|\+|CategoryInfo|FullyQualifiedErrorId|#< CLIXML|<Objs)/.test(l));
  return (lines[0] || "").replace(/^Exception calling "[^"]+" with "\d+" argument\(s\): /, "").replace(/^"|"$/g, "").slice(0, 300);
}
