// What the UI may ask of the main process (window.pelica).
import { contextBridge, ipcRenderer, webUtils } from "electron";

const call =
  (name: string) =>
  (...a: unknown[]) =>
    ipcRenderer.invoke(name, ...a);

const on = (name: string) => (fn: (v: any) => void) => {
  const f = (_e: unknown, v: any) => fn(v);
  ipcRenderer.on(name, f);
  return () => ipcRenderer.removeListener(name, f);
};

contextBridge.exposeInMainWorld("pelica", {
  win: { minimize: call("win:minimize"), maximize: call("win:maximize"), close: call("win:close"), isMaximized: call("win:isMaximized"), onState: on("win:state"), material: call("win:material") },
  settings: { get: call("settings:get"), set: call("settings:set") },
  folder: { pick: call("folder:pick"), open: call("folder:open") },
  chats: { list: call("chats:list"), load: call("chats:load"), save: call("chats:save"), remove: call("chats:delete") },
  fs: { list: call("fs:list"), read: call("fs:read"), write: call("fs:write"), edit: call("fs:edit"), search: call("fs:search"), find: call("fs:find"), raw: call("fs:raw"), notes: call("fs:notes"), trash: call("fs:trash"), reveal: call("fs:reveal") },
  cmd: { run: call("cmd:run"), kill: call("cmd:kill"), onData: on("cmd:data"), onExit: on("cmd:exit") },
  web: { search: call("web:search"), open: call("web:open") },
  mcp: { start: call("mcp:start"), list: call("mcp:list"), call: call("mcp:call"), restart: call("mcp:restart"), launchers: call("mcp:launchers") },
  lib: { search: call("lib:search"), files: call("lib:files"), download: call("lib:download"), downloads: call("lib:downloads"), cancel: call("lib:cancel"), pause: call("lib:pause"), resume: call("lib:resume"), pickDir: call("lib:pickDir"), onProgress: on("lib:progress") },
  engine: { log: call("engine:log"), restart: call("engine:restart") },
  media: { ocr: call("media:ocr"), dictateStart: call("media:dictateStart"), dictateStop: call("media:dictateStop"), onDictate: on("media:dictate") },
  app: { info: call("app:info"), busy: call("app:busy"), notify: call("app:notify"), hotkey: call("app:hotkey"), login: call("app:login"), onAction: on("app:action") },
  quick: { submit: call("quick:submit"), hide: call("quick:hide"), resize: call("quick:resize"), clip: call("quick:clip"), onShown: on("quick:shown") },
  sys: { info: call("sys:info"), open: call("sys:open"), clipRead: call("sys:clipRead"), clipWrite: call("sys:clipWrite"), external: call("sys:external"), saveAs: call("sys:saveAs"), reveal: call("sys:reveal") },
  // a dropped file's path on disk
  pathFor: (f: File) => webUtils.getPathForFile(f),
});
