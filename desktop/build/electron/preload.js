"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// What the UI may ask of the main process (window.pelica).
const electron_1 = require("electron");
const call = (name) => (...a) => electron_1.ipcRenderer.invoke(name, ...a);
const on = (name) => (fn) => {
    const f = (_e, v) => fn(v);
    electron_1.ipcRenderer.on(name, f);
    return () => electron_1.ipcRenderer.removeListener(name, f);
};
electron_1.contextBridge.exposeInMainWorld("pelica", {
    win: { minimize: call("win:minimize"), maximize: call("win:maximize"), close: call("win:close"), isMaximized: call("win:isMaximized"), onState: on("win:state"), material: call("win:material") },
    settings: { get: call("settings:get"), set: call("settings:set"), onPushed: on("settings:pushed") },
    account: { status: call("account:status"), signIn: call("account:signIn"), signOut: call("account:signOut"), syncNow: call("sync:now"), onChanged: on("account:changed"), onSynced: on("sync:changed") },
    folder: { pick: call("folder:pick"), open: call("folder:open"), restore: call("folder:restore") },
    chats: { list: call("chats:list"), load: call("chats:load"), save: call("chats:save"), remove: call("chats:delete"), search: call("chats:search"), usage: call("chats:usage"), backup: call("chats:backup"), restore: call("chats:restore"), import: call("chats:import") },
    fs: { list: call("fs:list"), read: call("fs:read"), write: call("fs:write"), edit: call("fs:edit"), search: call("fs:search"), find: call("fs:find"), raw: call("fs:raw"), notes: call("fs:notes"), trash: call("fs:trash"), reveal: call("fs:reveal") },
    cmd: { run: call("cmd:run"), kill: call("cmd:kill"), onData: on("cmd:data"), onExit: on("cmd:exit") },
    web: { search: call("web:search"), open: call("web:open"), repos: call("web:repos") },
    mcp: { start: call("mcp:start"), list: call("mcp:list"), call: call("mcp:call"), restart: call("mcp:restart"), launchers: call("mcp:launchers"), others: call("mcp:others"), parse: call("mcp:parse"), signIn: call("mcp:signIn"), signOut: call("mcp:signOut"), onChanged: on("mcp:changed") },
    lib: { search: call("lib:search"), files: call("lib:files"), download: call("lib:download"), downloads: call("lib:downloads"), cancel: call("lib:cancel"), pause: call("lib:pause"), resume: call("lib:resume"), uninstall: call("lib:uninstall"), repack: call("lib:repack"), repacks: call("lib:repacks"), repackStop: call("lib:repackStop"), onRepack: on("lib:repack"), pickDir: call("lib:pickDir"), hardware: call("lib:hardware"), drives: call("lib:drives"), space: call("lib:space"), move: call("lib:move"), moves: call("lib:moves"), moveStop: call("lib:moveStop"), onMove: on("lib:move"), onProgress: on("lib:progress") },
    phone: { status: call("phone:status"), push: call("phone:push"), newKey: call("phone:newKey"), onCmd: on("phone:cmd") },
    engine: { log: call("engine:log"), restart: call("engine:restart"), api: call("api:status") },
    docs: { index: call("docs:index"), info: call("docs:info"), search: call("docs:search"), forget: call("docs:forget"), onProgress: on("docs:progress") },
    media: { ocr: call("media:ocr"), dictateStart: call("media:dictateStart"), dictateStop: call("media:dictateStop"), onDictate: on("media:dictate") },
    app: { info: call("app:info"), busy: call("app:busy"), notify: call("app:notify"), hotkey: call("app:hotkey"), login: call("app:login"), onAction: on("app:action") },
    update: { check: call("update:check"), install: call("update:install"), cancel: call("update:cancel"), onProgress: on("update:progress") },
    quick: { submit: call("quick:submit"), hide: call("quick:hide"), resize: call("quick:resize"), clip: call("quick:clip"), onShown: on("quick:shown") },
    sys: { info: call("sys:info"), open: call("sys:open"), clipRead: call("sys:clipRead"), clipWrite: call("sys:clipWrite"), external: call("sys:external"), saveAs: call("sys:saveAs"), reveal: call("sys:reveal"), awake: call("sys:awake") },
    // a dropped file's path on disk
    pathFor: (f) => electron_1.webUtils.getPathForFile(f),
});
