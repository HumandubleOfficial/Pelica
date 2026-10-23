// Renders public/logo.svg into the app's icons with Electron's own Chromium,
// so every size is drawn from the vector: pelica.ico (16-256 px) for the
// exe, the shortcuts and the installer, and logo.png for the installer's
// window.  Run: electron installer/icons.cjs <logo.svg> <out dir>
const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const [svgPath, outDir] = process.argv.slice(-2);
app.setPath("userData", path.join(outDir, "icons-profile"));

const SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256];

/** An .ico holding PNG images (Windows Vista and later read these at every size). */
function ico(pngs) {
  const head = Buffer.alloc(6 + 16 * pngs.length);
  head.writeUInt16LE(0, 0);
  head.writeUInt16LE(1, 2);
  head.writeUInt16LE(pngs.length, 4);
  let offset = head.length;
  pngs.forEach(([size, data], i) => {
    const e = 6 + i * 16;
    head.writeUInt8(size >= 256 ? 0 : size, e);
    head.writeUInt8(size >= 256 ? 0 : size, e + 1);
    head.writeUInt8(0, e + 2);
    head.writeUInt8(0, e + 3);
    head.writeUInt16LE(1, e + 4);
    head.writeUInt16LE(32, e + 6);
    head.writeUInt32LE(data.length, e + 8);
    head.writeUInt32LE(offset, e + 12);
    offset += data.length;
  });
  return Buffer.concat([head, ...pngs.map((p) => p[1])]);
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false, webPreferences: { offscreen: true } });
  await win.loadURL("data:text/html,<!doctype html><html><body></body></html>");
  const svg = fs.readFileSync(svgPath, "utf8");
  const pngs = [];
  for (const size of SIZES) {
    // the SVG sized to the target, so it is rasterized there rather than scaled
    const sized = svg.replace("<svg ", `<svg width="${size}" height="${size}" `);
    const url = await win.webContents.executeJavaScript(`(async () => {
      const img = new Image();
      img.src = 'data:image/svg+xml;base64,${Buffer.from(sized).toString("base64")}';
      await img.decode();
      const c = document.createElement('canvas');
      c.width = c.height = ${size};
      c.getContext('2d').drawImage(img, 0, 0, ${size}, ${size});
      return c.toDataURL('image/png');
    })()`);
    pngs.push([size, Buffer.from(url.split(",")[1], "base64")]);
  }
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "pelica.ico"), ico(pngs));
  fs.writeFileSync(path.join(outDir, "logo.png"), pngs[pngs.length - 1][1]);
  console.log(`icons: ${SIZES.join(", ")} px`);
  app.quit();
});
