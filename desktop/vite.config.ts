import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built UI is served by pelica.exe (`pelica app --ui dist`), same origin as its API.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 2000 },
});
