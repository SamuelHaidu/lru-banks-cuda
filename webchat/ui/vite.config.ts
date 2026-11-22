// One self-contained dist/index.html (JS and CSS inlined) that the FastAPI server serves at "/".
// `npm run dev` serves the page with hot reload and forwards /api to a running server (`python -m webchat.server`).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
