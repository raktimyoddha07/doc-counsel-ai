import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "object-assign": path.resolve(__dirname, "object-assign.cjs"),
    },
  },
  server: {
    port: 5173,
  },
});

