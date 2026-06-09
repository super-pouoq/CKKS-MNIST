import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 前端 /api 转发到 Flask 后端, 避免跨域
      "/api": "http://localhost:5000",
    },
  },
});
