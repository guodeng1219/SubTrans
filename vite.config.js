import { defineConfig } from "vite";

// 前端打包配置：源码在 src/，产物输出到 dist/。
// tauri.conf.json 的 frontendDist 指向 ../dist；dev 时 Tauri 连到下面的 devUrl。
export default defineConfig({
  root: "src",
  // 输出目录相对 root(src)，即项目根的 dist/
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    target: "es2021",
  },
  // 不让 Vite 清屏，方便看到 Tauri/cargo 的编译输出
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Rust 后端交给 cargo 监视，Vite 不用管
      ignored: ["**/src-tauri/**"],
    },
  },
});
