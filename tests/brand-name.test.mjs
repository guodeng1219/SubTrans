import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

test("user-facing surfaces present the 译幕 brand consistently", () => {
  const tauri = JSON.parse(read("src-tauri/tauri.conf.json"));
  const html = read("src/index.html");
  const main = read("src/main.js");
  const releaseWorkflow = read(".github/workflows/release.yml");
  const readme = read("README.md");

  assert.equal(tauri.productName, "译幕");
  assert.equal(tauri.app.windows[0].title, "译幕");
  assert.match(html, /<title>译幕 · AI 字幕翻译<\/title>/);
  assert.match(html, /<div class="wizard-brand">YIMU<\/div>/);
  assert.match(html, /<span class="brand-mark">译<\/span>/);
  assert.match(html, /<span class="brand-text">译幕<\/span>/);
  assert.equal(main.match(/name: "译幕项目"/g)?.length, 2);
  assert.doesNotMatch(main, /SubTrans 项目/);
  assert.match(releaseWorkflow, /releaseName: "译幕 \$\{\{ github\.ref_name \}\}"/);
  assert.match(readme, /^# 译幕（本地 AI 视频字幕翻译）/u);
});

test("rename preserves compatibility identifiers and project format", () => {
  const tauri = JSON.parse(read("src-tauri/tauri.conf.json"));
  const packageJson = JSON.parse(read("package.json"));
  const main = read("src/main.js");

  assert.equal(tauri.identifier, "com.subtrans.desktop");
  assert.equal(packageJson.name, "subtrans");
  assert.match(main, /subtrans\.setupDone/);
  assert.match(main, /extensions: \["subtrans", "json"\]/);
});
