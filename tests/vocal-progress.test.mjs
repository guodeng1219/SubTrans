import test from "node:test";
import assert from "node:assert/strict";
import { formatVocalProgress } from "../src/vocal-progress.js";

test("vocal progress includes chunks and memory budget", () => {
  assert.equal(
    formatVocalProgress({
      message: "高精度人声分离 7/24",
      chunk_index: 7,
      chunk_total: 24,
      memory_bytes: 5 * 1024 ** 3,
      memory_budget_bytes: 8 * 1024 ** 3,
      warning: false,
      retrying_with_smaller_chunks: false,
    }),
    "高精度人声分离 7/24 · 内存 5.0/8.0 GB"
  );
});

test("retry progress explains why progress changed", () => {
  const text = formatVocalProgress({
    message: "正在改用更小分片重试",
    memory_bytes: 8 * 1024 ** 3,
    memory_budget_bytes: 8 * 1024 ** 3,
    warning: true,
    retrying_with_smaller_chunks: true,
  });
  assert.match(text, /内存占用过高/);
  assert.match(text, /更小分片/);
});
