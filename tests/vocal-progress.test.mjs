import test from "node:test";
import assert from "node:assert/strict";
import {
  formatVocalProgress,
  shouldAcceptSeparateProgress,
} from "../src/vocal-progress.js";

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

test("matching task id accepts the separation progress event", () => {
  assert.equal(
    shouldAcceptSeparateProgress("sep-1", { task_id: "sep-1", message: "分离中" }),
    true
  );
});

test("mismatched task id rejects the stale separation event", () => {
  assert.equal(
    shouldAcceptSeparateProgress("sep-2", { task_id: "sep-1", message: "分离中" }),
    false
  );
});

test("event without task id is rejected", () => {
  assert.equal(shouldAcceptSeparateProgress("sep-1", { message: "分离中" }), false);
});

test("cancelling task A invalidates its progress: A cannot update the UI", () => {
  // 模拟三个取消入口的行为：先 sepTaskId = null 再 cancel_vocal_separation。
  // 取消后（currentTaskId 为 null），A 的残留进度事件必须被丢弃，
  // 即使事件携带 A 的 task_id 且内容合法。
  const taskA = "sep-A";
  const eventFromA = { task_id: taskA, message: "分离中: 99%" };
  // 取消前：A 的事件有效
  assert.equal(shouldAcceptSeparateProgress(taskA, eventFromA), true);
  // 取消：sepTaskId 置 null
  const currentTaskIdAfterCancel = null;
  // 取消后：A 的残留进度被丢弃，不会覆盖新页面状态
  assert.equal(shouldAcceptSeparateProgress(currentTaskIdAfterCancel, eventFromA), false);
  // 新任务 B 接管后，只有 B 的事件通过
  assert.equal(
    shouldAcceptSeparateProgress("sep-B", { task_id: "sep-B", message: "分离中: 1%" }),
    true
  );
  assert.equal(shouldAcceptSeparateProgress("sep-B", eventFromA), false);
});
