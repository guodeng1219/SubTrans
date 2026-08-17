import test from "node:test";
import assert from "node:assert/strict";
import { captureStartSession, startFlowExpired } from "../src/start-guard.js";

test("double-click start: two captures yield distinct session tokens", () => {
  // 模拟快速双击：两个启动流程都在任何 await 之前捕获。
  // 入口按钮守卫（startBtn.disabled）之外，令牌唯一性保证即使两个流程
  // 都进入了函数，也不会共享同一会话号。
  const state = { session: 0, videoPath: "a.mp4" };
  const capturedA = captureStartSession(state);
  const capturedB = captureStartSession(state);
  assert.notEqual(capturedA.session, capturedB.session);
  // 先捕获的 A 相对当前状态已过期：恢复后不得继续
  assert.equal(startFlowExpired(capturedA, state), true);
  // B 有效
  assert.equal(startFlowExpired(capturedB, state), false);
});

test("switch video during metadata wait: A resumes but must not proceed", async () => {
  // 复现阻断问题：A 进入最长 15 秒的元数据等待；期间用户打开新视频 B；
  // A 恢复后按**捕获值**校验（而不是重读 state.session），判定过期，
  // 不得调用分离、不得修改 stream。
  const state = { session: 0, videoPath: "a.mp4" };
  // A 在任何 await 之前捕获
  const capturedA = captureStartSession(state);
  assert.deepEqual(capturedA, { session: 1, videoPath: "a.mp4" });
  // A 进入元数据等待（真实代码 await loadedmetadata，最长 15s）
  await Promise.resolve();
  // 等待期间用户打开新视频 B：换路径 + 新令牌
  state.videoPath = "b.mp4";
  const capturedB = captureStartSession(state);
  // A 恢复：对照捕获值 → 已过期（会话号与视频路径都不再匹配）
  assert.equal(startFlowExpired(capturedA, state), true);
  // B 的捕获对应当前状态 → 有效
  assert.equal(startFlowExpired(capturedB, state), false);
});

test("restart on the same video: old flow expires, new flow valid", () => {
  const state = { session: 0, videoPath: "a.mp4" };
  const capturedA = captureStartSession(state);
  // 同一视频重复点击开始：会话号 +1，路径不变
  const capturedB = captureStartSession(state);
  assert.equal(startFlowExpired(capturedA, state), true);
  assert.equal(startFlowExpired(capturedB, state), false);
});

test("no switch: captured flow stays valid", () => {
  const state = { session: 0, videoPath: "a.mp4" };
  const captured = captureStartSession(state);
  assert.equal(startFlowExpired(captured, state), false);
});
