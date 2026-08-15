// 高精度人声分离进度与失败策略的纯格式化逻辑（无 DOM 依赖，Node 内置测试可跑）。
//
// 失败策略不在此处分支：任何高精度分离失败都停止识别（绝不静默回退普通音频
// 或 Demucs），由 main.js 的 catch 内联实现——无条件常量不需要可测试的抽象。

/** 字节数 → GiB 数值文本（保留 1 位小数；单位由展示处统一追加，避免 "5.0 GB/8.0 GB"）。 */
export function formatBytesGiB(bytes) {
  return (Number(bytes || 0) / 1024 ** 3).toFixed(1);
}

/**
 * 人声分离进度文本：分片消息 + 内存占用；降片重试时明确解释进度变化原因，
 * 不伪装成普通进度回退。
 *
 * @param {object} payload progress 事件负载（stage === "separate" 且带 chunk_total）
 * @returns {string}
 */
export function formatVocalProgress(payload) {
  const used = formatBytesGiB(payload.memory_bytes);
  const budget = formatBytesGiB(payload.memory_budget_bytes);
  if (payload.retrying_with_smaller_chunks) {
    return `内存占用过高，正在改用更小分片重试 · 内存 ${used}/${budget} GB`;
  }
  const warning = payload.warning ? " · ⚠ 内存偏高" : "";
  return `${payload.message} · 内存 ${used}/${budget} GB${warning}`;
}
