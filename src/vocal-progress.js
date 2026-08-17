// 高精度人声分离进度与失败策略的纯格式化逻辑（无 DOM 依赖，Node 内置测试可跑）。
//
// 失败策略不在此处分支：任何高精度分离失败都停止识别（绝不静默回退普通音频
// 或 Demucs），由 main.js 的 catch 内联实现——无条件常量不需要可测试的抽象。

/** 字节数 → GiB 数值文本（保留 1 位小数；单位由展示处统一追加，避免 "5.0 GB/8.0 GB"）。 */
export function formatBytesGiB(bytes) {
  return (Number(bytes || 0) / 1024 ** 3).toFixed(1);
}

/**
 * 分离进度事件过滤：当前任务 id 为空、事件未携带 id、或 id 不匹配时丢弃。
 * 取消任务 A 后（currentTaskId 置 null），A 的残留进度（如 Demucs 后台
 * 读行任务在退出前发出的最后几行）必须被丢弃，不得更新新任务的 UI。
 *
 * @param {string|null} currentTaskId 当前分离任务的 request_id（取消后为 null）
 * @param {object} payload 后端 progress 事件的 payload
 * @returns {boolean} 是否接受该事件
 */
export function shouldAcceptSeparateProgress(currentTaskId, payload) {
  if (!currentTaskId) return false;
  return payload?.task_id === currentTaskId;
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
