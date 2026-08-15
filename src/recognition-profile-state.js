// 识别预设的纯状态逻辑：自动检测锁定、有界滚动上下文、版本 1 项目迁移。
//
// 本模块无 DOM 依赖、无副作用，可用 Node 内置测试跑（npm run test:frontend），
// 保证这些关键决策与 UI 解耦、可独立验证。

/**
 * 自动检测锁定决策（纯函数）。
 * - 手动预设（非 auto）永不改变：返回 null。
 * - auto 且已有锁：保持锁定（后续分片不再来回切换）。
 * - auto 且无锁：锁定首个有效检测结果；无检测结果返回 null。
 *
 * @param {string} selectedProfileId 用户当前选择的预设 ID
 * @param {string|null} lockedProfileId 会话已锁定的预设 ID（null = 未锁定）
 * @param {string|null} detectedProfileId 后端本分片返回的 detected_profile_id
 * @returns {string|null} 应写入 lockedProfileId 的新值；无变化或不可锁定时返回原锁
 */
export function lockDetectedProfile(selectedProfileId, lockedProfileId, detectedProfileId) {
  if (selectedProfileId !== "auto") return null;
  if (lockedProfileId) return lockedProfileId;
  return detectedProfileId || null;
}

/**
 * 会话重置（纯函数）：新视频 / 重新识别 / 打开项目时重建识别会话状态。
 * 只清空自动检测锁定；用户选择的预设与口音保持不变。
 *
 * @param {string} selectedProfileId 用户当前选择的预设 ID
 * @param {string} accentVariant 用户当前选择的口音变体 ID
 * @returns {{selectedProfileId: string, lockedProfileId: null, accentVariant: string}}
 */
export function resetRecognitionSession(selectedProfileId, accentVariant) {
  return {
    selectedProfileId: selectedProfileId || "auto",
    lockedProfileId: null,
    accentVariant: accentVariant || "auto",
  };
}

/**
 * 会话隔离的锁定决策（纯函数）：过期会话（currentSession !== responseSession）
 * 的检测结果不能改动当前锁定，返回原锁不变。
 *
 * @returns {string|null} 应写入 lockedProfileId 的新值
 */
export function applyDetectedProfileForSession(
  currentSession,
  responseSession,
  selectedProfileId,
  lockedProfileId,
  detectedProfileId
) {
  if (currentSession !== responseSession) return lockedProfileId;
  return lockDetectedProfile(selectedProfileId, lockedProfileId, detectedProfileId) ?? lockedProfileId;
}

/**
 * 有界滚动上下文（纯函数）：取最近 maxLines 条非空原文，按换行连接，
 * 再从左侧截断至 maxChars 个 Unicode 字符，保证最新对白存活。
 *
 * @param {Array<{original?: string}>} subtitles 按时间有序的字幕列表
 * @param {number} maxLines 最多取最近几行（默认 3）
 * @param {number} maxChars 总字符上限（默认 600，按码点计数）
 * @returns {string} 合成上下文；无有效原文时为空串
 */
export function buildRollingContext(subtitles, maxLines = 3, maxChars = 600) {
  const lines = (Array.isArray(subtitles) ? subtitles : [])
    .map((s) => (s && typeof s.original === "string" ? s.original.trim() : ""))
    .filter(Boolean);
  const recent = lines.slice(-Math.max(1, maxLines));
  const joined = recent.join("\n");
  // 按 Unicode 码点从左侧截断：旧对白先被丢弃，最新对白存活
  return [...joined].slice(-Math.max(0, maxChars)).join("");
}

const V1_SOURCE_TO_PROFILE = {
  zh: "zh-film",
  en: "en-film",
  ja: "ja-film",
  ko: "ko-film",
  fr: "fr-film",
  de: "de-film",
};

/**
 * 项目识别设置迁移（v1 → v2，纯函数）。
 * - 已有新字段（recognitionProfileId）时原样保留并补齐默认值。
 * - 版本 1 只有 sourceLang：zh/en/ja/ko/fr/de 映射到对应 *-film，
 *   空值映射到 auto，其余语言映射到 custom 并保留原 sourceLang。
 *
 * @param {object} [settings] 项目 settings（可能为空）
 * @returns {{recognitionProfileId: string, accentVariant: string, sourceLang: string}}
 */
export function migrateRecognitionSettings(settings) {
  const s = settings || {};
  const sourceLang = typeof s.sourceLang === "string" ? s.sourceLang : "";
  if (typeof s.recognitionProfileId === "string" && s.recognitionProfileId) {
    return {
      recognitionProfileId: s.recognitionProfileId,
      accentVariant: typeof s.accentVariant === "string" && s.accentVariant ? s.accentVariant : "auto",
      sourceLang,
    };
  }
  const recognitionProfileId = sourceLang
    ? V1_SOURCE_TO_PROFILE[sourceLang] || "custom"
    : "auto";
  return { recognitionProfileId, accentVariant: "auto", sourceLang };
}
