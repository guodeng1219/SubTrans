//! 多语言影视识别预设目录：本应用识别预设的唯一真相源。
//!
//! 前端只读 DTO（[`all_profiles`]）；每个分片由 [`resolve_profile`] 解析出语言与解码参数，
//! CPU（whisper.cpp）与 GPU（faster-whisper）共用同一解析结果，保证两条链路的语言、
//! 提示词与热词行为对齐。所有内容离线内置，不引入任何运行时依赖。

/// 提示词预算（Unicode 字符数）：滚动上下文、热词与预设固定文本之和的上限。
const PROMPT_MAX_CHARS: usize = 1_400;
/// 滚动上下文预算（前端已按此截断，后端兜底再截一次，按 Unicode 字符而非字节）。
const CONTEXT_MAX_CHARS: usize = 600;
/// 源语言热词预算。
const HOTWORDS_MAX_CHARS: usize = 400;

/// `custom` 预设允许的 Whisper 语言代码白名单（与前端 sourceLang 选项一致）。
const CUSTOM_LANGUAGES: &[&str] = &["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt"];

/// 口音变体（目录内部使用）。
struct AccentVariant {
    id: &'static str,
    label: &'static str,
    prompt_suffix: &'static str,
}

/// 预设目录条目（目录内部使用）。
struct LanguageProfile {
    id: &'static str,
    label: &'static str,
    language: Option<&'static str>,
    initial_prompt: &'static str,
    beam_size: u8,
    rolling_context_lines: u8,
    accent_variants: &'static [AccentVariant],
}

/// 前端展示用的口音变体 DTO。
#[derive(Clone, Debug, serde::Serialize)]
pub struct AccentVariantDto {
    pub id: &'static str,
    pub label: &'static str,
}

/// 前端展示用的预设 DTO。
#[derive(Clone, Debug, serde::Serialize)]
pub struct LanguageProfileDto {
    pub id: &'static str,
    pub label: &'static str,
    pub language: Option<&'static str>,
    pub accent_variants: Vec<AccentVariantDto>,
}

/// 解析后的预设：自有数据（owned），可安全移入阻塞任务 / 子进程请求。
#[derive(Clone, Debug)]
pub struct ResolvedLanguageProfile {
    pub id: String,
    pub language: Option<String>,
    /// 已包含口音后缀、尚未包含滚动上下文与热词的提示词（由 [`compose_initial_prompt`] 合成）。
    pub initial_prompt: String,
    pub beam_size: usize,
    pub rolling_context_lines: usize,
    /// 降级等非致命提示（如口音变体不适用）。
    pub warning: Option<String>,
}

const ACCENT_AUTO: AccentVariant =
    AccentVariant { id: "auto", label: "自动口音", prompt_suffix: "" };

const ACCENT_EN_GB: AccentVariant = AccentVariant {
    id: "en-gb",
    label: "英式英语",
    prompt_suffix: "The speakers use British English pronunciation and vocabulary.",
};

const ACCENT_EN_US: AccentVariant = AccentVariant {
    id: "en-us",
    label: "美式英语",
    prompt_suffix: "The speakers use American English pronunciation and vocabulary.",
};

const EN_ACCENTS: &[AccentVariant] = &[ACCENT_AUTO, ACCENT_EN_GB, ACCENT_EN_US];
const AUTO_ONLY: &[AccentVariant] = &[ACCENT_AUTO];

/// 首批预设目录。所有内置影视预设使用 beam_size=5、滚动上下文 3 行；
/// 预设只改善提示词与上下文，不替换底层 Whisper 模型。
static PROFILES: &[LanguageProfile] = &[
    LanguageProfile {
        id: "auto",
        label: "自动检测",
        language: None,
        initial_prompt: "",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "zh-film",
        label: "中文影视",
        language: Some("zh"),
        initial_prompt: "以下是影视对白，使用简体中文。人名、地名和术语应保持一致。",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "en-film",
        label: "English Film",
        language: Some("en"),
        initial_prompt: "This is film dialogue in English. Preserve character names, place names, contractions, and natural speech.",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: EN_ACCENTS,
    },
    LanguageProfile {
        id: "ja-film",
        label: "日本語映像",
        language: Some("ja"),
        initial_prompt: "これは日本語の映像作品の会話です。人名、地名、敬称を正確に書き起こしてください。",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "ko-film",
        label: "한국어 영상",
        language: Some("ko"),
        initial_prompt: "한국어 영상 대화입니다. 인명, 지명, 존댓말 어미를 정확히 기록합니다.",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "fr-film",
        label: "Français cinéma",
        language: Some("fr"),
        initial_prompt: "Dialogue de film en français. Conserver les noms propres, les élisions, les apostrophes et les accents.",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "de-film",
        label: "Deutsch Film",
        language: Some("de"),
        initial_prompt: "Deutschsprachiger Filmdialog. Eigennamen, zusammengesetzte Wörter und Großschreibung beibehalten.",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
    LanguageProfile {
        id: "custom",
        label: "自定义语言",
        language: None,
        initial_prompt: "",
        beam_size: 5,
        rolling_context_lines: 3,
        accent_variants: AUTO_ONLY,
    },
];

/// 全部预设（含 auto/custom），供前端填充下拉框。
pub fn all_profiles() -> Vec<LanguageProfileDto> {
    PROFILES
        .iter()
        .map(|p| LanguageProfileDto {
            id: p.id,
            label: p.label,
            language: p.language,
            accent_variants: p
                .accent_variants
                .iter()
                .map(|a| AccentVariantDto { id: a.id, label: a.label })
                .collect(),
        })
        .collect()
}

/// 检测语言 → 内置影视预设 ID；没有对应预设（如 es）返回 None，自动会话保持 auto。
pub fn profile_for_detected_language(language: Option<&str>) -> Option<&'static str> {
    let lang = language?;
    PROFILES.iter().find(|p| p.id.ends_with("-film") && p.language == Some(lang)).map(|p| p.id)
}

/// 解析预设 ID + 口音变体 + 自定义语言，返回可执行参数。
///
/// - 未知预设 ID 返回错误（不静默接受任意字符串）。
/// - 口音变体不属于该预设时降级为 `auto`，并写入 `warning`。
/// - `custom` 的语言来自 `custom_language`（对应前端 `sourceLang`），
///   仅接受白名单内的 Whisper 语言代码；空值表示自动检测。
pub fn resolve_profile(
    profile_id: &str,
    accent_variant: &str,
    custom_language: Option<&str>,
) -> Result<ResolvedLanguageProfile, String> {
    let profile = PROFILES
        .iter()
        .find(|p| p.id == profile_id)
        .ok_or_else(|| format!("未知识别预设: {profile_id}"))?;

    // 口音变体：不属于所选预设的 ID 降级为 auto，并在结果 warn 中提示
    let (accent, warning) = match profile.accent_variants.iter().find(|a| a.id == accent_variant) {
        Some(a) => (a, None),
        None => {
            let fallback =
                profile.accent_variants.iter().find(|a| a.id == "auto").unwrap_or(&ACCENT_AUTO);
            (
                fallback,
                Some(format!("口音变体 {accent_variant} 不适用于预设 {profile_id}，已使用自动")),
            )
        }
    };

    // 语言：内置预设固定；custom 走白名单校验的 sourceLang；auto 不强制
    let language = if let Some(l) = profile.language {
        Some(l.to_string())
    } else if profile.id == "custom" {
        match custom_language {
            None | Some("") => None,
            Some(l) if CUSTOM_LANGUAGES.contains(&l) => Some(l.to_string()),
            Some(other) => return Err(format!("不支持的 Whisper 语言代码: {other}")),
        }
    } else {
        None
    };

    // 提示词 = 预设基础提示 + 口音后缀
    let mut base = String::new();
    if !profile.initial_prompt.is_empty() {
        base.push_str(profile.initial_prompt);
    }
    if !accent.prompt_suffix.is_empty() {
        if !base.is_empty() {
            base.push(' ');
        }
        base.push_str(accent.prompt_suffix.trim());
    }

    Ok(ResolvedLanguageProfile {
        id: profile.id.to_string(),
        language,
        initial_prompt: base,
        beam_size: profile.beam_size as usize,
        rolling_context_lines: profile.rolling_context_lines as usize,
        warning,
    })
}

/// 把预设提示词、滚动上下文与源语言热词合成最终 initial_prompt。
///
/// 预算按 Unicode 字符执行：上下文先按预设行数（`rolling_context_lines`）取最近几行、
/// 再截断至 ≤600 字符（保留最新尾部），热词 ≤400，总长 ≤1400。
/// `hotwords` 是已解析的源语言热词（如 `glossary_hotwords` 输出），不注入术语表译文。
pub fn compose_initial_prompt(
    profile: &ResolvedLanguageProfile,
    context: &str,
    hotwords: &str,
) -> String {
    // 行预算：只保留最近 N 行（前端已按此取上下文，后端兜底再执行一次）
    let context = keep_last_lines(context, profile.rolling_context_lines);
    // 上下文保留最新的尾部（先到的是旧对白，应被截掉）
    let context = keep_last_chars(&context, CONTEXT_MAX_CHARS);
    let hotwords = keep_first_chars(hotwords.trim(), HOTWORDS_MAX_CHARS);

    let mut parts: Vec<String> = Vec::new();
    if !profile.initial_prompt.is_empty() {
        parts.push(profile.initial_prompt.clone());
    }
    if !context.is_empty() {
        parts.push(context);
    }
    if !hotwords.is_empty() {
        parts.push(hotwords);
    }
    keep_first_chars(&parts.join("\n"), PROMPT_MAX_CHARS)
}

/// 保留最近 `max` 行非空行（保持原有顺序）。
fn keep_last_lines(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let lines: Vec<&str> = s.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.len() <= max {
        return lines.join("\n");
    }
    lines[lines.len() - max..].join("\n")
}

/// 按 Unicode 字符保留开头至多 `max` 个字符。
fn keep_first_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// 按 Unicode 字符保留结尾至多 `max` 个字符（丢弃旧内容，保住最新对白）。
fn keep_last_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    s.chars().skip(count - max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_unique_expected_ids() {
        let ids: Vec<_> = all_profiles().into_iter().map(|p| p.id).collect();
        assert_eq!(
            ids,
            vec![
                "auto", "zh-film", "en-film", "ja-film", "ko-film", "fr-film", "de-film", "custom"
            ]
        );
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn detected_languages_map_only_to_built_in_profiles() {
        assert_eq!(profile_for_detected_language(Some("en")), Some("en-film"));
        assert_eq!(profile_for_detected_language(Some("ja")), Some("ja-film"));
        assert_eq!(profile_for_detected_language(Some("es")), None);
        assert_eq!(profile_for_detected_language(None), None);
    }

    #[test]
    fn english_variant_and_context_are_composed_and_bounded() {
        let resolved = resolve_profile("en-film", "en-gb", None).unwrap();
        let context = "x".repeat(800);
        let prompt = compose_initial_prompt(&resolved, &context, "Pemberton, Bank of England");
        assert!(prompt.contains("British English"));
        assert!(prompt.contains("Pemberton"));
        assert!(prompt.chars().count() <= 1_400);
    }

    #[test]
    fn custom_requires_a_supported_whisper_language_code() {
        assert!(resolve_profile("custom", "auto", Some("es")).is_ok());
        assert!(resolve_profile("custom", "auto", Some("../../bad")).is_err());
        assert!(resolve_profile("missing", "auto", None).is_err());
    }

    #[test]
    fn unknown_accent_degrades_to_auto_with_warning() {
        let resolved = resolve_profile("zh-film", "en-gb", None).unwrap();
        assert!(resolved.warning.is_some());
        // 降级后提示词只含中文预设文本，不含英式后缀
        assert!(!resolved.initial_prompt.contains("British English"));
    }

    #[test]
    fn context_truncation_keeps_the_newest_dialogue() {
        let resolved = resolve_profile("zh-film", "auto", None).unwrap();
        let context = "旧对白\n".to_string() + &"新".repeat(700);
        let prompt = compose_initial_prompt(&resolved, &context, "");
        assert!(!prompt.contains("旧对白"));
        assert!(prompt.contains(&"新".repeat(100)));
    }

    #[test]
    fn custom_empty_language_means_auto_detect() {
        let resolved = resolve_profile("custom", "auto", Some("")).unwrap();
        assert_eq!(resolved.language, None);
    }
}
