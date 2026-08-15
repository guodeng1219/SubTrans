//! ASR 引擎抽象层（目前仅 Whisper-CPU；中文/GPU 走 faster-whisper sidecar）。

use anyhow::Result;
use regex::Regex;
use serde::Serialize;
use std::path::Path;
use std::sync::LazyLock;

/// 一条字幕片段
#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// 过滤语气词/填充词：移除独立出现的"呃、啊、嗯、哦"等。
pub fn clean_fillers(text: &str) -> String {
    // 独立的中文语气词（前后是空白或标点）、英文 fillers、连续重复语气词
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(concat!(
            r"(?:^|[\s,，。！？、…\p{P}])[呃啊嗯哦唔噢呵哈嘿嗨诶哎唷嘛呢吧呀哇嘢](?:[\s,，。！？、…\p{P}]|$)",
            r"|\b(?:um|uh|ah|er|hmm|mm|hm)\b[,，。！？、…]*",
            r"|[呃啊嗯哦唔噢呵哈嘿嗨诶哎唷]{2,}",
        ))
        .unwrap()
    });
    // 句中标点前的多余空格："say , then" → "say, then"
    static SPACE_BEFORE_PUNCT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\s+([,，。！？、…])").unwrap());
    let result = RE.replace_all(text, |caps: &regex::Captures| {
        let m = caps.get(0).unwrap();
        let s = m.as_str();
        let puncts: Vec<char> = s
            .chars()
            .filter(|c| {
                c.is_ascii_punctuation()
                    || *c == '，'
                    || *c == '。'
                    || *c == '！'
                    || *c == '？'
                    || *c == '、'
                    || *c == '…'
            })
            .collect();
        // 段首/段尾的语气词连同边界标点一起去掉（避免留下前导逗号）；
        // 句中的语气词只保留一个边界标点（避免“你好，呃，再见。”变成双逗号或丢逗号）。
        if m.start() == 0 || m.end() == text.len() || puncts.is_empty() {
            String::new()
        } else {
            puncts.last().copied().map(|c| c.to_string()).unwrap_or_default()
        }
    });
    let result = result.to_string();
    let result = SPACE_BEFORE_PUNCT.replace_all(&result, "$1");
    // 连续重复标点："你好。。再见。" → "你好。再见。"
    let result = dedupe_punct(&result);
    let result = result.replace("  ", " ").replace("\n\n", "\n");
    result.trim().to_string()
}

/// 去掉连续重复的中文标点（如 "。。"、"？？"）。
fn dedupe_punct(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    for c in text.chars() {
        let is_punct = matches!(c, '，' | '。' | '！' | '？' | '、' | '…');
        if is_punct && prev == Some(c) {
            continue;
        }
        out.push(c);
        prev = if is_punct { Some(c) } else { None };
    }
    out
}

/// 是否为纯标点片段（连续重复标点按标点拆分时会被单独切出来，如 "Hello!!!" → "!"）。
fn is_pure_punct(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            matches!(c, '。' | '！' | '？' | '；' | '，' | '、' | ',' | ';' | '!' | '?' | '.' | '…')
        })
}

/// 拆分过长的 segment：Whisper 对连续朗读/唱歌内容可能把一整段话合成一个 segment，
/// 导致 overlay 一次显示一大坨文字。按标点符号拆成多个子段，时间按字数比例分配。
pub fn split_long_segments(segments: Vec<Segment>) -> Vec<Segment> {
    // 单段超过 20 字或 8 秒就拆
    const MAX_CHARS: usize = 20;
    const MAX_DUR: f64 = 8.0;
    // 拆分用的标点（句末 + 句中）
    // 中英文常用句末/句中标点都拆；英文句号 . 之前漏了
    static SPLIT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"([。！？；，,;!?.])").unwrap());

    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        let dur = seg.end - seg.start;
        let char_count = seg.text.chars().count();
        if char_count <= MAX_CHARS && dur <= MAX_DUR {
            out.push(seg);
            continue;
        }
        // 中/日/韩按“字”切（每个字是完整单元）；西文按“词”切，避免把英文单词切碎
        let cjk = is_cjk_dominant(&seg.text);

        // 1) 按标点拆分，标点保留在前一片末尾
        let mut parts: Vec<String> = Vec::new();
        let mut last = 0;
        for m in SPLIT_RE.find_iter(&seg.text) {
            let end = m.end();
            let chunk = seg.text[last..end].trim();
            if !chunk.is_empty() {
                parts.push(chunk.to_string());
            }
            last = end;
        }
        let tail = seg.text[last..].trim();
        if !tail.is_empty() {
            parts.push(tail.to_string());
        }
        if parts.is_empty() {
            parts.push(seg.text.clone());
        }

        // 2) 标点拆完仍超长的片，再按字/词二次切分
        let mut refined: Vec<String> = Vec::new();
        for p in parts {
            // 连续重复标点会拆出孤立的纯标点片（"Hello!!!" → "Hello!" "!" "!"），
            // 单独成段按词计数为 0 → 时间分配得到零时长字幕；把它并回前一片，
            // 文本与原文一致（标点本就相邻），不丢内容也不产生空段。
            if is_pure_punct(&p) {
                if let Some(prev) = refined.last_mut() {
                    prev.push_str(&p);
                } else {
                    refined.push(p); // 整段以标点开头：罕见，保留（时间分配有最小 1 单位兜底）
                }
                continue;
            }
            if p.chars().count() <= MAX_CHARS {
                refined.push(p);
                continue;
            }
            if cjk {
                let chars: Vec<char> = p.chars().collect();
                for i in (0..chars.len()).step_by(MAX_CHARS) {
                    let to = (i + MAX_CHARS).min(chars.len());
                    refined.push(chars[i..to].iter().collect());
                }
            } else {
                // 按词贪心组装，尽量在空格边界切，单词保持完整
                let mut cur = String::new();
                for w in p.split_whitespace() {
                    if !cur.is_empty() && cur.chars().count() + w.chars().count() + 1 > MAX_CHARS {
                        refined.push(std::mem::take(&mut cur));
                    }
                    if !cur.is_empty() {
                        cur.push(' ');
                    }
                    cur.push_str(w);
                }
                if !cur.is_empty() {
                    refined.push(cur);
                }
            }
        }

        // 3) 时间分配：CJK 按字符数（信息密度均匀），西文按单词数（词时长更接近）
        let unit_of = |p: &str| {
            if cjk {
                p.chars().count()
            } else {
                p.split_whitespace().count()
            }
        };
        let total_units: usize = refined.iter().map(|p| unit_of(p).max(1)).sum::<usize>().max(1);
        let mut t = seg.start;
        for (i, text) in refined.iter().enumerate() {
            // 每片至少 1 单位：纯标点片（西文按词计数为 0）也分到正时长，杜绝 start==end 的零时长字幕
            let ratio = (unit_of(text).max(1)) as f64 / total_units as f64;
            let sub_dur = dur * ratio;
            let sub_end = if i == refined.len() - 1 { seg.end } else { t + sub_dur };
            out.push(Segment {
                index: 0, // 后面会重编
                start: t,
                end: sub_end,
                text: text.clone(),
            });
            t = sub_end;
        }
    }
    // 重编 index
    for (i, s) in out.iter_mut().enumerate() {
        s.index = i + 1;
    }
    out
}

/// 判断文本是否以 CJK（中日韩）为主：存在足够多的汉字/假名/谚文字符。
/// 决定后续按“字”还是按“词”切分。
fn is_cjk_dominant(text: &str) -> bool {
    let total = text.chars().count();
    if total == 0 {
        return false;
    }
    let cjk = text
        .chars()
        .filter(|c| {
            let cp = *c as u32;
            (0x4E00..=0x9FFF).contains(&cp) // CJK 统一表意文字
                || (0x3400..=0x4DBF).contains(&cp) // 扩展 A
                || (0x3040..=0x30FF).contains(&cp) // 平假名/片假名
                || (0xAC00..=0xD7AF).contains(&cp) // 谚文
        })
        .count();
    cjk * 2 >= total
}

/// ASR 引擎统一接口
pub trait AsrEngine: Send + Sync {
    /// 加载模型，返回引擎实例。在阻塞线程里调用，避免卡异步执行器。
    fn load(model_path: &Path) -> Result<Self>
    where
        Self: Sized;

    /// 转写 f32 音频 → 结构化结果（字幕段 + 检测语言）。
    /// 语言、提示词、VAD 与解码参数统一由 [`TranscribeOptions`] 携带。
    fn transcribe(&self, audio: &[f32], options: TranscribeOptions<'_>) -> Result<Transcription>;
}

/// ASR 转写的结构化输入：把语言、提示词、解码参数打包，避免两条链路参数漂移。
pub struct TranscribeOptions<'a> {
    /// 强制语言（None = 自动检测）。
    pub language: Option<&'a str>,
    /// 预设/滚动上下文合成的初始提示词（None 或空串 = 不注入）。
    pub initial_prompt: Option<&'a str>,
    pub threads: i32,
    /// 分片局部时间戳 → 视频绝对时间的偏移秒数。
    pub time_offset_sec: f64,
    /// Silero VAD 模型路径（None = 不启用 VAD）。
    pub vad_model_path: Option<&'a str>,
    /// Greedy 采样候选数。
    pub best_of: i32,
}

/// ASR 转写的结构化输出：字幕段 + 检测语言。
/// 手动指定语言时回传该语言；自动检测时回传 whisper 判定结果。
pub struct Transcription {
    pub segments: Vec<Segment>,
    pub detected_lang: Option<String>,
}

pub mod whisper;

#[cfg(test)]
mod tests {
    use super::{clean_fillers, split_long_segments, Segment, TranscribeOptions, Transcription};

    #[test]
    fn fillers_at_start_leave_no_leading_punctuation() {
        assert_eq!(clean_fillers("呃，你好。"), "你好。");
        assert_eq!(clean_fillers("嗯 好的"), "好的");
        assert_eq!(clean_fillers("um, hello"), "hello");
    }

    #[test]
    fn fillers_in_middle_keep_single_boundary() {
        assert_eq!(clean_fillers("你好，呃，再见。"), "你好，再见。");
        assert_eq!(clean_fillers("say um, then"), "say, then");
    }

    #[test]
    fn repeated_fillers_are_removed() {
        assert_eq!(clean_fillers("嗯嗯嗯好的"), "好的");
        assert_eq!(clean_fillers("你好。呃嗯。再见。"), "你好。再见。");
    }

    #[test]
    fn non_filler_text_is_unchanged() {
        assert_eq!(clean_fillers("你好，世界。"), "你好，世界。");
        assert_eq!(clean_fillers("hello world"), "hello world");
    }

    #[test]
    fn english_long_segment_splits_on_word_boundaries() {
        // 无标点长英文句：必须按词切，不能从单词中间切开
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 0.0,
            end: 12.0,
            text: "this is a very long english sentence without any punctuation at all".to_string(),
        }]);
        assert!(segs.len() >= 2, "长英文句应被拆成多段");
        // 每个词都必须完整（出现在原文中且不是半截词）
        let words: Vec<&str> =
            "this is a very long english sentence without any punctuation at all"
                .split_whitespace()
                .collect();
        let split_words: Vec<&str> = segs.iter().flat_map(|s| s.text.split_whitespace()).collect();
        assert_eq!(split_words, words, "拆分段不能丢词或切碎单词");
        // 时间必须连续覆盖 [0, 12]
        assert!((segs.first().unwrap().start - 0.0).abs() < 1e-6);
        assert!((segs.last().unwrap().end - 12.0).abs() < 1e-6);
        for w in segs.windows(2) {
            assert!((w[1].start - w[0].end).abs() < 1e-6);
        }
    }

    #[test]
    fn english_time_split_by_word_count() {
        // 两段词数 2:3 → 时长 4:6
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 0.0,
            end: 10.0,
            text: "alpha beta. gamma delta epsilon".to_string(),
        }]);
        assert_eq!(segs.len(), 2);
        assert!((segs[0].end - 4.0).abs() < 1e-6);
        assert!((segs[1].start - 4.0).abs() < 1e-6);
        assert!((segs[1].end - 10.0).abs() < 1e-6);
    }

    #[test]
    fn chinese_long_segment_splits_by_chars() {
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 0.0,
            end: 8.0,
            text: "这是一个非常长的中文句子没有任何标点符号而且还要继续说下去直到超过二十个字为止"
                .to_string(),
        }]);
        assert!(segs.len() >= 2);
        // 每段不超过 20 字，且字数总和等于原文
        let total: usize = segs.iter().map(|s| s.text.chars().count()).sum();
        assert_eq!(
            total,
            "这是一个非常长的中文句子没有任何标点符号而且还要继续说下去直到超过二十个字为止"
                .chars()
                .count()
        );
        for s in &segs {
            assert!(s.text.chars().count() <= 20);
        }
    }

    #[test]
    fn short_segment_stays_whole() {
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 1.0,
            end: 3.0,
            text: "hello world".to_string(),
        }]);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
    }

    #[test]
    fn punctuation_only_parts_never_get_zero_duration() {
        // "Hello!!!" 按标点拆分会切出孤立的 "!" 片段，不能产生 start==end 的零时长字幕
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 0.0,
            end: 12.0,
            text: "Hello!!! World".to_string(),
        }]);
        for s in &segs {
            assert!(s.end > s.start, "零时长字幕: {s:?}");
        }
        // 标点不能丢：各段拼起来应与拆分前一致（孤立的 "!" 被并回前一片；
        // 尾片与标点间的空格被既有 trim 逻辑去除，属于拆分前的固有行为）
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, "Hello!!!World");
    }

    #[test]
    fn leading_punctuation_part_gets_positive_duration() {
        // 段首纯标点 + 无标点长句：首片按词计数为 0，时间分配必须有最小 1 单位兜底
        let segs = split_long_segments(vec![Segment {
            index: 1,
            start: 0.0,
            end: 12.0,
            text: "!!!hello world hello world hello world".to_string(),
        }]);
        for s in &segs {
            assert!(s.end > s.start, "零时长字幕: {s:?}");
        }
    }

    #[test]
    fn transcription_carries_detected_language() {
        let result = Transcription {
            segments: Vec::new(),
            detected_lang: Some("en".to_string()),
        };
        assert_eq!(result.detected_lang.as_deref(), Some("en"));
    }

    #[test]
    fn transcribe_options_keep_prompt_and_decode_values() {
        let options = TranscribeOptions {
            language: Some("en"),
            initial_prompt: Some("British film dialogue"),
            threads: 4,
            time_offset_sec: 10.0,
            vad_model_path: None,
            best_of: 5,
        };
        assert_eq!(options.initial_prompt, Some("British film dialogue"));
        assert_eq!(options.best_of, 5);
    }
}
