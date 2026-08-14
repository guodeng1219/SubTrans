//! SRT / VTT 字幕文件解析（导入现有字幕用）。

use serde::Serialize;

/// 一条导入的字幕（translated 留空，导入的视为原文）。
#[derive(Debug, Clone, Serialize)]
pub struct ParsedSubtitle {
    pub start: f64,
    pub end: f64,
    pub original: String,
    pub translated: String,
}

/// 解析时间戳：支持 "HH:MM:SS,mmm" / "HH:MM:SS.mmm" / "MM:SS.mmm"（SRT 与 VTT 混用）
fn parse_ts(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (
            parts[0].parse::<f64>().ok()?,
            parts[1].parse::<f64>().ok()?,
            parts[2].parse::<f64>().ok()?,
        ),
        2 => (0.0, parts[0].parse::<f64>().ok()?, parts[1].parse::<f64>().ok()?),
        _ => return None,
    };
    Some(h * 3600.0 + m * 60.0 + sec)
}

/// 解析一行 "HH:MM:SS,mmm --> HH:MM:SS.mmm"。
/// 每侧只取第一个空白分隔的 token：VTT 时间轴后可带 cue 设置（如 align:start position:50%），
/// 直接整串解析会失败导致整条字幕被丢弃。
fn parse_ts_line(line: &str) -> Option<(f64, f64)> {
    let mut it = line.split("-->");
    let a = it.next()?;
    let b = it.next()?;
    let a = a.split_whitespace().next()?;
    let b = b.split_whitespace().next()?;
    Some((parse_ts(a)?, parse_ts(b)?))
}

/// 按块解析（SRT/VTT 共用）：空行分隔，序号行可选，时间轴行 --> 文本行。
fn parse_blocks(text: &str) -> Vec<ParsedSubtitle> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        // 跳过空行
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        let mut ts_line = lines[i];
        // 纯数字的序号行（SRT）：下一行才是时间轴
        if ts_line.trim().parse::<u64>().is_ok() {
            i += 1;
            if i >= lines.len() {
                break;
            }
            ts_line = lines[i];
        }
        if !ts_line.contains("-->") {
            i += 1;
            continue;
        }
        let Some((start, end)) = parse_ts_line(ts_line) else {
            i += 1;
            continue;
        };
        i += 1;
        let mut text = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text.push(lines[i].trim().to_string());
            i += 1;
        }
        if text.is_empty() || end <= start {
            continue;
        }
        out.push(ParsedSubtitle {
            start,
            end,
            original: text.join("\n"),
            translated: String::new(),
        });
    }
    out
}

/// 按扩展名解析字幕文件内容（UTF-8 文本；先统一换行符并去掉 BOM）。
pub fn parse_subtitle_file_ext(ext: &str, text: &str) -> Result<Vec<ParsedSubtitle>, String> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    match ext {
        "srt" => Ok(parse_blocks(text)),
        "vtt" => {
            // 去掉 WEBVTT 头（到第一个空行为止）；没有空行时退化为跳过首行，
            // 不能整文件丢弃（畸形但常见的 "WEBVTT\n<时间轴>…" 应尽量解析出内容）
            let body = if text.trim_start().starts_with("WEBVTT") {
                match text.find("\n\n") {
                    Some(p) => &text[p + 2..],
                    None => text.split_once('\n').map(|(_, rest)| rest).unwrap_or(""),
                }
            } else {
                text
            };
            Ok(parse_blocks(body))
        }
        other => Err(format!("不支持的字幕格式: .{other}（仅支持 srt/vtt）")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_srt_standard() {
        let srt = "1\n00:00:01,000 --> 00:00:03,500\n第一句\n\n2\n00:00:04,000 --> 00:00:06,000\n第二句\n第三行\n";
        let subs = parse_subtitle_file_ext("srt", srt).unwrap();
        assert_eq!(subs.len(), 2);
        assert!((subs[0].start - 1.0).abs() < 1e-9);
        assert!((subs[0].end - 3.5).abs() < 1e-9);
        assert_eq!(subs[0].original, "第一句");
        assert_eq!(subs[1].original, "第二句\n第三行");
    }

    #[test]
    fn parse_srt_without_index_and_bom() {
        let srt = "\u{feff}00:01:00,000 --> 00:01:02,000\n只有时间轴没有序号\n";
        let subs = parse_subtitle_file_ext("srt", srt).unwrap();
        assert_eq!(subs.len(), 1);
        assert!((subs[0].start - 60.0).abs() < 1e-9);
    }

    #[test]
    fn parse_vtt_with_header() {
        let vtt = "WEBVTT\nKind: captions\nLanguage: zh\n\n00:00.500 --> 00:02.000\n第一条\n\n00:03.000 --> 00:04.000\n第二条\n";
        let subs = parse_subtitle_file_ext("vtt", vtt).unwrap();
        assert_eq!(subs.len(), 2);
        assert!((subs[0].start - 0.5).abs() < 1e-9);
        assert_eq!(subs[1].original, "第二条");
    }

    #[test]
    fn parse_vtt_crlf() {
        let vtt = "WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\nCRLF 文件\r\n";
        let subs = parse_subtitle_file_ext("vtt", vtt).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].original, "CRLF 文件");
    }

    #[test]
    fn parse_vtt_with_cue_settings() {
        // 时间轴带 cue 设置（align/position）是合法 VTT，不能整条丢弃
        let vtt = "WEBVTT\n\n00:00.500 --> 00:02.000 align:start position:20%\n带设置的条目\n\n00:03.000 --> 00:04.000 line:90%\n第二条\n";
        let subs = parse_subtitle_file_ext("vtt", vtt).unwrap();
        assert_eq!(subs.len(), 2);
        assert!((subs[0].start - 0.5).abs() < 1e-9);
        assert_eq!(subs[0].original, "带设置的条目");
        assert_eq!(subs[1].original, "第二条");
    }

    #[test]
    fn rejects_unknown_ext() {
        assert!(parse_subtitle_file_ext("ass", "x").is_err());
    }
}
