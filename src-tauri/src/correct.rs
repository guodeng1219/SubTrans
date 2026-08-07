//! 中文字幕同音字/错别字校对：把识别文本交给 LLM 按上下文修正。
//! 复用 translate::Engine（DeepSeek / 本地 Ollama）。Free 引擎不支持校对，原样返回。

use crate::translate::{strip_think, Engine};
use anyhow::{anyhow, Result};
use regex::Regex;
use std::sync::LazyLock;
use std::time::Duration;

/// 整批校对一个分片的句子。返回与输入**等长**的结果；
/// 任何解析不一致都整批退回原文（绝不因 LLM 乱格式而损坏字幕）。
pub async fn correct_lines(
    client: &reqwest::Client,
    engine: &Engine,
    lines: &[String],
    glossary: &str,
) -> Result<Vec<String>> {
    if lines.is_empty() {
        return Ok(vec![]);
    }
    // Free 不是 LLM，无法校对
    if matches!(engine, Engine::Free) {
        return Ok(lines.to_vec());
    }

    // 整批编号：「序号|文本」
    let numbered: String = lines
        .iter()
        .enumerate()
        .map(|(i, l)| format!("{}|{}", i + 1, l))
        .collect::<Vec<_>>()
        .join("\n");

    let glossary = glossary.trim();
    let glossary_clause = if glossary.is_empty() {
        String::new()
    } else {
        // 截断过长输入并移除引号，避免 LLM prompt 注入或 token 浪费
        let safe: String = glossary
            .chars()
            .take(500)
            .map(|c| if c == '"' || c == '\'' { ' ' } else { c })
            .collect();
        format!("\n这些是正确的专有名词/人名/术语，遇到同音词必须改成它们：{safe}。")
    };

    let sys = format!(
        "你是中文字幕校对。下面每行是「序号|文本」格式的语音识别结果，可能有同音字或错别字（例如 在/再、的/得/地、是/事/市）。\
         请只纠正明显的同音字和错别字，不要改变原意、不要翻译、不要增删或合并行、不要改动标点和断句。\
         必须逐行输出，保持「序号|文本」格式，序号与行数和输入完全一致，只允许改动文本内容。{glossary_clause}"
    );

    // 与翻译一致：给 LLM 请求加超时，避免模型挂起时整个分片流水线永久阻塞。
    let raw = tokio::time::timeout(Duration::from_secs(120), async {
        match engine {
            Engine::DeepSeek { api_key, model } => {
                correct_deepseek(client, api_key, model, &sys, &numbered).await
            }
            Engine::Ollama { host, model } => {
                correct_ollama(client, host, model, &sys, &numbered).await
            }
            Engine::Free => unreachable!(),
        }
    })
    .await
    .map_err(|_| anyhow!("纠错超时（120s）"))??;

    Ok(parse_or_fallback(&raw, lines))
}

// 匹配「序号|文本」行（兼容全角竖线）
static LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(\d+)\s*[|｜](.*)$").unwrap());

/// 解析 LLM 输出回填到对应行；行号越界、行数对不上 → 整批退回原文。
fn parse_or_fallback(raw: &str, original: &[String]) -> Vec<String> {
    let mut out = original.to_vec();
    let mut covered = 0usize;
    for caps in LINE_RE.captures_iter(raw) {
        let idx: usize = match caps[1].parse() {
            Ok(n) => n,
            Err(_) => return original.to_vec(),
        };
        if idx < 1 || idx > original.len() {
            return original.to_vec();
        }
        let text = caps[2].trim();
        // 模型把整句删空了 → 该行保留原文，避免丢字幕
        if !text.is_empty() {
            out[idx - 1] = text.to_string();
        }
        covered += 1;
    }
    // 必须每一行都被覆盖，否则放弃（保证不串行、不丢行）
    if covered != original.len() {
        return original.to_vec();
    }
    out
}

async fn correct_deepseek(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    sys: &str,
    user: &str,
) -> Result<String> {
    if api_key.is_empty() {
        return Err(anyhow!("DeepSeek API Key 未配置"));
    }
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 4096
    });
    let resp = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;
    // 与 translate.rs 一致：先看 HTTP 状态，否则 401/429 的错误体会被当成"格式异常"误报
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("DeepSeek API 返回 HTTP {status}: {body}"));
    }
    let v: serde_json::Value = resp.json().await?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(strip_think)
        .ok_or_else(|| anyhow!("DeepSeek 返回格式异常: {v}"))
}

async fn correct_ollama(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    sys: &str,
    user: &str,
) -> Result<String> {
    let body = serde_json::json!({
        "model": model,
        "system": sys,
        "prompt": user,
        "stream": false,
        "options": { "temperature": 0.0 }
    });
    let resp = client.post(format!("{host}/api/generate")).json(&body).send().await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Ollama 返回 HTTP {status}: {body}"));
    }
    let v: serde_json::Value = resp.json().await?;
    v["response"].as_str().map(strip_think).ok_or_else(|| anyhow!("Ollama 返回格式异常: {v}"))
}
