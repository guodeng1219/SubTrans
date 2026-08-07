//! 翻译引擎：免费在线 (MyMemory) / DeepSeek API / 本地 Ollama。

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Engine {
    /// 免费在线翻译，无需 Key
    Free,
    /// DeepSeek API，需要 key
    DeepSeek { api_key: String, model: String },
    /// 本地 Ollama，完全离线
    Ollama { host: String, model: String },
}

fn lang_cn_name(code: &str) -> &str {
    match code {
        "zh" => "中文",
        "en" => "英文",
        "ja" => "日文",
        "ko" => "韩文",
        "fr" => "法文",
        "de" => "德文",
        "es" => "西班牙文",
        "ru" => "俄文",
        "pt" => "葡萄牙文",
        _ => "中文",
    }
}

/// 去掉推理模型(如 deepseek-r1)输出里的 `<think>...</think>` 段，避免把思考过程当成
/// 译文/纠错结果写进字幕。对不带 think 标签的普通模型(如 qwen2.5)无副作用。
pub(crate) fn strip_think(s: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    RE.replace_all(s, "").trim().to_string()
}

/// 翻译单条文本。source_lang 为 None 时表示自动检测（仅 LLM 引擎支持，MyMemory 会回退用 en）。
pub async fn translate(
    client: &reqwest::Client,
    engine: &Engine,
    text: &str,
    target_lang: &str,
    source_lang: Option<&str>,
) -> Result<String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(String::new());
    }
    // 统一超时 60s，避免 API 挂起时整个分片处理永久阻塞
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        translate_inner(client, engine, text, target_lang, source_lang),
    )
    .await;
    match result {
        Ok(r) => r,
        Err(_) => Err(anyhow!("翻译超时（60s）")),
    }
}

async fn translate_inner(
    client: &reqwest::Client,
    engine: &Engine,
    text: &str,
    target_lang: &str,
    source_lang: Option<&str>,
) -> Result<String> {
    match engine {
        Engine::Free => translate_free(client, text, target_lang, source_lang).await,
        Engine::DeepSeek { api_key, model } => {
            translate_deepseek(client, api_key, model, text, target_lang).await
        }
        Engine::Ollama { host, model } => {
            translate_ollama(client, host, model, text, target_lang).await
        }
    }
}

async fn translate_free(
    client: &reqwest::Client,
    text: &str,
    target: &str,
    source: Option<&str>,
) -> Result<String> {
    let tgt = match target {
        "zh" => "zh-CN",
        other => other,
    };
    // MyMemory 不支持 "auto" 源语言，必须传具体语言代码；
    // 未指定时根据文本内容简单推断：含大量 CJK 字符 → zh，否则 en
    let src = source.unwrap_or_else(|| {
        let cjk = text
            .chars()
            .filter(|c| {
                let cp = *c as u32;
                (0x4E00..=0x9FFF).contains(&cp)
                    || (0x3040..=0x309F).contains(&cp)
                    || (0x30A0..=0x30FF).contains(&cp)
                    || (0xAC00..=0xD7AF).contains(&cp)
            })
            .count();
        if cjk * 2 > text.chars().count() {
            "zh"
        } else {
            "en"
        }
    });
    let url = "https://api.mymemory.translated.net/get";
    let resp =
        client.get(url).query(&[("q", text), ("langpair", &format!("{src}|{tgt}"))]).send().await?;
    // MyMemory 限流可能直接回 HTTP 429；先判 HTTP 状态，给清晰报错而非含糊的 JSON 解析失败
    if !resp.status().is_success() {
        return Err(anyhow!("免费翻译请求失败：HTTP {}", resp.status().as_u16()));
    }
    let v: serde_json::Value = resp.json().await?;
    // MyMemory 配额/错误用 HTTP 200 + responseStatus 表达（如 429 当日额度用尽），
    // 不检查就会把 "MYMEMORY WARNING: YOU USED ALL ..." 当成译文写进字幕。
    let status = v["responseStatus"]
        .as_i64()
        .or_else(|| v["responseStatus"].as_str().and_then(|s| s.parse().ok()));
    if let Some(code) = status {
        if code != 200 {
            let detail = v["responseDetails"].as_str().unwrap_or("未知错误");
            return Err(anyhow!("免费翻译失败（{code}）：{detail}"));
        }
    }
    v["responseData"]["translatedText"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("免费翻译返回格式异常"))
}

async fn translate_deepseek(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    text: &str,
    target: &str,
) -> Result<String> {
    if api_key.is_empty() {
        return Err(anyhow!("DeepSeek API Key 未配置"));
    }
    let sys = format!(
        "你是专业字幕翻译。直接输出{}翻译结果，不要任何解释、引号或前后缀。",
        lang_cn_name(target)
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    });
    let resp = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("DeepSeek API 返回 HTTP {status}: {body}"));
    }
    let v: serde_json::Value = resp.json().await?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| strip_think(s.trim()))
        .ok_or_else(|| anyhow!("DeepSeek 返回格式异常: {v}"))
}

async fn translate_ollama(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    text: &str,
    target: &str,
) -> Result<String> {
    let prompt =
        format!("将下面的字幕翻译成{}，只输出译文本身，不要解释：\n{}", lang_cn_name(target), text);
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": { "temperature": 0.1 }
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
