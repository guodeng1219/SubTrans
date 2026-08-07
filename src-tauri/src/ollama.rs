//! Ollama 管理：检测是否安装/运行、列出模型、拉取模型。
//!
//! 一键安装的实现策略（跨平台）：
//!
//! - macOS：下载官方 Ollama.app（或随包内置 sidecar 二进制）
//! - Windows：下载官方 OllamaSetup.exe 静默安装
//!
//! 这里提供"检测 + 拉模型 + 进度"的核心逻辑，安装器下载由前端向导触发。

use anyhow::{anyhow, Result};
use serde::Serialize;
use std::time::Duration;

const DEFAULT_HOST: &str = "http://localhost:11434";

#[derive(Debug, Serialize)]
pub struct OllamaStatus {
    pub running: bool,
    pub models: Vec<String>,
    /// 检测异常时的诊断信息（如代理拦截、非 Ollama 响应等）
    pub error: Option<String>,
}

/// 检测 Ollama 服务是否在跑，并列出已安装模型。
pub async fn status(client: &reqwest::Client, host: Option<&str>) -> OllamaStatus {
    let host = host.unwrap_or(DEFAULT_HOST);
    match client
        .get(format!("{host}/api/tags"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                let code = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                return OllamaStatus {
                    running: false,
                    models: vec![],
                    error: Some(format!("HTTP {code}: {body}")),
                };
            }
            match resp.json::<serde_json::Value>().await {
                Ok(v) => {
                    let models = v["models"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m["name"].as_str().map(String::from))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    OllamaStatus { running: true, models, error: None }
                }
                Err(e) => OllamaStatus {
                    running: false,
                    models: vec![],
                    error: Some(format!("解析 Ollama 响应失败: {e}")),
                },
            }
        }
        Err(e) => {
            OllamaStatus {
                running: false, models: vec![], error: Some(format!("连接失败: {e}"))
            }
        }
    }
}

/// 拉取模型。Ollama 的 /api/pull 是流式的，逐行返回 JSON 进度。
/// 通过回调把进度百分比报告给上层（再转发到前端）。
pub async fn pull_model<F>(
    client: &reqwest::Client,
    host: Option<&str>,
    model: &str,
    mut on_progress: F,
) -> Result<()>
where
    F: FnMut(f64, String),
{
    use futures_util::StreamExt;

    let host = host.unwrap_or(DEFAULT_HOST);
    let resp = client
        .post(format!("{host}/api/pull"))
        .json(&serde_json::json!({ "model": model, "stream": true }))
        .send()
        .await?;
    // HTTP 错误（模型不存在、Ollama 未启动等）必须直接报错，不能当成功
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Ollama 拉取失败：HTTP {status}: {body}"));
    }

    // 整个拉取过程加超时（模型大、网络慢时可能很久，给 2 小时兜底）
    let result = tokio::time::timeout(Duration::from_secs(2 * 3600), async {
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // 按行解析 JSON
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    // Ollama 出错时返回 {"error": "..."}，不报错会误判成拉取成功
                    if let Some(err) = v["error"].as_str() {
                        return Err(anyhow!("Ollama 拉取失败：{err}"));
                    }
                    let status = v["status"].as_str().unwrap_or("").to_string();
                    let total = v["total"].as_f64().unwrap_or(0.0);
                    let completed = v["completed"].as_f64().unwrap_or(0.0);
                    let pct = if total > 0.0 { completed / total * 100.0 } else { 0.0 };
                    on_progress(pct, status);
                }
            }
        }
        Ok(())
    })
    .await;
    match result {
        Ok(r) => r,
        Err(_) => Err(anyhow!("Ollama 拉取超时（2 小时）")),
    }
}

/// 官方安装器下载地址（向导用）。
pub fn installer_url() -> Result<&'static str> {
    #[cfg(target_os = "windows")]
    {
        Ok("https://ollama.com/download/OllamaSetup.exe")
    }
    #[cfg(target_os = "macos")]
    {
        Ok("https://ollama.com/download/Ollama-darwin.zip")
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        Err(anyhow::anyhow!("当前平台请手动安装 Ollama: https://ollama.com"))
    }
}
