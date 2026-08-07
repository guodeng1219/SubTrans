//! faster-whisper GPU 识别：常驻 Python 进程 IPC。
//!
//! 内嵌 fw_server.py 到二进制，运行时写入临时文件启动。
//! 模型常驻显存，逐片通过 stdin/stdout 行协议喂音频。

use crate::asr;
use crate::emit_progress;
use std::sync::{Arc, Mutex};

/// 内嵌的 faster-whisper 服务脚本（编译进二进制，运行时写到临时文件，自包含）。
const FW_SERVER_PY: &str = include_str!("../../python/fw_server.py");

/// IPC 超时：单片识别（含 large-v3 在 CPU 上的极端情况）给足 5 分钟。
const IPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// 启动加载超时：large-v3 首次加载（含模型读盘）给 3 分钟。
const STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// 常驻的 faster-whisper 服务进程（模型常驻显存，逐片喂音频）。
pub(crate) struct FwServer {
    key: (String, String), // (model, device)
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    /// stderr 尾部缓存（用于崩溃时报错诊断）
    stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>>,
}

#[derive(Default, Clone)]
pub struct FwState(pub Arc<tokio::sync::Mutex<Option<FwServer>>>);

impl FwState {
    /// 优雅关闭：先发 quit 命令让 Python 释放显存，再强杀。
    pub async fn shutdown(&self) {
        use tokio::io::AsyncWriteExt;
        let mut guard = self.0.lock().await;
        if let Some(mut srv) = guard.take() {
            let _ = srv.stdin.write_all(b"{\"cmd\":\"quit\"}\n").await;
            let _ = srv.stdin.flush().await;
            // 给 Python 1 秒清理时间，然后强杀
            let _ = tokio::time::timeout(std::time::Duration::from_secs(1), srv.child.wait()).await;
            let _ = srv.child.kill().await;
        }
    }
}

/// 确保 faster-whisper 服务已按指定 model/device 启动（懒加载；换模型则重启）。
pub async fn fw_ensure(
    state: &FwState,
    python_exe: &str,
    model: &str,
    device: &str,
    app: &tauri::AppHandle,
) -> Result<(), String> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut guard = state.0.lock().await;
    if let Some(srv) = guard.as_ref() {
        if srv.key == (model.to_string(), device.to_string()) {
            return Ok(());
        }
    }
    if let Some(mut old) = guard.take() {
        let _ = old.child.kill().await;
    }

    emit_progress(app, "fw", 5.0, &format!("加载识别模型 {model}（{device}，首次会下载）..."));
    let compute = if device == "cuda" { "float16" } else { "int8" };
    // 使用 PID 区分临时脚本，避免多实例竞争
    let script_path =
        std::env::temp_dir().join(format!("subtrans_fw_server_{}.py", std::process::id()));
    std::fs::write(&script_path, FW_SERVER_PY).map_err(|e| format!("写入识别服务脚本失败: {e}"))?;
    let mut cmd = tokio::process::Command::new(python_exe);
    let model_root = crate::fw_model_root().to_string_lossy().to_string();
    cmd.arg(&script_path)
        .arg(model)
        .arg(device)
        .arg(compute)
        .arg(&model_root)
        .env("PYTHONUTF8", "1") // 与 fw_server.py 内部 reconfigure 双保险，防 cp932 管道乱码
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "启动 faster-whisper 失败: {e}（确认 Python 路径，且已 pip install faster-whisper）"
        )
    })?;
    let stdin = child.stdin.take().ok_or("无法获取 stdin")?;
    let stdout = child.stdout.take().ok_or("无法获取 stdout")?;
    let stderr = child.stderr.take().ok_or("无法获取 stderr")?;
    let mut lines = BufReader::new(stdout).lines();

    // 后台读取 stderr 并缓存末尾几行（用于崩溃诊断）
    let stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>> =
        Arc::new(Mutex::new(std::collections::VecDeque::new()));
    {
        let tail = stderr_tail.clone();
        tokio::spawn(async move {
            let mut err_lines = BufReader::new(stderr).lines();
            loop {
                match err_lines.next_line().await {
                    Ok(Some(line)) => {
                        let line = line.trim().to_string();
                        if !line.is_empty() {
                            let mut t = tail.lock().unwrap();
                            t.push_back(line);
                            while t.len() > 10 {
                                t.pop_front();
                            }
                        }
                    }
                    Ok(None) => break,
                    // 单行坏字节不能终止诊断采集，否则后面的 traceback 全丢
                    Err(_) => continue,
                }
            }
        });
    }

    // 等就绪信号（large-v3 首次要下载+加载，可能较久）——带超时避免永久挂起
    let ready = match tokio::time::timeout(STARTUP_TIMEOUT, lines.next_line()).await {
        Ok(Ok(Some(l))) => l,
        Ok(Ok(None)) => {
            let detail = stderr_diag(&stderr_tail);
            return Err(format!("faster-whisper 未输出就绪信号（进程已退出）{detail}"));
        }
        Ok(Err(e)) => return Err(format!("读取就绪信号失败: {e}")),
        Err(_) => {
            let _ = child.kill().await;
            let detail = stderr_diag(&stderr_tail);
            return Err(format!(
                "faster-whisper 加载超时（{}s）{detail}",
                STARTUP_TIMEOUT.as_secs()
            ));
        }
    };
    let v: serde_json::Value = serde_json::from_str(&ready)
        .map_err(|e| format!("解析就绪信号失败: {e}（输出: {ready}）"))?;
    if v["ready"].as_bool() != Some(true) {
        return Err(format!(
            "faster-whisper 加载失败: {}",
            v["error"].as_str().unwrap_or("未知错误")
        ));
    }
    emit_progress(app, "fw", 10.0, "识别模型就绪");
    *guard = Some(FwServer {
        key: (model.to_string(), device.to_string()),
        child,
        stdin,
        stdout: lines,
        stderr_tail,
    });
    Ok(())
}

/// 从 stderr 缓存取末尾几行作为诊断信息。
fn stderr_diag(tail: &Arc<Mutex<std::collections::VecDeque<String>>>) -> String {
    let t = tail.lock().unwrap();
    if t.is_empty() {
        String::new()
    } else {
        format!("\nstderr: {}", t.iter().cloned().collect::<Vec<_>>().join(" | "))
    }
}

/// 让常驻服务识别一个 wav，返回相对该 wav 的字幕段（时间从 0 起，调用方再加 offset）。
/// separated: 音频是否已经过人声分离（决定 Python 端是否开 VAD）。
pub(crate) struct FwTranscription {
    pub segments: Vec<asr::Segment>,
    /// 模型检测到的语言（自动检测时才有意义，前端用来提示用户是否误判）
    pub language: Option<String>,
    pub language_probability: Option<f64>,
}

pub async fn fw_transcribe_one(
    state: &FwState,
    audio_path: &str,
    language: Option<&str>,
    separated: bool,
    vad_enabled: bool,
    hotwords: Option<&str>,
) -> Result<FwTranscription, String> {
    use tokio::io::AsyncWriteExt;

    let mut guard = state.0.lock().await;
    let srv = guard.as_mut().ok_or("faster-whisper 服务未启动")?;

    let req = serde_json::json!({
        "audio": audio_path,
        "language": language,
        "separated": separated,
        "vad_enabled": vad_enabled,
        "hotwords": hotwords,
    });
    if let Err(e) = srv.stdin.write_all(format!("{req}\n").as_bytes()).await {
        *guard = None;
        return Err(format!("faster-whisper 写入失败（进程已退出，将自动重启）: {e}"));
    }
    if let Err(e) = srv.stdin.flush().await {
        *guard = None;
        return Err(format!("faster-whisper 写入失败: {e}"));
    }

    // 带超时等待响应，避免 Python 卡死时应用永久挂起
    let resp = match tokio::time::timeout(IPC_TIMEOUT, srv.stdout.next_line()).await {
        Ok(Ok(Some(l))) => l,
        Ok(Ok(None)) | Ok(Err(_)) => {
            let detail = stderr_diag(&srv.stderr_tail);
            *guard = None;
            return Err(format!("faster-whisper 无响应（进程已退出，将自动重启）{detail}"));
        }
        Err(_) => {
            let detail = stderr_diag(&srv.stderr_tail);
            *guard = None;
            return Err(format!(
                "faster-whisper 识别超时（{}s，将自动重启）{detail}",
                IPC_TIMEOUT.as_secs()
            ));
        }
    };
    let v: serde_json::Value =
        serde_json::from_str(&resp).map_err(|e| format!("解析响应失败: {e}"))?;
    if let Some(err) = v["error"].as_str() {
        return Err(format!("faster-whisper 识别错误: {err}"));
    }
    let arr = v["segments"].as_array().ok_or("响应缺少 segments")?;
    let out = arr
        .iter()
        .enumerate()
        .map(|(i, s)| asr::Segment {
            index: i + 1,
            start: s["start"].as_f64().unwrap_or(0.0),
            end: s["end"].as_f64().unwrap_or(0.0),
            text: s["text"].as_str().unwrap_or("").trim().to_string(),
        })
        .collect();
    Ok(FwTranscription {
        segments: out,
        language: v["language"].as_str().map(|s| s.to_string()),
        language_probability: v["language_probability"].as_f64(),
    })
}
