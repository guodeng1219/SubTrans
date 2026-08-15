//! 常驻 BS-RoFormer 人声分离 worker：嵌入式 Python 服务生命周期 + 类型化 JSON Lines IPC。
//!
//! 与 `fw_ipc.rs` 相同的模式，但请求发送与响应读取分离，供 Task 5 的内存监控
//! 在等待响应期间以 `tokio::select!` 轮询。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// 内嵌的人声分离服务脚本（编译进二进制，运行时写到临时文件，自包含）。
const VOCAL_SERVER_PY: &str = include_str!("../../python/vocal_server.py");

/// 启动加载超时：BS-RoFormer 首次加载模型可能较久。
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
/// 单片分离超时。
const CHUNK_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// 就绪等待期间的取消轮询间隔（与管线其余部分的 CANCEL_TICK 一致）。
const CANCEL_TICK: Duration = Duration::from_millis(500);

/// 一次成功分离响应的类型化结果。
#[derive(Clone, Debug)]
pub struct VocalResponse {
    pub vocals_path: PathBuf,
    pub elapsed_ms: u64,
}

/// 类型化 worker 错误：`code` 为稳定错误码（`separation_failed` / `memory_error` /
/// `vocal_worker_exited` / `vocal_chunk_timeout` / `vocal_protocol_error`），
/// `message` 为可展示的诊断信息。
#[derive(Clone, Debug)]
pub struct VocalWorkerError {
    pub code: String,
    pub message: String,
}

/// 解析一行成功/失败分离响应。
/// - `ok: true` 且 `request_id` 匹配 → `VocalResponse`。
/// - `request_id` 不匹配 → 协议错误（消息包含 request_id 供诊断）。
/// - `ok: false` → 原样透传 `error_code` / `message`。
pub fn parse_response(row: &str, request_id: &str) -> Result<VocalResponse, VocalWorkerError> {
    let v: serde_json::Value = serde_json::from_str(row).map_err(|e| VocalWorkerError {
        code: "vocal_protocol_error".into(),
        message: e.to_string(),
    })?;
    let got_id = v.get("request_id").and_then(|x| x.as_str()).unwrap_or("");
    if got_id != request_id {
        return Err(VocalWorkerError {
            code: "vocal_protocol_error".into(),
            message: format!("响应 request_id 不匹配：期望 {request_id}，实际 {got_id}"),
        });
    }
    if v.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        return Err(VocalWorkerError {
            code: v
                .get("error_code")
                .and_then(|x| x.as_str())
                .unwrap_or("vocal_protocol_error")
                .to_string(),
            message: v
                .get("message")
                .and_then(|x| x.as_str())
                .unwrap_or("未知分离错误")
                .to_string(),
        });
    }
    let vocals_path =
        v.get("vocals_path").and_then(|x| x.as_str()).ok_or_else(|| VocalWorkerError {
            code: "vocal_protocol_error".into(),
            message: "响应缺少 vocals_path".into(),
        })?;
    Ok(VocalResponse {
        vocals_path: PathBuf::from(vocals_path),
        elapsed_ms: v.get("elapsed_ms").and_then(|x| x.as_u64()).unwrap_or(0),
    })
}

/// 解析启动就绪行。
/// - `ready: true` → 返回 PID。
/// - `ready: false` → 错误消息统一带 `vocal_model_unavailable` 前缀，
///   使 Rust 端能把模型下载/组件缺失与内存限制等其它失败区分开。
pub fn parse_ready(row: &str) -> Result<u32, String> {
    let v: serde_json::Value =
        serde_json::from_str(row).map_err(|e| format!("解析就绪信号失败: {e}（输出: {row}）"))?;
    if v.get("ready").and_then(|x| x.as_bool()) == Some(true) {
        return v
            .get("pid")
            .and_then(|x| x.as_u64())
            .map(|p| p as u32)
            .ok_or_else(|| format!("就绪信号缺少 pid（输出: {row}）"));
    }
    let err = v.get("error").and_then(|x| x.as_str()).unwrap_or("未知错误");
    let code = v.get("error_code").and_then(|x| x.as_str()).unwrap_or("model_load_failed");
    Err(format!("vocal_model_unavailable: [{code}] {err}"))
}

/// 常驻的 BS-RoFormer 分离服务进程。
pub struct VocalWorker {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    /// stderr 尾部缓存（最多 20 行非空，用于崩溃诊断）
    stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>>,
    script_path: PathBuf,
}

/// 已 spawn、尚未完成就绪握手的服务。
/// 调用方先取 `pid()` 绑定 Job 硬限，再调用 `wait_ready()`：
/// `wait_ready` 发出 `{"op":"load"}` 启动屏障指令后，Python 才开始加载模型——
/// 「Job 在模型加载前生效」由协议保证，而不是 spawn/PID/绑 Job 之间的竞态侥幸。
/// 字段用 Option 包裹：wait_ready 取走所有权，而 Drop 在「Job 绑定失败 /
/// future 被取消」等未到达 wait_ready 的路径上兜底清理临时脚本。
pub struct PendingWorker {
    child: Option<tokio::process::Child>,
    stdin: Option<tokio::process::ChildStdin>,
    stdout: Option<tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>>,
    stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>>,
    script_path: Option<PathBuf>,
}

impl PendingWorker {
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().and_then(|c| c.id())
    }

    /// 发加载指令并等待就绪信号；`token` 失效（取消/换代）时立即终止。
    /// 失败/超时/取消都先杀子进程并清理临时脚本。
    /// 注意：`script_path` 一直留在 self 里直到函数收尾同步取出——
    /// 若整个 future 被外层取消（drop），PendingWorker::Drop 仍能删掉临时脚本，
    /// 本地 PathBuf 析构不会清理磁盘文件。
    pub async fn wait_ready(
        mut self,
        state: &crate::vocal_pipeline::VocalState,
        token: u64,
    ) -> Result<VocalWorker, String> {
        use tokio::io::AsyncWriteExt;
        let mut child =
            self.child.take().ok_or_else(|| "人声分离服务状态异常（child 已缺失）".to_string())?;
        let mut stdin =
            self.stdin.take().ok_or_else(|| "人声分离服务状态异常（stdin 已缺失）".to_string())?;
        let mut stdout = self
            .stdout
            .take()
            .ok_or_else(|| "人声分离服务状态异常（stdout 已缺失）".to_string())?;

        // 取消可能发生在 spawn 之后：进入等待前先校验一次令牌
        if state.is_cancelled(token) {
            let _ = child.kill().await;
            self.remove_script();
            return Err("任务已取消".into());
        }

        // 启动屏障：Job 硬限此刻已绑定（调用方在 wait_ready 之前完成），
        // 才允许 Python 加载模型。
        let mut load_sent = stdin.write_all(b"{\"op\":\"load\"}\n").await;
        if load_sent.is_ok() {
            load_sent = stdin.flush().await;
        }
        if let Err(e) = load_sent {
            let _ = child.kill().await;
            self.remove_script();
            return Err(format!("写入加载指令失败（进程可能已退出）: {e}"));
        }

        // 就绪等待：取消用 500ms 轮询（与管线一致的可靠模式），
        // 取消后旧模型进程在 500ms 内被终止，不会与新任务叠加加载
        let line = loop {
            tokio::select! {
                r = tokio::time::timeout(STARTUP_TIMEOUT, stdout.next_line()) => {
                    match r {
                        Ok(Ok(Some(l))) => break l,
                        Ok(Ok(None)) => {
                            let _ = child.kill().await;
                            self.remove_script();
                            return Err(format!(
                                "人声分离服务未输出就绪信号（进程已退出）{}",
                                stderr_diag_of(&self.stderr_tail)
                            ));
                        }
                        Ok(Err(e)) => {
                            let _ = child.kill().await;
                            self.remove_script();
                            return Err(format!("读取就绪信号失败: {e}"));
                        }
                        Err(_) => {
                            let _ = child.kill().await;
                            self.remove_script();
                            return Err(format!(
                                "人声分离服务加载超时（{}s）{}",
                                STARTUP_TIMEOUT.as_secs(),
                                stderr_diag_of(&self.stderr_tail)
                            ));
                        }
                    }
                }
                _ = tokio::time::sleep(CANCEL_TICK) => {
                    if state.is_cancelled(token) {
                        let _ = child.kill().await;
                        self.remove_script();
                        return Err("任务已取消".into());
                    }
                }
            }
        };
        // 同步收尾：取走脚本路径后不再有任何 await，外层取消无法介入
        let script_path = self
            .script_path
            .take()
            .ok_or_else(|| "人声分离服务状态异常（脚本路径已缺失）".to_string())?;
        match parse_ready(&line) {
            Ok(_pid) => Ok(VocalWorker {
                child,
                stdin,
                stdout,
                stderr_tail: self.stderr_tail.clone(),
                script_path,
            }),
            Err(e) => {
                // 先删脚本再 kill：即便此 await 期间 future 被 drop，脚本也已清理
                let _ = std::fs::remove_file(&script_path);
                let _ = child.kill().await;
                Err(e)
            }
        }
    }

    fn remove_script(&mut self) {
        if let Some(script) = self.script_path.take() {
            let _ = std::fs::remove_file(script);
        }
    }
}

impl Drop for PendingWorker {
    fn drop(&mut self) {
        // wait_ready 未到达（Job 绑定失败 / future 被取消）时兜底清理临时脚本；
        // 子进程由 spawn 时设置的 kill_on_drop(true) 负责终止。
        if let Some(script) = self.script_path.take() {
            let _ = std::fs::remove_file(script);
        }
    }
}

impl VocalWorker {
    /// spawn 子进程（不等待就绪）：返回的 [`PendingWorker`] 可先取 PID
    /// 用于在模型加载前绑定 Job 硬限，再调用 `wait_ready()` 完成握手。
    pub async fn spawn(
        python_exe: &str,
        model_dir: &Path,
        model: &str,
        device: &str,
    ) -> Result<PendingWorker, String> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let script_path = std::env::temp_dir().join(format!(
            "subtrans_vocal_server_{}_{}.py",
            std::process::id(),
            stamp
        ));
        std::fs::write(&script_path, VOCAL_SERVER_PY)
            .map_err(|e| format!("写入人声分离服务脚本失败: {e}"))?;

        let mut cmd = tokio::process::Command::new(python_exe);
        cmd.arg(&script_path)
            .arg(model_dir)
            .arg(model)
            .arg(device)
            .env("PYTHONUTF8", "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(target_os = "windows")]
        {
            cmd.creation_flags(0x0800_0000);
        }
        let mut child = cmd.spawn().map_err(|e| {
            format!("启动人声分离服务失败: {e}（确认 Python 路径与 audio-separator 已安装）")
        })?;
        let stdin = child.stdin.take().ok_or("无法获取 stdin")?;
        let stdout = child.stdout.take().ok_or("无法获取 stdout")?;
        let stderr = child.stderr.take().ok_or("无法获取 stderr")?;
        use tokio::io::AsyncBufReadExt;
        let lines = tokio::io::BufReader::new(stdout).lines();

        // 后台读取 stderr 尾部（崩溃诊断）
        let stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        {
            let tail = stderr_tail.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut err_lines = tokio::io::BufReader::new(stderr).lines();
                loop {
                    match err_lines.next_line().await {
                        Ok(Some(line)) => {
                            let line = line.trim().to_string();
                            if !line.is_empty() {
                                let mut t = tail.lock().unwrap();
                                t.push_back(line);
                                while t.len() > 20 {
                                    t.pop_front();
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(_) => continue, // 坏字节不终止诊断采集
                    }
                }
            });
        }

        Ok(PendingWorker {
            child: Some(child),
            stdin: Some(stdin),
            stdout: Some(lines),
            stderr_tail,
            script_path: Some(script_path),
        })
    }

    /// 发送一次分离请求（不等待响应；响应由 [`Self::read_response`] 读取）。
    pub async fn send_separate(
        &mut self,
        request_id: &str,
        input_path: &Path,
        output_dir: &Path,
    ) -> Result<(), String> {
        use tokio::io::AsyncWriteExt;
        let req = serde_json::json!({
            "op": "separate",
            "request_id": request_id,
            "input_path": input_path.to_string_lossy(),
            "output_dir": output_dir.to_string_lossy(),
        });
        self.stdin
            .write_all(format!("{req}\n").as_bytes())
            .await
            .map_err(|e| format!("人声分离写入失败（进程可能已退出）: {e}"))?;
        self.stdin.flush().await.map_err(|e| format!("人声分离写入失败: {e}"))
    }

    /// 读取并解析指定 request_id 的响应（带单片超时）。
    pub async fn read_response(
        &mut self,
        request_id: &str,
    ) -> Result<VocalResponse, VocalWorkerError> {
        let line = match tokio::time::timeout(CHUNK_TIMEOUT, self.stdout.next_line()).await {
            Ok(Ok(Some(l))) => l,
            Ok(Ok(None)) | Ok(Err(_)) => {
                return Err(VocalWorkerError {
                    code: "vocal_worker_exited".into(),
                    message: format!("人声分离服务已退出 {}", self.stderr_diag()),
                });
            }
            Err(_) => {
                return Err(VocalWorkerError {
                    code: "vocal_chunk_timeout".into(),
                    message: format!("人声分离单片超时（{}s）", CHUNK_TIMEOUT.as_secs()),
                });
            }
        };
        parse_response(&line, request_id)
    }

    /// 优雅关闭：发 shutdown、等 1 秒，仍存活则强杀；随后清理临时脚本。
    pub async fn shutdown(&mut self) {
        use tokio::io::AsyncWriteExt;
        let _ = self.stdin.write_all(b"{\"op\":\"shutdown\"}\n").await;
        let _ = self.stdin.flush().await;
        let _ = tokio::time::timeout(Duration::from_secs(1), self.child.wait()).await;
        let _ = self.child.kill().await;
        let _ = std::fs::remove_file(&self.script_path);
    }

    /// 立即强杀进程（不清理脚本，交由 Drop/后续 shutdown 兜底）。
    pub async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }

    pub fn stderr_diag(&self) -> String {
        stderr_diag_of(&self.stderr_tail)
    }
}

impl Drop for VocalWorker {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.script_path);
    }
}

fn stderr_diag_of(tail: &Arc<Mutex<std::collections::VecDeque<String>>>) -> String {
    let t = tail.lock().unwrap();
    if t.is_empty() {
        String::new()
    } else {
        format!("stderr: {}", t.iter().cloned().collect::<Vec<_>>().join(" | "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_response_preserves_request_and_path() {
        let row =
            r#"{"request_id":"sep-0007","ok":true,"vocals_path":"C:/tmp/v.wav","elapsed_ms":42}"#;
        let parsed = parse_response(row, "sep-0007").unwrap();
        assert_eq!(parsed.vocals_path, std::path::PathBuf::from("C:/tmp/v.wav"));
        assert_eq!(parsed.elapsed_ms, 42);
    }

    #[test]
    fn mismatched_request_id_and_worker_error_are_rejected() {
        let mismatch = r#"{"request_id":"other","ok":true,"vocals_path":"v.wav","elapsed_ms":1}"#;
        assert!(parse_response(mismatch, "expected").unwrap_err().message.contains("request_id"));
        let failed =
            r#"{"request_id":"a","ok":false,"error_code":"separation_failed","message":"boom"}"#;
        assert_eq!(parse_response(failed, "a").unwrap_err().code, "separation_failed");
        let oom = r#"{"request_id":"a","ok":false,"error_code":"memory_error","message":"CUDA out of memory"}"#;
        assert_eq!(parse_response(oom, "a").unwrap_err().code, "memory_error");
    }

    #[test]
    fn startup_requires_ready_true() {
        assert!(parse_ready(r#"{"ready":true,"pid":123}"#).is_ok());
        let err = parse_ready(
            r#"{"ready":false,"error_code":"model_load_failed","error":"model missing"}"#,
        )
        .unwrap_err();
        assert!(err.contains("vocal_model_unavailable"));
    }
}
