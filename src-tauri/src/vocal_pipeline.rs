//! 有界人声分离编排器：规范 PCM 解码 → 整数切窗 → 常驻 worker 逐片分离 →
//! 整数裁芯 → FFmpeg 流式拼接 → 校验发布。带取消、一次性 120 秒降片重试与内存守卫。

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::ffmpeg;
use crate::process_memory::{
    classify_worker_failure, effective_memory_budget_bytes, JobGuard, MemoryDecision, MemoryGuard,
    ProcessMemoryProbe, WorkerFailureClass,
};
use crate::task_log::TaskLogger;
use crate::vocal_chunk::{
    build_chunk_plan, build_remaining_plan, VocalChunkPlan, DEFAULT_CORE_SEC, GUARD_SEC,
    RETRY_CORE_SEC,
};
use crate::vocal_worker::VocalWorker;
use crate::{emit_vocal_progress, VocalProgressFields};

const CANONICAL_DECODE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const WINDOW_EXTRACT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const CONCAT_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const MEMORY_TICK: Duration = Duration::from_secs(2);
const CANCEL_TICK: Duration = Duration::from_millis(500);

/// 会话级取消/发布状态：换代使旧任务立即失效，已发布的人声轨由会话生命周期管理。
/// 取消感知统一用「500ms 轮询 + is_cancelled」模式（与 run_cancellable 一致），
/// 不引入 watch 通道——避免 receiver 生命周期与 tokio 无订阅者丢值等边界。
#[derive(Default, Clone)]
pub struct VocalState {
    generation: Arc<AtomicU64>,
    published_output: Arc<Mutex<Option<PathBuf>>>,
}

impl VocalState {
    /// 开始新任务：只分配换代令牌，**不**删除旧发布——旧人声轨仍归旧会话所有，
    /// 由 `cancel()`（显式丢弃）、`publish_output()`（成功发布时原子替换）或
    /// `release_output()`（归属校验后释放）负责清理。
    /// 这样预检失败的新任务不会破坏「发布只在成功时被替换」的不变量：
    /// 令牌先行 + 预检失败 = 旧会话的人声轨原样保留。
    pub fn begin(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// 取消当前任务：换代与删除旧发布在**同一锁临界区**内原子完成——
    /// 旧 begin/cancel 不可能在「换代之后、删除之前」被新任务抢占，
    /// 从而绝不会误删新任务刚发布的人声轨。
    pub fn cancel(&self) {
        let mut guard = self.published_output.lock().unwrap();
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(p) = guard.take() {
            let _ = std::fs::remove_file(&p);
        }
    }

    pub fn is_cancelled(&self, token: u64) -> bool {
        self.generation.load(Ordering::SeqCst) != token
    }

    /// 发布最终人声轨路径：令牌检查与替换发布在**同一把锁内**原子完成——
    /// 若取消恰好在两者之间发生，过期人声轨被直接删除并报错，绝不进入已发布状态。
    pub fn publish_output(&self, token: u64, path: PathBuf) -> Result<(), String> {
        let mut guard = self.published_output.lock().unwrap();
        if self.generation.load(Ordering::SeqCst) != token {
            drop(guard);
            let _ = std::fs::remove_file(&path);
            return Err("vocal_cancelled".into());
        }
        if let Some(old) = guard.take() {
            let _ = std::fs::remove_file(&old);
        }
        *guard = Some(path);
        Ok(())
    }

    /// 释放已发布的人声轨（识别会话结束后调用）。
    /// 按路径校验归属：只有 `expected` 与当前发布路径一致才删除——
    /// 旧会话延迟到达的释放指令不会误删新会话的人声轨，也不会清空其发布状态。
    pub fn release_output(&self, expected: &Path) {
        let mut guard = self.published_output.lock().unwrap();
        let matches = match &*guard {
            Some(current) => current == expected,
            None => false,
        };
        if matches {
            if let Some(p) = guard.take() {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
}

/// 稳定错误码 + 用户可读信息。
#[derive(Debug)]
pub struct VocalPipelineError {
    code: &'static str,
    message: String,
}

impl VocalPipelineError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn user_message(&self) -> String {
        self.message.clone()
    }
}

/// 一次尝试的重试状态机：记录已完成前缀，只允许一次 240→120 秒降片。
pub struct PipelineAttemptState {
    total_frames: u64,
    completed_until_frame: u64,
    retry_used: bool,
}

impl PipelineAttemptState {
    pub fn new(total_frames: u64) -> Self {
        Self { total_frames, completed_until_frame: 0, retry_used: false }
    }

    pub fn mark_completed(&mut self, frame: u64) {
        self.completed_until_frame = self.completed_until_frame.max(frame);
    }

    pub fn retry_used(&self) -> bool {
        self.retry_used
    }

    /// 内存超限：第一次返回 120 秒核心的剩余计划；第二次返回稳定错误码。
    pub fn on_memory_limit(&mut self) -> Result<VocalChunkPlan, VocalPipelineError> {
        if self.retry_used {
            return Err(VocalPipelineError::new(
                "vocal_memory_limit_exceeded",
                "120 秒分片仍内存超限，任务终止（请尝试 GPU 或关闭高精度模式）",
            ));
        }
        self.retry_used = true;
        build_remaining_plan(
            self.total_frames,
            self.completed_until_frame,
            RETRY_CORE_SEC,
            GUARD_SEC,
        )
        .map_err(|e| VocalPipelineError::new("vocal_memory_limit_exceeded", e))
    }
}

/// 编排器配置（路径均借自 separate_vocals_inner 的栈上变量）。
pub struct VocalPipelineConfig<'a> {
    pub video_path: &'a str,
    pub python_exe: &'a str,
    pub model: &'a str,
    pub device: &'a str,
    pub ffmpeg_bin: &'a str,
    pub model_dir: &'a Path,
    pub work_dir: &'a Path,
    pub stable_output: &'a Path,
}

pub struct VocalPipelineResult {
    pub output_path: PathBuf,
    pub peak_private_bytes: u64,
    pub chunk_count: usize,
    pub retried_with_smaller_chunks: bool,
}

/// 一次 worker 尝试的上下文（worker + Job 硬限 + 采样器 + 软限守卫）。
struct WorkerCtx {
    worker: VocalWorker,
    job: JobGuard,
    probe: ProcessMemoryProbe,
    memory_guard: MemoryGuard,
    budget: u64,
}

impl WorkerCtx {
    async fn start(
        config: &VocalPipelineConfig<'_>,
        logger: &TaskLogger,
        state: &VocalState,
        token: u64,
    ) -> Result<Self, VocalPipelineError> {
        if state.is_cancelled(token) {
            return Err(VocalPipelineError::new("vocal_cancelled", "任务已取消"));
        }
        // 启动屏障三阶段：spawn → 取 PID 绑 Job 硬限 → wait_ready 才发 load 指令。
        // Python 端在收到 load 之前绝不加载模型，因此「模型加载前 Job 已生效」是
        // 协议保证而非竞态下的侥幸；等待就绪期间取消会立即杀掉旧进程。
        let pending =
            VocalWorker::spawn(config.python_exe, config.model_dir, config.model, config.device)
                .await
                .map_err(|e| {
                    let code = if e.contains("vocal_model_unavailable") {
                        "vocal_model_unavailable"
                    } else {
                        "vocal_worker_start_failed"
                    };
                    VocalPipelineError::new(code, e)
                })?;
        let pid = pending
            .pid()
            .ok_or_else(|| VocalPipelineError::new("vocal_worker_start_failed", "worker 无 PID"))?;
        let mut probe = ProcessMemoryProbe::new(pid);
        let physical = probe.total_physical_bytes();
        let budget = effective_memory_budget_bytes(physical);
        // Job 绑定失败：pending 随作用域 Drop（kill_on_drop 杀子进程 + 清理临时脚本）
        let job = JobGuard::assign(pid, budget)
            .map_err(|e| VocalPipelineError::new("vocal_worker_start_failed", e))?;
        let memory_guard = MemoryGuard::new(budget);
        let worker = pending.wait_ready(state, token).await.map_err(|e| {
            let code = if e.contains("vocal_model_unavailable") {
                "vocal_model_unavailable"
            } else if e.contains("任务已取消") {
                "vocal_cancelled"
            } else {
                "vocal_worker_start_failed"
            };
            VocalPipelineError::new(code, e)
        })?;
        let _ = logger.write(
            "worker_started",
            &serde_json::json!({"pid": pid, "physical_bytes": physical, "budget_bytes": budget}),
        );
        Ok(Self { worker, job, probe, memory_guard, budget })
    }
}

/// 单片处理结果：Done（继续）或 MemoryLimit（终止当前 worker 并走降片重试）。
enum ChunkOutcome {
    Done,
    MemoryLimit,
}

/// 可取消/可超时的异步操作包装：每 500ms 检查会话取消，超时映射到指定错误码。
/// 显式 Send 约束：Tauri 命令的 future 必须可跨线程（在 async 运行时上迁移）。
pub async fn run_cancellable<T, E, F>(
    state: &VocalState,
    token: u64,
    timeout: Duration,
    err_code: &'static str,
    fut: F,
) -> Result<T, VocalPipelineError>
where
    E: ToString + Send,
    F: Future<Output = Result<T, E>> + Send,
    T: Send,
{
    tokio::pin!(fut);
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        tokio::select! {
            r = &mut fut => {
                return r.map_err(|e| VocalPipelineError::new(err_code, e.to_string()));
            }
            _ = tokio::time::sleep(CANCEL_TICK) => {
                if state.is_cancelled(token) {
                    return Err(VocalPipelineError::new("vocal_cancelled", "任务已取消"));
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(VocalPipelineError::new(err_code, "操作超时"));
                }
            }
        }
    }
}

/// 进度封装：pct 按已完成核心帧数计算。
#[allow(clippy::too_many_arguments)] // 分片进度展示所需的完整上下文
fn emit_chunk_progress(
    app: &tauri::AppHandle,
    done: usize,
    total: usize,
    completed_frames: u64,
    total_frames: u64,
    peak_private_bytes: u64,
    budget_bytes: u64,
    retrying: bool,
    warning: bool,
) {
    let pct =
        if total_frames > 0 { completed_frames as f64 / total_frames as f64 * 100.0 } else { 0.0 };
    emit_vocal_progress(
        app,
        pct,
        &format!("高精度人声分离 {}/{}", done + 1, total),
        VocalProgressFields {
            chunk_index: done,
            chunk_total: total,
            memory_bytes: peak_private_bytes,
            memory_peak_bytes: peak_private_bytes,
            memory_budget_bytes: budget_bytes,
            retrying_with_smaller_chunks: retrying,
            warning,
        },
    );
}

#[allow(clippy::too_many_arguments)] // 单片步骤的上下文参数，保持扁平与计划一致
async fn process_one_chunk(
    app: &tauri::AppHandle,
    state: &VocalState,
    token: u64,
    source_wav: &Path,
    chunk: &crate::vocal_chunk::VocalChunk,
    global_index: usize,
    chunk_total: usize,
    completed_frames: u64,
    total_frames: u64,
    work_dir: &Path,
    config: &VocalPipelineConfig<'_>,
    ctx: &mut WorkerCtx,
    logger: &TaskLogger,
) -> Result<ChunkOutcome, VocalPipelineError> {
    let inputs_dir = work_dir.join("inputs");
    let separated_dir = work_dir.join("separated");
    let cores_dir = work_dir.join("cores");
    std::fs::create_dir_all(&inputs_dir).ok();
    std::fs::create_dir_all(&separated_dir).ok();
    std::fs::create_dir_all(&cores_dir).ok();

    // 1) 从规范 PCM 整数切窗
    let input_part = inputs_dir.join(format!("input_{global_index:04}.wav.part"));
    let input_wav = inputs_dir.join(format!("input_{global_index:04}.wav"));
    run_cancellable(
        state,
        token,
        WINDOW_EXTRACT_TIMEOUT,
        "vocal_chunk_extract_failed",
        ffmpeg::extract_vocal_window(
            source_wav,
            &input_part,
            config.ffmpeg_bin,
            chunk.extract_start_frame,
            chunk.extract_frames,
        ),
    )
    .await?;
    let input_info = ffmpeg::validate_pcm_wav(&input_part)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_extract_failed", e.to_string()))?;
    if input_info.frames != chunk.extract_frames {
        let _ = std::fs::remove_file(&input_part);
        return Err(VocalPipelineError::new(
            "vocal_chunk_extract_failed",
            format!("提取窗口帧数不符：期望 {}，实际 {}", chunk.extract_frames, input_info.frames),
        ));
    }
    std::fs::rename(&input_part, &input_wav)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_extract_failed", e.to_string()))?;

    // 2) 送 worker 分离，等待期间采样内存 / 响应取消。
    // 关键：read_response future 只创建并 pin 一次——它内部的 30 分钟单片超时
    // 从创建时刻起算，select! 循环迭代不能重建它（否则每次取消轮询都会重置超时，
    // 卡死的分离只能等整体 2 小时兜底）。
    let output_dir = separated_dir.join(format!("output_{global_index:04}"));
    let request_id = format!("sep-{global_index:04}");
    ctx.worker
        .send_separate(&request_id, &input_wav, &output_dir)
        .await
        .map_err(|e| VocalPipelineError::new("vocal_worker_exited", e))?;

    let resp = {
        let resp_future = ctx.worker.read_response(&request_id);
        tokio::pin!(resp_future);
        let result = loop {
            tokio::select! {
                r = &mut resp_future => {
                    break Some(r);
                }
                _ = tokio::time::sleep(MEMORY_TICK) => {
                    match ctx.probe.sample() {
                        Ok(sample) => {
                            match ctx.memory_guard.observe(sample) {
                                MemoryDecision::Warn => {
                                    let _ = logger.write(
                                        "memory_warn",
                                        &serde_json::json!({
                                            "budget_bytes": ctx.budget,
                                            "working_set_bytes": sample.working_set_bytes,
                                            "private_bytes": sample.private_bytes,
                                        }),
                                    );
                                    emit_chunk_progress(
                                        app,
                                        global_index,
                                        chunk_total,
                                        completed_frames,
                                        total_frames,
                                        ctx.memory_guard.peak_private_bytes(),
                                        ctx.budget,
                                        false,
                                        true,
                                    );
                                }
                                MemoryDecision::Exceeded => {
                                    let _ = logger.write("memory_exceeded", &serde_json::json!({"budget_bytes": ctx.budget}));
                                    // Job 整树终止（含 worker 根进程；resp_future 仍借用 ctx.worker，不能在此 kill）
                                    ctx.job.terminate(1);
                                    return Ok(ChunkOutcome::MemoryLimit);
                                }
                                MemoryDecision::Continue => {}
                            }
                        }
                        Err(_) => {
                            // root 消失 → read_response 会以 worker_exited 收场，交给下一次 select 分支
                        }
                    }
                }
                _ = tokio::time::sleep(CANCEL_TICK) => {
                    if state.is_cancelled(token) {
                        // 同上：整树终止，避免与 resp_future 的 &mut ctx.worker 借用冲突
                        ctx.job.terminate(1);
                        return Err(VocalPipelineError::new("vocal_cancelled", "任务已取消"));
                    }
                }
            }
        };
        // 内层作用域结束：future 本体随作用域销毁，&mut ctx.worker 借用随之释放
        result
    };

    // 3) 响应分类
    let response = match resp {
        None => {
            ctx.worker.kill().await;
            return Err(VocalPipelineError::new("vocal_worker_exited", "worker 无响应"));
        }
        Some(Ok(r)) => r,
        Some(Err(werr)) => {
            // 三证据分类：最后采样峰值 / Job 峰值 / 明确 OOM
            let job_peak = ctx.job.peak_job_memory_bytes().ok();
            let class = classify_worker_failure(
                ctx.memory_guard.peak_private_bytes(),
                job_peak,
                ctx.budget,
                werr.code == "memory_error",
            );
            if class == WorkerFailureClass::MemoryLimit {
                ctx.job.terminate(1);
                ctx.worker.kill().await;
                let _ = logger.write(
                    "worker_failure_classified_memory",
                    &serde_json::json!({
                        "worker_error_code": werr.code,
                        "job_peak_bytes": job_peak,
                        "budget_bytes": ctx.budget,
                    }),
                );
                return Ok(ChunkOutcome::MemoryLimit);
            }
            ctx.job.terminate(1);
            ctx.worker.kill().await;
            let code: &'static str = match werr.code.as_str() {
                "vocal_worker_exited" => "vocal_worker_exited",
                "vocal_chunk_timeout" => "vocal_chunk_timeout",
                "vocal_protocol_error" => "vocal_protocol_error",
                _ => "vocal_chunk_invalid",
            };
            return Err(VocalPipelineError::new(code, werr.message));
        }
    };

    // 4) 整数裁芯
    let separated_info = ffmpeg::validate_pcm_wav(&response.vocals_path)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_invalid", e.to_string()))?;
    let trim_end = chunk
        .trim_start_frame
        .checked_add(chunk.core_frames)
        .ok_or_else(|| VocalPipelineError::new("vocal_chunk_invalid", "裁剪终点溢出"))?;
    if separated_info.frames < trim_end {
        let _ = std::fs::remove_file(&input_wav);
        let _ = std::fs::remove_dir_all(&output_dir);
        return Err(VocalPipelineError::new(
            "vocal_chunk_invalid",
            format!("分离输出长度不足：期望 ≥{} 帧，实际 {} 帧", trim_end, separated_info.frames),
        ));
    }
    let core_part = cores_dir.join(format!("vocals_{global_index:04}.wav.part"));
    let core_wav = cores_dir.join(format!("vocals_{global_index:04}.wav"));
    run_cancellable(
        state,
        token,
        WINDOW_EXTRACT_TIMEOUT,
        "vocal_chunk_invalid",
        ffmpeg::trim_vocal_core(
            &response.vocals_path,
            &core_part,
            config.ffmpeg_bin,
            chunk.trim_start_frame,
            chunk.core_frames,
        ),
    )
    .await?;
    let core_info = ffmpeg::validate_pcm_wav(&core_part)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_invalid", e.to_string()))?;
    if core_info.frames != chunk.core_frames {
        let _ = std::fs::remove_file(&core_part);
        return Err(VocalPipelineError::new(
            "vocal_chunk_invalid",
            format!("核心片帧数不符：期望 {}，实际 {}", chunk.core_frames, core_info.frames),
        ));
    }
    std::fs::rename(&core_part, &core_wav)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_invalid", e.to_string()))?;

    // 5) 立即清理本片输入与未裁剪分离结果
    let _ = std::fs::remove_file(&input_wav);
    let _ = std::fs::remove_dir_all(&output_dir);

    let _ = logger.write(
        "chunk_done",
        &serde_json::json!({
            "chunk_index": global_index,
            "core_start_frame": chunk.core_start_frame,
            "core_frames": chunk.core_frames,
            "elapsed_ms": response.elapsed_ms,
        }),
    );
    Ok(ChunkOutcome::Done)
}

/// 有界人声分离主编排。
pub async fn run_bounded_vocal_pipeline(
    app: &tauri::AppHandle,
    state: &VocalState,
    token: u64,
    config: VocalPipelineConfig<'_>,
    logger: &TaskLogger,
) -> Result<VocalPipelineResult, VocalPipelineError> {
    if state.is_cancelled(token) {
        return Err(VocalPipelineError::new("vocal_cancelled", "任务已取消"));
    }

    // 1) 规范 PCM 一次解码（worker 未启动，模型不常驻）
    let source_part = config.work_dir.join("source.wav.part");
    let source_wav = config.work_dir.join("source.wav");
    run_cancellable(
        state,
        token,
        CANONICAL_DECODE_TIMEOUT,
        "vocal_source_decode_failed",
        ffmpeg::decode_canonical_audio(config.video_path, &source_part, config.ffmpeg_bin),
    )
    .await?;
    let source_info = ffmpeg::validate_pcm_wav(&source_part)
        .map_err(|e| VocalPipelineError::new("vocal_source_decode_failed", e.to_string()))?;
    std::fs::rename(&source_part, &source_wav)
        .map_err(|e| VocalPipelineError::new("vocal_source_decode_failed", e.to_string()))?;
    let _ = logger.write(
        "source_decoded",
        &serde_json::json!({
            "frames": source_info.frames,
            "bytes": source_wav.metadata().map(|m| m.len()).unwrap_or(0),
        }),
    );

    // 2) 计划 + 尝试循环（每次内存超限重启 worker，最多一次 120 秒降片）
    let mut attempt = PipelineAttemptState::new(source_info.frames);
    let mut plan = build_chunk_plan(source_info.frames, DEFAULT_CORE_SEC, GUARD_SEC)
        .map_err(|e| VocalPipelineError::new("vocal_chunk_invalid", e))?;
    let mut peak_private: u64 = 0;
    let mut chunk_count: usize = 0;
    let mut retried = false;
    // 展示用总分片数：降片重试后 = 已完成的旧片数 + 新计划片数，
    // 保证「第 N/总」单调递增，绝不出现 42/40 之类倒退。
    let mut chunk_total_display = plan.chunks.len();

    loop {
        let mut ctx = WorkerCtx::start(&config, logger, state, token).await?;
        let mut outcome: Result<ChunkOutcome, VocalPipelineError> = Ok(ChunkOutcome::Done);
        for chunk in &plan.chunks {
            match process_one_chunk(
                app,
                state,
                token,
                &source_wav,
                chunk,
                chunk_count,
                chunk_total_display,
                attempt.completed_until_frame,
                source_info.frames,
                config.work_dir,
                &config,
                &mut ctx,
                logger,
            )
            .await
            {
                Ok(ChunkOutcome::Done) => {
                    attempt.mark_completed(chunk.core_end_frame());
                    chunk_count += 1;
                    emit_chunk_progress(
                        app,
                        chunk_count.saturating_sub(1),
                        chunk_total_display,
                        attempt.completed_until_frame,
                        source_info.frames,
                        ctx.memory_guard.peak_private_bytes(),
                        ctx.budget,
                        retried,
                        false,
                    );
                }
                Ok(ChunkOutcome::MemoryLimit) => {
                    outcome = Ok(ChunkOutcome::MemoryLimit);
                    break;
                }
                Err(e) => {
                    outcome = Err(e);
                    break;
                }
            }
        }
        let job_peak = ctx.job.peak_job_memory_bytes().ok();
        peak_private = peak_private.max(ctx.memory_guard.peak_private_bytes());
        ctx.worker.shutdown().await;
        drop(ctx.job);
        let _ = logger.write(
            "attempt_finished",
            &serde_json::json!({
                "peak_private_bytes": peak_private,
                "job_peak_bytes": job_peak,
                "completed_until_frame": attempt.completed_until_frame,
                "retry_used": attempt.retry_used(),
            }),
        );

        match outcome? {
            ChunkOutcome::Done => break,
            ChunkOutcome::MemoryLimit => {
                let next = attempt.on_memory_limit()?;
                retried = true;
                chunk_total_display = chunk_count + next.chunks.len();
                let _ = logger.write(
                    "retry_with_smaller_chunks",
                    &serde_json::json!({"core_sec": next.core_sec, "total_frames": next.total_frames}),
                );
                // 进度不重置到 0%：按已完成核心帧占比继续（单调不减），
                // 分片编号接在已完成旧片之后（下一片是第 chunk_count+1/新总数 片）。
                let retry_pct = if source_info.frames > 0 {
                    attempt.completed_until_frame as f64 / source_info.frames as f64 * 100.0
                } else {
                    0.0
                };
                emit_vocal_progress(
                    app,
                    retry_pct,
                    "内存占用过高，正在改用更小分片重试",
                    VocalProgressFields {
                        chunk_index: chunk_count,
                        chunk_total: chunk_total_display,
                        memory_bytes: peak_private,
                        memory_peak_bytes: peak_private,
                        memory_budget_bytes: ctx.budget,
                        retrying_with_smaller_chunks: true,
                        warning: true,
                    },
                );
                plan = next;
            }
        }
    }

    // 3) 流式拼接 + 校验
    let vocals_part = config.work_dir.join("vocals.wav.part");
    run_cancellable(
        state,
        token,
        CONCAT_TIMEOUT,
        "vocal_concat_failed",
        ffmpeg::concat_vocal_cores(config.work_dir, chunk_count, &vocals_part, config.ffmpeg_bin),
    )
    .await?;
    let final_info = ffmpeg::validate_pcm_wav(&vocals_part)
        .map_err(|e| VocalPipelineError::new("vocal_output_invalid", e.to_string()))?;
    if final_info.frames != source_info.frames {
        return Err(VocalPipelineError::new(
            "vocal_output_invalid",
            format!("拼接总帧数不符：期望 {}，实际 {}", source_info.frames, final_info.frames),
        ));
    }

    // 4) 原子发布到任务目录外
    std::fs::rename(&vocals_part, config.stable_output)
        .map_err(|e| VocalPipelineError::new("vocal_output_invalid", e.to_string()))?;
    state
        .publish_output(token, config.stable_output.to_path_buf())
        .map_err(|e| VocalPipelineError::new("vocal_cancelled", e))?;
    let _ = logger.write(
        "task_done",
        &serde_json::json!({
            "output_frames": final_info.frames,
            "chunk_count": chunk_count,
            "peak_private_bytes": peak_private,
            "retry_used": retried,
        }),
    );
    Ok(VocalPipelineResult {
        output_path: config.stable_output.to_path_buf(),
        peak_private_bytes: peak_private,
        chunk_count,
        retried_with_smaller_chunks: retried,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocal_chunk::frames;

    fn unique_temp_wav(label: &str) -> std::path::PathBuf {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("subtrans-{label}-{}-{stamp}.wav", std::process::id()))
    }

    #[test]
    fn first_memory_limit_retries_remaining_audio_with_120_second_cores() {
        let mut state = PipelineAttemptState::new(frames(5657.258685));
        state.mark_completed(frames(480.0));
        let next = state.on_memory_limit().unwrap();
        assert_eq!(next.core_sec, RETRY_CORE_SEC);
        assert_eq!(next.chunks[0].core_start_frame, frames(480.0));
        assert!(state.retry_used());
    }

    #[test]
    fn second_memory_limit_returns_stable_error_code() {
        let mut state = PipelineAttemptState::new(frames(5657.258685));
        state.on_memory_limit().unwrap();
        let err = state.on_memory_limit().unwrap_err();
        assert_eq!(err.code(), "vocal_memory_limit_exceeded");
    }

    #[test]
    fn session_generation_cancels_only_older_tasks() {
        let state = VocalState::default();
        let first = state.begin();
        let second = state.begin();
        assert!(state.is_cancelled(first));
        assert!(!state.is_cancelled(second));
        state.cancel();
        assert!(state.is_cancelled(second));
    }

    #[test]
    fn trim_start_comes_directly_from_the_exact_frame_plan() {
        let plan = build_chunk_plan(frames(481.0), DEFAULT_CORE_SEC, GUARD_SEC).unwrap();
        assert_eq!(plan.chunks[0].trim_start_frame, 0);
        assert_eq!(plan.chunks[1].trim_start_frame, frames(GUARD_SEC as f64));
    }

    #[test]
    fn releasing_session_output_deletes_the_published_temp_file() {
        let state = VocalState::default();
        let token = state.begin();
        let path = unique_temp_wav("published-vocals");
        std::fs::write(&path, b"temporary vocals").unwrap();
        state.publish_output(token, path.clone()).unwrap();
        state.release_output(&path);
        assert!(!path.exists());
    }

    #[test]
    fn release_with_mismatched_path_keeps_current_published_output() {
        let state = VocalState::default();
        let token = state.begin();
        let published = unique_temp_wav("current-vocals");
        std::fs::write(&published, b"current vocals").unwrap();
        state.publish_output(token, published.clone()).unwrap();
        // 旧会话延迟到达的释放：路径不匹配 → 不动当前发布的人声轨
        let stale = unique_temp_wav("stale-vocals");
        std::fs::write(&stale, b"stale vocals").unwrap();
        state.release_output(&stale);
        assert!(published.exists());
        let _ = std::fs::remove_file(&stale);
        // 归属匹配的释放仍然生效
        state.release_output(&published);
        assert!(!published.exists());
    }

    #[test]
    fn release_with_no_published_output_is_a_noop() {
        let state = VocalState::default();
        state.begin();
        // 从未发布过任何输出：释放不 panic、不产生任何文件操作
        let stale = unique_temp_wav("never-published");
        std::fs::write(&stale, b"stale vocals").unwrap();
        state.release_output(&stale);
        assert!(stale.exists());
        let _ = std::fs::remove_file(&stale);
    }

    #[test]
    fn cancel_deletes_published_output() {
        let state = VocalState::default();
        let token = state.begin();
        let path = unique_temp_wav("cancel-vocals");
        std::fs::write(&path, b"temporary vocals").unwrap();
        state.publish_output(token, path.clone()).unwrap();
        state.cancel();
        assert!(!path.exists());
    }

    #[test]
    fn begin_does_not_delete_previous_published_output() {
        // 令牌先行 ≠ 销毁旧发布：预检失败的新任务不得破坏旧会话的人声轨，
        // 只有 cancel()、成功发布或归属释放才删除文件
        let state = VocalState::default();
        let token = state.begin();
        let path = unique_temp_wav("begin-keeps-vocals");
        std::fs::write(&path, b"previous session vocals").unwrap();
        state.publish_output(token, path.clone()).unwrap();
        let new_token = state.begin(); // 新任务只分配令牌（随后可能预检失败）
        assert!(path.exists());
        // 新任务成功发布时才替换旧文件
        let new_path = unique_temp_wav("begin-replaces-vocals");
        std::fs::write(&new_path, b"new session vocals").unwrap();
        state.publish_output(new_token, new_path.clone()).unwrap();
        assert!(!path.exists());
        assert!(new_path.exists());
        let _ = std::fs::remove_file(&new_path);
    }

    #[test]
    fn stale_session_publish_deletes_file_and_fails() {
        let state = VocalState::default();
        let token = state.begin();
        state.cancel(); // token 过期
        let path = unique_temp_wav("stale-vocals");
        std::fs::write(&path, b"temporary vocals").unwrap();
        assert!(state.publish_output(token, path.clone()).is_err());
        assert!(!path.exists());
    }
}
