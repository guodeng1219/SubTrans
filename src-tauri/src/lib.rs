//! Tauri 后端入口：暴露给前端的命令。

mod asr;
mod correct;
mod ffmpeg;
mod fw_ipc;
mod ollama;
mod python_setup;
mod translate;

use fw_ipc::FwState;

use asr::whisper::WhisperEngine;
use asr::AsrEngine;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use translate::Engine;

/// Whisper ggml 模型下载源
fn whisper_model_url(name: &str) -> String {
    format!("https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin")
}

/// Silero VAD 模型（~2MB，whisper.cpp 内置 VAD 用）
const VAD_MODEL_FILE: &str = "ggml-silero-v5.1.2.onnx";

fn vad_model_url() -> &'static str {
    "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-silero-v5.1.2.onnx"
}

fn data_dir(app: &tauri::AppHandle) -> PathBuf {
    app.path().app_data_dir().unwrap_or_else(|_| dirs::data_dir().unwrap().join("subtrans"))
}

/// 去掉 Windows 扩展路径前缀 `\\?\`（Tauri 的 resource_dir / python 的 sys.executable
/// 可能返回这种形式；留到子进程参数里容易出问题，统一转成普通路径）。
fn normalize_windows_path(s: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        s.strip_prefix(r"\\?\").unwrap_or(s).to_string()
    }
    #[cfg(not(target_os = "windows"))]
    {
        s.to_string()
    }
}

fn whisper_model_path(app: &tauri::AppHandle, name: &str) -> PathBuf {
    data_dir(app).join(format!("ggml-{name}.bin"))
}

fn vad_model_path(app: &tauri::AppHandle) -> PathBuf {
    data_dir(app).join(VAD_MODEL_FILE)
}

/// 查找内置模型：优先 resource_dir/models/（打包内置），回退 data_dir（运行时下载）。
fn bundled_model_path(app: &tauri::AppHandle, filename: &str) -> Option<PathBuf> {
    if let Ok(dir) = app.path().resource_dir() {
        let p = dir.join("models").join(filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// CUDA 升级成功后持久化的 Python 路径文件（保证 env_status 检测升级到的那个 Python）。
fn cuda_python_marker(app: &tauri::AppHandle) -> PathBuf {
    data_dir(app).join("cuda_python_path.txt")
}

fn read_cuda_python_marker(app: &tauri::AppHandle) -> Option<String> {
    let p = cuda_python_marker(app);
    std::fs::read_to_string(&p)
        .ok()
        .map(|s| normalize_windows_path(s.trim()))
        .filter(|s| !s.is_empty())
}

/// 确保 VAD 模型存在：优先用内置的，否则下载到 data_dir。
async fn ensure_vad_model(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    // 优先用打包内置的 VAD 模型
    if let Some(p) = bundled_model_path(app, VAD_MODEL_FILE) {
        return Ok(p);
    }
    let dest = vad_model_path(app);
    if dest.exists() {
        return Ok(dest);
    }
    let dir = data_dir(app);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(5 * 60))
        .build()
        .map_err(|e| e.to_string())?;
    let resp =
        client.get(vad_model_url()).send().await.map_err(|e| format!("下载 VAD 模型失败: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("下载 VAD 模型失败: HTTP {}", resp.status().as_u16()));
    }
    let total = resp.content_length().unwrap_or(0);
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    if total > 0 && bytes.len() as u64 != total {
        return Err(format!("下载 VAD 模型不完整: {}/{}", bytes.len(), total));
    }
    let part = dest.with_extension("part");
    std::fs::write(&part, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

// ── 进度事件 ──

#[derive(Serialize, Clone)]
struct ProgressEvent {
    stage: String,
    pct: f64,
    message: String,
}

pub(crate) fn emit_progress(app: &tauri::AppHandle, stage: &str, pct: f64, message: &str) {
    let _ = app.emit(
        "progress",
        ProgressEvent { stage: stage.to_string(), pct, message: message.to_string() },
    );
}

/// 把失败日志同时输出到控制台（dev 终端的 stderr）和日志文件
/// （`app_data_dir/logs/subtrans.log`，release 版没有控制台也能查）。
/// 日志超过 5MB 时轮转（删掉重建），避免无限膨胀。
pub(crate) fn log_err(app: &tauri::AppHandle, tag: &str, err: &str) {
    let line = format!("[{tag}] {err}");
    eprintln!("[subtrans] {line}");
    let Ok(dir) = app.path().app_data_dir() else { return };
    let log_dir = dir.join("logs");
    if std::fs::create_dir_all(&log_dir).is_err() {
        return;
    }
    let log_file = log_dir.join("subtrans.log");
    if log_file.metadata().map(|m| m.len() > 5 * 1024 * 1024).unwrap_or(false) {
        let _ = std::fs::remove_file(&log_file);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_file) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{ts}] {line}");
    }
}

/// Whisper 模型缓存（按模型名，只加载一次）。tokio Mutex 确保并发请求不会重复加载同一模型。
#[derive(Default)]
struct AsrCache {
    whisper: tokio::sync::Mutex<Option<(String, Arc<WhisperEngine>)>>,
}

impl AsrCache {
    async fn get_or_load_whisper(
        &self,
        model_path: PathBuf,
        model_name: String,
    ) -> Result<Arc<WhisperEngine>, String> {
        let mut guard = self.whisper.lock().await;
        if let Some((name, engine)) = guard.as_ref() {
            if *name == model_name {
                return Ok(engine.clone());
            }
        }
        let path = model_path.clone();
        let engine = tokio::task::spawn_blocking(move || WhisperEngine::load(&path))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let arc = Arc::new(engine);
        *guard = Some((model_name, arc.clone()));
        Ok(arc)
    }
}

// ── 命令 ──

#[tauri::command]
fn model_exists(app: tauri::AppHandle, name: String) -> bool {
    whisper_model_path(&app, &name).exists()
        || bundled_model_path(&app, &format!("ggml-{name}.bin")).is_some()
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, name: String) -> Result<String, String> {
    let res = download_model_inner(app.clone(), name.clone()).await;
    if let Err(e) = &res {
        log_err(&app, "download_model", e);
    }
    res
}

async fn download_model_inner(app: tauri::AppHandle, name: String) -> Result<String, String> {
    use futures_util::StreamExt;

    let dir = data_dir(&app);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let dest = whisper_model_path(&app, &name);
    // 先下到 .part，完成后再 rename，避免中断留下"看似已下载"的损坏文件
    let part = dest.with_extension("part");

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30 * 60))
        .build()
        .map_err(|e| e.to_string())?;
    let url = whisper_model_url(&name);
    let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
    // 校验 HTTP 状态：模型名拼错或镜像 404 时会返回错误页，不校验就会把错误页当成模型存盘，
    // 之后 model_exists 返回 true 却永远加载失败。
    if !resp.status().is_success() {
        return Err(format!("下载识别模型失败：HTTP {} ({url})", resp.status().as_u16()));
    }
    let total = resp.content_length().unwrap_or(0);

    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    use std::io::Write;
    let download_result: Result<(), String> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            let pct = if total > 0 { downloaded as f64 / total as f64 * 100.0 } else { 0.0 };
            emit_progress(&app, "download_model", pct, &format!("下载识别模型 {name}..."));
        }
        Ok(())
    }
    .await;
    if let Err(e) = download_result {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    drop(file);
    if total > 0 && downloaded != total {
        let _ = std::fs::remove_file(&part);
        return Err(format!("下载识别模型不完整: {downloaded}/{total}（请重试）"));
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("重命名模型文件失败: {e}"))?;
    emit_progress(&app, "download_model", 100.0, "模型下载完成");
    Ok(dest.to_string_lossy().to_string())
}

#[derive(Serialize, Clone)]
struct OutSegment {
    index: usize,
    start: f64,
    end: f64,
    original: String,
    translated: String,
}

#[derive(Serialize)]
struct ChunkResult {
    segments: Vec<OutSegment>,
    extract_ms: u64,
    transcribe_ms: u64,
    correct_ms: u64,
    translate_ms: u64,
    engine_used: String,
    /// 自动检测到的源语言（faster-whisper），供前端提示用户是否误判
    detected_lang: Option<String>,
    detected_lang_probability: Option<f64>,
    /// 校对失败提示（如 Ollama 模型未拉取）；成功为 None
    warn: Option<String>,
}

// ── faster-whisper(CT2) 模型：Rust 端直下 ──
//
// fw_server.py 里 huggingface_hub 在国内网络下连镜像都常 SSL/EOF 断连（实测），
// 而 Rust reqwest 从 hf-mirror 直下一直稳定（CPU 模型 download_model 同路）。
// 所以模型下载放在 Rust 端做完，Python 端永远走 local_files_only，不碰网络。

/// 应用自管的 fw 模型根目录（与 fw_server.py resolve_model 的查找路径保持一致）。
pub(crate) fn fw_model_root() -> PathBuf {
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("subtrans-models")
}

/// 在所有已知根目录里找已下载好的 fw 模型（以 model.bin 存在为准）。
fn fw_model_dir_existing(name: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("SUBTRANS_MODELS") {
        if !v.is_empty() {
            roots.push(PathBuf::from(v));
        }
    }
    roots.push(fw_model_root());
    for root in roots {
        let d = root.join(format!("faster-whisper-{name}"));
        if d.join("model.bin").is_file() {
            return Some(d);
        }
    }
    None
}

/// 流式下载一个文件（.part + rename 防半成品）。HTTP 非 2xx 返回 Ok(false) 交调用方裁决。
async fn http_download(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    label: &str,
    report: bool,
) -> Result<bool, String> {
    use futures_util::StreamExt;
    use std::io::Write;
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Ok(false);
    }
    let total = resp.content_length().unwrap_or(0);
    let part = dest.with_extension("part");
    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut got: u64 = 0;
    let mut stream = resp.bytes_stream();
    let download_result: Result<(), String> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            got += chunk.len() as u64;
            if report && total > 0 {
                emit_progress(
                    app,
                    "fw",
                    got as f64 / total as f64 * 100.0,
                    &format!(
                        "下载 GPU 识别模型 {label}（{:.0}/{:.0} MB）...",
                        got as f64 / 1e6,
                        total as f64 / 1e6
                    ),
                );
            }
        }
        Ok(())
    }
    .await;
    if let Err(e) = download_result {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    drop(file);
    if total > 0 && got != total {
        let _ = std::fs::remove_file(&part);
        return Err(format!("下载 {label} 不完整: {got}/{total}（请重试）"));
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    Ok(true)
}

/// 确保所选模型的 CT2 文件齐备；缺则从 hf-mirror 下载到本地模型目录。
/// 小文件先下、model.bin 最后下——fw_server 以 model.bin 存在为"本地完整"标志，
/// 这样中断的下载不会被误判成完整模型。
async fn ensure_fw_model(app: &tauri::AppHandle, name: &str) -> Result<(), String> {
    if fw_model_dir_existing(name).is_some() {
        return Ok(());
    }
    let dir = fw_model_root().join(format!("faster-whisper-{name}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let base = format!("https://hf-mirror.com/Systran/faster-whisper-{name}/resolve/main");
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30 * 60))
        .build()
        .map_err(|e| e.to_string())?;

    emit_progress(app, "fw", 1.0, &format!("准备 GPU 识别模型 {name}（首次需下载）..."));
    // 词表命名两种都试：large-v3 用 vocabulary.json，其余多为 vocabulary.txt，至少要有一个
    let mut have_vocab = false;
    for (fname, required) in [
        ("config.json", true),
        ("tokenizer.json", true),
        ("vocabulary.txt", false),
        ("vocabulary.json", false),
        ("preprocessor_config.json", false),
    ] {
        let dest = dir.join(fname);
        let ok = if dest.is_file() {
            true
        } else {
            http_download(app, &client, &format!("{base}/{fname}"), &dest, name, false).await?
        };
        if ok && fname.starts_with("vocabulary") {
            have_vocab = true;
        }
        if !ok && required {
            return Err(format!(
                "下载模型文件失败: {fname}（hf-mirror 不可达或模型 {name} 不存在）"
            ));
        }
    }
    if !have_vocab {
        return Err("下载模型词表失败（vocabulary.txt / vocabulary.json 均不可得）".into());
    }
    let model_bin = dir.join("model.bin");
    if !model_bin.is_file()
        && !http_download(app, &client, &format!("{base}/model.bin"), &model_bin, name, true)
            .await?
    {
        return Err(format!("下载 model.bin 失败（模型 {name}）"));
    }
    emit_progress(app, "fw", 100.0, "GPU 识别模型就绪");
    Ok(())
}

/// 校验前端传来的 Python 路径，给出比系统 "找不到路径(os error 3)" 更明确的提示。
fn check_python_path(python_exe: &str) -> Result<(), String> {
    let p = python_exe.trim();
    if p.is_empty() {
        return Err(
            "未配置 Python 路径：请到「引擎」页填写 python.exe 的完整路径，或用「一键安装 GPU 加速组件」。"
                .into(),
        );
    }
    // 裸命令名（python/python3/py）交给系统在 PATH 里找；带路径分隔符的按具体文件校验是否存在
    if (p.contains('\\') || p.contains('/')) && !std::path::Path::new(p).exists() {
        return Err(format!("Python 路径不存在：{p}（请到「引擎」页改成正确的 python.exe 路径）"));
    }
    Ok(())
}

/// 解析 bundled Python 路径：生产环境用 python-bundle，开发环境用 PATH。
fn resolve_python(app: &tauri::AppHandle) -> String {
    if let Ok(dir) = app.path().resource_dir() {
        let bundled = dir.join("python-bundle").join("python.exe");
        if bundled.exists() {
            return normalize_windows_path(&bundled.to_string_lossy());
        }
    }
    String::new() // 返回空 → 调用方回退到 PATH 或用户配置
}

/// 解析 ffmpeg 路径：生产环境用 sidecar（与 exe 同级），开发环境用 PATH。
fn resolve_ffmpeg(app: &tauri::AppHandle) -> String {
    // Tauri 2 的 externalBin 把 sidecar 放在主 exe 同级目录
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sidecar = dir.join("ffmpeg.exe");
            if sidecar.exists() {
                return sidecar.to_string_lossy().to_string();
            }
        }
    }
    // 回退：resource_dir（兼容旧布局）
    if let Ok(dir) = app.path().resource_dir() {
        let bundled = dir.join("ffmpeg.exe");
        if bundled.exists() {
            return bundled.to_string_lossy().to_string();
        }
    }
    "ffmpeg".to_string()
}

/// 处理视频的一个时间段。识别引擎：use_fw=true → faster-whisper(GPU)，否则 Whisper(CPU)。
#[tauri::command]
#[allow(clippy::too_many_arguments)] // Tauri 命令参数由前端按名传入，拆成结构体会牵动前端，得不偿失
async fn process_chunk(
    app: tauri::AppHandle,
    cache: tauri::State<'_, AsrCache>,
    fw_state: tauri::State<'_, FwState>,
    video_path: String,
    audio_source: Option<String>,
    model_name: String,
    start_sec: f64,
    duration_sec: f64,
    lead_in_sec: f64,
    total_sec: f64,
    source_lang: Option<String>,
    target_lang: String,
    engine: Engine,
    translate_enabled: bool,
    correct_enabled: bool,
    correct_engine: Option<Engine>,
    glossary: String,
    use_fw: bool,
    fw_python: String,
    fw_device: String,
    vad_enabled: bool,
) -> Result<ChunkResult, String> {
    let res = process_chunk_inner(
        app.clone(),
        cache,
        fw_state,
        video_path,
        audio_source,
        model_name,
        start_sec,
        duration_sec,
        lead_in_sec,
        total_sec,
        source_lang,
        target_lang,
        engine,
        translate_enabled,
        correct_enabled,
        correct_engine,
        glossary,
        use_fw,
        fw_python,
        fw_device,
        vad_enabled,
    )
    .await;
    if let Err(e) = &res {
        log_err(&app, "process_chunk", e);
    }
    res
}

#[allow(clippy::too_many_arguments)]
async fn process_chunk_inner(
    app: tauri::AppHandle,
    cache: tauri::State<'_, AsrCache>,
    fw_state: tauri::State<'_, FwState>,
    video_path: String,
    audio_source: Option<String>,
    model_name: String,
    start_sec: f64,
    duration_sec: f64,
    lead_in_sec: f64,
    total_sec: f64,
    source_lang: Option<String>,
    target_lang: String,
    engine: Engine,
    translate_enabled: bool,
    correct_enabled: bool,
    correct_engine: Option<Engine>,
    glossary: String,
    use_fw: bool,
    fw_python: String,
    fw_device: String,
    vad_enabled: bool,
) -> Result<ChunkResult, String> {
    use futures_util::stream::StreamExt;

    let engine_label = if use_fw { "faster-whisper" } else { "whisper" };
    // GPU 路径要起 Python 子进程，先校验路径；空参数 → 自动用 bundled Python
    let fw_python = if fw_python.is_empty() { resolve_python(&app) } else { fw_python };
    if use_fw {
        check_python_path(&fw_python)?;
    }

    // lead-in 仅用于给 ASR 提供上下文，最终丢弃 start_sec 之前的段
    // 5s 重叠（而非 3s）：给分片边界处的句子更多上下文，减少切断导致的识别错误
    let real_start = (start_sec - lead_in_sec).max(0.0);
    let real_dur = duration_sec + (start_sec - real_start);

    // 1) 抽取音频
    let t_ex = std::time::Instant::now();
    // 临时文件带时间戳后缀：切换视频/并发请求时同一 start 不会互相覆盖
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!(
        "subtrans_{}_{}_{}.wav",
        std::process::id(),
        start_sec as u64,
        stamp
    ));
    // 高精度模式下从分离好的人声轨取音；否则直接取原视频
    let asr_src = audio_source.as_deref().unwrap_or(video_path.as_str());
    let ffmpeg_bin = resolve_ffmpeg(&app);
    // ffmpeg 是阻塞子进程，放进 spawn_blocking，避免占住 async runtime 的 worker
    let (asr_src_owned, tmp_owned, ffmpeg_owned) =
        (asr_src.to_string(), tmp.clone(), ffmpeg_bin.clone());
    // 所有失败路径都删掉临时 wav，避免 temp 里堆半成品（成功时文件还要给 ASR 用，不能删）
    let extract_result = tokio::time::timeout(
        std::time::Duration::from_secs(180),
        tokio::task::spawn_blocking(move || {
            ffmpeg::extract_audio_range(
                &asr_src_owned,
                &tmp_owned,
                &ffmpeg_owned,
                real_start,
                real_dur,
            )
        }),
    )
    .await;
    match extract_result {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(e))) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.to_string());
        }
        Ok(Err(e)) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.to_string());
        }
        Err(_) => {
            let _ = std::fs::remove_file(&tmp);
            return Err("抽取音频超时（180s），请检查视频文件或 ffmpeg".to_string());
        }
    }
    let extract_ms = t_ex.elapsed().as_millis() as u64;

    // 2) 识别
    let t_tr = std::time::Instant::now();
    let src_owned: Option<String> = source_lang.filter(|s| s != "auto");
    let offset = real_start;

    let mut vad_warn: Option<String> = None;
    let mut detected_lang: Option<String> = None;
    let mut detected_lang_probability: Option<f64> = None;
    let segments: Vec<asr::Segment> = if use_fw {
        // faster-whisper：把分片 wav 路径喂给常驻 GPU 服务（模型常驻显存）。
        // 模型由 Rust 端先下好（python 端 huggingface_hub 在国内网络下 SSL 不稳）。
        let tmp_str = tmp.to_string_lossy().to_string();
        ensure_fw_model(&app, &model_name).await?;
        fw_ipc::fw_ensure(&fw_state, &fw_python, &model_name, &fw_device, &app).await?;
        // 术语表同时作为 faster-whisper 热词（截断防超长），人名/专有名词识别更准
        let hotwords = glossary.trim();
        let hotwords = if hotwords.is_empty() {
            None
        } else {
            Some(hotwords.chars().take(200).collect::<String>())
        };
        let transcribe = fw_ipc::fw_transcribe_one(
            &fw_state,
            &tmp_str,
            src_owned.as_deref(),
            audio_source.is_some(),
            vad_enabled,
            hotwords.as_deref(),
        )
        .await;
        let _ = std::fs::remove_file(&tmp);
        let detected = transcribe?;
        detected_lang = detected.language;
        detected_lang_probability = detected.language_probability;
        let mut s = detected.segments;
        // 与 CPU(Whisper) 路径保持一致：去掉语气词/填充词，再丢弃因此变空的段
        for seg in &mut s {
            seg.start += offset;
            seg.end += offset;
            seg.text = asr::clean_fillers(&seg.text);
        }
        s.retain(|seg| !seg.text.is_empty());
        s
    } else {
        let tmp2 = tmp.clone();
        let read_result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || ffmpeg::read_wav_as_f32(&tmp2)),
        )
        .await;
        let _ = std::fs::remove_file(&tmp);
        let audio: Vec<f32> = read_result
            .map_err(|_| "读取音频超时（30s）".to_string())?
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let threads = num_cpus::get().saturating_sub(1).max(1) as i32;
        let wp = whisper_model_path(&app, &model_name);
        if !wp.exists() {
            return Err(format!("识别模型不存在: {}", wp.display()));
        }
        let eng = cache.get_or_load_whisper(wp, model_name.clone()).await?;
        // 仅在已经过人声分离（高精度模式）时启用 VAD：
        // 分离后是纯人声，VAD 可安全过滤静音段幻觉；
        // 未分离的原始视频可能含音乐/歌声，VAD 会把唱歌当“非语音”跳过。
        // VAD：已分离人声轨必开；未分离音频按用户开关（音乐多的视频建议开）
        let vad_path = if audio_source.is_some() || vad_enabled {
            match ensure_vad_model(&app).await {
                Ok(p) => Some(p),
                Err(e) => {
                    vad_warn = Some(format!("VAD 模型不可用（{e}），已跳过静音过滤"));
                    None
                }
            }
        } else {
            None
        };
        let vad_str = vad_path.as_ref().map(|p| p.to_string_lossy().to_string());
        let lang = src_owned.clone();
        tokio::task::spawn_blocking(move || {
            eng.transcribe(&audio, lang.as_deref(), threads, offset, vad_str.as_deref())
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?
    };
    let transcribe_ms = t_tr.elapsed().as_millis() as u64;

    // 拆分过长的 segment：Whisper 对连续朗读/唱歌可能把一整段话合成一个 segment，
    // 导致 overlay 一次显示一大坨文字。按标点拆成多个子段，时间按字数比例分配。
    let mut segments = asr::split_long_segments(segments);

    // 丢弃落在 lead-in 重叠区的段；同时丢弃分片末尾不确定的段（尾部 1s 内的段可能因截断而不完整）
    // 但最后一个分片不做尾部截断（否则视频末尾字幕丢失，没有“下一片”来补）
    let chunk_end = start_sec + duration_sec;
    let is_last_chunk = chunk_end >= total_sec - 0.1;
    if is_last_chunk {
        segments.retain(|s| s.end > start_sec - 0.05);
    } else {
        segments.retain(|s| s.end > start_sec - 0.05 && s.start < chunk_end - 1.0);
    }

    // 2.5) 可选：LLM 同音字校对（按上下文整批修正后再翻译）
    let mut warn: Option<String> = vad_warn;
    let t_co = std::time::Instant::now();
    if correct_enabled {
        if let Some(ce) = &correct_engine {
            let texts: Vec<String> = segments.iter().map(|s| s.text.clone()).collect();
            if !texts.is_empty() {
                let cclient = reqwest::Client::new();
                match correct::correct_lines(&cclient, ce, &texts, &glossary).await {
                    Ok(fixed) => {
                        for (seg, t) in segments.iter_mut().zip(fixed) {
                            seg.text = t;
                        }
                    }
                    Err(e) => warn = Some(format!("纠错失败: {e}")),
                }
            }
        }
    }
    let correct_ms = t_co.elapsed().as_millis() as u64;

    // 3) 并发翻译
    let t_tx = std::time::Instant::now();
    let out: Vec<OutSegment> = if translate_enabled {
        let client = Arc::new(reqwest::Client::new());
        let engine = Arc::new(engine);
        let target = Arc::new(target_lang);
        let src_lang: Arc<Option<String>> = Arc::new(src_owned.clone());
        let seg_count = segments.len();
        // Ollama 本地模型并发能力有限，给 2；DeepSeek/免费 API 可承受更高并发
        let cap = if matches!(*engine, translate::Engine::Ollama { .. }) { 2 } else { 6 };
        let max_concurrent = seg_count.clamp(1, cap);
        // 翻译失败计数：失败时 translated 留空（前端会回退显示原文），
        // 避免把错误字符串塞进字幕内容污染 SRT 导出和 overlay 显示。
        let fail_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let out_vec: Vec<OutSegment> = futures_util::stream::iter(segments)
            .map(|seg| {
                let client = client.clone();
                let engine = engine.clone();
                let target = target.clone();
                let fail_count = fail_count.clone();
                let src_lang = src_lang.clone();
                async move {
                    let translated = match translate::translate(
                        &client,
                        &engine,
                        &seg.text,
                        target.as_str(),
                        src_lang.as_deref(),
                    )
                    .await
                    {
                        Ok(t) => t,
                        Err(_e) => {
                            fail_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            String::new()
                        }
                    };
                    OutSegment {
                        index: seg.index,
                        start: seg.start,
                        end: seg.end,
                        original: seg.text,
                        translated,
                    }
                }
            })
            .buffered(max_concurrent)
            .collect()
            .await;
        let fails = fail_count.load(std::sync::atomic::Ordering::Relaxed);
        if fails > 0 {
            warn = Some(format!("翻译失败 {fails}/{seg_count} 条（已回退显示原文）"));
        }
        out_vec
    } else {
        segments
            .into_iter()
            .map(|seg| OutSegment {
                index: seg.index,
                start: seg.start,
                end: seg.end,
                original: seg.text,
                translated: String::new(),
            })
            .collect()
    };
    let translate_ms = t_tx.elapsed().as_millis() as u64;

    Ok(ChunkResult {
        segments: out,
        extract_ms,
        transcribe_ms,
        correct_ms,
        translate_ms,
        engine_used: engine_label.to_string(),
        detected_lang,
        detected_lang_probability,
        warn,
    })
}

/// 检测 faster-whisper 是否可用 + CUDA 状态。
#[tauri::command]
async fn fw_check(app: tauri::AppHandle, python_exe: String) -> Result<String, String> {
    let python_exe = if python_exe.is_empty() { resolve_python(&app) } else { python_exe };
    check_python_path(&python_exe)?;
    let mut command = tokio::process::Command::new(&python_exe);
    command.args([
        "-c",
        "import torch, faster_whisper; print('faster-whisper OK | torch', torch.__version__, '| CUDA', torch.cuda.is_available())",
    ]);
    command.env("PYTHONUTF8", "1");
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(0x0800_0000);
    }
    let out = tokio::time::timeout(std::time::Duration::from_secs(30), command.output())
        .await
        .map_err(|_| "faster-whisper 检测超时（30s）".to_string())?
        .map_err(|e| format!("无法运行 Python: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

// ── Python 环境探测 ──

#[derive(Serialize, Clone)]
struct PythonEnv {
    found: bool,
    exe_path: Option<String>,
    version: Option<String>,
    has_faster_whisper: bool,
    has_demucs: bool,
    has_torch_cuda: bool,
}

/// 一键安装 Python 环境（下载 embeddable Python + pip install 依赖包）。
#[tauri::command]
async fn python_setup(app: tauri::AppHandle) -> Result<String, String> {
    let res = python_setup_inner(app.clone()).await;
    if let Err(e) = &res {
        log_err(&app, "python_setup", e);
    }
    res
}

async fn python_setup_inner(app: tauri::AppHandle) -> Result<String, String> {
    let path = python_setup::setup(&app).await?;
    remember_cuda_python(&app, &path).await;
    Ok(path)
}

/// 往已检测到的 Python 一键安装 GPU 识别组件（CUDA 版 torch + faster-whisper/demucs/soundfile）。
#[tauri::command]
async fn install_gpu_packages(app: tauri::AppHandle, python_exe: String) -> Result<String, String> {
    let res = install_gpu_packages_inner(app.clone(), python_exe).await;
    if let Err(e) = &res {
        log_err(&app, "install_gpu_packages", e);
    }
    res
}

async fn install_gpu_packages_inner(
    app: tauri::AppHandle,
    python_exe: String,
) -> Result<String, String> {
    let path = python_setup::install_gpu_packages(&app, &python_exe).await?;
    if python_setup::torch_cuda_ready(std::path::Path::new(&path)).await {
        remember_cuda_python(&app, &path).await;
        Ok(path)
    } else {
        Err("组件已安装，但 CUDA 版 torch 不可用（已回退 CPU 版）。\
             若确认 NVIDIA 驱动正常，请检查网络后重试"
            .into())
    }
}

/// 安装成功后若 torch 确实可用 CUDA，则持久化标记，让 env_status 优先探测这个 Python。
async fn remember_cuda_python(app: &tauri::AppHandle, python_exe: &str) {
    if python_setup::torch_cuda_ready(std::path::Path::new(python_exe)).await {
        let _ = std::fs::write(cuda_python_marker(app), normalize_windows_path(python_exe));
    }
}

/// 自动探测系统上的 Python 安装，并检查 GPU 功能所需的包。
#[tauri::command]
async fn python_detect(app: tauri::AppHandle) -> Result<PythonEnv, String> {
    let not_found = PythonEnv {
        found: false,
        exe_path: None,
        version: None,
        has_faster_whisper: false,
        has_demucs: false,
        has_torch_cuda: false,
    };

    // 收集候选 python（bundled → PATH → py launcher → 常见目录 → 本应用 embeddable），去重
    let mut candidates: Vec<String> = Vec::new();

    // 0) 随包安装的 python-bundle（优先使用）
    let bundled = resolve_python(&app);
    if !bundled.is_empty() && test_python_bin(&bundled).await.is_some() {
        candidates.push(bundled);
    }

    // 1) PATH 上的 python / python3 / py
    for name in ["python", "python3", "py"] {
        if let Some(p) = test_python_bin(name).await {
            if !candidates.contains(&p) {
                candidates.push(p);
            }
        }
    }
    for dir in python_candidate_dirs() {
        if let Some(p) = test_python_bin(&dir).await {
            if !candidates.contains(&p) {
                candidates.push(p);
            }
        }
    }
    if let Ok(data_dir) = app.path().app_data_dir() {
        let embedded = data_dir.join("python").join("python.exe");
        if let Some(p) = test_python_bin(&embedded.to_string_lossy()).await {
            if !candidates.contains(&p) {
                candidates.push(p);
            }
        }
    }

    // 优先选「已装 GPU 栈」的 python：否则 PATH 上排第一的裸 python（如新版安装器的 3.14、
    // 或 WindowsApps 的 Store 别名）会盖过真正装了 faster-whisper/demucs 的那个（如 Python39）。
    // torch 冷启动很慢（单个最多 30s），多个候选并行探测，避免启动检测成倍变慢。
    use futures_util::stream::StreamExt;
    let concurrency = candidates.len().clamp(1, 4);
    let envs: Vec<PythonEnv> = futures_util::stream::iter(candidates)
        .map(|path| {
            let path = path.clone();
            async move {
                let (v, fw, dm, cuda) = probe_python(&path).await;
                PythonEnv {
                    found: true,
                    exe_path: Some(path),
                    version: v,
                    has_faster_whisper: fw,
                    has_demucs: dm,
                    has_torch_cuda: cuda,
                }
            }
        })
        .buffered(concurrency)
        .collect()
        .await;
    if let Some(env) = envs.iter().find(|e| e.has_faster_whisper || e.has_demucs) {
        return Ok(env.clone()); // 装了 GPU 栈，直接选它
    }
    Ok(envs.into_iter().next().unwrap_or(not_found))
}

async fn test_python_bin(path_or_name: &str) -> Option<String> {
    let mut cmd = tokio::process::Command::new(path_or_name);
    cmd.args(["-c", "import sys; print(sys.executable); print(sys.version.split()[0])"])
        // 日文 locale 下 python 管道默认 cp932，会把含「南山」的路径解坏（南山→??R）；强制 UTF-8
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    let out =
        tokio::time::timeout(std::time::Duration::from_secs(15), cmd.output()).await.ok()?.ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    if lines.len() >= 2 {
        Some(normalize_windows_path(lines[0].trim()))
    } else {
        None
    }
}

async fn probe_python(python: &str) -> (Option<String>, bool, bool, bool) {
    let version = get_python_version(python).await;
    let (fw, dm, cuda) = check_python_packages(python).await;
    (version, fw, dm, cuda)
}

async fn get_python_version(python: &str) -> Option<String> {
    let mut cmd = tokio::process::Command::new(python);
    cmd.args(["-c", "import sys; print(sys.version.split()[0])"])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    let out =
        tokio::time::timeout(std::time::Duration::from_secs(15), cmd.output()).await.ok()?.ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

async fn check_python_packages(python: &str) -> (bool, bool, bool) {
    // 注意：必须 `import importlib.util`（光 `import importlib` 不会加载 util 子模块，
    // 否则 `importlib.util` 抛 AttributeError，脚本崩 → 探测永远返回全 false）。
    let script = "import importlib.util\nfw = importlib.util.find_spec('faster_whisper') is not None\ndm = importlib.util.find_spec('demucs') is not None\ncuda = False\ntry:\n    import torch\n    cuda = torch.cuda.is_available()\nexcept Exception:\n    pass\nprint(f'{fw}|{dm}|{cuda}')";
    let mut cmd = tokio::process::Command::new(python);
    cmd.args(["-c", script])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    // torch 冷启动 import + CUDA 初始化实测可达 ~14s，给足 30s 免得被超时砍成"没 GPU"
    let out = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output()).await {
        Ok(Ok(o)) => o,
        _ => return (false, false, false),
    };
    if !out.status.success() {
        return (false, false, false);
    }
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<&str> = stdout.split('|').collect();
    (
        parts.first().is_some_and(|s| *s == "True"),
        parts.get(1).is_some_and(|s| *s == "True"),
        parts.get(2).is_some_and(|s| *s == "True"),
    )
}

fn python_candidate_dirs() -> Vec<String> {
    let mut dirs = Vec::new();
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        // python.org per-user install
        let base = std::path::PathBuf::from(&local).join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("Python3") || name.starts_with("Python") {
                    let exe = entry.path().join("python.exe");
                    dirs.push(exe.to_string_lossy().to_string());
                }
            }
        }
    }
    // System-wide python.org installs
    for ver in &["314", "313", "312", "311", "310", "39", "38"] {
        dirs.push(format!("C:\\Python{ver}\\python.exe"));
    }
    dirs
}

// ── 高精度模式：人声分离（audio-separator / BS-RoFormer，回退 demucs） ──

/// 检测人声分离组件是否可用（优先 audio-separator，回退 demucs）。
#[tauri::command]
async fn demucs_check(app: tauri::AppHandle, python_exe: String) -> Result<String, String> {
    use tokio::process::Command as TokioCommand;
    let python_exe = if python_exe.is_empty() { resolve_python(&app) } else { python_exe };
    check_python_path(&python_exe)?;
    // 先检测 audio-separator（更先进的分离模型）
    let mut command = TokioCommand::new(&python_exe);
    command.args([
        "-c",
        // bundle 里 audio_separator 无 dist-info，也没有 __version__ 属性，只验证可导入即可
        "import audio_separator; print('audio-separator OK')",
    ]);
    command.env("PYTHONUTF8", "1");
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(0x0800_0000);
    }
    let out = tokio::time::timeout(std::time::Duration::from_secs(30), command.output())
        .await
        .map_err(|_| "audio-separator 检测超时（30s）".to_string())?
        .map_err(|e| format!("无法运行 Python ({python_exe}): {e}"))?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    // 回退检测 demucs
    let mut command2 = TokioCommand::new(&python_exe);
    command2.args([
        "-c",
        "import torch, demucs; print('demucs OK | torch', torch.__version__, '| CUDA', torch.cuda.is_available())",
    ]);
    command2.env("PYTHONUTF8", "1");
    #[cfg(target_os = "windows")]
    {
        command2.creation_flags(0x0800_0000);
    }
    let out2 = tokio::time::timeout(std::time::Duration::from_secs(30), command2.output())
        .await
        .map_err(|_| "demucs 检测超时（30s）".to_string())?
        .map_err(|e| format!("无法运行 Python ({python_exe}): {e}"))?;
    if out2.status.success() {
        Ok(String::from_utf8_lossy(&out2.stdout).trim().to_string())
    } else {
        Err(format!("人声分离组件不可用：{}", String::from_utf8_lossy(&out2.stderr).trim()))
    }
}

/// 用 audio-separator（BS-RoFormer）或 demucs 把整段音频分离出纯人声轨，返回 vocals.wav 路径。
/// 优先使用 audio-separator（分离质量更高），不可用时回退 demucs。
#[tauri::command]
async fn separate_vocals(
    app: tauri::AppHandle,
    video_path: String,
    python_exe: String,
    model: String,
    device: String,
) -> Result<String, String> {
    let res = separate_vocals_inner(app.clone(), video_path, python_exe, model, device).await;
    if let Err(e) = &res {
        log_err(&app, "separate_vocals", e);
    }
    res
}

async fn separate_vocals_inner(
    app: tauri::AppHandle,
    video_path: String,
    python_exe: String,
    model: String,
    device: String,
) -> Result<String, String> {
    let python_exe = if python_exe.is_empty() { resolve_python(&app) } else { python_exe };
    check_python_path(&python_exe)?;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command as TokioCommand;
    // 每次分离用独立目录（带 PID + 时间戳），避免多实例/并发互相删目录；
    // 结束后把 vocals 复制到稳定路径，再清掉工作目录。
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    // 清理本进程之前留下的 vocals 文件（每次分离只保留最新一份，避免 temp 堆大文件）
    if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
        let prefix = format!("subtrans_vocals_{}_", std::process::id());
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".wav") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let work = std::env::temp_dir().join(format!("subtrans_sep_{}_{}", std::process::id(), stamp));
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let audio = work.join("audio.wav");
    let vocals_dest =
        std::env::temp_dir().join(format!("subtrans_vocals_{}_{}.wav", std::process::id(), stamp));

    // 1) 提取整段音频（44.1k 立体声）
    emit_progress(&app, "separate", 3.0, "提取音频中...");
    let ffmpeg_bin = resolve_ffmpeg(&app);
    let (vp, fb, ap) = (video_path.clone(), ffmpeg_bin, audio.clone());
    tokio::time::timeout(
        std::time::Duration::from_secs(10 * 60),
        tokio::task::spawn_blocking(move || ffmpeg::extract_audio_full(&vp, &ap, &fb)),
    )
    .await
    .map_err(|_| "提取整段音频超时（10 分钟），请检查视频文件".to_string())?
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    // 2) 检测用哪个分离引擎：demucs 系模型名走 demucs；
    //    空模型（默认 BS-RoFormer）或其它 ckpt 走 audio-separator，
    //    audio-separator 不可用时才回退 demucs 默认模型。
    let is_demucs_model = model.starts_with("htdemucs") || model.starts_with("mdx_extra");
    let use_audio_separator = check_audio_separator(&python_exe).await && !is_demucs_model;

    let out_dir = work.join("out");
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    // audio-separator 默认把模型下到 /tmp/audio-separator-models（Windows 上会落到
    // C:\tmp，可能无权限且每次重下）；改放到应用数据目录，跨次运行可复用。
    let sep_models_dir = app
        .path()
        .app_data_dir()
        .map(|d| d.join("models").join("audio-separator"))
        .unwrap_or_else(|_| std::env::temp_dir().join("audio-separator-models"));
    std::fs::create_dir_all(&sep_models_dir).map_err(|e| e.to_string())?;

    let mut command = TokioCommand::new(&python_exe);
    // demucs 回退时要求模型名非空：空模型名（BS-RoFormer 意图）在没有 audio-separator
    // 时回退到 demucs 默认 htdemucs，避免 `demucs -n ""` 启动即失败。
    let demucs_model = if model.is_empty() { "htdemucs".to_string() } else { model.clone() };
    if use_audio_separator {
        // audio-separator + BS-RoFormer：分离质量显著优于 demucs
        emit_progress(&app, "separate", 8.0, "分离人声中（BS-RoFormer，首次会下载模型）...");
        let sep_model = if model.is_empty() {
            // 未指定模型 → 用默认 BS-RoFormer（demucs 系模型名已在上面的 is_demucs_model 分流走 demucs）
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt".to_string()
        } else {
            model.clone()
        };
        // bundle 里 audio_separator 没有 dist-info，CLI 入口（console script / -m）都会挂；
        // 直接用 Separator API 直调，不依赖命令行解析和包元数据
        let script = format!(
            "from audio_separator.separator import Separator\n\
             sep = Separator(model_file_dir=r'{mdir}', output_dir=r'{odir}', output_format='WAV')\n\
             sep.load_model(model_filename=r'{model}')\n\
             files = sep.separate([r'{audio}'])\n\
             print('OK', files)",
            mdir = sep_models_dir.display(),
            odir = out_dir.display(),
            model = sep_model,
            audio = audio.display(),
        );
        command.arg("-c").arg(&script);
        if device == "cpu" {
            // audio-separator CLI 没有 --cpu 参数（argparse 会直接报错）；
            // 用 CUDA_VISIBLE_DEVICES 清空让 torch.cuda.is_available() 为 false，
            // 从而强制走 CPU，与 demucs 的 -d cpu 行为对齐。
            command.env("CUDA_VISIBLE_DEVICES", "");
        }
    } else {
        // 回退 demucs
        emit_progress(
            &app,
            "separate",
            8.0,
            &format!("分离人声中（demucs {demucs_model}, {device}）..."),
        );
        command.args([
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            &demucs_model,
            "-d",
            &device,
            "-o",
            out_dir.to_str().ok_or("输出路径无效")?,
            audio.to_str().ok_or("音频路径无效")?,
        ]);
    }
    command
        .env("PYTHONUTF8", "1")
        // demucs 模型从 HuggingFace 下载（adefossez/HTDemucs），
        // 不设镜像会去连主站（国内常不可达）→ 首次分离必失败
        .env("HF_ENDPOINT", "https://hf-mirror.com")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(0x0800_0000);
    }
    let mut child = command.spawn().map_err(|e| {
        format!("启动人声分离失败: {e}（请确认已 pip install audio-separator 或 demucs）")
    })?;

    // 读取 stdout+stderr 实时转发进度
    let tail = Arc::new(Mutex::new(std::collections::VecDeque::<String>::new()));
    let mut readers = Vec::new();
    for pipe in [
        child.stdout.take().map(|s| Box::new(s) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
        child.stderr.take().map(|s| Box::new(s) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let app2 = app.clone();
        let tail2 = tail.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = BufReader::new(pipe).lines();
            loop {
                let line = match lines.next_line().await {
                    Ok(Some(l)) => l,
                    Ok(None) => break,
                    Err(_) => continue,
                };
                let line = line.trim().to_string();
                if !line.is_empty() {
                    emit_progress(&app2, "separate", 50.0, &format!("分离中: {line}"));
                    let mut t = tail2.lock().unwrap();
                    t.push_back(line);
                    while t.len() > 8 {
                        t.pop_front();
                    }
                }
            }
        }));
    }

    // 分离可能很慢（长视频 + CPU），给 2 小时兜底超时，避免进程卡死时界面永久等待
    let status =
        match tokio::time::timeout(std::time::Duration::from_secs(2 * 3600), child.wait()).await {
            Ok(s) => s.map_err(|e| e.to_string())?,
            Err(_) => {
                let _ = child.kill().await;
                let _ = std::fs::remove_dir_all(&work);
                return Err("人声分离超时（2 小时），已终止进程".to_string());
            }
        };
    for r in readers {
        let _ = r.await;
    }
    if !status.success() {
        let detail = {
            let t = tail.lock().unwrap();
            t.iter().cloned().collect::<Vec<_>>().join(" | ")
        };
        let _ = std::fs::remove_dir_all(&work);
        return Err(format!("人声分离失败（退出码 {:?}）：{detail}", status.code()));
    }

    // 3) 定位人声轨文件
    let vocals = if use_audio_separator {
        // audio-separator 输出: {input_name}_(Vocals)_{model}.wav
        match find_vocals_in_dir(&out_dir) {
            Some(p) => p,
            None => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(format!("未找到分离结果（在 {} 中）", out_dir.display()));
            }
        }
    } else {
        // demucs 输出: out/{model}/audio/vocals.wav
        let p = out_dir.join(&demucs_model).join("audio").join("vocals.wav");
        if !p.exists() {
            let _ = std::fs::remove_dir_all(&work);
            return Err(format!("未找到分离结果: {}", p.display()));
        }
        p
    };
    // 复制到工作目录之外的稳定路径，然后清理工作目录（避免 temp 里堆整段音频）
    std::fs::copy(&vocals, &vocals_dest).map_err(|e| {
        let _ = std::fs::remove_dir_all(&work);
        format!("保存人声轨失败: {e}")
    })?;
    let _ = std::fs::remove_dir_all(&work);
    emit_progress(&app, "separate", 100.0, "人声分离完成");
    Ok(vocals_dest.to_string_lossy().to_string())
}

/// 检测 audio-separator 是否可用。
async fn check_audio_separator(python_exe: &str) -> bool {
    let mut cmd = tokio::process::Command::new(python_exe);
    cmd.args(["-c", "import audio_separator"])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output())
        .await
        .map(|r| r.map(|o| o.status.success()).unwrap_or(false))
        .unwrap_or(false)
}

/// 在目录里找含 "Vocals" 的 wav/flac 文件（audio-separator 输出命名）。
fn find_vocals_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if name.contains("vocals") && (name.ends_with(".wav") || name.ends_with(".flac")) {
            return Some(entry.path());
        }
    }
    None
}

/// 把文本写到指定路径（导出 SRT 用）。走后端 std::fs，绕过 fs 插件 scope 限制——
/// saveDialog 已让用户确认路径，可安全写入任意盘符（否则保存到 $HOME/$APPDATA 之外会失败）。
/// 写入时附加 UTF-8 BOM，确保 Windows 播放器（PotPlayer 等）正确显示中文字幕。
#[tauri::command]
fn save_text_file(path: String, content: String) -> Result<(), String> {
    use std::io::Write;
    // 基本安全校验：必须是绝对路径且扩展名合理，防止 WebView 注入后写任意文件
    let p = std::path::Path::new(&path);
    if !p.is_absolute() {
        return Err("保存路径必须是绝对路径".into());
    }
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if !matches!(ext.as_str(), "srt" | "txt" | "ass" | "vtt") {
        return Err(format!("不支持的文件类型: .{ext}（仅支持 srt/txt/ass/vtt）"));
    }
    let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    file.write_all(&[0xEF, 0xBB, 0xBF]).map_err(|e| e.to_string())?;
    file.write_all(content.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Ollama 命令（不变） ──

#[tauri::command]
async fn ollama_status(host: Option<String>) -> Result<ollama::OllamaStatus, String> {
    let client = reqwest::Client::new();
    Ok(ollama::status(&client, host.as_deref()).await)
}

#[tauri::command]
async fn ollama_pull(
    app: tauri::AppHandle,
    host: Option<String>,
    model: String,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    ollama::pull_model(&client, host.as_deref(), &model, |pct, status| {
        emit_progress(&app, "ollama_pull", pct, &format!("拉取模型: {status}"));
    })
    .await
    .map_err(|e| e.to_string())?;
    emit_progress(&app, "ollama_pull", 100.0, "模型就绪");
    Ok(())
}

#[tauri::command]
fn ollama_installer_url() -> Result<String, String> {
    ollama::installer_url().map(|s| s.to_string()).map_err(|e| e.to_string())
}

// ── 环境自适应命令 ──

#[derive(Serialize)]
struct EnvStatus {
    python_path: String,
    python_bundled: bool,
    has_gpu: bool,
    cuda_torch_ready: bool,
    fw_ready: bool,
    demucs_ready: bool,
    audio_sep_ready: bool,
    models: Vec<String>,
    bundled_tiny: bool,
}

/// 综合环境状态检测：Python 路径、GPU、各组件可用性、已下载模型。
#[tauri::command]
async fn env_status(app: tauri::AppHandle) -> Result<EnvStatus, String> {
    let bundled_python = resolve_python(&app);
    let python_bundled = !bundled_python.is_empty();

    // 检测 GPU（nvidia-smi）
    let has_gpu = {
        let mut cmd = std::process::Command::new("nvidia-smi");
        cmd.arg("--query-gpu=name").arg("--format=csv,noheader");
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        cmd.output().map(|o| o.status.success()).unwrap_or(false)
    };

    // 用 python_detect 的结果填充组件状态
    let mut env = python_detect(app.clone()).await.unwrap_or(PythonEnv {
        found: false,
        exe_path: None,
        version: None,
        has_faster_whisper: false,
        has_demucs: false,
        has_torch_cuda: false,
    });

    // 优先检测 CUDA 升级持久化的 Python 路径：
    // python_detect 可能选错 Python（如选了没装 CUDA torch 的），
    // 而升级成功后我们明确知道装到了哪个 Python，直接 probe 它。
    if let Some(marker_path) = read_cuda_python_marker(&app) {
        if marker_path != env.exe_path.as_deref().unwrap_or("") {
            let (v, fw, dm, cuda) = probe_python(&marker_path).await;
            if cuda {
                env = PythonEnv {
                    found: true,
                    exe_path: Some(marker_path),
                    version: v,
                    has_faster_whisper: fw,
                    has_demucs: dm,
                    has_torch_cuda: cuda,
                };
            }
        }
    }

    let python_path = env.exe_path.unwrap_or_else(|| bundled_python.clone());

    // 检测 audio-separator
    let audio_sep_ready =
        if !python_path.is_empty() { check_audio_separator(&python_path).await } else { false };

    // 已下载的 whisper 模型
    let mut models = Vec::new();
    for name in ["tiny", "base", "small", "medium", "large-v3"] {
        if whisper_model_path(&app, name).exists()
            || bundled_model_path(&app, &format!("ggml-{name}.bin")).is_some()
        {
            models.push(name.to_string());
        }
    }

    let bundled_tiny = bundled_model_path(&app, "ggml-tiny.bin").is_some();

    Ok(EnvStatus {
        python_path,
        python_bundled,
        has_gpu,
        cuda_torch_ready: env.has_torch_cuda,
        fw_ready: env.has_faster_whisper,
        demucs_ready: env.has_demucs,
        audio_sep_ready,
        models,
        bundled_tiny,
    })
}

/// 预估处理时间：根据视频时长、模型、是否 GPU 返回人类可读字符串。
#[tauri::command]
fn estimate_time(duration_sec: f64, model_name: String, use_fw: bool) -> String {
    // 速度倍率：每秒能处理多少秒音频（越大越快）。
    // 预估处理时长 = 视频时长 / 倍率。
    let factor: f64 = if use_fw {
        5.0 // GPU + large-v3: ~5x 实时
    } else {
        match model_name.as_str() {
            "tiny" => 0.8,
            "base" => 0.5,
            "small" => 0.35,
            "medium" => 0.2,
            "large-v3" => 0.08,
            _ => 0.3,
        }
    };
    let est_sec = duration_sec / factor.max(0.01);
    if est_sec < 60.0 {
        format!("预计 {} 秒", est_sec.round() as u64)
    } else if est_sec < 3600.0 {
        format!("预计 {} 分钟", (est_sec / 60.0).ceil() as u64)
    } else {
        format!("预计 {} 小时", (est_sec / 3600.0).ceil() as u64)
    }
}

/// 后台升级 CUDA torch：卸载 CPU 版 → 安装 CUDA 版。
#[tauri::command]
async fn upgrade_cuda(app: tauri::AppHandle, python_exe: String) -> Result<String, String> {
    let res = upgrade_cuda_inner(app.clone(), python_exe).await;
    if let Err(e) = &res {
        log_err(&app, "upgrade_cuda", e);
    }
    res
}

async fn upgrade_cuda_inner(app: tauri::AppHandle, python_exe: String) -> Result<String, String> {
    let python_exe = if python_exe.is_empty() { resolve_python(&app) } else { python_exe };
    if python_exe.is_empty() {
        return Err("未找到 Python 路径".into());
    }
    // 确保 pip 可用（embeddable/精简 Python 可能没装 pip）
    python_setup::ensure_pip(std::path::Path::new(&python_exe), &std::env::temp_dir()).await?;

    emit_progress(&app, "cuda_upgrade", 5.0, "卸载 CPU 版 torch...");

    // 1. 卸载 CPU 版
    let mut cmd = tokio::process::Command::new(&python_exe);
    cmd.args(["-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    let _ = tokio::time::timeout(std::time::Duration::from_secs(300), cmd.output()).await; // 忽略错误（可能本来就没装）

    emit_progress(&app, "cuda_upgrade", 10.0, "下载 CUDA 版 torch（约 2.5GB，请耐心等待）...");

    // 2. 安装 CUDA 版（多镜像尝试）
    let mirrors = [
        ("https://mirror.sjtu.edu.cn/pytorch-wheels/cu124", "mirror.sjtu.edu.cn"),
        ("https://mirrors.nju.edu.cn/pytorch-wheels/cu124", "mirrors.nju.edu.cn"),
        ("https://download.pytorch.org/whl/cu124", "download.pytorch.org"),
    ];
    let pypi = "https://mirrors.aliyun.com/pypi/simple/";
    let mut ok = false;
    let mut last_err = String::new(); // 记录最后一次失败的详细 stderr
    for (idx, host) in &mirrors {
        emit_progress(&app, "cuda_upgrade", 15.0, &format!("尝试镜像: {host}..."));
        let mut cmd2 = tokio::process::Command::new(&python_exe);
        cmd2.args([
            "-m",
            "pip",
            "install",
            "--index-url",
            idx,
            "--no-cache-dir",
            "torch",
            "torchaudio",
        ])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
        #[cfg(target_os = "windows")]
        {
            cmd2.creation_flags(0x0800_0000);
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1800), // 30 分钟超时
            cmd2.output(),
        )
        .await;
        match result {
            Ok(Ok(out)) if out.status.success() => {
                // 退出码 0 不代表装对了：混源时 pip 会挑 PyPI 的 CPU 版。
                // 必须实测 torch.cuda.is_available() 为 True 才算成功。
                if python_setup::torch_cuda_ready(std::path::Path::new(&python_exe)).await {
                    ok = true;
                    break;
                }
                last_err = format!(
                    "[{host}] 安装完成但 torch.cuda.is_available() 为 False（装到了 CPU 版）"
                );
            }
            Ok(Ok(out)) => {
                // 捕获 stderr 末尾几行作为详细报错
                let stderr = String::from_utf8_lossy(&out.stderr);
                let tail: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
                let start = tail.len().saturating_sub(4);
                last_err = format!("[{host}] {}", tail[start..].join(" | "));
            }
            Ok(Err(e)) => {
                last_err = format!("[{host}] 启动 pip 失败: {e}");
            }
            Err(_) => {
                last_err = format!("[{host}] 安装超时（30分钟）");
            }
        }
        // 清掉这个源留下的半成品 / CPU 版，避免干扰下一个源
        let mut un = tokio::process::Command::new(&python_exe);
        un.args(["-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
            .env("PYTHONUTF8", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(target_os = "windows")]
        {
            un.creation_flags(0x0800_0000);
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(300), un.output()).await;
    }

    if !ok {
        // 回退：重装 CPU 版
        emit_progress(&app, "cuda_upgrade", 80.0, "CUDA 镜像不可达，回退 CPU 版...");
        let mut cmd3 = tokio::process::Command::new(&python_exe);
        cmd3.args(["-m", "pip", "install", "-i", pypi, "torch", "torchaudio"])
            .env("PYTHONUTF8", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(target_os = "windows")]
        {
            cmd3.creation_flags(0x0800_0000);
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1800), cmd3.output()).await;
        eprintln!("[subtrans] GPU 升级失败，详细报错：{last_err}");
        return Err(format!("所有 CUDA 镜像安装失败，已回退 CPU 版 torch。最后报错：{last_err}"));
    }

    emit_progress(&app, "cuda_upgrade", 100.0, "CUDA 版 torch 安装完成！");
    // 持久化升级到的 Python 路径，供 env_status 准确检测
    let _ = std::fs::write(cuda_python_marker(&app), normalize_windows_path(&python_exe));
    let _ = app.emit("cuda-ready", true);
    Ok(python_exe)
}

// ── 启动 ──

/// 启动时确保 ffmpeg 在进程 PATH 里。winget 装的 ffmpeg 写进了用户持久 PATH，
/// 但双击启动的 GUI 进程可能继承的是没刷新的旧 PATH，导致抽音频静默失败。
/// 这里探测一下，找不到就把 winget 的 ffmpeg 目录加进当前进程 PATH。
fn ensure_ffmpeg_on_path() {
    let mut probe = std::process::Command::new("ffmpeg");
    probe.arg("-version");
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        probe.creation_flags(0x0800_0000);
    }
    if probe.output().map(|o| o.status.success()).unwrap_or(false) {
        return; // 已经能找到
    }

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let base = std::path::PathBuf::from(&local).join("Microsoft").join("WinGet");
        candidates.push(base.join("Links")); // winget 生成的 shim 目录
                                             // Gyan.FFmpeg 实际安装目录（含版本号）：.../Packages/Gyan.FFmpeg_*/ffmpeg-*/bin
        if let Ok(pkgs) = std::fs::read_dir(base.join("Packages")) {
            for e in pkgs.flatten() {
                if e.file_name().to_string_lossy().starts_with("Gyan.FFmpeg") {
                    if let Ok(sub) = std::fs::read_dir(e.path()) {
                        for s in sub.flatten() {
                            candidates.push(s.path().join("bin"));
                        }
                    }
                }
            }
        }
    }
    for c in candidates {
        if c.join("ffmpeg.exe").exists() {
            let old = std::env::var("PATH").unwrap_or_default();
            std::env::set_var("PATH", format!("{};{}", c.display(), old));
            return;
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    ensure_ffmpeg_on_path();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AsrCache::default())
        .manage(FwState::default())
        .invoke_handler(tauri::generate_handler![
            model_exists,
            download_model,
            process_chunk,
            demucs_check,
            separate_vocals,
            fw_check,
            python_detect,
            python_setup,
            install_gpu_packages,
            save_text_file,
            ollama_status,
            ollama_pull,
            ollama_installer_url,
            env_status,
            estimate_time,
            upgrade_cuda,
        ])
        .on_window_event(|window, event| {
            // 窗口关闭时优雅关闭 faster-whisper 服务，释放 GPU 显存
            if let tauri::WindowEvent::Destroyed = event {
                let app = window.app_handle();
                if let Some(fw_state) = app.try_state::<FwState>() {
                    let fw = fw_state.inner().clone();
                    // 在阻塞上下文里无法 await，用 spawn 异步关闭
                    tauri::async_runtime::spawn(async move {
                        fw.shutdown().await;
                    });
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}

#[cfg(test)]
mod tests {
    use super::estimate_time;

    #[test]
    fn gpu_estimate_is_faster_than_realtime() {
        // GPU large-v3 约 5x 实时：1 分钟视频应预估约 12 秒
        assert_eq!(estimate_time(60.0, "large-v3".to_string(), true), "预计 12 秒");
    }

    #[test]
    fn cpu_large_v3_estimate_is_slower_than_realtime() {
        let s = estimate_time(60.0, "large-v3".to_string(), false);
        assert!(s.contains("分钟") || s.contains("小时"), "got {s}");
    }
}
