//! 从视频中提取音频 / 烧录字幕，供识别引擎与导出使用。
//!
//! 所有 ffmpeg 子进程一律走 tokio::process（异步等待 + kill_on_drop），
//! 超时丢弃 future 时子进程会被立刻杀死，不会留下孤儿进程或半成品文件。

use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::process::Command;

/// 给 tokio 命令补 Windows CREATE_NO_WINDOW（GUI 程序 spawn 子进程防闪黑窗）。
/// tokio::process::Command 自带 creation_flags 方法，无需 std 扩展 trait。
#[cfg(target_os = "windows")]
fn hide_window(cmd: &mut Command) {
    cmd.creation_flags(0x0800_0000);
}
#[cfg(not(target_os = "windows"))]
fn hide_window(_cmd: &mut Command) {}

/// 跑 ffmpeg 命令并校验：退出码成功 **且** 输出文件非空。
/// tokio 的 output() 会并发读取 stdout+stderr，不存在经典双管道死锁。
async fn run_ffmpeg(mut cmd: Command, out_wav: &Path) -> Result<()> {
    hide_window(&mut cmd);
    cmd.kill_on_drop(true);
    let output = cmd
        .output()
        .await
        .map_err(|e| anyhow!("启动 ffmpeg 失败: {e}（请确认 ffmpeg 在 PATH 或已安装）"))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg 失败（退出码 {:?}）: {}",
            output.status.code(),
            stderr_tail(&output.stderr)
        ));
    }
    // 偶有退出码 0 却没真正产出文件（无音轨 / 输入异常 / 路径问题）——显式校验，避免后续误报
    if out_wav.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
        return Err(anyhow!(
            "ffmpeg 未产出有效音频（可能视频无音轨或输入异常）: {}",
            stderr_tail(&output.stderr)
        ));
    }
    Ok(())
}

/// 取 ffmpeg stderr 末尾几行非空内容，作为错误提示。
fn stderr_tail(stderr: &[u8]) -> String {
    let s = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = s.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let start = lines.len().saturating_sub(3);
    lines[start..].join(" | ")
}

/// 构建提取音频用的滤镜链：高通(去低频乐器/嗡声) → [可选降噪]。
/// denoise_filter: "afftdn" / "anlmdn" / "arnndn"，其余（含 "none"）不加降噪。
/// arnndn 只把**文件名**写进滤镜图（滤镜图的单引号机制无法表达路径里的特殊字符，
/// 如盘符冒号/单引号），模型目录通过 ffmpeg 的 current_dir 指定，见 extract_audio_range。
fn build_af(denoise_filter: &str, arnndn_model: Option<&Path>) -> Result<String> {
    let mut af = String::from("highpass=f=80");
    match denoise_filter {
        "afftdn" => {
            // FFT 谱减：nf=-25 是语音降噪常用档位（默认 -50 过于保守）
            af.push_str(",afftdn=nf=-25");
        }
        "anlmdn" => {
            // 非局部均值：s 为强度（1e-5..10000，默认 1e-5 几乎无效果），
            // s=1 是适中的语音降噪档位；patch/research/smooth 用默认值
            af.push_str(",anlmdn=s=1");
        }
        "arnndn" => {
            let model = arnndn_model.ok_or_else(|| anyhow!("arnndn 降噪模型未就绪"))?;
            let file = model
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| anyhow!("arnndn 模型路径无效"))?;
            af.push_str(&format!(",arnndn=m={file}"));
        }
        _ => {}
    }
    Ok(af)
}

/// 调用 ffmpeg 抽取指定时间段的音轨成临时 WAV（16kHz / mono / s16），
/// 滤镜链含高通与可选降噪（见 build_af）。
pub async fn extract_audio_range(
    video_path: &str,
    out_wav: &Path,
    ffmpeg_bin: &str,
    start_sec: f64,
    duration_sec: f64,
    denoise_filter: &str,
    arnndn_model: Option<&Path>,
) -> Result<()> {
    let out = out_wav.to_str().ok_or_else(|| anyhow!("无效输出路径"))?;
    let af = build_af(denoise_filter, arnndn_model)?;
    // arnndn 模型路径只传文件名，把工作目录切到模型所在目录（输入/输出都是绝对路径，不受影响）
    let model_dir = arnndn_model.and_then(|m| m.parent().map(|p| p.to_path_buf()));

    // 主路径：双 -ss（输入端快速 seek + 输出端精确裁剪）——
    // 仅输入端 -ss 会落在关键帧上，长 GOP 视频窗口会整体前移导致字幕时间轴漂移；
    // 输出端再 -ss 一次保证从精确的 start_sec 起算。
    let mut cmd = Command::new(ffmpeg_bin);
    if let Some(dir) = &model_dir {
        cmd.current_dir(dir);
    }
    cmd.args([
        "-y",
        "-ss",
        &format!("{start_sec}"),
        "-i",
        video_path,
        "-ss",
        &format!("{start_sec}"),
        "-t",
        &format!("{duration_sec}"),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        &af,
        "-resampler",
        "soxr",
        "-acodec",
        "pcm_s16le",
        out,
    ]);
    let primary = run_ffmpeg(cmd, out_wav).await;
    if primary.is_ok() {
        return Ok(());
    }

    // 回退：少数视频在输入端 seek 下音轨“零包”（时间戳异常 / 默认选轨不对 / soxr 不适配该参数）。
    // 改用更稳的组合：纯输出端 -ss（精确、必定读包）+ 显式选第一条音轨 + 默认重采样器（swr）。
    let mut fb = Command::new(ffmpeg_bin);
    if let Some(dir) = &model_dir {
        fb.current_dir(dir);
    }
    fb.args([
        "-y",
        "-i",
        video_path,
        "-ss",
        &format!("{start_sec}"),
        "-t",
        &format!("{duration_sec}"),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        &af,
        "-acodec",
        "pcm_s16le",
        out,
    ]);
    run_ffmpeg(fb, out_wav).await.map_err(|fb_err| {
        // 两条路都失败：把主路径的报错也带上，便于判断是无音轨/编解码不支持还是其它
        anyhow!("{fb_err}（主路径亦失败：{})", primary.unwrap_err())
    })
}

/// 提取整段音频为 44.1kHz 立体声 WAV，作为 demucs 人声分离的输入（保留更多信息，效果更好）。
pub async fn extract_audio_full(video_path: &str, out_wav: &Path, ffmpeg_bin: &str) -> Result<()> {
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.args([
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "2", // 立体声
        "-ar",
        "44100", // demucs 训练采样率
        "-c:a",
        "pcm_s16le",
        out_wav.to_str().ok_or_else(|| anyhow!("无效输出路径"))?,
    ]);
    run_ffmpeg(cmd, out_wav).await
}

// ── 高精度人声分离：规范 PCM + 整数采样切窗（不重复 seek 压缩源） ──

/// 44.1 kHz 双声道 16-bit 整数 PCM WAV 的校验信息。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcmWavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub frames: u64,
}

/// 校验 PCM WAV：44.1 kHz、双声道、16-bit 整数、非零帧数。
/// 帧数按 WAV 头数据块精确读出，不做秒级取整。
pub fn validate_pcm_wav(path: &Path) -> Result<PcmWavInfo> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 44_100 {
        return Err(anyhow!("采样率必须为 44100，实际 {}", spec.sample_rate));
    }
    if spec.channels != 2 {
        return Err(anyhow!("声道数必须为 2，实际 {}", spec.channels));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(anyhow!("编码必须为 16-bit 整数 PCM"));
    }
    let frames = u64::from(reader.duration());
    if frames == 0 {
        return Err(anyhow!("PCM 帧数为 0"));
    }
    Ok(PcmWavInfo {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        frames,
    })
}

/// 规范 PCM 解码参数：顺序解码整条音轨，不带 `-ss`/`-t`。
pub fn build_canonical_decode_args(
    video_path: &str,
    out_part: &Path,
) -> Result<Vec<String>, String> {
    let out = out_part.to_str().ok_or_else(|| "无效输出路径".to_string())?;
    Ok(vec![
        "-y".into(),
        "-hide_banner".into(),
        "-i".into(),
        video_path.to_string(),
        "-vn".into(),
        "-ac".into(),
        "2".into(),
        "-ar".into(),
        "44100".into(),
        "-c:a".into(),
        "pcm_s16le".into(),
        "-f".into(),
        "wav".into(),
        out.to_string(),
    ])
}

/// 一次性把压缩源音轨流式解码为规范 PCM WAV（44.1 kHz 双声道 16-bit，只落盘不载入内存）。
pub async fn decode_canonical_audio(
    video_path: &str,
    out_part: &Path,
    ffmpeg_bin: &str,
) -> Result<()> {
    let args = build_canonical_decode_args(video_path, out_part).map_err(|e| anyhow!("{e}"))?;
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.args(&args);
    run_ffmpeg(cmd, out_part).await
}

/// 构造 atrim 整数采样裁剪参数：`atrim=start_sample=S:end_sample=E,asetpts=PTS-STARTPTS`。
/// 输入必须与输出同为 PCM，采样网格完全由整数决定，不使用 `-ss`/`-t`。
fn build_atrim_args(
    input: &Path,
    out_part: &Path,
    start_sample: u64,
    end_sample: u64,
) -> Result<Vec<String>, String> {
    if end_sample <= start_sample {
        return Err("无效采样范围：终点必须大于起点".into());
    }
    let input = input.to_str().ok_or_else(|| "无效输入路径".to_string())?;
    let out = out_part.to_str().ok_or_else(|| "无效输出路径".to_string())?;
    Ok(vec![
        "-y".into(),
        "-hide_banner".into(),
        "-i".into(),
        input.to_string(),
        "-af".into(),
        format!("atrim=start_sample={start_sample}:end_sample={end_sample},asetpts=PTS-STARTPTS"),
        "-c:a".into(),
        "pcm_s16le".into(),
        "-f".into(),
        "wav".into(),
        out.to_string(),
    ])
}

/// 纯参数构造：从规范 PCM 用整数 atrim 切窗（`extract_end_sample` 为开区间终点）。
pub fn build_vocal_window_args(
    canonical_wav: &Path,
    out_part: &Path,
    extract_start_sample: u64,
    extract_end_sample: u64,
) -> Result<Vec<String>, String> {
    build_atrim_args(canonical_wav, out_part, extract_start_sample, extract_end_sample)
}

/// 纯参数构造：按 `trim_start_sample + core_frames` 的整数采样范围裁剪核心片。
pub fn build_vocal_trim_args(
    separated_wav: &Path,
    out_part: &Path,
    trim_start_sample: u64,
    core_frames: u64,
) -> Result<Vec<String>, String> {
    let end =
        trim_start_sample.checked_add(core_frames).ok_or_else(|| "裁剪终点溢出".to_string())?;
    build_atrim_args(separated_wav, out_part, trim_start_sample, end)
}

/// 从规范 PCM 按整数采样范围切出模型输入窗口。
pub async fn extract_vocal_window(
    canonical_wav: &Path,
    out_part: &Path,
    ffmpeg_bin: &str,
    extract_start_sample: u64,
    extract_frames: u64,
) -> Result<()> {
    let end =
        extract_start_sample.checked_add(extract_frames).ok_or_else(|| anyhow!("提取终点溢出"))?;
    let args = build_vocal_window_args(canonical_wav, out_part, extract_start_sample, end)
        .map_err(|e| anyhow!("{e}"))?;
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.args(&args);
    run_ffmpeg(cmd, out_part).await
}

/// 从分离结果按整数采样偏移裁出核心片。
pub async fn trim_vocal_core(
    separated_wav: &Path,
    out_part: &Path,
    ffmpeg_bin: &str,
    trim_start_sample: u64,
    core_frames: u64,
) -> Result<()> {
    let args = build_vocal_trim_args(separated_wav, out_part, trim_start_sample, core_frames)
        .map_err(|e| anyhow!("{e}"))?;
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.args(&args);
    run_ffmpeg(cmd, out_part).await
}

/// 生成 concat 清单：只包含按固定模板生成的相对路径，用户路径绝不进入 concat 语法。
pub fn concat_manifest(core_count: usize) -> String {
    (0..core_count).map(|i| format!("file 'cores/vocals_{i:04}.wav'\n")).collect()
}

/// 用 FFmpeg concat demuxer 流式拼接核心片（`-c:a copy`，整轨 PCM 始终只落盘）。
pub async fn concat_vocal_cores(
    work_dir: &Path,
    core_count: usize,
    out_part: &Path,
    ffmpeg_bin: &str,
) -> Result<()> {
    let manifest = work_dir.join("concat.txt");
    std::fs::write(&manifest, concat_manifest(core_count))?;
    let out = out_part.to_str().ok_or_else(|| anyhow!("无效输出路径"))?;
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.current_dir(work_dir);
    cmd.args([
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "concat.txt",
        "-c:a",
        "copy",
        "-f",
        "wav",
        out,
    ]);
    run_ffmpeg(cmd, out_part).await
}

/// 把 SRT 字幕烧录进视频（硬字幕）。srt 只传文件名，工作目录切到其所在目录
/// （文件名固定为 subs.srt，不含特殊字符；滤镜图引号机制无法表达 Windows 路径特殊字符）。
/// font_color/outline_color 为 ASS &HAABBGGRR 字符串；extra_vf 为追加的滤镜链
/// （如竖屏 scale+pad，空串表示不追加）。进度通过 tokio mpsc 发送 0..100。
/// stderr 由后台任务持续泄流，避免编码告警（VFR 等）写满管道导致死锁。
#[allow(clippy::too_many_arguments)] // 参数与 Tauri 命令一一对应，保持扁平便于调用
pub async fn burn_subtitles(
    video_path: &str,
    srt_path: &Path,
    out_path: &str,
    ffmpeg_bin: &str,
    font_size: u32,
    margin_v: u32,
    alignment: u32,
    font_color: &str,
    outline_color: &str,
    extra_vf: &str,
    total_sec: f64,
    tx: tokio::sync::mpsc::UnboundedSender<f64>,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let srt = srt_path.file_name().and_then(|n| n.to_str()).unwrap_or("subs.srt");
    // Alignment: 2=底部居中 8=顶部居中 5=中间；Outline/Shadow 保证任意画面可读
    let style = format!(
        "FontName=Microsoft YaHei,FontSize={font_size},MarginV={margin_v},Outline=2,Shadow=1,Alignment={alignment},PrimaryColour={font_color},OutlineColour={outline_color}"
    );
    let mut vf = format!("subtitles={srt}:force_style='{style}'");
    if !extra_vf.is_empty() {
        vf.push(',');
        vf.push_str(extra_vf);
    }

    // 首选直接拷贝音轨；容器/编码不兼容时回退 AAC 重编码
    let mut last_err: Option<anyhow::Error> = None;
    for (idx, audio_codec) in ["copy", "aac"].iter().enumerate() {
        let mut cmd = Command::new(ffmpeg_bin);
        if let Some(dir) = srt_path.parent() {
            cmd.current_dir(dir);
        }
        hide_window(&mut cmd);
        cmd.args([
            "-y",
            "-hide_banner",
            "-i",
            video_path,
            "-vf",
            &vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            audio_codec,
            "-progress",
            "pipe:1",
            "-nostats",
            out_path,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| anyhow!("启动 ffmpeg 失败: {e}"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("无法读取 ffmpeg 进度"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("无法读取 ffmpeg 错误输出"))?;
        // 后台泄流 stderr：编码期间的告警（VFR "Past duration too large" 等）可能成百上千行，
        // 无人读取会写满管道缓冲（约 4KB）导致 ffmpeg 阻塞 → 死锁
        let tail: Arc<Mutex<std::collections::VecDeque<String>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        {
            let tail = tail.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut t = tail.lock().unwrap();
                    t.push_back(line);
                    while t.len() > 10 {
                        t.pop_front();
                    }
                }
            });
        }
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line == "progress=end" {
                break;
            }
            let t_sec = line
                .strip_prefix("out_time_us=")
                .and_then(|v| v.trim().parse::<f64>().ok())
                .map(|us| us / 1_000_000.0)
                .or_else(|| {
                    line.strip_prefix("out_time_ms=")
                        .and_then(|v| v.trim().parse::<f64>().ok())
                        .map(|ms| ms / 1_000.0)
                })
                .or_else(|| line.strip_prefix("out_time=").and_then(parse_hhmmss));
            if let Some(t) = t_sec {
                if total_sec > 0.0 {
                    let _ = tx.send((t / total_sec * 100.0).clamp(0.0, 99.0));
                }
            }
        }
        let status = child.wait().await.map_err(|e| anyhow!("等待 ffmpeg 失败: {e}"))?;
        if status.success() {
            return Ok(());
        }
        let detail = {
            let t = tail.lock().unwrap();
            t.iter().cloned().collect::<Vec<_>>().join(" | ")
        };
        last_err = Some(anyhow!(
            "ffmpeg 烧录失败（退出码 {:?}，编码器 {audio_codec}）: {detail}",
            status.code()
        ));
        if idx == 0 {
            // 拷贝失败→重编码重试
            continue;
        }
        break;
    }
    Err(last_err
        .unwrap_or_else(|| anyhow!("ffmpeg 烧录失败"))
        .context("音轨直接拷贝亦失败，已尝试 AAC 重编码"))
}

/// 解析 "HH:MM:SS.microseconds" 为秒（ffmpeg -progress 的 out_time 格式）
fn parse_hhmmss(s: &str) -> Option<f64> {
    let s = s.trim();
    let (hms, frac) = match s.split_once('.') {
        Some((a, b)) => (a, format!("0.{b}")),
        None => (s, "0".to_string()),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts[2].parse().ok()?;
    let frac: f64 = frac.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sec + frac)
}

/// 把 PCM WAV 读成 whisper 需要的 f32 单声道样本（归一化到 [-1, 1]）。
/// 支持常见整数位深（16/24/32）与 32 位浮点，不再假设一定是 16-bit。
pub fn read_wav_as_f32(wav_path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?,
            24 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 8_388_608.0))
                .collect::<Result<_, _>>()?,
            32 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 2_147_483_648.0))
                .collect::<Result<_, _>>()?,
            b => return Err(anyhow!("不支持的整数采样位深: {b}")),
        },
        hound::SampleFormat::Float => match spec.bits_per_sample {
            // hound 只支持 32 位浮点采样；64 位浮点 WAV 明确报错，避免读成错误数据
            32 => reader.samples::<f32>().collect::<Result<_, _>>()?,
            b => return Err(anyhow!("不支持的浮点采样位深: {b}")),
        },
    };

    // whisper 要求单声道；若意外是多声道则做下混
    if spec.channels > 1 {
        let ch = spec.channels as usize;
        let mono: Vec<f32> =
            samples.chunks(ch).map(|frame| frame.iter().sum::<f32>() / ch as f32).collect();
        Ok(mono)
    } else {
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_wav_path(label: &str) -> std::path::PathBuf {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("subtrans-{label}-{}-{stamp}.wav", std::process::id()))
    }

    fn write_test_wav(path: &Path, frames: u64) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..frames * 2 {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn canonical_decode_args_stream_one_complete_pcm_track() {
        let args = build_canonical_decode_args("movie.mp4", Path::new("source.wav.part")).unwrap();
        let joined = args.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(" ");
        assert!(!joined.contains("-ss"));
        assert!(!joined.contains("-t"));
        assert!(joined.contains("-vn"));
        assert!(joined.contains("-ac 2"));
        assert!(joined.contains("-ar 44100"));
        assert!(joined.contains("-c:a pcm_s16le"));
        assert!(joined.contains("-f wav"));
    }

    #[test]
    fn vocal_window_args_cut_canonical_pcm_by_integer_samples() {
        let args = build_vocal_window_args(
            Path::new("source.wav"),
            Path::new("input.wav.part"),
            10_363_500,
            21_388_500,
        )
        .unwrap();
        let joined = args.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("atrim=start_sample=10363500:end_sample=21388500"));
        assert!(joined.contains("asetpts=PTS-STARTPTS"));
        assert!(!joined.contains("-ss"));
    }

    #[test]
    fn pcm_validator_reports_frames_not_rounded_seconds() {
        let path = test_wav_path("validator");
        write_test_wav(&path, 44_101);
        let info = validate_pcm_wav(&path).unwrap();
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.frames, 44_101);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn concat_manifest_uses_generated_relative_paths_only() {
        let text = concat_manifest(3);
        assert_eq!(
            text,
            "file 'cores/vocals_0000.wav'\nfile 'cores/vocals_0001.wav'\nfile 'cores/vocals_0002.wav'\n"
        );
    }

    #[test]
    fn vocal_trim_args_use_integer_atrim_sample_offsets() {
        let args = build_vocal_trim_args(
            Path::new("vocals.wav"),
            Path::new("core.wav.part"),
            220_485,
            10_584_000,
        )
        .unwrap();
        let joined = args.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(" ");
        assert!(joined.contains("atrim=start_sample=220485"));
        assert!(joined.contains("end_sample=10804485"));
        assert!(joined.contains("-c:a pcm_s16le"));
        assert!(joined.contains("-f wav"));
    }

    #[test]
    fn hhmmss_parses_ffmpeg_progress_time() {
        assert!((parse_hhmmss("00:01:02.500000").unwrap() - 62.5).abs() < 1e-9);
        assert!((parse_hhmmss("01:00:00").unwrap() - 3600.0).abs() < 1e-9);
        assert!(parse_hhmmss("garbage").is_none());
    }
}
