//! 从视频中提取音频，供识别引擎使用。
//!
//! 用系统 ffmpeg 把任意视频转成识别需要的 PCM 音频。app 启动时会尝试把
//! ffmpeg 目录注入 PATH（见 lib.rs ensure_ffmpeg_on_path），双击启动也能用。

use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Command;

/// 跑 ffmpeg 命令并校验：退出码成功 **且** 输出文件非空。
/// 失败时带上 ffmpeg stderr 末尾几行，便于定位（无音轨 / 路径异常 / 找不到 ffmpeg 等）。
fn run_ffmpeg(mut cmd: Command, out_wav: &Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // GUI 程序 spawn 子进程会闪命令行窗口；分片处理频繁调用，用 CREATE_NO_WINDOW 隐藏。
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let output = cmd
        .output()
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

/// 调用 ffmpeg 抽取指定时间段的音轨成临时 WAV（16kHz / mono / s16）。
pub fn extract_audio_range(
    video_path: &str,
    out_wav: &Path,
    ffmpeg_bin: &str,
    start_sec: f64,
    duration_sec: f64,
) -> Result<()> {
    let out = out_wav.to_str().ok_or_else(|| anyhow!("无效输出路径"))?;

    // 主路径：输入端 -ss 快速 seek + soxr 高质量重采样 + 高通滤波去低频乐器干扰
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.args([
        "-y",
        "-ss",
        &format!("{start_sec}"),
        "-i",
        video_path,
        "-t",
        &format!("{duration_sec}"),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=80",
        "-resampler",
        "soxr",
        "-acodec",
        "pcm_s16le",
        out,
    ]);
    let primary = run_ffmpeg(cmd, out_wav);
    if primary.is_ok() {
        return Ok(());
    }

    // 回退：少数视频在输入端 seek 下音轨“零包”（时间戳异常 / 默认选轨不对 / soxr 不适配该参数）。
    // 改用更稳的组合：输出端 -ss（精确、必定读包）+ 显式选第一条音轨 + 默认重采样器（swr）。
    let mut fb = Command::new(ffmpeg_bin);
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
        "highpass=f=80",
        "-acodec",
        "pcm_s16le",
        out,
    ]);
    run_ffmpeg(fb, out_wav).map_err(|fb_err| {
        // 两条路都失败：把主路径的报错也带上，便于判断是无音轨/编解码不支持还是其它
        anyhow!("{fb_err}（主路径亦失败：{})", primary.unwrap_err())
    })
}

/// 提取整段音频为 44.1kHz 立体声 WAV，作为 demucs 人声分离的输入（保留更多信息，效果更好）。
pub fn extract_audio_full(video_path: &str, out_wav: &Path, ffmpeg_bin: &str) -> Result<()> {
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
    run_ffmpeg(cmd, out_wav)
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
