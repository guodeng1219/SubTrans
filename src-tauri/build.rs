use std::path::{Path, PathBuf};

fn main() {
    download_ffmpeg_if_missing();
    download_models_if_missing();
    tauri_build::build();
}

fn download_ffmpeg_if_missing() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let ext = if target.contains("windows") { ".exe" } else { "" };
    let name = format!("ffmpeg-{target}{ext}");
    let dest = PathBuf::from("binaries").join(&name);

    if dest.exists() {
        println!("cargo:warning=ffmpeg sidecar already exists: {}", dest.display());
        return;
    }

    std::fs::create_dir_all("binaries").ok();
    println!("cargo:warning=Downloading ffmpeg for {target}...");

    #[cfg(target_os = "windows")]
    {
        download_ffmpeg_windows(&dest);
    }
    #[cfg(target_os = "macos")]
    {
        download_ffmpeg_macos(&dest);
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        println!(
            "cargo:warning=No auto-download for this platform. Place ffmpeg at {dest:?} manually."
        );
    }
}

/// 尝试从多个 URL 下载 zip，返回成功下载的文件内容（临时文件路径）。
/// URL 按优先级排列，任一成功即返回；全部失败则 panic。
#[cfg(target_os = "windows")]
fn try_download_zip(urls: &[&str], tmp: &Path) {
    let ps = find_powershell();
    for url in urls {
        println!("cargo:warning=Trying: {url}");
        let result = std::process::Command::new(&ps)
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "$ProgressPreference='SilentlyContinue'; try {{ Invoke-WebRequest -Uri '{}' -OutFile '{}' -TimeoutSec 300; exit 0 }} catch {{ exit 1 }}",
                    url,
                    tmp.display()
                ),
            ])
            .status();
        let ok = match result {
            Ok(s) => s.success(),
            Err(_) => false,
        };
        if ok && tmp.exists() && tmp.metadata().map(|m| m.len() > 1000).unwrap_or(false) {
            return;
        }
        let _ = std::fs::remove_file(tmp);
        println!("cargo:warning=Failed, trying next mirror...");
    }
    panic!("All ffmpeg download mirrors failed. Place ffmpeg.exe at binaries/ manually.");
}

#[cfg(target_os = "windows")]
fn download_ffmpeg_windows(dest: &Path) {
    let tmp = PathBuf::from("binaries/ffmpeg_temp.zip");
    let extract = PathBuf::from("binaries/ffmpeg_extracted");

    let urls = [
        // 清华 TUNA BtbN mirror (国内优先)
        "https://mirrors.tuna.tsinghua.edu.cn/github-release/BtbN/FFmpeg-Builds/LatestRelease/ffmpeg-master-latest-win64-gpl.zip",
        // USTC mirror
        "https://mirrors.ustc.edu.cn/ffmpeg/releases/win64/ffmpeg-master-latest-win64-gpl.zip",
        // gyan.dev 原始
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    ];

    try_download_zip(&urls, &tmp);

    // Extract
    let ps = find_powershell();
    let _ = std::fs::remove_dir_all(&extract);
    let status = std::process::Command::new(&ps)
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}'",
                tmp.display(),
                extract.display()
            ),
        ])
        .status()
        .expect("Failed to extract ffmpeg zip");
    if !status.success() {
        panic!("ffmpeg extraction failed.");
    }

    let ffmpeg_exe =
        find_in_dir(&extract, "ffmpeg.exe").expect("ffmpeg.exe not found in downloaded archive");

    std::fs::copy(&ffmpeg_exe, dest).expect("Failed to copy ffmpeg.exe to binaries/");

    // Cleanup
    let _ = std::fs::remove_dir_all(&extract);
    let _ = std::fs::remove_file(&tmp);

    println!("cargo:warning=ffmpeg sidecar ready: {}", dest.display());
}

#[cfg(target_os = "macos")]
fn download_ffmpeg_macos(dest: &Path) {
    // evermeet.cx 只有 Intel (x86_64) 构建；Apple Silicon 需要 ffmpeg-static 的
    // ffmpeg-darwin-arm64 单文件。按 TARGET 环境变量选择正确的架构文件。
    let target = std::env::var("TARGET").unwrap_or_default();
    let asset =
        if target.contains("aarch64") { "ffmpeg-darwin-arm64" } else { "ffmpeg-darwin-x64" };
    println!("cargo:warning=Downloading {asset} for {target}...");

    let urls = [
        // 清华 TUNA github-release 镜像 (国内优先)
        format!("https://mirrors.tuna.tsinghua.edu.cn/github-release/eugeneware/ffmpeg-static/b6.1.1/{asset}"),
        // GitHub 原始 release
        format!("https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/{asset}"),
    ];

    let tmp = PathBuf::from("binaries/ffmpeg_temp_macos");
    for url in &urls {
        println!("cargo:warning=Trying: {url}");
        let status = std::process::Command::new("curl")
            .args([
                "-L",
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "-o",
                tmp.to_str().unwrap(),
                "-s",
                "-S",
                url,
            ])
            .status();
        let ok = match status {
            Ok(s) => s.success(),
            Err(_) => false,
        };
        // ffmpeg-static 产物约 30-50MB，远大于 1KB；用 1MB 下限排除错误页。
        if ok && tmp.exists() && tmp.metadata().map(|m| m.len() > 1_000_000).unwrap_or(false) {
            std::fs::copy(&tmp, dest).expect("Failed to copy ffmpeg to binaries/");
            let _ =
                std::process::Command::new("chmod").args(["+x", dest.to_str().unwrap()]).status();
            let _ = std::fs::remove_file(&tmp);
            println!("cargo:warning=ffmpeg sidecar ready: {}", dest.display());
            return;
        }
        let _ = std::fs::remove_file(&tmp);
        println!("cargo:warning=Failed, trying next mirror...");
    }
    panic!("All ffmpeg download mirrors failed. Place ffmpeg at binaries/ manually.");
}

#[cfg(target_os = "windows")]
fn find_powershell() -> String {
    let candidates = [
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\Program Files\PowerShell\7\pwsh.exe",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return c.to_string();
        }
    }
    "powershell".to_string()
}

fn find_in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&current) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if entry.file_name().to_string_lossy() == name {
                    return Some(path);
                }
            }
        }
    }
    None
}

// ── 内置模型下载（tiny 应急模型 + VAD 模型） ──

fn download_models_if_missing() {
    std::fs::create_dir_all("models").ok();

    let tiny = PathBuf::from("models/ggml-tiny.bin");
    if !tiny.exists() {
        println!("cargo:warning=Downloading ggml-tiny.bin (~75MB)...");
        download_file_simple(
            "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            &tiny,
        );
    } else {
        println!("cargo:warning=ggml-tiny.bin already exists");
    }

    let vad = PathBuf::from("models/ggml-silero-v5.1.2.onnx");
    if !vad.exists() {
        println!("cargo:warning=Downloading ggml-silero-v5.1.2.onnx (~2MB)...");
        download_file_simple(
            "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-silero-v5.1.2.onnx",
            &vad,
        );
    } else {
        println!("cargo:warning=ggml-silero-v5.1.2.onnx already exists");
    }
}

/// 简单文件下载（用 curl 或 certutil），失败不 panic（模型可在运行时再下载）。
fn download_file_simple(url: &str, dest: &Path) {
    let part = dest.with_extension("part");
    #[cfg(target_os = "windows")]
    {
        // 优先 curl
        let status = std::process::Command::new("curl")
            .args([
                "-L",
                "-o",
                part.to_str().unwrap(),
                url,
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "-s",
                "-S",
            ])
            .status();
        if status.map(|s| s.success()).unwrap_or(false) && part.exists() {
            let _ = std::fs::rename(&part, dest);
            println!("cargo:warning=Downloaded: {}", dest.display());
            return;
        }
        // 回退 certutil
        let _ = std::fs::remove_file(&part);
        let status = std::process::Command::new("certutil")
            .args(["-urlcache", "-split", "-f", url, part.to_str().unwrap()])
            .status();
        if status.map(|s| s.success()).unwrap_or(false) && part.exists() {
            let _ = std::fs::rename(&part, dest);
            println!("cargo:warning=Downloaded (certutil): {}", dest.display());
            return;
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("curl")
            .args([
                "-L",
                "-o",
                part.to_str().unwrap(),
                url,
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "-s",
                "-S",
            ])
            .status();
        if status.map(|s| s.success()).unwrap_or(false) && part.exists() {
            let _ = std::fs::rename(&part, dest);
            println!("cargo:warning=Downloaded: {}", dest.display());
            return;
        }
    }
    let _ = std::fs::remove_file(&part);
    println!(
        "cargo:warning=WARNING: Could not download {}. Will be downloaded at runtime.",
        dest.display()
    );
}
