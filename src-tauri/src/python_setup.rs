//! Python 环境一键安装：下载 embeddable Python → 配置 pip → 安装依赖包。
//! 所有下载优先使用国内镜像。

use crate::emit_progress;
use std::path::Path;
use tauri::Manager;

const PYTHON_VERSION: &str = "3.12.9";
const PYTHON_ZIP: &str = "python-3.12.9-embed-amd64.zip";

/// 下载 + 安装 Python 环境，返回 python.exe 路径。
pub async fn setup(app: &tauri::AppHandle) -> Result<String, String> {
    let data_dir = app.path().app_data_dir().map_err(|e| format!("获取数据目录失败: {e}"))?;
    let py_dir = data_dir.join("python");
    let py_exe = py_dir.join("python.exe");

    // 已存在时不能直接返回：上次安装可能中断（缺 pip / 缺依赖包），
    // 先补齐 ._pth、pip，再检查 GPU 栈，缺什么补什么。
    if py_exe.exists() {
        emit_progress(app, "python_setup", 10.0, "检测到已有 Python，检查组件完整性...");
        let _ = configure_embed_pth(&py_dir);
        ensure_pip(&py_exe, &py_dir).await?;
        if !python_has_stack(&py_exe).await {
            emit_progress(app, "python_setup", 35.0, "检测到 GPU 组件缺失，开始补齐...");
            install_gpu_stack(app, &py_exe).await?;
        }
        emit_progress(app, "python_setup", 100.0, "Python 环境就绪");
        return Ok(py_exe.to_string_lossy().to_string());
    }

    std::fs::create_dir_all(&py_dir).map_err(|e| format!("创建目录失败: {e}"))?;

    // 1) 下载 Python embeddable
    emit_progress(app, "python_setup", 5.0, "下载 Python 环境...");
    let zip_path = py_dir.join(PYTHON_ZIP);
    download_python_zip(&zip_path).await?;

    // 2) 解压
    emit_progress(app, "python_setup", 20.0, "解压 Python 环境...");
    extract_zip(&zip_path, &py_dir)?;
    let _ = std::fs::remove_file(&zip_path);

    // 3) 配置 pip：修改 ._pth 文件取消注释 import site
    emit_progress(app, "python_setup", 25.0, "配置 pip...");
    configure_embed_pth(&py_dir)?;

    // 4) 安装 pip
    emit_progress(app, "python_setup", 30.0, "安装 pip...");
    let get_pip = py_dir.join("get-pip.py");
    download_get_pip(&get_pip).await?;
    run_python(&py_exe, &[get_pip.to_string_lossy().as_ref()], &py_dir).await?;
    let _ = std::fs::remove_file(&get_pip);

    // 5) 安装 GPU 识别栈（CUDA 版 torch + faster-whisper/demucs/soundfile）
    install_gpu_stack(app, &py_exe).await?;

    emit_progress(app, "python_setup", 100.0, "Python 环境就绪");
    Ok(py_exe.to_string_lossy().to_string())
}

async fn download_python_zip(dest: &Path) -> Result<(), String> {
    let urls = [
        format!("https://mirrors.tuna.tsinghua.edu.cn/python/{PYTHON_VERSION}/{PYTHON_ZIP}"),
        format!("https://mirrors.huaweicloud.com/python/{PYTHON_VERSION}/{PYTHON_ZIP}"),
        format!("https://www.python.org/ftp/python/{PYTHON_VERSION}/{PYTHON_ZIP}"),
    ];

    for url in &urls {
        match download_file(url, dest).await {
            Ok(()) => {
                // ZIP 魔数 + 体积双校验：镜像"软 404"（HTTP 200 错误页）会被当完整文件
                let head = std::fs::read(dest).unwrap_or_default();
                if head.len() > 5_000_000 && head.starts_with(b"PK\x03\x04") {
                    return Ok(());
                }
                let _ = std::fs::remove_file(dest);
                eprintln!("python_setup: {url}: 下载内容不是有效 ZIP（魔数/体积校验失败）");
            }
            Err(e) => {
                let _ = std::fs::remove_file(dest);
                eprintln!("python_setup: {url}: {e}");
            }
        }
    }
    Err("所有 Python 下载镜像均失败，请检查网络连接".into())
}

async fn download_get_pip(dest: &Path) -> Result<(), String> {
    let urls =
        ["https://mirrors.aliyun.com/pypi/get-pip.py", "https://bootstrap.pypa.io/get-pip.py"];
    for url in &urls {
        if download_file(url, dest).await.is_ok() {
            return Ok(());
        }
        let _ = std::fs::remove_file(dest);
    }
    Err("下载 get-pip.py 失败".into())
}

async fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    use futures_util::StreamExt;
    use std::io::Write;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    let total = resp.content_length().unwrap_or(0);
    // 流式写入，避免大文件一次性加载进内存
    let part = dest.with_extension("part");
    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut got: u64 = 0;
    let download_result: Result<(), String> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            got += chunk.len() as u64;
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
        return Err(format!("下载不完整: {got}/{total}"));
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    Ok(())
}

/// 用 tar 或 Expand-Archive 解压 zip。
fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), String> {
    // 优先用 tar（Windows 10 1803+ 内置）
    let mut tar = std::process::Command::new("tar");
    tar.args(["-xf", &zip_path.to_string_lossy(), "-C", &dest_dir.to_string_lossy()]);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        tar.creation_flags(0x0800_0000);
    }
    if tar.status().map(|s| s.success()).unwrap_or(false) {
        return Ok(());
    }

    // 回退 PowerShell Expand-Archive。
    // 路径通过环境变量（$env:...）传入并用 -LiteralPath：
    // 用户目录含单引号/空格（如 C:\Users\D'Artagnan）时不再有转义地狱，也不会拼进命令行造成注入。
    let ps_path = find_powershell_exe();
    let mut ps = std::process::Command::new(&ps_path);
    ps.args([
        "-NoProfile",
        "-Command",
        "Expand-Archive -LiteralPath $env:SUBTRANS_ZIP -DestinationPath $env:SUBTRANS_DEST -Force",
    ])
    .env("SUBTRANS_ZIP", zip_path)
    .env("SUBTRANS_DEST", dest_dir);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        ps.creation_flags(0x0800_0000);
    }
    let status = ps.status().map_err(|e| format!("解压失败: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("解压 Python 失败".into())
    }
}

fn find_powershell_exe() -> String {
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

/// 修改 embeddable Python 的 ._pth 文件，取消注释 `#import site` 使 pip 可用。
fn configure_embed_pth(py_dir: &Path) -> Result<(), String> {
    // embeddable 的 ._pth 文件名只含 major+minor（如 python312._pth），不含 patch 版本号
    let parts: Vec<&str> = PYTHON_VERSION.split('.').collect();
    let ver_short = format!("{}{}", parts[0], parts.get(1).unwrap_or(&"0"));
    let pth_name = format!("python{ver_short}._pth");
    let pth_path = py_dir.join(&pth_name);

    let content =
        std::fs::read_to_string(&pth_path).map_err(|e| format!("读取 ._pth 失败: {e}"))?;
    // 把 `#import site` / `# import site`（含空格变体）改成 `import site`
    let modified =
        content.replace("#import site", "import site").replace("# import site", "import site");
    let final_content = if modified == content && !content.contains("import site") {
        // 文件里没有这行，手动追加
        format!("{}\nimport site\n", content)
    } else {
        modified
    };
    std::fs::write(&pth_path, final_content).map_err(|e| format!("写入 ._pth 失败: {e}"))?;
    Ok(())
}

async fn run_python(python: &Path, args: &[&str], cwd: &Path) -> Result<(), String> {
    let mut cmd = tokio::process::Command::new(python);
    cmd.args(args)
        .current_dir(cwd)
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true); // pip_install 超时丢弃 future 时杀掉子进程，避免孤儿 pip 持续后台下载
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().await.map_err(|e| format!("运行 Python 失败: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!("Python 命令失败: {stderr}\n{stdout}"));
    }
    Ok(())
}

/// 确保 pip 可用；embeddable Python 需要先装 get-pip.py。
pub(crate) async fn ensure_pip(py_exe: &Path, py_dir: &Path) -> Result<(), String> {
    if run_python(py_exe, &["-m", "pip", "--version"], py_dir).await.is_ok() {
        return Ok(());
    }
    let get_pip = py_dir.join("get-pip.py");
    download_get_pip(&get_pip).await?;
    run_python(py_exe, &[get_pip.to_string_lossy().as_ref()], py_dir).await?;
    let _ = std::fs::remove_file(&get_pip);
    Ok(())
}

/// 检查 Python 是否已装 GPU 栈的核心包（faster-whisper / demucs / audio-separator）。
/// 三个都装齐才算完整；否则半成品续装时会把缺的那个补上。
async fn python_has_stack(py_exe: &Path) -> bool {
    let script = "import importlib.util; print(all(importlib.util.find_spec(m) is not None for m in ('faster_whisper', 'demucs', 'audio_separator')))";
    let mut cmd = tokio::process::Command::new(py_exe);
    cmd.args(["-c", script])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    cmd.output()
        .await
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "True")
        .unwrap_or(false)
}

/// 往已有的 Python 里一键安装 GPU 识别组件（供"有 Python 但缺包"时调用）。
pub async fn install_gpu_packages(
    app: &tauri::AppHandle,
    python_exe: &str,
) -> Result<String, String> {
    let py = std::path::PathBuf::from(python_exe);
    if !py.exists() {
        return Err(format!("指定的 Python 不存在: {python_exe}"));
    }
    // 确保 pip 可用（系统 Python 一般自带；个别精简环境才需 ensurepip）
    emit_progress(app, "python_setup", 5.0, "检查 pip...");
    if run_python(&py, &["-m", "pip", "--version"], &std::env::temp_dir()).await.is_err() {
        let _ = run_python(&py, &["-m", "ensurepip", "--upgrade"], &std::env::temp_dir()).await;
    }
    install_gpu_stack(app, &py).await?;
    emit_progress(app, "python_setup", 100.0, "GPU 识别组件就绪");
    Ok(py.to_string_lossy().to_string())
}

/// 往指定 Python 装 GPU 识别栈。顺序很关键：
/// 先装 **CUDA 版** torch/torchaudio（用上 N 卡，且必须先装，否则随后装 demucs 会顺带
/// 拉 CPU 版 torch 覆盖），再装 faster-whisper/demucs/soundfile（soundfile 是 torchaudio 2.x
/// 的音频后端，缺它 demucs 读音频会失败）。
///
/// CUDA 镜像在国内经常不可达 → 多个 CUDA 源依次尝试 → 全部失败则降级 CPU torch（GPU 不可用
/// 但其余功能正常），不会让整个安装流程报错退出。
async fn install_gpu_stack(app: &tauri::AppHandle, py_exe: &Path) -> Result<(), String> {
    let pypi = "https://mirrors.aliyun.com/pypi/simple/";

    // 0) 预检：先装 certifi（~200KB，验证 pip + 网络 + SSL 是否正常）
    emit_progress(app, "python_setup", 35.0, "验证网络连通性...");
    pip_install(
        app,
        py_exe,
        &["-m", "pip", "install", "-i", pypi, "--no-cache-dir", "certifi"],
        38.0,
        "preflight",
    )
    .await
    .map_err(|e| format!("网络预检失败（pip 无法从阿里云镜像安装包）: {e}"))?;

    // 1) CUDA 版 torch：多个 CUDA wheel 源依次尝试，全失败则用 CPU 版兜底
    emit_progress(
        app,
        "python_setup",
        40.0,
        "下载 CUDA 版 torch（约 2.5GB，较慢，请耐心等待，勿关闭窗口）...",
    );
    let cu_mirrors: &[&str] = &[
        "https://mirror.sjtu.edu.cn/pytorch-wheels/cu124",
        "https://mirrors.nju.edu.cn/pytorch-wheels/cu124",
        "https://download.pytorch.org/whl/cu124",
    ];
    let mut torch_ok = false;
    let mut torch_source: Option<&str> = None;
    for cu_idx in cu_mirrors {
        if !mirror_reachable(cu_idx).await {
            continue;
        }
        // 只用 CUDA 源安装，绝不加 --extra-index-url（PyPI 只有 CPU 版 torch，
        // 混源时 pip 会挑版本更新的 CPU 版 → “装成功但 torch.cuda 不可用”）。
        let install_ok = pip_install(
            app,
            py_exe,
            &[
                "-m",
                "pip",
                "install",
                "--index-url",
                cu_idx,
                "--no-cache-dir",
                "torch",
                "torchaudio",
            ],
            60.0,
            "torch",
        )
        .await
        .is_ok();
        // 装完必须实测 CUDA，防止源本身没有 CUDA 轮子或依赖解析异常
        if install_ok && torch_cuda_ready(py_exe).await {
            torch_ok = true;
            torch_source = Some(cu_idx);
            break;
        }
        // 清理残留（半成品 / CPU 版），换下一个源重试
        let _ = run_python(
            py_exe,
            &["-m", "pip", "uninstall", "-y", "torch", "torchaudio"],
            &std::env::temp_dir(),
        )
        .await;
    }
    if !torch_ok {
        emit_progress(
            app,
            "python_setup",
            65.0,
            "CUDA 镜像不可达，改用 CPU 版 torch（GPU 识别不可用，其余功能正常）...",
        );
        pip_install(
            app,
            py_exe,
            &["-m", "pip", "install", "-i", pypi, "torch", "torchaudio"],
            70.0,
            "torch-cpu",
        )
        .await?;
    }

    // 2) 其余包（audio-separator 用于 BS-RoFormer 人声分离，demucs 作为回退）
    emit_progress(
        app,
        "python_setup",
        85.0,
        "安装 faster-whisper / audio-separator / demucs / soundfile...",
    );
    // CUDA torch 装成功才用 onnxruntime-gpu（audio-separator[gpu]）；
    // 纯 CPU 环境装 [gpu] 会引入用不了的 onnxruntime-gpu。
    let audio_separator_pkg = if torch_ok { "audio-separator[gpu]" } else { "audio-separator" };
    pip_install(
        app,
        py_exe,
        &[
            "-m",
            "pip",
            "install",
            "-i",
            pypi,
            "faster-whisper",
            audio_separator_pkg,
            "demucs",
            "soundfile",
        ],
        90.0,
        "packages",
    )
    .await?;

    // 3) 其余包可能把 torch 覆盖成 CPU 版（pip 依赖解析优先 PyPI 新版本）：
    //    装完再验证一次，丢了 CUDA 就重装 CUDA 版。
    if torch_ok && !torch_cuda_ready(py_exe).await {
        emit_progress(
            app,
            "python_setup",
            92.0,
            "检测到 torch 被覆盖成 CPU 版，正在重装 CUDA 版...",
        );
        let _ = run_python(
            py_exe,
            &["-m", "pip", "uninstall", "-y", "torch", "torchaudio"],
            &std::env::temp_dir(),
        )
        .await;
        pip_install(
            app,
            py_exe,
            &[
                "-m",
                "pip",
                "install",
                "--index-url",
                torch_source.unwrap_or("https://download.pytorch.org/whl/cu124"),
                "--no-cache-dir",
                "torch",
                "torchaudio",
            ],
            95.0,
            "torch-repair",
        )
        .await?;
    }
    Ok(())
}

/// 检查 torch 是否可用 CUDA（装完必须验证，防止 pip 混源装了 CPU 版）。
pub(crate) async fn torch_cuda_ready(py_exe: &Path) -> bool {
    let mut cmd = tokio::process::Command::new(py_exe);
    cmd.args(["-c", "import torch; print(torch.cuda.is_available())"])
        .env("PYTHONUTF8", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x0800_0000);
    }
    tokio::time::timeout(std::time::Duration::from_secs(60), cmd.output())
        .await
        .map(|r| {
            r.map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "True")
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// 快速探测 URL 是否可达（5s 超时）。
/// 部分 wheel 索引对 HEAD 返回 405（GET 正常），HEAD 失败时回退带 Range 的 GET 探测，
/// 避免 CUDA 镜像被误判不可达而白白降级 CPU torch。
async fn mirror_reachable(url: &str) -> bool {
    let client = match reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.head(url).send().await {
        Ok(resp) if resp.status().is_success() || resp.status().is_redirection() => true,
        _ => match client.get(url).header("Range", "bytes=0-0").send().await {
            Ok(resp) => {
                // 只取 1 字节的 Range 请求，206 表示服务器正常响应
                resp.status().is_success()
                    || resp.status().is_redirection()
                    || resp.status().as_u16() == 206
            }
            Err(_) => false,
        },
    }
}

/// 跑 pip install，使用 tokio timeout + output() 避免子进程管道问题导致崩溃（exit code 120）。
async fn pip_install(
    app: &tauri::AppHandle,
    python: &Path,
    args: &[&str],
    pct: f64,
    label: &str,
) -> Result<(), String> {
    emit_progress(app, "python_setup", pct, &format!("{label}: 正在下载（请耐心等待）..."));
    match tokio::time::timeout(
        std::time::Duration::from_secs(1800), // 30 分钟，够下载 2.5GB torch
        run_python(python, args, &std::env::temp_dir()),
    )
    .await
    {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(format!("{label}: 安装超时（30 分钟）")),
    }
}
