# 字幕翻译工具 (Tauri 桌面应用)

本地视频自动加翻译字幕。

- **CPU 识别**：whisper.cpp 编译进二进制，**零依赖**，装完即用。
- **GPU 识别 / 高精度模式**（可选）：GPU 识别（faster-whisper）和人声分离（demucs / BS-RoFormer）依赖 **Python + faster-whisper/demucs/audio-separator/torch**；应用内有「一键安装 Python 环境」按钮，普通用户不必手动配置。
- **翻译引擎**：免费在线（MyMemory）/ DeepSeek / 本地 Ollama，按需选。

最终产物：**Windows `.exe` 安装包** + **macOS `.dmg`**。

---

## 这个工程已经帮你做好了什么

- ✅ 完整 Tauri 2 项目结构（Rust 后端 + Vite 打包的 WebView 前端）
- ✅ CPU 语音识别：`whisper-rs`（whisper.cpp 绑定，编译进二进制，零依赖）
- ✅ GPU 语音识别：faster-whisper（常驻 Python 进程，模型常驻显存，large-v3 也能秒级）
- ✅ 高精度模式：人声分离（demucs / BS-RoFormer，先抽纯人声轨再识别，应对动漫/影视 BGM）
- ✅ 中文 AI 纠错：LLM 按上下文修同音字/错别字（DeepSeek / 本地 Ollama）
- ✅ 边播边译流式管线：按时间段分片，边播放边识别 + 翻译，字幕实时叠加
- ✅ 三种翻译引擎：免费在线 / DeepSeek / 本地 Ollama
- ✅ 一键安装 Python 环境：下载内嵌版 Python + 装 faster-whisper/demucs/audio-separator/torch
- ✅ 首次启动向导：一键下载识别模型、可选安装 Ollama + 拉翻译模型
- ✅ 视频播放器 + 字幕实时叠加 + SRT 导出
- ✅ ffmpeg 已内置：构建时自动下载并作为 sidecar 打包，用户零配置
- ✅ 打包配置：Windows NSIS 安装包 + macOS DMG
- ✅ GitHub Actions CI：推一个 tag 就云端同时出两个平台安装包

## 还需要你做的（一次性）

1. 安装开发环境（仅打包时需要，最终用户不需要）
2. 跑一次 build（本机）或推 tag（云端 CI）得到安装包

---

## 一、本机打包（出当前系统的安装包）

### 前置环境

```bash
# 1. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Node.js 20+（用于 Tauri CLI、Vite 打包、前端依赖）

# 3. 平台编译依赖
#   - Windows: 安装 "Visual Studio Build Tools"（含 C++）、CMake、WebView2(Win10/11自带)
#   - macOS:   xcode-select --install && brew install cmake
```

> whisper.cpp 编译需要 **CMake + C++ 编译器**，这是唯一的"重"依赖，只在打包机器上需要。

### 生成图标（首次，一次即可）

```bash
npm install
npm run tauri icon src-tauri/icons/icon-source.png
```

这会把源图标自动生成所有平台规范尺寸（含 .ico / .icns）。

### 打包

```bash
npm install
npm run tauri build
```

> 前端用 Vite 打包（`vite build` → `dist/`），由 `tauri.conf.json` 的 `beforeBuildCommand` 自动触发，无需手动跑。

产物位置：
- **Windows**：`src-tauri/target/release/bundle/nsis/*.exe`（双击即可安装）
- **macOS**：`src-tauri/target/release/bundle/dmg/*.dmg`

开发时实时预览：`npm run tauri dev`（会先起 Vite dev server，Tauri 再连上去）。

---

## 二、云端同时出两个平台安装包（推荐）

因为 macOS 的 `.dmg` 必须在 macOS 上构建、Windows `.exe` 在 Windows 上构建，
**同时出两个平台**最省事的方式是 GitHub Actions（已配好 `.github/workflows/release.yml`）：

```bash
git init && git add . && git commit -m "init"
git remote add origin <你的仓库地址>
git push -u origin main

# 打个版本 tag 触发自动构建
git tag v0.1.0
git push origin v0.1.0
```

几分钟后，在仓库的 **Releases** 页面会出现一个 draft release，
里面挂着 Windows `.exe`、macOS Intel `.dmg`、macOS Apple Silicon `.dmg`。
（也可以在 Actions 页面点 "Run workflow" 手动触发。）

---

## 三、ffmpeg：已内置，无需手动处理

Whisper 需要先把视频转成音频，这一步用 ffmpeg。**现在已自动内置**，打包者无需任何手工操作：

- 构建时 `src-tauri/build.rs` 会自动下载对应平台的 ffmpeg 静态二进制到 `src-tauri/binaries/`
  （国内优先走清华 TUNA / USTC 镜像，失败再回退官方源；已下载则跳过）。
- `tauri.conf.json` 已配 `"externalBin": ["binaries/ffmpeg"]`，把它作为 sidecar 打进安装包。
- 运行时后端 `resolve_ffmpeg()` 自动定位：生产环境用打包进去的 ffmpeg，开发环境回退系统 PATH。
  前端不再传 ffmpeg 路径。

> 运行时定位顺序：主 exe 同级（sidecar）→ `resource_dir()`（旧布局）→ 系统 PATH。
> 前端不再传 ffmpeg 路径。

---

## 四、GPU 识别 / 高精度模式：Python sidecar（可选增强）

CPU 识别（whisper.cpp）零依赖；**GPU 识别（faster-whisper）和人声分离（demucs / BS-RoFormer）需要 Python**：

- 依赖包：`faster-whisper`、`demucs`、`audio-separator`、`torch`。
- 应用内「一键安装 Python 环境」按钮（后端 `python_setup`）：下载内嵌版 Python 3.12 → 配 pip
  → 优先装 CUDA 版 torch，镜像不可达时自动降级 CPU 版，再装 faster-whisper/demucs/audio-separator。
  普通用户点一下即可，也可让应用自动探测系统已有的 Python（`python_detect`），
  或对已有 Python 用「安装 GPU 加速组件」补齐依赖。
- 识别服务脚本 `python/fw_server.py` 用 `include_str!` 内嵌进二进制，运行时写到临时文件启动
  （自包含，打包者不必单独分发它）；模型常驻显存，逐分片通过 stdin/stdout 行协议喂音频。

> 注：一键安装会先尝试 CUDA(cu124) 版 torch；若 CUDA 镜像不可达才降级 CPU 版
> （此时 `torch.cuda.is_available()` 为 false，GPU 识别/人声分离实际跑在 CPU）。
>
> 分发提醒：当前 `.gitignore` 排除了 `python-bundle/Lib` 与 `Scripts`，CI 构建不会携带完整 Python 依赖；
> 最终用户按需通过应用内「一键安装 Python 环境」安装即可。

---

## 常见编译问题

### `Unable to find libclang` 或 结构体尺寸 overflow (`_IO_FILE` / `_G_fpos_t`) (Windows)

`whisper-rs-sys` 用 bindgen 现场生成 C 绑定，需要 LLVM 的 `libclang.dll`。
**正确做法是装 LLVM 让 bindgen 跑起来**（不要用 `WHISPER_DONT_GENERATE_BINDINGS`，
那会改用 Linux 预生成绑定，在 Windows 上因结构体尺寸不匹配而编译失败）：

```powershell
winget install LLVM.LLVM
$env:LIBCLANG_PATH="C:\Program Files\LLVM\bin"
npm run tauri build
```

macOS 的 Xcode Command Line Tools 已自带 libclang，一般无需额外安装。
打包机器才需要 LLVM；最终用户的电脑不需要任何依赖。

## 翻译引擎说明

| 引擎 | 联网 | 需要 Key | 体验 |
|------|------|---------|------|
| 免费在线 (MyMemory) | 是 | 否 | 开箱即用，默认 |
| DeepSeek API | 是 | 是 | 翻译质量最好，费用极低 |
| 本地 Ollama | 否（离线）| 否 | 向导里一键装，模型约 4.7GB |

> 中文 AI 纠错（同音字/错别字）也复用 DeepSeek / Ollama 两种 LLM 引擎；免费在线引擎不支持纠错。

## 识别模型

首次启动向导下载，存到应用数据目录，之后离线运行。下表是向导里的选项
（CPU 用 whisper.cpp 的 ggml 模型；勾选「用 faster-whisper 在显卡上跑」时，
GPU 引擎首次会自动下载同名模型的 CTranslate2 格式）：

| 模型 | 大小 | 说明 |
|------|------|------|
| tiny | 75MB | 极速，精度低 |
| base | 142MB | 速度最快 |
| small | 466MB | 速度快 |
| medium | 1.5GB | 中文精度显著提升 |
| large-v3 | 2.9GB | 最高精度，默认 / 推荐（尤其中文）|

---

## 项目结构

```
subtrans/
├── package.json                 前端依赖 + tauri / vite 脚本
├── vite.config.js               前端打包配置（源码 src/ → 产物 dist/）
├── dist/                        Vite 构建产物（npm run build 生成，frontendDist 指向它）
├── src/                         前端（Vite 打包）
│   ├── index.html               主界面 + 首启动向导
│   ├── styles.css
│   └── main.js                  调用 Rust 命令、播放同步、字幕渲染、流式分片调度
├── python/
│   └── fw_server.py             faster-whisper 常驻识别服务（被 include_str! 内嵌进二进制）
├── src-tauri/                   Rust 后端
│   ├── Cargo.toml
│   ├── tauri.conf.json          打包配置（nsis + dmg + ffmpeg sidecar）
│   ├── build.rs                 构建时自动下载 ffmpeg 到 binaries/
│   ├── binaries/                ffmpeg sidecar（build.rs 自动放置）
│   ├── capabilities/default.json 权限
│   ├── icons/                   图标（用 tauri icon 重新生成）
│   └── src/
│       ├── main.rs
│       ├── lib.rs               Tauri 命令入口（process_chunk、separate_vocals、python_setup 等）
│       ├── asr/
│       │   ├── mod.rs           ASR 抽象层 + 语气词清理
│       │   └── whisper.rs       whisper.cpp(CPU) 识别
│       ├── fw_ipc.rs            faster-whisper(GPU) 常驻进程 IPC
│       ├── correct.rs           中文同音字 / 错别字 LLM 纠错
│       ├── python_setup.rs      一键安装内嵌版 Python + 依赖包
│       ├── translate.rs         三种翻译引擎
│       ├── ollama.rs            Ollama 检测 / 拉模型
│       └── ffmpeg.rs            音频提取（调用 sidecar ffmpeg）
└── .github/workflows/release.yml  跨平台自动打包 CI
```
