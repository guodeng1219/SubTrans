import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";

// ───────── 简易本地配置（仅记一个"是否完成向导"标记） ─────────
const SETUP_KEY = "subtrans.setupDone";
const isSetupDone = () => localStorage.getItem(SETUP_KEY) === "1";
const markSetupDone = () => localStorage.setItem(SETUP_KEY, "1");

// ───────── 全局状态 ─────────
const state = {
  videoPath: "",
  modelName: "large-v3",
  subtitles: [],
};

// 边播边译的流水线状态
const stream = {
  chunkLen: 120,     // 当前分片秒数（慢机降级到 45）
  readyUntil: 0,     // 已生成字幕覆盖到的视频秒数
  total: 0,          // 视频总时长
  processing: false, // 是否正在处理某一片
  pumping: false,    // 连续流水线是否在跑
  running: false,    // 本次会话是否已开始
  done: false,       // 是否已处理到结尾
  waiting: false,    // 播放因等字幕而暂停
  smallSlices: false,
  audioSource: null, // 高精度模式：分离出的人声轨路径（ASR 用它，播放仍用原视频）
};

// GPU 升级状态机
let gpuUpgrading = false;     // 是否正在升级中
let gpuUpgradeError = "";     // 升级失败的错误信息（空 = 无错误）

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitUntil(cond, step = 200) {
  while (!cond()) await sleep(step);
}

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ───────── 步骤导航 ─────────
function goStep(n) {
  $$(".step-item").forEach((it) => it.classList.toggle("active", Number(it.dataset.step) === n));
  $$(".step-panel").forEach((p) => p.classList.toggle("hidden", Number(p.dataset.panel) !== n));
}
function setStepDone(n) {
  const item = document.querySelector(`.step-item[data-step="${n}"]`);
  if (item && !item.classList.contains("done")) {
    item.classList.add("done");
    item.querySelector(".step-dot").textContent = "✓";
  }
}

// ═══════════ 进度事件总线 ═══════════
listen("progress", (e) => {
  const { stage, pct, message } = e.payload;
  if (stage === "download_model") {
    $("#modelBar")?.classList.remove("hidden");
    setFill("#modelBar .bar-fill", pct);
    $("#modelMsg").textContent = message;
  } else if (stage === "ollama_pull") {
    $("#ollamaBar")?.classList.remove("hidden");
    setFill("#ollamaBar .bar-fill", pct);
    $("#ollamaMsg").textContent = message;
  } else if (stage === "cuda_upgrade") {
    // GPU 升级进度 → 更新徽章为黄色升级中状态 + 进度条
    $("#gpuBadge").className = "gpu-badge upgrading";
    $("#gpuBadgeText").textContent = "GPU 升级中...";
    const prog = $("#gpuProgress");
    prog.classList.remove("hidden");
    $("#gpuProgressFill").style.width = `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
    $("#gpuProgressText").textContent = message;
  } else if (stage === "python_setup") {
    // Python 环境安装进度 → 更新引擎配置区的进度条
    $("#pyBar")?.classList.remove("hidden");
    setFill("#pyBarFill", pct);
    const st = $("#pyStatus");
    if (st) st.textContent = message;
  } else {
    setFill("#runFill", pct);
    $("#runMsg").textContent = message;
  }
});

// CUDA 升级完成事件
listen("cuda-ready", () => {
  gpuUpgrading = false;
  gpuUpgradeError = "";
  $("#gpuBadge").className = "gpu-badge ready";
  $("#gpuBadge").style.color = "";
  $("#gpuBadgeText").textContent = "GPU 就绪";
  // 隐藏进度条和错误
  $("#gpuProgress").classList.add("hidden");
  $("#gpuProgressFill").style.width = "0%";
  $("#gpuError").classList.add("hidden");
  $("#gpuUpgradeBtn").classList.add("hidden");
  // 重新检测环境以启用 GPU 选项
  detectEnv();
});

function setFill(sel, pct) {
  const el = $(sel);
  if (el) el.style.width = `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
}

// ═══════════ 启动：决定显示向导还是主界面 ═══════════
async function boot() {
  // 用上次向导实际下载的模型名（否则默认 large-v3 可能并未下载，会误退回向导）
  state.modelName = localStorage.getItem("subtrans.model") || state.modelName;
  if (isSetupDone()) {
    const exists = await invoke("model_exists", { name: state.modelName }).catch(() => false);
    if (exists) return showApp();
  }
  showWizard();
}

function showWizard() {
  $("#wizard").classList.remove("hidden");
  $("#app").classList.add("hidden");
  initWizard();
}
function showApp() {
  $("#wizard").classList.add("hidden");
  $("#app").classList.remove("hidden");
  initApp();
}

// ═══════════ 向导逻辑 ═══════════
function initWizard() {
  let step = 0;
  const steps = $$(".wizard-step");
  const goto = (i) => {
    step = i;
    steps.forEach((s, idx) => s.classList.toggle("hidden", idx !== i));
  };

  // 步骤1：下载识别模型
  $("#dlModelBtn").addEventListener("click", async () => {
    const name = document.querySelector('input[name="wm"]:checked').value;
    state.modelName = name;
    localStorage.setItem("subtrans.model", name); // 记住选的模型，启动时据此校验
    $("#dlModelBtn").disabled = true;
    try {
      const exists = await invoke("model_exists", { name });
      if (!exists) await invoke("download_model", { name });
      goto(1);
    } catch (err) {
      console.error("[subtrans] download_model failed:", name, err);
      $("#modelMsg").textContent = `下载失败: ${err}`;
    } finally {
      $("#dlModelBtn").disabled = false;
    }
  });

  // 步骤2：翻译方式切换
  $$(".opt-card").forEach((card) => {
    card.addEventListener("click", () => {
      $$(".opt-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      const isOllama = card.dataset.opt === "ollama";
      $("#ollamaSetup").classList.toggle("hidden", !isOllama);
      if (isOllama) checkOllama();
    });
  });

  // 检测环境：GPU 状态 + 内置模型提示
  (async () => {
    try {
      const env = await invoke("env_status");
      if (env.has_gpu && env.cuda_torch_ready && (env.fw_ready || env.demucs_ready)) {
        $("#wizardPyStatus").textContent = "✓ 检测到 GPU 加速可用。";
        $("#wizardPyStatus").style.color = "var(--accent)";
      } else if (env.bundled_tiny) {
        $("#wizardPyStatus").textContent = "已内置 tiny 应急模型，可先使用；推荐下载 large-v3 获得最佳精度。";
        $("#wizardPyStatus").style.color = "var(--muted)";
        // 无 GPU 且有内置模型：默认改选 medium（CPU 上 faster）
        const med = document.querySelector('input[name="wm"][value="medium"]');
        const lg = document.querySelector('input[name="wm"][value="large-v3"]');
        if (med && lg?.checked) med.checked = true;
      } else {
        $("#wizardPyStatus").textContent = "";
        const med = document.querySelector('input[name="wm"][value="medium"]');
        const lg = document.querySelector('input[name="wm"][value="large-v3"]');
        if (med && lg?.checked) med.checked = true;
      }
    } catch {
      $("#wizardPyStatus").textContent = "";
    }
  })();

  $("#installOllamaBtn").addEventListener("click", async () => {
    try {
      const url = await invoke("ollama_installer_url");
      await openUrl(url); // 打开官方安装器下载，用户双击安装
      $("#ollamaMsg").textContent = "已打开 Ollama 安装器下载页，安装后点「测试」。";
    } catch (err) {
      $("#ollamaMsg").textContent = `${err}`;
    }
  });

  $("#pullModelBtn").addEventListener("click", async () => {
    $("#pullModelBtn").disabled = true;
    try {
      await invoke("ollama_pull", { model: "qwen2.5:7b" });
    } catch (err) {
      $("#ollamaMsg").textContent = `拉取失败: ${err}`;
    } finally {
      $("#pullModelBtn").disabled = false;
    }
  });

  $("#wizardDone").addEventListener("click", () => {
    markSetupDone();
    showApp();
  });

  goto(0);
}

async function checkOllama() {
  $("#ollamaMsg").textContent = "正在检测 Ollama...";
  const st = await invoke("ollama_status", {}).catch(() => ({ running: false, models: [], error: "调用失败" }));
  if (st.running) {
    $("#ollamaMsg").textContent = st.models.length
      ? `已就绪，已安装模型: ${st.models.join(", ")}`
      : "Ollama 已安装，但还没有翻译模型，请点下方下载。";
  } else {
    const detail = st.error ? `（${st.error}）` : "";
    $("#ollamaMsg").textContent = `未检测到 Ollama${detail}，请先点「下载安装」。`;
  }
}

// ═══════════ 主界面逻辑 ═══════════
function initApp() {
  // 步骤导航
  $$(".step-item").forEach((item) => {
    item.addEventListener("click", () => goStep(Number(item.dataset.step)));
  });
  // 拖放区点击 = 打开视频
  $("#dropZone").addEventListener("click", () => $("#openBtn").click());

  $("#whisperModel").value = state.modelName;

  // 打开视频
  $("#openBtn").addEventListener("click", openVideo);
  $("#startBtn").addEventListener("click", startProcess);
  $("#exportBtn").addEventListener("click", exportSrt);
  $("#testOllama").addEventListener("click", testOllama);
  $("#pyInstallBtn").addEventListener("click", installPython);
  $("#gpuPackagesBtn").addEventListener("click", installGpuPackages);

  // 源语言切换 → 引擎提示
  $("#sourceLang").addEventListener("change", updateEngineHint);
  updateEngineHint();

  // 模型切换 → 更新预估时间
  $("#whisperModel").addEventListener("change", updateEstimate);

  // 纠错开关提示
  $("#correctEnabled").addEventListener("change", updateCorrectHint);

  // 高精度模式（人声分离）
  $("#hiQuality").addEventListener("change", () => {
    $("#hiQualityHint").textContent = $("#hiQuality").checked
      ? "开始时会先分离人声（BS-RoFormer / demucs，GPU 较快），再边播边译。可在「引擎」页检测/配置。"
      : "";
  });
  $("#checkDemucs").addEventListener("click", async () => {
    $("#demucsStatus").textContent = "检测中...";
    try {
      const msg = await invoke("demucs_check", { pythonExe: $("#pyPython").value.trim() });
      $("#demucsStatus").textContent = "✓ " + msg;
      $("#demucsStatus").style.color = "var(--accent)";
    } catch (err) {
      $("#demucsStatus").textContent = "✗ " + err;
      $("#demucsStatus").style.color = "var(--danger)";
    }
  });

  // GPU 加速识别（faster-whisper）
  $("#useFw").addEventListener("change", () => {
    $("#useFwHint").textContent = $("#useFw").checked
      ? "识别走 GPU（faster-whisper），large-v3 也能很快；首次会下载对应模型。可在「引擎」页检测。"
      : "";
    updateEngineHint();
    updateEstimate();
  });
  $("#fwCheck").addEventListener("click", async () => {
    $("#fwStatus").textContent = "检测中...";
    try {
      const msg = await invoke("fw_check", { pythonExe: $("#pyPython").value.trim() });
      $("#fwStatus").textContent = "✓ " + msg;
      $("#fwStatus").style.color = "var(--accent)";
    } catch (err) {
      $("#fwStatus").textContent = "✗ " + err;
      $("#fwStatus").style.color = "var(--danger)";
    }
  });

  // 恢复上次检测/安装到的 Python 路径（detectEnv 会用新探测结果覆盖）
  const savedPy = localStorage.getItem("subtrans.pythonPath");
  if (savedPy) $("#pyPython").value = savedPy;

  // 探测环境
  detectEnv();

  // GPU 升级按钮 + 重试按钮
  $("#gpuUpgradeBtn").addEventListener("click", startGpuUpgrade);
  $("#gpuRetryBtn").addEventListener("click", startGpuUpgrade);

  // 播放器字幕同步 + 追上未生成区域时暂停等待
  const video = $("#video");
  video.addEventListener("timeupdate", () => {
    updateOverlay(video.currentTime);
    if (
      stream.running &&
      !stream.done &&
      !video.paused &&
      !stream.waiting &&
      video.currentTime >= stream.readyUntil - 0.3
    ) {
      stream.waiting = true;
      video.pause();
      $("#runMsg").textContent = "字幕生成中…";
    }
  });
  // 向后跳转到尚未生成处：把流水线重定位到该分片继续
  video.addEventListener("seeking", () => {
    if (!stream.running) return;
    const t = video.currentTime;
    if (t > stream.readyUntil) {
      stream.readyUntil = Math.floor(t / stream.chunkLen) * stream.chunkLen;
      stream.done = false;
      pump();
    }
  });
}

async function openVideo() {
  const path = await openDialog({
    filters: [{ name: "视频", extensions: ["mp4", "mkv", "avi", "mov", "webm", "flv", "m4v"] }],
  });
  if (!path) return;
  // 切换视频前停掉旧流水线，避免旧任务继续往新列表里塞字幕
  stream.running = false;
  stream.done = true;
  stream.processing = false;
  stream.pumping = false;
  stream.audioSource = null; // 旧视频的人声轨不能留给新视频
  state.videoPath = path;
  state.subtitles = [];
  renderSubList();
  $("#fileName").textContent = path.split(/[\\/]/).pop();
  $("#video").src = convertFileSrc(path);
  $("#stageEmpty").classList.add("hidden");
  $("#startBtn").disabled = false;
  $("#exportBtn").disabled = true;
  // 视频加载后更新预估时间
  $("#video").addEventListener("loadedmetadata", () => updateEstimate(), { once: true });
  // 步骤 1 完成 → 自动进入步骤 2
  setStepDone(1);
  goStep(2);
}

function currentEngine() {
  const sel = $("#engine").value;
  if (sel === "deepseek")
    return { kind: "DeepSeek", api_key: $("#dsKey").value.trim(), model: $("#dsModel").value };
  if (sel === "ollama")
    return { kind: "Ollama", host: $("#olHost").value.trim(), model: $("#olModel").value.trim() };
  return { kind: "Free" };
}

// 纠错引擎：优先 DeepSeek（填了 Key），否则本地 Ollama（host 默认已填）；都没有 → null
function correctionEngine() {
  const key = $("#dsKey").value.trim();
  if (key) return { kind: "DeepSeek", api_key: key, model: $("#dsModel").value };
  const host = $("#olHost").value.trim();
  if (host) return { kind: "Ollama", host, model: $("#olModel").value.trim() };
  return null;
}

// 调用后端处理一个时间段，把返回的字幕追加进列表，返回后端结果
async function invokeChunk(start, dur) {
  const ce = $("#correctEnabled").checked ? correctionEngine() : null;
  const videoAtCall = state.videoPath; // 在飞请求返回时可能已切换视频，丢弃旧结果
  const res = await invoke("process_chunk", {
    videoPath: state.videoPath,
    audioSource: stream.audioSource,
    modelName: $("#whisperModel").value,
    startSec: start,
    durationSec: dur,
    leadInSec: start > 0 ? 5.0 : 0.0,
    totalSec: stream.total,
    sourceLang: $("#sourceLang").value || null,
    targetLang: $("#targetLang").value,
    engine: currentEngine(),
    translateEnabled: $("#transEnabled").checked,
    correctEnabled: !!ce,
    correctEngine: ce,
    glossary: $("#glossary").value || "",
    useFw: $("#useFw").checked,
    vadEnabled: $("#vadFilter").checked,
    // 优先用探测/用户填写的 Python（否则检测是绿的、实际运行却找不到解释器）；
    // 留空时后端才回退 bundled Python。
    fwPython: $("#pyPython").value.trim(),
    fwDevice: $("#fwDevice").value,
  });
  if (state.videoPath === videoAtCall) res.segments.forEach(addSubtitle);
  return res;
}

// 处理"下一片"（从 readyUntil 开始），返回墙钟耗时 ms；无事可做返回 null
async function processNextChunk() {
  if (stream.processing || stream.done) return null;
  const start = stream.readyUntil;
  const dur = Math.min(stream.chunkLen, stream.total - start);
  if (dur <= 0.05) {
    stream.done = true;
    return null;
  }
  stream.processing = true;
  const t0 = performance.now();
  const videoAtStart = state.videoPath; // 处理期间可能切换视频，返回后丢弃旧结果
  try {
    const res = await invokeChunk(start, dur);
    // 处理期间用户可能向前 seek 改大了 readyUntil，不能被旧分片覆盖
    stream.readyUntil = Math.max(stream.readyUntil, start + dur);
    if (state.videoPath !== videoAtStart) return null; // 已切视频，丢弃本次结果
    updateStatus(res);
    if (stream.waiting) {
      // 之前因等字幕暂停了，现在新片就绪 → 继续播放
      stream.waiting = false;
      $("#video").play().catch(() => {});
    }
    return performance.now() - t0;
  } catch (err) {
    console.error("[subtrans] process_chunk failed:", start, dur, err);
    // 跳过出错分片，继续处理后续分片（而非停止整个流水线）
    stream.readyUntil = Math.max(stream.readyUntil, start + dur); // 跳过这一段
    if (!stream.failedChunks) stream.failedChunks = [];
    stream.failedChunks.push({ start, dur, err: String(err) });
    $("#runMsg").textContent = `分片 ${fmt(start)} 出错（已跳过）: ${err}`;
    if (stream.waiting) {
      stream.waiting = false;
      $("#video").play().catch(() => {});
    }
    return null;
  } finally {
    stream.processing = false;
  }
}

// 连续流水线：一片处理完立即下一片，直到结尾（与播放并行）
async function pump() {
  if (stream.pumping) return;
  stream.pumping = true;
  while (stream.running && !stream.done) {
    const ms = await processNextChunk();
    if (ms === null) {
      if (stream.done) break;
      await sleep(300); // 出错后稍等再试
    }
  }
  stream.pumping = false;
  if (stream.done) {
    const failed = stream.failedChunks || [];
    const failMsg = failed.length ? `（${failed.length} 个分片出错已跳过：${failed.map(f => fmt(f.start)).join(", ")}）` : "";
    $("#runMsg").textContent = `字幕已全部生成（至 ${fmt(stream.total)}）${failMsg}`;
    $("#startBtn").disabled = false;
    $("#exportBtn").disabled = state.subtitles.length === 0;
    stream.failedChunks = [];
    // 步骤 2 完成 → 自动跳到步骤 3 查看结果
    setStepDone(2);
    goStep(3);
  }
}

function updateStatus(res) {
  const pct = stream.total ? (stream.readyUntil / stream.total) * 100 : 0;
  setFill("#runFill", pct);
  const langInfo = res?.detected_lang
    ? ` · 检测语言 ${res.detected_lang}${
        res.detected_lang_probability != null
          ? ` (${(res.detected_lang_probability * 100).toFixed(0)}%)`
          : ""
      }`
    : "";
  const last = res
    ? ` · 识别 ${(res.transcribe_ms / 1000).toFixed(0)}s${res.correct_ms ? `/纠错 ${(res.correct_ms / 1000).toFixed(0)}s` : ""}/翻译 ${(res.translate_ms / 1000).toFixed(0)}s${langInfo}`
    : "";
  let msg = `字幕已生成至 ${fmt(stream.readyUntil)} / ${fmt(stream.total)}${stream.smallSlices ? " · 1分片" : ""}${last}`;
  if (res && res.warn) msg += `　⚠ ${res.warn}`;
  $("#runMsg").textContent = msg;
}

async function startProcess() {
  if (!state.videoPath) return;
  const video = $("#video");
  if (!video.duration || isNaN(video.duration)) {
    // 等待元数据加载，带超时避免损坏视频导致 UI 卡死
    const ok = await Promise.race([
      new Promise((res) => video.addEventListener("loadedmetadata", () => res(true), { once: true })),
      new Promise((res) => setTimeout(() => res(false), 15000)),
    ]);
    if (!ok || !video.duration || isNaN(video.duration)) {
      $("#runMsg").textContent = "无法读取视频时长（文件可能损坏或格式不支持）";
      return;
    }
  }
  state.subtitles = [];
  renderSubList();
  $("#startBtn").disabled = true;
  $("#exportBtn").disabled = true;
  setFill("#runFill", 0);

  Object.assign(stream, {
    chunkLen: 120,
    readyUntil: 0,
    total: video.duration,
    processing: false,
    pumping: false,
    running: true,
    done: false,
    waiting: false,
    smallSlices: false,
    audioSource: null,
    failedChunks: [],
  });

  // 高精度模式：先用 demucs 分离人声，ASR 改用纯人声轨（播放仍用原视频）
  if ($("#hiQuality").checked) {
    $("#runMsg").textContent = "高精度模式：分离人声中（首次会加载模型，请稍候）...";
    try {
      stream.audioSource = await invoke("separate_vocals", {
        videoPath: state.videoPath,
        pythonExe: $("#pyPython").value.trim(), // 探测/用户填写的 Python；留空才回退 bundled
        model: $("#demucsModel").value,
        device: $("#demucsDevice").value,
      });
    } catch (err) {
      console.error("[subtrans] separate_vocals failed:", err);
      // 分离失败不阻塞主流程：本次先按普通模式识别，
      // 但不取消用户的高精度勾选（保留选择，方便查看错误后调整设置重试）
      stream.audioSource = null;
      $("#runMsg").textContent = `人声分离失败（本次已按普通模式识别，请检查设置后重试）: ${err}`;
      $("#demucsStatus").textContent = "✗ 分离失败: " + err;
      $("#demucsStatus").style.color = "var(--danger)";
    }
  }

  // 先处理头一片，再开播 + 启动后台连续流水线
  $("#runMsg").textContent = `处理第一段（约${Math.round(stream.chunkLen / 60)}分钟）...`;
  const ms = await processNextChunk();
  if (ms !== null && ms > stream.chunkLen * 1000) {
    // 处理耗时超过分片时长（慢于实时）→ 后续改更小分片
    stream.smallSlices = true;
    stream.chunkLen = 45;
  }
  $("#exportBtn").disabled = state.subtitles.length === 0;
  try {
    await video.play();
  } catch {}
  pump(); // 不 await，后台持续处理
}

// ───────── 字幕渲染 ─────────
function addSubtitle(sub) {
  // 按 start 时间有序插入（补洞/seek 可能乱序到达），保证 overlay 二分查找正确
  const arr = state.subtitles;
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].start < sub.start) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, sub);
  // 切到字幕 tab 不强制，但追加渲染
  const list = $("#subList");
  $(".empty")?.remove();
  const row = document.createElement("div");
  row.className = "sub-row";
  row.dataset.start = sub.start;
  row.innerHTML = `
    <div class="sub-time">${fmt(sub.start)}</div>
    <div class="sub-text">
      <div class="sub-orig">${escapeHtml(sub.original)}</div>
      ${sub.translated ? `<div class="sub-trans">${escapeHtml(sub.translated)}</div>` : ""}
    </div>`;
  row.addEventListener("click", () => {
    $("#video").currentTime = sub.start;
  });
  // 按 start 位置插入 DOM 行，保持列表与 state.subtitles 同序
  // （补洞/seek 乱序到达时不会出现“时间排序正确但列表乱序”）
  const next = list.children[lo];
  if (next) list.insertBefore(row, next);
  else list.appendChild(row);
  // 只有本来就接近底部时才自动滚动，避免打断用户向上翻看
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  if (nearBottom) list.scrollTop = list.scrollHeight;
  $("#subCount").textContent = `· ${state.subtitles.length} 条`;
}

function renderSubList() {
  const list = $("#subList");
  list.innerHTML = state.subtitles.length
    ? ""
    : '<div class="empty">识别后字幕会显示在这里</div>';
  $("#subCount").textContent = state.subtitles.length ? `· ${state.subtitles.length} 条` : "";
}

function updateOverlay(t) {
  // 字幕按 start 排序：先二分找 start <= t 的最右一条，再向前小范围扫描，
  // 兼容补洞/seek 可能产生的重叠字幕，避免显示到“碰巧命中”的旧字幕。
  const subs = state.subtitles;
  let cur = null;
  let lo = 0, hi = subs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (subs[mid].start <= t) lo = mid + 1;
    else hi = mid;
  }
  for (let i = lo - 1; i >= 0 && i >= lo - 50; i--) {
    const s = subs[i];
    if (s.end >= t) {
      cur = s;
      break;
    }
  }
  const overlay = $("#overlay");
  const showOrig = $("#showOrig").checked;
  const showTrans = $("#showTrans").checked;
  if (!cur) return overlay.classList.add("hidden");
  let html = "";
  if (showOrig && cur.original) html += `<div class="ol-orig">${escapeHtml(cur.original)}</div>`;
  if (showTrans && cur.translated) html += `<div class="ol-trans">${escapeHtml(cur.translated)}</div>`;
  if (html) {
    overlay.innerHTML = html;
    overlay.classList.remove("hidden");
  } else overlay.classList.add("hidden");
}

// ───────── 导出 SRT ─────────
async function exportSrt() {
  if (!state.videoPath) return;
  const total = $("#video").duration || stream.total;
  $("#exportBtn").disabled = true;

  // 1) 后台流水线还在跑则等它真正结束（含分片间短暂休眠的间隙），避免补洞与流水线并发
  if (stream.running && !stream.done) {
    $("#runMsg").textContent = "等待后台字幕生成完成...";
    await waitUntil(() => stream.done);
  }
  if (stream.pumping || stream.processing) {
    $("#runMsg").textContent = "等待后台字幕生成完成...";
    await waitUntil(() => !stream.pumping && !stream.processing);
  }
  // 2) 补全所有未覆盖的时间段（含因跳转产生的空洞）后再导出
  if (total) {
    const step = stream.chunkLen || 120; // 用实际分片大小，而非硬编码
    for (let s = 0; s < total - 0.05; s += step) {
      const e = Math.min(s + step, total);
      const covered = windowCovered(s, e);
      if (!covered) {
        $("#runMsg").textContent = `补全字幕 ${fmt(s)} / ${fmt(total)}`;
        try {
          await invokeChunk(s, e - s);
        } catch (err) {
          console.error("[subtrans] 导出补全分片失败:", s, err);
        }
      }
    }
  }

  if (!state.subtitles.length) {
    $("#exportBtn").disabled = false;
    return;
  }
  // 分片可能乱序到达（补洞），按时间排序；并把“时间重叠且原文相同”的相邻字幕合并
  // （lead-in 跨边界段可能被前后两个分片各生成一次，start/end 有微小差异）
  const merged = [];
  for (const s of [...state.subtitles].sort((a, b) => a.start - b.start)) {
    const prev = merged[merged.length - 1];
    if (prev && s.original === prev.original && s.start < prev.end - 0.05) {
      prev.end = Math.max(prev.end, s.end);
      if (!prev.translated && s.translated) prev.translated = s.translated;
      continue;
    }
    merged.push({ ...s });
  }
  const subs = merged;
  const path = await saveDialog({ defaultPath: "subtitle.srt", filters: [{ name: "SRT", extensions: ["srt"] }] });
  if (!path) {
    $("#exportBtn").disabled = false;
    return;
  }
  const showOrig = $("#showOrig").checked;
  const lines = [];
  subs.forEach((s, i) => {
    lines.push(String(i + 1));
    lines.push(`${ts(s.start)} --> ${ts(s.end)}`);
    if (showOrig && s.original && s.translated) {
      // 原文 + 译文双行（仅有译文时才双行，避免无翻译时原文重复）
      lines.push(s.original);
      lines.push(s.translated);
    } else {
      lines.push(s.translated || s.original);
    }
    lines.push("");
  });
  await invoke("save_text_file", { path, content: lines.join("\n") });
  $("#runMsg").textContent = `已导出: ${path}`;
  $("#exportBtn").disabled = false;
}

// 判断 [s, e) 窗口是否被现有字幕完整覆盖（合并重叠段后逐段检查，不能“窗口里有一条就算覆盖”）
function windowCovered(s, e) {
  const ints = state.subtitles
    .filter((sub) => sub.end > s + 0.1 && sub.start < e - 0.1)
    .map((sub) => [Math.max(sub.start, s), Math.min(sub.end, e)])
    .sort((a, b) => a[0] - b[0]);
  let cur = s;
  for (const [a, b] of ints) {
    if (a > cur + 0.1) return false;
    cur = Math.max(cur, b);
    if (cur >= e - 0.1) return true;
  }
  return cur >= e - 0.1;
}

async function testOllama() {
  $("#olStatus").textContent = "检测中...";
  const st = await invoke("ollama_status", { host: $("#olHost").value.trim() }).catch(() => null);
  if (st?.running) {
    $("#olStatus").textContent = `已连接。模型: ${st.models.join(", ") || "（无，请先 ollama pull）"}`;
  } else {
    const detail = st?.error ? `（${st.error}）` : "";
    $("#olStatus").textContent = `无法连接${detail}，请确认 Ollama 已启动`;
  }
}

// ───────── 环境探测（三级 GPU 状态 + 预估时间）─────────
async function detectEnv() {
  let env;
  try {
    env = await invoke("env_status");
  } catch {
    env = { python_path: "", python_bundled: false, has_gpu: false, cuda_torch_ready: false,
            fw_ready: false, demucs_ready: false, audio_sep_ready: false, models: [], bundled_tiny: false };
  }

  const gpuReady = env.has_gpu && env.cuda_torch_ready && (env.fw_ready || env.demucs_ready);
  const cudaInstalled = env.has_gpu && env.cuda_torch_ready; // CUDA torch 已装（不管 fw/demucs 检测结果）
  const fwReady = gpuReady && env.fw_ready;
  const demucsReady = gpuReady && env.demucs_ready;

  // 记录/回填 Python 路径（探测结果优先）
  if (env.python_path) {
    $("#pyPython").value = env.python_path;
    localStorage.setItem("subtrans.pythonPath", env.python_path);
  }

  // GPU 选项显示/隐藏
  showGroup("gpuRecGroup", fwReady);
  showGroup("hiQualityGroup", demucsReady);
  showGroup("fwCfgGroup", gpuReady);
  showGroup("demucsCfgGroup", gpuReady);
  // Python 安装区：GPU 未就绪时展示，让用户有明确的修复入口
  showGroup("pySetupGroup", !gpuReady);

  // GPU 徽章四级状态（就绪 / 升级中 / 升级失败 / 可升级 / CPU模式）
  const badge = $("#gpuBadge");
  const badgeText = $("#gpuBadgeText");
  const gpuProg = $("#gpuProgress");
  const gpuUpgBtn = $("#gpuUpgradeBtn");
  const gpuErr = $("#gpuError");

  if (gpuReady) {
    badge.className = "gpu-badge ready";
    badgeText.textContent = "GPU 就绪";
    gpuProg.classList.add("hidden");
    gpuUpgBtn.classList.add("hidden");
    gpuErr.classList.add("hidden");
    $("#pyStatus").textContent = "✓ GPU 加速就绪";
    $("#fwDevice").value = "cuda";
    $("#demucsDevice").value = "cuda";
    if (fwReady && !stream.running) $("#useFw").checked = true;
    $("#useFwHint").textContent = "✓ GPU 加速可用";
    $("#useFwHint").style.color = "var(--accent)";
  } else if (cudaInstalled) {
    // CUDA torch 已装但 fw/demucs 缺失 → 不能谎称就绪，给出安装入口
    badge.className = "gpu-badge ready";
    badgeText.textContent = "GPU 组件不完整";
    gpuProg.classList.add("hidden");
    gpuUpgBtn.classList.add("hidden");
    gpuErr.classList.add("hidden");
    $("#pyStatus").textContent = "检测到 CUDA 版 torch，但 faster-whisper/demucs 未安装，请用「安装 GPU 加速组件」补齐。";
    $("#useFw").checked = false;
    $("#hiQuality").checked = false;
  } else if (gpuUpgrading) {
    badge.className = "gpu-badge upgrading";
    badgeText.textContent = "GPU 升级中...";
    gpuProg.classList.remove("hidden");
    gpuUpgBtn.classList.add("hidden");
    gpuErr.classList.add("hidden");
    $("#useFw").checked = false;
    $("#hiQuality").checked = false;
  } else if (gpuUpgradeError) {
    badge.className = "gpu-badge";
    badgeText.textContent = "GPU 升级失败";
    badge.style.color = "var(--danger)";
    gpuProg.classList.add("hidden");
    gpuUpgBtn.classList.add("hidden");
    gpuErr.classList.remove("hidden");
    $("#gpuErrorText").textContent = gpuUpgradeError;
    $("#useFw").checked = false;
    $("#hiQuality").checked = false;
  } else if (env.has_gpu && !env.cuda_torch_ready) {
    // 有 GPU 但 CUDA torch 未安装 → 显示升级按钮
    badge.className = "gpu-badge";
    badgeText.textContent = "检测到 GPU，可启用加速";
    gpuProg.classList.add("hidden");
    gpuUpgBtn.classList.remove("hidden");
    gpuErr.classList.add("hidden");
    $("#pyStatus").textContent = env.python_path
      ? "检测到 GPU，可点击「安装 GPU 加速组件」或左上角「安装 GPU 加速」。"
      : "未检测到 Python，请先点「一键安装 Python 环境」。";
    $("#useFw").checked = false;
    $("#hiQuality").checked = false;
  } else {
    badge.className = "gpu-badge";
    badgeText.textContent = env.python_bundled ? "CPU 模式（内置）" : "CPU 模式";
    gpuProg.classList.add("hidden");
    gpuUpgBtn.classList.add("hidden");
    gpuErr.classList.add("hidden");
    $("#pyStatus").textContent = env.python_path
      ? "CPU 模式：未检测到 GPU，或 GPU 组件未就绪。"
      : "CPU 模式：需要 Python 环境才能启用 GPU 识别 / 人声分离。";
    $("#useFw").checked = false;
    $("#hiQuality").checked = false;
  }

  // 更新预估时间
  updateEstimate();
  updateEngineHint();
}

// 一键安装 Python 环境（后端会自动装好 GPU 识别/人声分离组件）
async function installPython() {
  const btn = $("#pyInstallBtn");
  btn.disabled = true;
  $("#pyStatus").textContent = "正在安装 Python 环境（首次需下载，约 10-30 分钟）...";
  try {
    const path = await invoke("python_setup");
    $("#pyPython").value = path;
    localStorage.setItem("subtrans.pythonPath", path);
    $("#pyStatus").textContent = `✓ Python 环境就绪: ${path}`;
    await detectEnv();
  } catch (err) {
    console.error("[subtrans] python_setup failed:", err);
    $("#pyStatus").textContent = `✗ 安装失败: ${err}`;
  } finally {
    btn.disabled = false;
  }
}

// 往已检测到的 Python 里安装 GPU 识别/人声分离组件
async function installGpuPackages() {
  const btn = $("#gpuPackagesBtn");
  const py = $("#pyPython").value.trim();
  if (!py) {
    $("#pyStatus").textContent = "请先填写 Python 路径，或点击「一键安装 Python 环境」";
    return;
  }
  btn.disabled = true;
  $("#pyStatus").textContent = "正在安装 faster-whisper / demucs / audio-separator（大文件，请耐心等待）...";
  try {
    await invoke("install_gpu_packages", { pythonExe: py });
    $("#pyStatus").textContent = "✓ GPU 加速组件安装完成";
    await detectEnv();
  } catch (err) {
    console.error("[subtrans] install_gpu_packages failed:", err);
    $("#pyStatus").textContent = `✗ 安装失败: ${err}`;
  } finally {
    btn.disabled = false;
  }
}

// 触发 GPU 升级（下载 CUDA torch 替换 CPU 版）
async function startGpuUpgrade() {
  gpuUpgrading = true;
  gpuUpgradeError = "";
  // 立即更新 UI
  $("#gpuBadge").className = "gpu-badge upgrading";
  $("#gpuBadgeText").textContent = "GPU 升级中...";
  $("#gpuBadge").style.color = "";
  $("#gpuProgress").classList.remove("hidden");
  $("#gpuProgressFill").style.width = "0%";
  $("#gpuProgressText").textContent = "准备中...";
  $("#gpuUpgradeBtn").classList.add("hidden");
  $("#gpuError").classList.add("hidden");

  try {
    let pythonPath = $("#pyPython")?.value?.trim() || "";
    if (!pythonPath) {
      // 没有 Python：先一键安装（后端安装流程本身会装 CUDA 版 torch + 组件）
      $("#gpuProgressText").textContent = "未检测到 Python，先安装 Python 环境...";
      pythonPath = await invoke("python_setup");
      $("#pyPython").value = pythonPath;
      localStorage.setItem("subtrans.pythonPath", pythonPath);
      gpuUpgrading = false;
      $("#gpuBadge").className = "gpu-badge";
      $("#gpuProgress").classList.add("hidden");
      await detectEnv();
      return;
    }
    await invoke("upgrade_cuda", { pythonExe: pythonPath });
    // 成功由 cuda-ready 事件处理
  } catch (err) {
    console.error("[subtrans] upgrade_cuda failed:", err);
    gpuUpgrading = false;
    gpuUpgradeError = String(err);
    $("#gpuBadge").className = "gpu-badge";
    $("#gpuBadgeText").textContent = "GPU 升级失败";
    $("#gpuBadge").style.color = "var(--danger)";
    $("#gpuProgress").classList.add("hidden");
    $("#gpuError").classList.remove("hidden");
    $("#gpuErrorText").textContent = gpuUpgradeError;
  }
}

// 预估处理时间显示
async function updateEstimate() {
  const el = $("#estimateTime");
  if (!el) return;
  const video = $("#video");
  const dur = video?.duration;
  if (!dur || isNaN(dur) || dur <= 0) {
    el.textContent = "";
    return;
  }
  const model = $("#whisperModel")?.value || "large-v3";
  const useFw = $("#useFw")?.checked || false;
  try {
    const est = await invoke("estimate_time", { durationSec: dur, modelName: model, useFw });
    el.textContent = est;
  } catch {
    el.textContent = "";
  }
}

function showGroup(id, show) {
  document.getElementById(id)?.classList.toggle("hidden", !show);
}

// ───────── 识别引擎提示 ─────────
function updateEngineHint() {
  const hint = $("#engineHint");
  const gpuVisible = !document.getElementById("gpuRecGroup")?.classList.contains("hidden");
  if (gpuVisible && $("#useFw")?.checked) {
    hint.textContent = "识别引擎：faster-whisper（GPU，快）";
  } else if (gpuVisible) {
    hint.textContent = "识别引擎：Whisper（CPU；可勾「GPU 加速识别」提速）";
  } else {
    hint.textContent = "识别引擎：Whisper（CPU）";
  }
  hint.style.color = "var(--muted)";
}

function updateCorrectHint() {
  const hint = $("#correctHint");
  if (!$("#correctEnabled").checked) {
    hint.textContent = "";
    return;
  }
  const ce = correctionEngine();
  if (!ce) {
    hint.textContent = "⚠ 未配置 DeepSeek Key 或 Ollama，纠错不会生效（到「引擎」页填写）。";
    hint.style.color = "var(--accent-2)";
  } else {
    hint.textContent =
      ce.kind === "DeepSeek"
        ? "将用 DeepSeek 纠错"
        : `将用本地 Ollama（${ce.model}）纠错，需先 ollama pull 该模型`;
    hint.style.color = "var(--muted)";
  }
}

// ───────── 工具函数 ─────────
function fmt(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function ts(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60),
    s = Math.floor(sec % 60), ms = Math.floor((sec % 1) * 1000);
  return `${p(h)}:${p(m)}:${p(s)},${String(ms).padStart(3, "0")}`;
}
const p = (n) => String(n).padStart(2, "0");
function escapeHtml(str) {
  return str.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

boot();
