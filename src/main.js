import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  lockDetectedProfile,
  buildRollingContext,
  migrateRecognitionSettings,
} from "./recognition-profile-state.js";

// ───────── 简易本地配置（仅记一个"是否完成向导"标记） ─────────
const SETUP_KEY = "subtrans.setupDone";
const isSetupDone = () => localStorage.getItem(SETUP_KEY) === "1";
const markSetupDone = () => localStorage.setItem(SETUP_KEY, "1");

// ───────── 全局状态 ─────────
const state = {
  videoPath: "",
  modelName: "large-v3",
  subtitles: [],
  // 会话令牌：重新打开同一视频 / 重启识别 / 打开指向同一视频的项目时 +1，
  // 在飞的异步分片据此判过期（仅比较视频路径无法识别这些情况）
  session: 0,
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
  holes: [],         // 被跳过的 [start,end) 区间（seek 重定位 / 分片出错），收尾统一补全
  filling: false,    // 收尾补洞进行中（防多个收尾块重复补）
  consecFails: 0,    // 连续分片失败计数（≥3 停止流水线，防止配置错误时空转）
  // ── 识别预设会话状态 ──
  selectedProfileId: "auto", // 用户选择的预设（新会话开始时的快照）
  lockedProfileId: null,     // auto 模式首个成功分片后锁定的内置预设（手动模式恒 null）
  accentVariant: "auto",     // 口音变体
};

// ───────── 识别预设目录（后端唯一真相源，前端只读展示元数据） ─────────
let profileCatalogById = new Map(); // id → LanguageProfileDto

// 按所选预设刷新口音/自定义语言组的可见性与选项（保留合法选择，非法回退 auto）
function updateRecognitionGroups() {
  const sel = $("#recognitionProfile");
  const profileId = sel?.value || "auto";
  const profile = profileCatalogById.get(profileId);
  const accentGroup = $("#accentVariantGroup");
  const accentSel = $("#accentVariant");
  const variants = profile && Array.isArray(profile.accent_variants) ? profile.accent_variants : [];
  if (variants.length > 1) {
    const current = accentSel.value;
    accentSel.innerHTML = "";
    for (const a of variants) {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      accentSel.appendChild(opt);
    }
    accentSel.value = [...accentSel.options].some((o) => o.value === current) ? current : "auto";
    accentGroup.classList.remove("hidden");
  } else {
    accentSel.value = "auto";
    accentGroup.classList.add("hidden");
  }
  // 自定义语言组：仅 custom 预设显示（保留旧 sourceLang 选择行为）
  $("#customLanguageGroup")?.classList.toggle("hidden", profileId !== "custom");
}

// 从后端加载预设目录；失败保留 HTML 里的静态 auto/custom 兜底项
async function loadRecognitionProfiles() {
  try {
    const profiles = await invoke("list_language_profiles");
    if (!Array.isArray(profiles) || !profiles.length) return;
    profileCatalogById = new Map(profiles.map((p) => [p.id, p]));
    const sel = $("#recognitionProfile");
    const current = sel.value;
    sel.innerHTML = "";
    for (const p of profiles) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      sel.appendChild(opt);
    }
    // 保留当前选择（若仍有效）；否则回退 auto
    sel.value = [...sel.options].some((o) => o.value === current) ? current : "auto";
    updateRecognitionGroups();
  } catch (err) {
    console.error("[subtrans] list_language_profiles failed:", err);
    // 目录不可用时保留静态 auto/custom 兜底项，用户仍可按旧方式选源语言
    updateRecognitionGroups();
  }
}

// GPU 升级状态机
let gpuUpgrading = false;     // 是否正在升级中
let gpuUpgradeError = "";     // 升级失败的错误信息（空 = 无错误）

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// 轮询等待条件；timeoutMs>0 时超时返回 false，避免后端异常导致永久挂起
async function waitUntil(cond, step = 200, timeoutMs = 0) {
  const start = performance.now();
  while (!cond()) {
    if (timeoutMs > 0 && performance.now() - start > timeoutMs) return false;
    await sleep(step);
  }
  return true;
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
    if (exists) {
      showApp();
      restoreAutosave();
      return;
    }
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

  // 离线/下载失败时的逃生通道：先用内置 tiny 模型进入主界面（按钮仅在检测到内置模型时显示）
  $("#skipModelBtn").addEventListener("click", () => {
    state.modelName = "tiny";
    localStorage.setItem("subtrans.model", "tiny");
    goto(1);
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
      if (env.bundled_tiny) $("#skipModelBtn").classList.remove("hidden");
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
  $("#saveProjectBtn").addEventListener("click", saveProjectDialog);
  $("#openProjectBtn").addEventListener("click", openProjectDialog);
  $("#burnBtn").addEventListener("click", burnSubtitles);
  $("#importSubBtn").addEventListener("click", () => importSubtitleFile(false));
  $("#importSubReplaceBtn").addEventListener("click", () => importSubtitleFile(true));
  $("#applyShiftBtn").addEventListener("click", applyShift);
  $("#applyFpsBtn").addEventListener("click", applyFps);
  $("#applyReplaceBtn").addEventListener("click", applyReplace);
  $("#batchTranslateBtn").addEventListener("click", batchTranslate);
  $("#qcBtn").addEventListener("click", checkSubtitles);
  $("#fixOverlapBtn").addEventListener("click", fixOverlaps);
  $("#spkA").addEventListener("click", () => assignSpeaker("甲"));
  $("#spkB").addEventListener("click", () => assignSpeaker("乙"));
  $("#spkC").addEventListener("click", () => assignSpeaker("丙"));
  $("#spkD").addEventListener("click", () => assignSpeaker("丁"));
  $("#spkClear").addEventListener("click", () => assignSpeaker(""));
  // 烧录预设：切换时把预设参数回填到输入框（仅提示，实际由后端按预设处理）
  $("#burnPreset").addEventListener("change", () => {
    const p = $("#burnPreset").value;
    if (p === "douyin") {
      $("#burnFontSize").value = "28";
      $("#burnMarginV").value = "160";
      $("#burnPosition").value = "bottom";
    } else if (p === "bilibili") {
      $("#burnFontSize").value = "20";
      $("#burnMarginV").value = "32";
      $("#burnPosition").value = "bottom";
    } else if (p === "youtube") {
      $("#burnFontSize").value = "18";
      $("#burnMarginV").value = "36";
      $("#burnPosition").value = "bottom";
    }
  });
  $("#testOllama").addEventListener("click", testOllama);
  $("#pyInstallBtn").addEventListener("click", installPython);
  $("#gpuPackagesBtn").addEventListener("click", installGpuPackages);

  // 源语言切换 → 引擎提示
  $("#sourceLang").addEventListener("change", updateEngineHint);
  updateEngineHint();

  // 识别预设切换 → 刷新口音/自定义语言组的可见性与选项
  $("#recognitionProfile").addEventListener("change", updateRecognitionGroups);
  // 加载后端预设目录（不阻塞启动：失败保留静态 auto/custom 兜底项）
  loadRecognitionProfiles();

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
  // 向后跳转到尚未生成处：把流水线重定位到该分片继续。
  // 被跳过的区间记入 stream.holes，流水线收尾时统一回头补全（不再静默丢内容）。
  video.addEventListener("seeking", () => {
    if (!stream.running) return;
    const t = video.currentTime;
    if (t > stream.readyUntil) {
      // chunkLen 可能中途由 120 降为 45，readyUntil 不一定是 chunkLen 的倍数，
      // 用 max 保证游标只前移、不产生负长度区间
      const newReady = Math.max(stream.readyUntil, Math.floor(t / stream.chunkLen) * stream.chunkLen);
      if (newReady > stream.readyUntil) {
        stream.holes.push({ start: stream.readyUntil, end: newReady });
        const gapStart = stream.readyUntil;
        stream.readyUntil = newReady;
        stream.done = false;
        $("#runMsg").textContent = `已跳转至 ${fmt(newReady)}，${fmt(gapStart)}–${fmt(newReady)} 将在收尾时补全`;
      }
      pump();
    }
  });

  // ── 校对快捷键 + 循环播放（字幕工作站 ①） ──
  const isTyping = () => {
    const el = document.activeElement;
    return (
      el &&
      (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)
    );
  };
  // 连续按 L 判定：按键次数与最近一次按键时刻分开存
  //（曾共用一个 lastL，第一次按键后 lastL 变成计数 1，下一次 now-1 恒 >500ms，永远停在 1x）
  let lCount = 0;
  let lastLTime = 0;
  document.addEventListener("keydown", (e) => {
    // Ctrl+Z/Y 撤销重做（输入框内交给原生撤销，不抢占）
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
      const k = e.key.toLowerCase();
      if (k === "z" && !isTyping()) {
        e.preventDefault();
        undo();
      } else if (k === "y" && !isTyping()) {
        e.preventDefault();
        redo();
      }
      return;
    }
    if (isTyping() || e.ctrlKey || e.metaKey || e.altKey) return;
    const v = $("#video");
    if (!v || !v.duration) return;
    switch (e.key) {
      case " ": {
        // 焦点在按钮/视频元素上时，交给原生行为（按钮=点击，视频=原生播放切换），
        // 避免全局快捷键与原生行为双触发互相抵消
        const ae = document.activeElement;
        if (ae && (ae.tagName === "BUTTON" || ae === v)) return;
        e.preventDefault();
        if (v.paused) v.play().catch(() => {});
        else v.pause();
        break;
      }
      case "1":
        assignSpeaker("甲");
        break;
      case "2":
        assignSpeaker("乙");
        break;
      case "3":
        assignSpeaker("丙");
        break;
      case "4":
        assignSpeaker("丁");
        break;
      case "0":
        assignSpeaker("");
        break;
      case "ArrowLeft":
        v.currentTime = Math.max(0, v.currentTime - (e.shiftKey ? 0.1 : 1));
        break;
      case "ArrowRight":
        v.currentTime = Math.min(v.duration, v.currentTime + (e.shiftKey ? 0.1 : 1));
        break;
      case "ArrowUp":
      case "ArrowDown": {
        e.preventDefault();
        const i = currentSubIndex();
        const next = e.key === "ArrowDown" ? i + 1 : Math.max(0, i - 1);
        if (state.subtitles[next]) {
          v.currentTime = state.subtitles[next].start;
          v.pause();
        }
        break;
      }
      case "j":
      case "J":
        v.currentTime = Math.max(0, v.currentTime - 10);
        v.playbackRate = 1;
        lCount = 0;
        lastLTime = 0;
        break;
      case "k":
      case "K":
        v.pause();
        break;
      case "l":
      case "L": {
        const now = performance.now();
        lCount = now - lastLTime < 500 ? lCount + 1 : 1;
        lastLTime = now;
        if (lCount >= 3) v.playbackRate = 2;
        else if (lCount === 2) v.playbackRate = 1.5;
        else v.currentTime = Math.min(v.duration, v.currentTime + 10);
        break;
      }
      case "e":
      case "E": {
        const i = currentSubIndex();
        if (i >= 0) {
          $("#subsDrawer").classList.remove("hidden");
          startEdit(i, "original");
        }
        break;
      }
      case "Delete": {
        const i = currentSubIndex();
        if (i >= 0) deleteSubtitle(i);
        break;
      }
      case "r":
      case "R":
        toggleLoopCurrent();
        break;
      default:
        break;
    }
  });

  // 循环播放：到循环终点跳回起点
  video.addEventListener("timeupdate", () => {
    if (video.dataset.loopEnd && video.currentTime >= Number(video.dataset.loopEnd)) {
      video.currentTime = Number(video.dataset.loopStart) || 0;
    }
  });

  // 波形图：播放跟随 + 点击拖动定位
  video.addEventListener("timeupdate", () => scheduleWaveformDraw());
  const wfWrap = $("#waveformWrap");
  wfWrap.addEventListener("mousedown", (e) => {
    wfDrag = true;
    seekWaveform(e);
  });
  // mousemove 挂 document：拖拽移出波形条外仍能连续定位
  document.addEventListener("mousemove", (e) => {
    if (wfDrag) seekWaveform(e);
  });
  document.addEventListener("mouseup", () => {
    wfDrag = false;
  });
  window.addEventListener("resize", () => scheduleWaveformDraw());
}

async function openVideo() {
  const path = await openDialog({
    filters: [{ name: "视频", extensions: ["mp4", "mkv", "avi", "mov", "webm", "flv", "m4v"] }],
  });
  if (!path) return;
  // 先提交未完成的行内编辑（对象已脱离数组，直接丢弃）；再落盘当前会话草稿
  commitInlineEdit();
  await saveAutosave();
  // 切换视频前停掉旧流水线，避免旧任务继续往新列表里塞字幕
  stream.running = false;
  stream.done = true;
  stream.processing = false;
  stream.pumping = false;
  stream.waiting = false;
  stream.total = 0;
  stream.readyUntil = 0;
  stream.failedChunks = [];
  stream.audioSource = null; // 旧视频的人声轨不能留给新视频
  stream.holes = [];
  stream.filling = false;
  stream.consecFails = 0;
  stream.lockedProfileId = null; // 新会话：只清自动锁定，不动用户选择的预设
  resetUndo(); // 旧会话字幕快照不得跨视频生效
  state.session++; // 会话令牌：即使重新打开同一个视频，旧分片也不得写入新列表
  state.videoPath = path;
  state.subtitles = [];
  renderSubList();
  $("#fileName").textContent = path.split(/[\\/]/).pop();
  $("#video").src = convertFileSrc(path);
  $("#stageEmpty").classList.add("hidden");
  $("#startBtn").disabled = false;
  $("#exportBtn").disabled = true;
  // 视频加载后更新预估时间 + 提取波形 + 读取视频信息
  wf = null;
  $("#waveformWrap").classList.add("hidden");
  $("#video").addEventListener(
    "loadedmetadata",
    () => {
      updateEstimate();
      loadWaveform();
      loadVideoInfo();
    },
    { once: true }
  );
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

// 调用后端处理一个时间段，把返回的字幕追加进列表，返回后端结果。
// 会话切换后返回 null（调用方据此丢弃结果，不触碰任何流水线状态）。
async function invokeChunk(start, dur) {
  const ce = $("#correctEnabled").checked ? correctionEngine() : null;
  const sessionAtCall = state.session; // 在飞请求返回时可能已切换会话，丢弃旧结果
  // 识别预设：auto 会话锁定后使用锁定的预设；手动预设直接使用用户选择。
  // sourceLang 继续发送：custom 预设与版本 1 兼容行为仍依赖它。
  const selected = $("#recognitionProfile")?.value || "auto";
  const effectiveProfileId = stream.lockedProfileId || selected;
  const accentVariant = $("#accentVariant")?.value || "auto";
  const contextPrompt = buildRollingContext(state.subtitles, 3, 600);
  const res = await invoke("process_chunk", {
    videoPath: state.videoPath,
    audioSource: stream.audioSource,
    modelName: $("#whisperModel").value,
    startSec: start,
    durationSec: dur,
    leadInSec: start > 0 ? 5.0 : 0.0,
    totalSec: stream.total,
    sourceLang: $("#sourceLang").value || null,
    recognitionProfileId: effectiveProfileId,
    accentVariant,
    contextPrompt,
    targetLang: $("#targetLang").value,
    engine: currentEngine(),
    translateEnabled: $("#transEnabled").checked,
    correctEnabled: !!ce,
    correctEngine: ce,
    glossary: $("#glossary").value || "",
    useFw: $("#useFw").checked,
    vadEnabled: $("#vadFilter").checked,
    denoiseFilter: $("#denoiseFilter")?.value || "none",
    // 优先用探测/用户填写的 Python（否则检测是绿的、实际运行却找不到解释器）；
    // 留空时后端才回退 bundled Python。
    fwPython: $("#pyPython").value.trim(),
    fwDevice: $("#fwDevice").value,
  });
  if (state.session !== sessionAtCall) return null; // 已切换会话：丢弃过期结果
  // 仅当前会话的成功分片可以锁定检测预设：过期分片（会话不匹配）不改锁，
  // 手动预设永不锁定（helper 内部处理）
  const nextLock = applyDetectedProfileForSession(
    state.session,
    sessionAtCall,
    selected,
    stream.lockedProfileId,
    res.detected_profile_id
  );
  if (nextLock !== stream.lockedProfileId) {
    stream.lockedProfileId = nextLock;
    if (nextLock) {
      const label = profileCatalogById.get(nextLock)?.label || nextLock;
      $("#runMsg").textContent = `已检测并锁定：${label}`;
    }
  }
  appendSubtitles(res.segments);
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
  const sessionAtStart = state.session; // 会话切换后不回写任何流水线状态
  try {
    const res = await invokeChunk(start, dur);
    if (!res || state.session !== sessionAtStart) return null;
    // 处理期间用户可能向前 seek 改大了 readyUntil，不能被旧分片覆盖
    stream.readyUntil = Math.max(stream.readyUntil, start + dur);
    stream.consecFails = 0; // 成功一片即清零连续失败计数
    updateStatus(res);
    if (stream.waiting) {
      // 之前因等字幕暂停了，现在新片就绪 → 继续播放
      stream.waiting = false;
      $("#video").play().catch(() => {});
    }
    return performance.now() - t0;
  } catch (err) {
    console.error("[subtrans] process_chunk failed:", start, dur, err);
    if (state.session !== sessionAtStart) return null; // 错误属于旧会话，不回写
    // 跳过出错分片，继续处理后续分片（而非停止整个流水线）
    stream.readyUntil = Math.max(stream.readyUntil, start + dur); // 跳过这一段
    if (!stream.failedChunks) stream.failedChunks = [];
    stream.failedChunks.push({ start, dur, err: String(err) });
    stream.holes.push({ start, end: start + dur }); // 出错的窗口记入洞，收尾时再试一次
    stream.consecFails = (stream.consecFails || 0) + 1;
    if (stream.consecFails >= 3) {
      // 连续失败（如 Python 路径错、ffmpeg 不可用）时不再空转烧时间，收尾补洞还会再试一次
      stream.done = true;
      $("#runMsg").textContent = `连续 ${stream.consecFails} 个分片失败，已停止（请检查引擎设置后重试）: ${err}`;
      return null;
    }
    $("#runMsg").textContent = `分片 ${fmt(start)} 出错（已跳过，收尾时重试）: ${err}`;
    if (stream.waiting) {
      stream.waiting = false;
      $("#video").play().catch(() => {});
    }
    return null;
  } finally {
    // 旧会话的 finally 不能把新会话正在进行的 processing 标志清掉，
    // 否则新流水线会出现两个分片并发处理
    if (state.session === sessionAtStart) stream.processing = false;
  }
}

// 连续流水线：一片处理完立即下一片，直到结尾（与播放并行）
async function pump() {
  if (stream.pumping) return;
  stream.pumping = true;
  const sessionAtPump = state.session; // 归属会话：会话切换后本 pump 静默退出
  while (stream.running && !stream.done && state.session === sessionAtPump) {
    const ms = await processNextChunk();
    if (ms === null) {
      if (stream.done) break;
      await sleep(300); // 出错后稍等再试
    }
  }
  stream.pumping = false;
  // openVideo 会把 running 置 false：收尾块只对"真正跑完"的会话生效
  if (stream.done && stream.running && state.session === sessionAtPump) {
    if (stream.filling) return; // 另一个收尾块正在补洞，避免重复补
    stream.filling = true;
    const fillStats = await fillHoles();
    stream.filling = false;
    if (state.session !== sessionAtPump || !stream.running) return; // 补洞期间会话已切换
    const failed = stream.failedChunks || [];
    const failMsg = failed.length ? `（${failed.length} 个分片出错已跳过：${failed.map(f => fmt(f.start)).join(", ")}）` : "";
    const fillMsg = fillStats && fillStats.filled ? ` · 收尾已补全 ${fillStats.filled} 个跳过分片` : "";
    $("#runMsg").textContent = `字幕已全部生成（至 ${fmt(stream.total)}）${fillMsg}${failMsg} · 左侧抽屉中双击可编辑`;
    $("#startBtn").disabled = false;
    $("#exportBtn").disabled = state.subtitles.length === 0;
    stream.failedChunks = [];
    // 步骤 2 完成 → 自动跳到步骤 3 查看结果
    setStepDone(2);
    goStep(3);
    // 自动打开字幕抽屉：让用户立刻看到生成的字幕和编辑入口
    if (state.subtitles.length) $("#subsDrawer").classList.remove("hidden");
  }
}

// 收尾补洞：seek 重定位 / 分片出错跳过的区间（stream.holes）在流水线结束后统一回头补全。
// 每个洞按 chunkLen 切成窗口，已被现有字幕覆盖的窗口跳过（判定与导出补洞共用
// windowCovered）。补洞期间 stream.done 保持 true，播放器可正常播放已生成的部分。
// 返回 { holeCount, filled, failed }；会话切换（invokeChunk 返回 null）时提前返回 null。
async function fillHoles() {
  if (!stream.holes.length) return null;
  // 合并重叠/相邻区间（多次 seek 的洞可能互相覆盖）
  const merged = [];
  for (const h of [...stream.holes].sort((a, b) => a.start - b.start)) {
    const prev = merged[merged.length - 1];
    if (prev && h.start <= prev.end + 0.1) prev.end = Math.max(prev.end, h.end);
    else merged.push({ start: h.start, end: h.end });
  }
  stream.holes = []; // 一次性取走，重试失败不再入队（避免死循环）
  const total = stream.total || 0;
  if (!total) return null;
  let filled = 0;
  let failed = 0;
  for (const h of merged) {
    const lo = Math.max(0, Math.min(h.start, total));
    const hi = Math.max(0, Math.min(h.end, total));
    if (hi - lo <= 0.05) continue;
    const step = stream.chunkLen || 120;
    for (let s = lo; s < hi - 0.05; s += step) {
      const e = Math.min(s + step, hi);
      if (windowCovered(s, e)) continue; // seek 前后可能已有局部覆盖
      $("#runMsg").textContent = `补全字幕 ${fmt(s)} / ${fmt(total)}`;
      try {
        const res = await invokeChunk(s, e - s);
        if (!res) return null; // 会话已切换：由调用方检查 session 后静默收尾
        filled++;
      } catch (err) {
        console.error("[subtrans] 补洞分片失败:", s, err);
        failed++;
        stream.failedChunks.push({ start: s, dur: e - s, err: String(err) });
      }
    }
  }
  return { holeCount: merged.length, filled, failed };
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
  state.session++; // 新一轮识别：上一次运行的在飞分片立即过期（同一视频重复点开始同理）
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
  resetUndo(); // 新一轮识别：旧字幕/旧撤销历史全部作废
  $("#startBtn").disabled = true;
  $("#exportBtn").disabled = true;
  setFill("#runFill", 0);

  const sessionAtProc = state.session; // 本次启动归属的会话：中途切换则静默放弃
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
    holes: [],
    filling: false,
    consecFails: 0,
  });
  // 新一轮识别：只清自动检测锁定，用户选择的预设与口音保持不变
  const rec = resetRecognitionSession(
    $("#recognitionProfile")?.value || "auto",
    $("#accentVariant")?.value || "auto"
  );
  stream.selectedProfileId = rec.selectedProfileId;
  stream.lockedProfileId = rec.lockedProfileId;
  stream.accentVariant = rec.accentVariant;

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
  if (state.session !== sessionAtProc) return; // 处理期间已切换会话：不播放、不起流水线
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

// ───────── 字幕渲染 + 编辑（字幕工作站 ①/⑤） ─────────

// 项目是否被修改（供自动保存/项目保存使用）
let projectDirty = false;
let autosaveTimer = null;
function markDirty() {
  projectDirty = true;
  if (autosaveTimer) return;
  autosaveTimer = setTimeout(() => {
    autosaveTimer = null;
    projectDirty = false;
    saveAutosave();
  }, 3000);
}

function fmtTimeInput(sec) {
  // 用总厘秒数整体计算并进位（避免 "SS.100" / "SS.10" 之类非法时间码），
  // 厘秒精度保证打开编辑器→直接回车不会明显改动时间码（最大漂移 5ms）
  const totalCs = Math.round(sec * 100);
  const m = Math.floor(totalCs / 6000);
  const s = Math.floor((totalCs % 6000) / 100);
  const c = totalCs % 100;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(c).padStart(2, "0")}`;
}

function parseTimeInput(str) {
  const m = String(str).trim().match(/^(\d{1,3}):([0-5]?\d)(?:[.:](\d{1,3}))?$/);
  if (!m) return null;
  const frac = m[3] ? Number(`0.${m[3]}`) : 0;
  return Number(m[1]) * 60 + Number(m[2]) + frac;
}

function findSubIndexAt(t) {
  const subs = state.subtitles;
  let lo = 0, hi = subs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (subs[mid].start <= t) lo = mid + 1;
    else hi = mid;
  }
  for (let i = lo - 1; i >= 0 && i >= lo - 50; i--) {
    if (subs[i].end >= t) return i;
  }
  return -1;
}

function currentSubIndex() {
  return findSubIndexAt($("#video").currentTime);
}

function rowFor(idx) {
  return document.querySelector(`.sub-row[data-idx="${idx}"]`);
}

function buildRow(sub, idx) {
  const row = document.createElement("div");
  row.className = "sub-row";
  row.dataset.idx = String(idx);
  row.innerHTML = `
    <div class="sub-time" title="点击编辑时间">${fmt(sub.start)}</div>
    <div class="sub-text">
      ${sub.speaker ? `<span class="speaker-badge">${escapeHtml(sub.speaker)}</span>` : ""}
      <div class="sub-orig ${sub.original ? "" : "dim"}">${sub.original ? escapeHtml(sub.original) : "（空）"}</div>
      ${sub.translated ? `<div class="sub-trans">${escapeHtml(sub.translated)}</div>` : '<div class="sub-trans dim">（未翻译，双击输入）</div>'}
    </div>
    <div class="sub-actions">
      <button class="mini-btn" data-act="retr" title="用当前引擎重新翻译本条">🔄</button>
      <button class="mini-btn" data-act="merge" title="与下一条合并">⤵</button>
      <button class="mini-btn" data-act="split" title="按时间中点拆分为两条">⤴</button>
      <button class="mini-btn danger" data-act="del" title="删除本条">✕</button>
    </div>`;
  row.addEventListener("click", (e) => {
    if (e.target.closest(".sub-actions") || e.target.closest(".sub-edit") || e.target.closest(".time-edit")) return;
    $("#video").currentTime = sub.start;
  });
  row.addEventListener("dblclick", (e) => {
    // 双击命中文字区域即可编辑；点译文编辑译文，否则编辑原文
    if (e.target.closest(".sub-actions") || e.target.closest(".sub-time")) return;
    if (e.target.closest(".sub-trans")) startEdit(idx, "translated");
    else startEdit(idx, "original");
  });
  row.querySelector(".sub-time").addEventListener("click", (e) => {
    e.stopPropagation();
    startTimeEdit(idx);
  });
  row.querySelector('[data-act="merge"]').addEventListener("click", (e) => {
    e.stopPropagation();
    mergeWithNext(idx);
  });
  row.querySelector('[data-act="split"]').addEventListener("click", (e) => {
    e.stopPropagation();
    splitSubtitle(idx);
  });
  row.querySelector('[data-act="del"]').addEventListener("click", (e) => {
    e.stopPropagation();
    deleteSubtitle(idx);
  });
  row.querySelector('[data-act="retr"]').addEventListener("click", (e) => {
    e.stopPropagation();
    retranslateOne(idx);
  });
  return row;
}

// 批量追加（识别分片返回）：按 start 有序插入后整体重建，保证 data-idx 与数组下标一致。
// 每个批次推入撤销栈：Ctrl+Z 可整体撤掉最近一批（避免追加与手动编辑的快照脱节）。
function appendSubtitles(segs, opts = {}) {
  if (!segs.length) return;
  if (!opts.noUndo) pushUndo();
  for (const s of segs) {
    let lo = 0, hi = state.subtitles.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (state.subtitles[mid].start < s.start) lo = mid + 1;
      else hi = mid;
    }
    state.subtitles.splice(lo, 0, s);
  }
  markDirty();
  refreshSubList(true);
}

function refreshSubList(autoScroll) {
  const list = $("#subList");
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
  list.innerHTML = "";
  if (!state.subtitles.length) {
    list.innerHTML = '<div class="empty">识别后字幕会显示在这里</div>';
  } else {
    state.subtitles.forEach((sub, i) => list.appendChild(buildRow(sub, i)));
  }
  $("#subCount").textContent = state.subtitles.length ? `· ${state.subtitles.length} 条` : "";
  if (autoScroll && nearBottom) list.scrollTop = list.scrollHeight;
  scheduleWaveformDraw();
}

// ── 撤销 / 重做（快照式：字幕数组 JSON，上限 50 步） ──
const undoStack = [];
const redoStack = [];
// 每个识别分片批次也是一步撤销，长视频分片多，上限放宽防手动编辑历史被挤出
const UNDO_MAX = 100;
let lastSnapshot = null;

function snapshotSubs() {
  return JSON.stringify(state.subtitles);
}

function pushUndo() {
  const snap = snapshotSubs();
  if (snap === lastSnapshot) return; // 同一状态不重复入栈
  lastSnapshot = snap;
  undoStack.push(snap);
  if (undoStack.length > UNDO_MAX) undoStack.shift();
  redoStack.length = 0;
}

function restoreSnapshot(snap) {
  try {
    state.subtitles = JSON.parse(snap);
  } catch {
    return;
  }
  markDirty();
  refreshSubList();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(snapshotSubs());
  const prev = undoStack.pop();
  lastSnapshot = prev;
  invalidateLoop();
  restoreSnapshot(prev);
  $("#runMsg").textContent = "已撤销";
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(snapshotSubs());
  const next = redoStack.pop();
  lastSnapshot = next;
  invalidateLoop();
  restoreSnapshot(next);
  $("#runMsg").textContent = "已重做";
}

// 会话切换（新视频/重新识别/打开项目）时必须清空：否则旧会话的字幕快照会经 Ctrl+Z 灌进新会话
function resetUndo() {
  undoStack.length = 0;
  redoStack.length = 0;
  lastSnapshot = null;
}

// ── 行内编辑 ──
let editState = null;

function commitInlineEdit() {
  if (!editState) return;
  const { sub, field, textarea } = editState;
  editState = null;
  const text = textarea.value.trim();
  const div = document.createElement("div");
  div.className = field === "original" ? "sub-orig" : "sub-trans";
  const placeholder = field === "original" ? "（空）" : "（未翻译，双击输入）";
  div.textContent = text || placeholder;
  if (!text) div.classList.add("dim");
  // 行可能因列表重建已脱离 DOM（编辑中途来了新分片）：直接提交到对象并整表刷新
  const detached = !textarea.isConnected;
  textarea.replaceWith(div);
  if (sub && (sub[field] || "") !== text) {
    pushUndo();
    sub[field] = text;
    markDirty();
  }
  if (detached) refreshSubList();
}

function startEdit(idx, field) {
  commitInlineEdit();
  const row = rowFor(idx);
  const el = row?.querySelector(field === "original" ? ".sub-orig" : ".sub-trans");
  // 捕获对象引用而非索引：编辑期间列表可能因新分片插入而重排，
  // 索引会移位导致提交写进错误的条目
  const sub = state.subtitles[idx];
  if (!el || !sub) return;
  const ta = document.createElement("textarea");
  ta.className = "sub-edit";
  ta.value = sub[field] || "";
  el.replaceWith(ta);
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  editState = { sub, field, textarea: ta };
  ta.addEventListener("blur", () => commitInlineEdit());
  ta.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ta.blur();
    } else if (e.key === "Escape") {
      ta.value = sub[field] || "";
      ta.blur();
    }
  });
}

function startTimeEdit(idx) {
  commitInlineEdit();
  const row = rowFor(idx);
  const timeEl = row?.querySelector(".sub-time");
  const s = state.subtitles[idx];
  if (!timeEl || !s) return;
  const wrap = document.createElement("span");
  wrap.className = "time-edit";
  wrap.innerHTML =
    `<input class="time-input" value="${fmtTimeInput(s.start)}">` +
    `<span class="time-sep">→</span>` +
    `<input class="time-input" value="${fmtTimeInput(s.end)}">`;
  timeEl.replaceWith(wrap);
  const [inA, inB] = wrap.querySelectorAll("input");
  inA.focus();
  inA.select();
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    const ns = parseTimeInput(inA.value);
    const ne = parseTimeInput(inB.value);
    // 仅在值实际变化时提交，避免"打开→回车"无谓入撤销栈
    const changed =
      ns != null && ne != null && ne > ns &&
      (Math.abs(ns - s.start) > 1e-6 || Math.abs(ne - s.end) > 1e-6);
    if (changed) {
      pushUndo();
      // 直接改捕获的对象引用，列表重排不影响正确性
      s.start = ns;
      s.end = ne;
      state.subtitles.sort((a, b) => a.start - b.start);
      markDirty();
    }
    refreshSubList();
  };
  // 两框互切不提交（焦点仍在 wrap 内）；点击外部/回车/Esc 才提交
  inA.addEventListener("blur", () => {
    if (!wrap.contains(document.activeElement)) finish();
  });
  inB.addEventListener("blur", () => {
    if (!wrap.contains(document.activeElement)) finish();
  });
  wrap.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") {
      e.preventDefault();
      finish();
    } else if (e.key === "Escape") {
      done = true;
      refreshSubList();
    }
  });
}

// ── 条目操作 ──
function deleteSubtitle(idx) {
  commitInlineEdit();
  if (!state.subtitles[idx]) return;
  pushUndo();
  invalidateLoop();
  state.subtitles.splice(idx, 1);
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = "已删除字幕（Ctrl+Z 撤销）";
}

function mergeWithNext(idx) {
  commitInlineEdit();
  const a = state.subtitles[idx];
  const b = state.subtitles[idx + 1];
  if (!a || !b) return;
  pushUndo();
  invalidateLoop();
  // 用 max 防后一条完全包在前一条内时时长回缩
  a.end = Math.max(a.end, b.end);
  a.original = [a.original, b.original].filter(Boolean).join("\n");
  a.translated = [a.translated, b.translated].filter(Boolean).join("\n");
  state.subtitles.splice(idx + 1, 1);
  markDirty();
  refreshSubList();
}

function splitSubtitle(idx) {
  commitInlineEdit();
  const s = state.subtitles[idx];
  if (!s || s.end - s.start < 0.4) return;
  pushUndo();
  invalidateLoop();
  const mid = (s.start + s.end) / 2;
  // 按码点切分：直接按 UTF-16 长度切会截断 emoji 等代理对
  const orig = Array.from(s.original);
  const trans = Array.from(s.translated || "");
  const halfO = Math.floor(orig.length / 2);
  const halfT = Math.floor(trans.length / 2);
  const a = {
    ...s,
    end: mid,
    original: orig.slice(0, halfO).join("").trim(),
    translated: trans.slice(0, halfT).join("").trim(),
  };
  const b = {
    ...s,
    start: mid,
    original: orig.slice(halfO).join("").trim(),
    translated: trans.slice(halfT).join("").trim(),
  };
  state.subtitles.splice(idx, 1, a, b);
  markDirty();
  refreshSubList();
}

// ── 循环播放当前字幕 ──
function invalidateLoop() {
  const v = $("#video");
  if (v.dataset.loopIdx == null) return;
  delete v.dataset.loopIdx;
  delete v.dataset.loopStart;
  delete v.dataset.loopEnd;
  $("#runMsg").textContent = "循环播放已停止（字幕结构已变化）";
}

function toggleLoopCurrent() {
  const v = $("#video");
  const i = currentSubIndex();
  if (i < 0) {
    $("#runMsg").textContent = "当前没有播放中的字幕，先播放到某条字幕再按 R";
    return;
  }
  const s = state.subtitles[i];
  if (v.dataset.loopIdx === String(i)) {
    delete v.dataset.loopIdx;
    delete v.dataset.loopStart;
    delete v.dataset.loopEnd;
    $("#runMsg").textContent = "循环播放已关闭";
  } else {
    v.dataset.loopIdx = String(i);
    v.dataset.loopStart = String(Math.max(0, s.start - 0.2));
    v.dataset.loopEnd = String(s.end + 0.3);
    $("#runMsg").textContent = "循环播放本条字幕（按 R 关闭）";
  }
}

function renderSubList() {
  const list = $("#subList");
  list.innerHTML = state.subtitles.length
    ? ""
    : '<div class="empty">识别后字幕会显示在这里</div>';
  $("#subCount").textContent = state.subtitles.length ? `· ${state.subtitles.length} 条` : "";
}

function updateOverlay(t) {
  const subs = state.subtitles;
  const idx = findSubIndexAt(t);
  const cur = idx >= 0 ? subs[idx] : null;
  // 抽屉跟随：高亮当前字幕；用户上翻时不强制滚动，只做最近边缘对齐
  document.querySelectorAll(".sub-row.active").forEach((r) => r.classList.remove("active"));
  if (idx >= 0) {
    const row = rowFor(idx);
    if (row) {
      row.classList.add("active");
      const body = document.querySelector(".drawer-body");
      if (body && body.scrollHeight - body.scrollTop - body.clientHeight < 80) {
        row.scrollIntoView({ block: "nearest" });
      }
    }
  }
  const overlay = $("#overlay");
  const showOrig = $("#showOrig").checked;
  const showTrans = $("#showTrans").checked;
  if (!cur) return overlay.classList.add("hidden");
  let html = "";
  const spk = cur.speaker ? `【${escapeHtml(cur.speaker)}】` : "";
  if (showOrig && cur.original) html += `<div class="ol-orig">${spk}${escapeHtml(cur.original)}</div>`;
  if (showTrans && cur.translated) {
    // 说话人前缀只出现一次：有原文行时在原文行，否则在译文行
    const transSpk = !showOrig || !cur.original ? spk : "";
    html += `<div class="ol-trans">${transSpk}${escapeHtml(cur.translated)}</div>`;
  }
  if (html) {
    overlay.innerHTML = html;
    overlay.classList.remove("hidden");
  } else overlay.classList.add("hidden");
}

// ───────── 音频波形图（字幕工作站 ⑤） ─────────
let wf = null; // Float32Array（100Hz 单声道样本）
let wfRate = 100;
let wfLastDraw = 0;
let wfDrag = false;

function scheduleWaveformDraw() {
  if (!wf) return;
  const now = performance.now();
  if (now - wfLastDraw < 150) return; // 最多 ~7fps，播放头足够流畅且省 CPU
  wfLastDraw = now;
  requestAnimationFrame(drawWaveform);
}

async function loadWaveform() {
  const vp = state.videoPath;
  if (!vp) return;
  try {
    const d = await invoke("extract_waveform", { videoPath: vp });
    if (state.videoPath !== vp) return; // 等待期间已切换视频：丢弃旧结果
    const bin = atob(d.samples_b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    wf = new Float32Array(bytes.buffer, 0, Math.floor(bin.length / 4));
    wfRate = d.sample_rate || 100;
    $("#waveformWrap").classList.remove("hidden");
    drawWaveform();
  } catch (e) {
    console.error("[subtrans] load waveform failed:", e);
    wf = null;
  }
}

function drawWaveform() {
  const wrap = $("#waveformWrap");
  const cv = $("#waveform");
  if (!wf || wrap.classList.contains("hidden")) return;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (!w || !h) return;
  if (cv.width !== Math.floor(w * dpr) || cv.height !== Math.floor(h * dpr)) {
    cv.width = Math.floor(w * dpr);
    cv.height = Math.floor(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const dur = $("#video").duration || wf.length / wfRate;
  const n = wf.length;
  // 每像素取峰值画对称波形
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor((x * n) / w);
    const i1 = Math.max(i0 + 1, Math.floor(((x + 1) * n) / w));
    let amp = 0;
    for (let i = i0; i < i1; i++) {
      const v = wf[i] < 0 ? -wf[i] : wf[i];
      if (v > amp) amp = v;
    }
    const bh = Math.max(1, Math.min(h / 2 - 3, amp * (h / 2 - 3)));
    ctx.fillStyle = "rgba(233,237,243,0.25)";
    ctx.fillRect(x, h / 2 - bh, 1, bh * 2);
  }
  // 已有字幕覆盖区（金色底）
  if (dur > 0 && state.subtitles.length) {
    ctx.fillStyle = "rgba(232,184,75,0.4)";
    for (const s of state.subtitles) {
      const x0 = (s.start / dur) * w;
      const x1 = Math.max(x0 + 1, (s.end / dur) * w);
      ctx.fillRect(x0, 3, x1 - x0, h - 6);
    }
  }
  // 播放头
  if (dur > 0) {
    const t = $("#video").currentTime || 0;
    const px = Math.min(w, Math.max(0, (t / dur) * w));
    ctx.fillStyle = "#f2c460";
    ctx.fillRect(px - 1, 0, 2, h);
  }
}

function seekWaveform(e) {
  const wrap = $("#waveformWrap");
  const dur = $("#video").duration;
  if (!dur) return;
  const rect = wrap.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
  $("#video").currentTime = frac * dur;
  drawWaveform();
}

// ───────── 项目保存 / 自动保存（字幕工作站 ②） ─────────

// includeSecret=true 时保存 API Key（仅用户显式保存的项目文件）；
// 自动保存草稿不落盘密钥，避免明文泄露
function collectProject(includeSecret = false) {
  return {
    version: 2,
    videoPath: state.videoPath,
    totalSec: stream.total || $("#video").duration || 0,
    subtitles: state.subtitles,
    settings: {
      modelName: $("#whisperModel").value,
      // v2 识别设置：预设 + 口音；sourceLang 继续保存供旧版本/自定义语言使用
      recognitionProfileId: $("#recognitionProfile").value,
      accentVariant: $("#accentVariant")?.value || "auto",
      sourceLang: $("#sourceLang").value,
      targetLang: $("#targetLang").value,
      engine: $("#engine").value,
      dsKey: includeSecret ? $("#dsKey").value : "",
      dsModel: $("#dsModel").value,
      olHost: $("#olHost").value,
      olModel: $("#olModel").value,
      transEnabled: $("#transEnabled").checked,
      correctEnabled: $("#correctEnabled").checked,
      glossary: $("#glossary").value,
      useFw: $("#useFw").checked,
      vadEnabled: $("#vadFilter").checked,
      hiQuality: $("#hiQuality").checked,
      denoiseFilter: $("#denoiseFilter").value,
      fwDevice: $("#fwDevice").value,
      demucsDevice: $("#demucsDevice").value,
      demucsModel: $("#demucsModel").value,
      pyPython: $("#pyPython").value,
      showOrig: $("#showOrig").checked,
      showTrans: $("#showTrans").checked,
    },
  };
}

function applyProject(p) {
  if (!p || typeof p !== "object") return false;
  // 字幕（对损坏字段做兜底，不让一个坏条目毁掉整个项目）
  if (Array.isArray(p.subtitles)) {
    state.subtitles = p.subtitles
      .filter((s) => s && Number.isFinite(+s.start) && Number.isFinite(+s.end))
      .map((s) => ({
        index: 0,
        start: Math.max(0, +s.start),
        end: Math.max(+s.start, +s.end),
        original: String(s.original ?? ""),
        translated: String(s.translated ?? ""),
        speaker: String(s.speaker ?? ""),
      }))
      .sort((a, b) => a.start - b.start);
    refreshSubList();
  }
  const s = p.settings || {};
  const setVal = (sel, v) => {
    const el = $(sel);
    if (el && v != null) el.value = v;
  };
  const setChk = (sel, v) => {
    const el = $(sel);
    if (el && v != null) el.checked = !!v;
  };
  setVal("#whisperModel", s.modelName);
  // v1 → v2 识别设置迁移（映射 sourceLang；不常见语言保留为 custom）
  const rec = migrateRecognitionSettings(s);
  setVal("#recognitionProfile", rec.recognitionProfileId);
  setVal("#accentVariant", rec.accentVariant);
  setVal("#sourceLang", rec.sourceLang);
  updateRecognitionGroups();
  setVal("#targetLang", s.targetLang);
  setVal("#engine", s.engine);
  setVal("#dsKey", s.dsKey);
  setVal("#dsModel", s.dsModel);
  setVal("#olHost", s.olHost);
  setVal("#olModel", s.olModel);
  setVal("#glossary", s.glossary);
  setVal("#denoiseFilter", s.denoiseFilter);
  setVal("#fwDevice", s.fwDevice);
  setVal("#demucsDevice", s.demucsDevice);
  setVal("#demucsModel", s.demucsModel);
  setVal("#pyPython", s.pyPython);
  setChk("#transEnabled", s.transEnabled);
  setChk("#correctEnabled", s.correctEnabled);
  setChk("#useFw", s.useFw);
  setChk("#vadFilter", s.vadEnabled); // 真实控件是 #vadFilter（vadEnabled 只是项目 JSON 里的字段名）
  setChk("#hiQuality", s.hiQuality);
  setChk("#showOrig", s.showOrig);
  setChk("#showTrans", s.showTrans);
  // 视频
  if (p.videoPath) {
    state.videoPath = p.videoPath;
    $("#fileName").textContent = p.videoPath.split(/[\\/]/).pop();
    $("#video").src = convertFileSrc(p.videoPath);
    $("#stageEmpty").classList.add("hidden");
    $("#startBtn").disabled = false;
    // 与 openVideo 一致：元数据就绪后加载波形/视频信息/预估
    wf = null;
    $("#waveformWrap").classList.add("hidden");
    $("#video").addEventListener(
      "loadedmetadata",
      () => {
        updateEstimate();
        loadWaveform();
        loadVideoInfo();
      },
      { once: true }
    );
  }
  if (Number.isFinite(+p.totalSec) && +p.totalSec > 0) {
    stream.total = +p.totalSec;
    stream.readyUntil = +p.totalSec; // 视为已处理到末尾；导出时缺口会自动补全
    stream.done = true;
  }
  setStepDone(1);
  if (state.subtitles.length) setStepDone(2);
  updateEngineHint();
  updateCorrectHint();
  updateEstimate();
  return true;
}

async function saveAutosave() {
  try {
    const p = collectProject();
    // 空会话（无视频且无字幕）不写草稿，避免覆盖更早的有效草稿
    if (!p.videoPath && !p.subtitles.length) return;
    await invoke("save_autosave", { json: JSON.stringify(p) });
  } catch (e) {
    console.error("[subtrans] autosave failed:", e);
  }
}

async function restoreAutosave() {
  try {
    const p = await invoke("load_autosave");
    if (p && applyProject(p)) {
      $("#runMsg").textContent = `已恢复上次会话（${state.subtitles.length} 条字幕）${
        state.videoPath ? "" : "· 原视频已移动，字幕仍保留"
      }`;
    }
  } catch (e) {
    console.error("[subtrans] restore autosave failed:", e);
  }
}

async function saveProjectDialog() {
  const path = await saveDialog({
    defaultPath: "subtrans-project.subtrans",
    filters: [{ name: "SubTrans 项目", extensions: ["subtrans", "json"] }],
  });
  if (!path) return;
  try {
    await invoke("save_project_file", { path, json: JSON.stringify(collectProject(true)) });
    $("#runMsg").textContent = `项目已保存: ${path}`;
  } catch (err) {
    $("#runMsg").textContent = `保存项目失败: ${err}`;
  }
}

async function openProjectDialog() {
  const path = await openDialog({
    filters: [{ name: "SubTrans 项目", extensions: ["subtrans", "json"] }],
  });
  if (!path) return;
  try {
    const p = await invoke("open_project_file", { path });
    // 打开项目前停掉旧流水线，避免旧任务往新字幕里塞数据
    stream.running = false;
    stream.done = true;
    stream.processing = false;
    stream.pumping = false;
    stream.audioSource = null;
    stream.holes = [];
    stream.filling = false;
    stream.consecFails = 0;
    stream.lockedProfileId = null; // 打开的项目是全新会话：只清自动锁定
    state.videoPath = "";
    state.subtitles = [];
    resetUndo(); // 打开的项目是全新会话，旧撤销历史不得混入
    state.session++; // 会话令牌：项目可能指向与当前相同的视频，旧分片不得写入新列表
    if (applyProject(p)) {
      lastSnapshot = snapshotSubs(); // 以载入状态为撤销基准
      goStep(3);
      $("#runMsg").textContent = `项目已打开（${state.subtitles.length} 条字幕）`;
    }
  } catch (err) {
    $("#runMsg").textContent = `打开项目失败: ${err}`;
  }
}

// ───────── 导出 SRT ─────────
async function exportSrt() {
  if (!state.videoPath) return;
  const total = $("#video").duration || stream.total;
  $("#exportBtn").disabled = true;

  // 1) 后台流水线还在跑则等它真正结束（含分片间短暂休眠的间隙），避免补洞与流水线并发。
  //    1 小时兜底超时：后端异常时不至于永远卡在"等待字幕生成完成"
  const WAIT_TIMEOUT = 60 * 60 * 1000;
  if (stream.running && !stream.done) {
    $("#runMsg").textContent = "等待后台字幕生成完成...";
    if (!(await waitUntil(() => stream.done, 200, WAIT_TIMEOUT))) {
      $("#exportBtn").disabled = false;
      $("#runMsg").textContent = "等待字幕生成超时，请先停止/重试识别";
      return;
    }
  }
  if (stream.pumping || stream.processing || stream.filling) {
    $("#runMsg").textContent = "等待后台字幕生成完成...";
    if (!(await waitUntil(() => !stream.pumping && !stream.processing && !stream.filling, 200, WAIT_TIMEOUT))) {
      $("#exportBtn").disabled = false;
      $("#runMsg").textContent = "等待字幕生成超时，请先停止/重试识别";
      return;
    }
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
  const format = $("#exportFormat").value || "srt";
  const path = await saveDialog({
    defaultPath: `subtitle.${format}`,
    filters: [{ name: format.toUpperCase(), extensions: [format] }],
  });
  if (!path) {
    $("#exportBtn").disabled = false;
    return;
  }
  const showOrig = $("#showOrig").checked;
  let content;
  if (format === "vtt") content = buildVttContent(subs, showOrig);
  else if (format === "ass") content = buildAssContent(subs, showOrig);
  else content = buildSrtContent(subs, showOrig);
  // 可选：繁体输出（OpenCC 简体→繁体）
  if ($("#tradOutput").checked) {
    try {
      content = await invoke("convert_traditional", { text: content });
    } catch (err) {
      $("#runMsg").textContent = `繁体转换失败（已按简体导出）: ${err}`;
    }
  }
  await invoke("save_text_file", { path, content });
  $("#runMsg").textContent = `已导出: ${path}`;
  $("#exportBtn").disabled = false;
}

// ── 字幕文本构建（SRT / VTT / ASS） ──
function subLines(s, showOrig) {
  const spk = s.speaker ? `【${s.speaker}】` : "";
  if (showOrig && s.original && s.translated) {
    return [spk + s.original, s.translated];
  }
  return [spk + (s.translated || s.original)];
}

function buildSrtContent(subs, showOrig) {
  const lines = [];
  subs.forEach((s, i) => {
    lines.push(String(i + 1));
    lines.push(`${ts(s.start)} --> ${ts(s.end)}`);
    lines.push(...subLines(s, showOrig));
    lines.push("");
  });
  return lines.join("\n");
}

function vttTs(sec) {
  // 用总毫秒数整体计算并进位，避免 (sec%60).toFixed 出现 "60.000" 非法时间码
  const ms = Math.round(sec * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const f = String(ms % 1000).padStart(3, "0");
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${f}`;
}

function buildVttContent(subs, showOrig) {
  const lines = ["WEBVTT", ""];
  subs.forEach((s) => {
    lines.push(`${vttTs(s.start)} --> ${vttTs(s.end)}`);
    lines.push(...subLines(s, showOrig));
    lines.push("");
  });
  return lines.join("\n");
}

function assTs(sec) {
  // 用总厘秒数整体计算并进位，避免 (sec%1)*100 四舍五入到 100 的非法时间码
  const cs = Math.round(sec * 100);
  const h = Math.floor(cs / 360000);
  const m = Math.floor((cs % 360000) / 6000);
  const s = Math.floor((cs % 6000) / 100);
  const f = String(cs % 100).padStart(2, "0");
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${f}`;
}

function buildAssContent(subs, showOrig) {
  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,48,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
  const lines = [header];
  for (const s of subs) {
    const text = subLines(s, showOrig)
      .map((t) =>
        t
          .replace(/\{/g, "（")
          .replace(/\}/g, "）")
          .replace(/\r\n|\r/g, "")
          .replace(/\n/g, "\\N") // 物理换行必须转成 ASS 软换行 \N，否则一条 Dialogue 被拆成多行
      )
      .join("\\N");
    if (!text) continue;
    lines.push(`Dialogue: 0,${assTs(s.start)},${assTs(s.end)},Default,,0,0,0,,${text}`);
  }
  return lines.join("\n");
}

// ── 硬字幕烧录（③） ──
async function burnSubtitles() {
  if (!state.videoPath) {
    $("#runMsg").textContent = "请先打开视频";
    return;
  }
  if (!state.subtitles.length) {
    $("#runMsg").textContent = "没有可烧录的字幕（先识别或导入字幕）";
    return;
  }
  const path = await saveDialog({
    defaultPath: "subtitled.mp4",
    filters: [{ name: "MP4 视频", extensions: ["mp4"] }],
  });
  if (!path) return;
  const btn = $("#burnBtn");
  btn.disabled = true;
  try {
    // 等待后台流水线结束，避免烧录中途字幕还在变（1 小时兜底超时）
    if (stream.running && !stream.done) {
      if (!(await waitUntil(() => stream.done, 200, 60 * 60 * 1000))) {
        $("#runMsg").textContent = "等待字幕生成超时，已取消烧录";
        return;
      }
    }
    if (stream.pumping || stream.processing || stream.filling) {
      if (!(await waitUntil(() => !stream.pumping && !stream.processing && !stream.filling, 200, 60 * 60 * 1000))) {
        $("#runMsg").textContent = "等待字幕生成超时，已取消烧录";
        return;
      }
    }
    const sorted = [...state.subtitles].sort((a, b) => a.start - b.start);
    const srt = buildSrtContent(sorted, $("#showOrig").checked);
    const total = $("#video").duration || stream.total;
    const out = await invoke("burn_subtitles", {
      videoPath: state.videoPath,
      srtContent: srt,
      outPath: path,
      fontSize: parseInt($("#burnFontSize").value, 10) || 18,
      marginV: parseInt($("#burnMarginV").value, 10) || 24,
      position: $("#burnPosition").value,
      totalSec: total || 0,
      preset: $("#burnPreset").value || "custom",
      fontColor: $("#burnFontColor").value || "white",
      outlineColor: $("#burnOutlineColor").value || "black",
    });
    $("#runMsg").textContent = `烧录完成: ${out}`;
  } catch (err) {
    console.error("[subtrans] burn_subtitles failed:", err);
    $("#runMsg").textContent = `烧录失败: ${err}`;
  } finally {
    btn.disabled = false;
  }
}

// ── 导入字幕（③） ──
async function importSubtitleFile(replace) {
  const path = await openDialog({
    filters: [
      { name: "字幕文件", extensions: ["srt", "vtt"] },
    ],
  });
  if (!path) return;
  try {
    const subs = await invoke("parse_subtitle_file", { path });
    if (!Array.isArray(subs) || !subs.length) {
      $("#runMsg").textContent = "字幕文件里没有条目";
      return;
    }
    pushUndo();
    invalidateLoop();
    if (replace) state.subtitles = [];
    const mapped = subs.map((s) => ({
      index: 0,
      start: +s.start,
      end: +s.end,
      original: String(s.original ?? ""),
      translated: String(s.translated ?? ""),
    }));
    // 导入自身已推过撤销快照，追加不再重复入栈（一次 Ctrl+Z 完整撤回导入）
    appendSubtitles(mapped, { noUndo: true });
    goStep(3);
    $("#runMsg").textContent = `已导入 ${mapped.length} 条字幕${replace ? "（已替换）" : "（已追加）"}`;
    markDirty();
  } catch (err) {
    console.error("[subtrans] parse_subtitle_file failed:", err);
    $("#runMsg").textContent = `导入失败: ${err}`;
  }
}

// ── 时间轴工具包（④） ──
function applyShift() {
  const ms = Number($("#shiftMs").value);
  if (!Number.isFinite(ms)) {
    $("#runMsg").textContent = "请输入有效的偏移毫秒数（如 500 或 -300）";
    return;
  }
  if (ms === 0) {
    $("#runMsg").textContent = "偏移为 0，无需处理";
    return;
  }
  const delta = ms / 1000;
  pushUndo();
  invalidateLoop();
  let n = 0;
  for (const s of state.subtitles) {
    const dur = s.end - s.start;
    const ns = Math.max(0, s.start + delta);
    s.start = ns;
    s.end = ns + dur;
    n++;
  }
  state.subtitles.sort((a, b) => a.start - b.start);
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = `已整体偏移 ${ms}ms（${n} 条）`;
}

function applyFps() {
  const from = Number($("#fpsFrom").value);
  const to = Number($("#fpsTo").value);
  if (!from || !to) {
    $("#runMsg").textContent = "请选择源/目标帧率";
    return;
  }
  if (from === to) {
    $("#runMsg").textContent = "源/目标帧率相同，无需转换";
    return;
  }
  const ratio = from / to;
  pushUndo();
  invalidateLoop();
  let n = 0;
  for (const s of state.subtitles) {
    s.start *= ratio;
    s.end *= ratio;
    n++;
  }
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = `已按 ${from} → ${to} 重算时间码（${n} 条）`;
}

function applyReplace() {
  const find = $("#findText").value;
  if (!find) {
    $("#runMsg").textContent = "请输入要查找的文本";
    return;
  }
  const rep = $("#replaceText").value;
  // 先探测命中再入撤销栈：无命中时不应制造"Ctrl+Z 没反应"的假象
  const hasHit = state.subtitles.some(
    (s) => s.original.includes(find) || (s.translated || "").includes(find)
  );
  if (!hasHit) {
    $("#runMsg").textContent = `没有找到「${find}」`;
    return;
  }
  pushUndo();
  invalidateLoop();
  let hits = 0;
  for (const s of state.subtitles) {
    if (s.original.includes(find)) {
      s.original = s.original.split(find).join(rep);
      hits++;
    }
    if (s.translated && s.translated.includes(find)) {
      s.translated = s.translated.split(find).join(rep);
      hits++;
    }
  }
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = `已替换 ${hits} 处`;
}

// ── 翻译现有字幕 / 单条重译（第二批 ①） ──
async function retranslateOne(idx) {
  const s = state.subtitles[idx];
  if (!s || !s.original.trim()) return;
  const btn = rowFor(idx)?.querySelector('[data-act="retr"]');
  if (btn) btn.disabled = true;
  try {
    const t = await invoke("translate_text", {
      text: s.original,
      targetLang: $("#targetLang").value,
      engine: currentEngine(),
      sourceLang: $("#sourceLang").value || null,
      glossary: $("#glossary").value || "",
    });
    // 翻译期间条目可能已被删除/合并：对象已脱离数组则丢弃结果
    if (!state.subtitles.includes(s)) {
      $("#runMsg").textContent = "该条目已被删除，重译结果已丢弃";
      return;
    }
    pushUndo();
    s.translated = t;
    markDirty();
    refreshSubList();
  } catch (err) {
    console.error("[subtrans] retranslate failed:", err);
    $("#runMsg").textContent = `重译失败: ${err}`;
  }
}

async function batchTranslate() {
  const targets = state.subtitles.filter((s) => !s.translated && s.original.trim());
  if (!targets.length) {
    $("#runMsg").textContent = "没有需要翻译的条目（都有译文了）";
    return;
  }
  const engine = currentEngine();
  const targetLang = $("#targetLang").value;
  const sourceLang = $("#sourceLang").value || null;
  const glossary = $("#glossary").value || "";
  const btn = $("#batchTranslateBtn");
  btn.disabled = true;
  const pool = 6;
  const queue = [...targets];
  let done = 0;
  let failed = 0;
  pushUndo();
  const worker = async () => {
    while (queue.length) {
      const s = queue.shift();
      try {
        const t = await invoke("translate_text", {
          text: s.original,
          targetLang,
          engine,
          sourceLang,
          glossary,
        });
        s.translated = t;
      } catch (e) {
        failed++;
        console.error("[subtrans] batch translate item failed:", e);
      }
      done++;
      $("#runMsg").textContent = `翻译现有字幕 ${done}/${targets.length}${failed ? `（失败 ${failed}）` : ""}`;
    }
  };
  await Promise.all(Array.from({ length: Math.min(pool, targets.length) }, worker));
  btn.disabled = false;
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = `批量翻译完成：${targets.length - failed}/${targets.length} 条`;
}

// ── 说话人标注（第二批 ④） ──
function assignSpeaker(label) {
  const i = currentSubIndex();
  if (i < 0) {
    $("#runMsg").textContent = "当前没有播放中的字幕，先播放到某条字幕";
    return;
  }
  const s = state.subtitles[i];
  // 与当前标注一致时不入撤销栈（避免无变化操作制造"Ctrl+Z 没反应"）
  if ((s.speaker || "") === label) {
    $("#runMsg").textContent = label ? `当前字幕已是「${label}」` : "当前字幕没有说话人标注";
    return;
  }
  pushUndo();
  if (label) {
    s.speaker = label;
    $("#runMsg").textContent = `已把当前字幕标注为「${label}」`;
  } else {
    delete s.speaker;
    $("#runMsg").textContent = "已清除当前字幕的说话人标注";
  }
  markDirty();
  refreshSubList();
}

// ── 字幕质检（第二批 ④） ──
function checkSubtitles() {
  const issues = [];
  const subs = state.subtitles;
  const CPS_MAX = 12; // 读速上限：字符/秒（中文 4-6 字/秒为舒适，12 为宽限）
  for (let i = 0; i < subs.length; i++) {
    const s = subs[i];
    const dur = s.end - s.start;
    if (dur <= 0) {
      issues.push({ idx: i, type: "时长无效", msg: `${fmt(s.start)} 时长≤0` });
      continue;
    }
    if (dur < 0.2) issues.push({ idx: i, type: "过短", msg: `${fmt(s.start)} 时长 ${dur.toFixed(2)}s < 0.2s` });
    const text = (s.translated || s.original || "").replace(/\s/g, "");
    const cps = text.length / dur;
    if (cps > CPS_MAX) issues.push({ idx: i, type: "读速过快", msg: `${fmt(s.start)} ${cps.toFixed(1)} 字/秒（>${CPS_MAX}）` });
    if (i > 0) {
      const p = subs[i - 1];
      if (s.start < p.end - 0.02) issues.push({ idx: i, type: "重叠", msg: `${fmt(p.start)} 与下一条重叠 ${((p.end - s.start) * 1000).toFixed(0)}ms` });
    }
  }
  const el = $("#qcResults");
  if (!issues.length) {
    el.innerHTML = '<div class="muted small">✓ 未发现明显问题</div>';
    $("#runMsg").textContent = "质检通过";
    return;
  }
  el.innerHTML = "";
  for (const it of issues) {
    const d = document.createElement("div");
    d.className = "qc-item";
    const t = document.createElement("span");
    t.className = "qc-type";
    t.textContent = it.type;
    const m = document.createElement("span");
    m.className = "qc-msg";
    m.textContent = it.msg;
    d.append(t, m);
    d.addEventListener("click", () => {
      // 质检后列表可能已被编辑：点击定位前做越界保护
      const sub = state.subtitles[it.idx];
      if (!sub) return;
      $("#video").currentTime = sub.start;
      $("#subsDrawer").classList.remove("hidden");
      // 收起设置浮层，让用户立刻看到定位的画面
      $("#settingsOverlay").classList.add("hidden");
    });
    el.appendChild(d);
  }
  $("#runMsg").textContent = `质检发现 ${issues.length} 个问题（点击定位）`;
}

function fixOverlaps() {
  // 先探测是否有重叠：无重叠不推撤销栈（避免"Ctrl+Z 没反应"的假象）
  const hasOverlap = state.subtitles.some(
    (s, i) => i > 0 && s.start < state.subtitles[i - 1].end - 0.02
  );
  if (!hasOverlap) {
    $("#runMsg").textContent = "没有发现重叠";
    return;
  }
  // 快照必须在修改前推入
  pushUndo();
  invalidateLoop();
  let fixed = 0;
  for (let i = 1; i < state.subtitles.length; i++) {
    const s = state.subtitles[i];
    const p = state.subtitles[i - 1];
    if (s.start < p.end - 0.02) {
      s.start = Math.max(0, p.end - 0.05);
      // 完全被包含/严重重叠：保底最小时长，绝不产生 start>end
      if (s.end <= s.start) s.end = s.start + 0.2;
      fixed++;
    }
  }
  state.subtitles.sort((a, b) => a.start - b.start); // 修复可能扰动顺序，重排保证二分查找有效
  markDirty();
  refreshSubList();
  $("#runMsg").textContent = `已修复 ${fixed} 处重叠`;
}

// ── 视频信息（第二批 ③） ──
async function loadVideoInfo() {
  const vp = state.videoPath;
  if (!vp) return;
  try {
    const info = await invoke("video_info", { videoPath: vp });
    if (state.videoPath !== vp) return; // 等待期间已切换视频：丢弃旧结果
    const el = $("#videoInfo");
    if (!info || !info.ok) {
      el.textContent = "无法读取视频信息（ffprobe 不可用）";
      return;
    }
    const parts = [];
    if (info.width && info.height) parts.push(`${info.width}×${info.height}`);
    if (info.fps) parts.push(`${info.fps.toFixed(3).replace(/\.?0+$/, "")}fps`);
    if (info.video_codec) parts.push(String(info.video_codec).toUpperCase());
    if (info.audio_codec) parts.push(`音频 ${String(info.audio_codec).toUpperCase()}`);
    if (info.duration_sec) parts.push(fmt(info.duration_sec));
    if (info.size_mb != null) parts.push(`${info.size_mb.toFixed(0)}MB`);
    el.textContent = parts.join(" · ");
  } catch (e) {
    console.error("[subtrans] video_info failed:", e);
    $("#videoInfo").textContent = "无法读取视频信息";
  }
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

// ═══════════ 影院外壳交互（方案 A 新增：仅浮层开关，不触碰既有逻辑） ═══════════
{
  const overlay = $("#settingsOverlay");
  const drawer = $("#subsDrawer");
  // 焦点管理：打开浮层时焦点移入（关闭按钮），关闭时还给触发按钮，Tab 不再掉到背景控件
  const openSettings = () => {
    overlay.classList.remove("hidden");
    $("#settingsClose").focus();
  };
  const closeSettings = () => {
    overlay.classList.add("hidden");
    $("#settingsBtn").focus();
  };
  const openDrawer = () => {
    drawer.classList.remove("hidden");
    $("#subsClose").focus();
  };
  const closeDrawer = () => {
    drawer.classList.add("hidden");
    $("#subsBtn").focus();
  };

  $("#settingsBtn").addEventListener("click", openSettings);
  $("#settingsClose").addEventListener("click", closeSettings);
  $("#settingsBackdrop").addEventListener("click", closeSettings);
  $("#subsBtn").addEventListener("click", openDrawer);
  $("#subsClose").addEventListener("click", closeDrawer);
  $("#openDrawerFromPanel").addEventListener("click", openDrawer);
  // 点击 GPU 徽章直达引擎状态
  $("#gpuBadge").addEventListener("click", openSettings);
  // Esc 关闭浮层
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeSettings();
      closeDrawer();
    }
  });
  // 自动保存兜底：修改后 3s（markDirty）+ 每 30s 周期 + 关闭前尽力保存
  setInterval(() => saveAutosave(), 30 * 1000);
  window.addEventListener("beforeunload", () => saveAutosave());
}
