#!/usr/bin/env python3
"""faster-whisper 常驻识别服务（行协议）。

由 Rust 后端启动一次，把模型常驻显存，然后逐行处理请求：
  stdin  : 每行一个 JSON，如 {"audio": "C:/path/chunk.wav", "language": "zh"}
           或 {"cmd": "quit"}
  stdout : 每行一个 JSON，如 {"segments": [{"start":..,"end":..,"text":..}, ...]}
           启动就绪时先输出 {"ready": true}；出错输出 {"error": "..."}

用法: python fw_server.py <model> <device> <compute_type> [model_root]
  例:  python fw_server.py large-v3 cuda float16 C:/path/to/models
"""
import os
import sys
import json

# stdin/stdout 强制 UTF-8：Rust 端按 UTF-8 写 JSON 请求，而中文 Windows 上 python 默认用
# cp936 解码 stdin，会把含中文的路径（如 C:\Users\南山\...）解成乱码 → 文件找不到（Errno 2）。
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 国内直连 HuggingFace 的大文件 CDN 经常下不动 → 默认走镜像（用户若已设 HF_ENDPOINT 则尊重其设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Windows 关键点：ctranslate2 走 GPU 需要 cuDNN/cuBLAS DLL。
# torch(cu124) 自带这些库，把它的 lib 目录加进 DLL 搜索路径即可，无需另装 cuDNN。
try:
    import torch

    _libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_libdir) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_libdir)
except Exception:
    pass

from faster_whisper import WhisperModel


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def resolve_model(name, model_root=None):
    """本地预下载的模型优先（绕过被墙的 HF 主站 API）。"""
    roots = [
        os.environ.get("SUBTRANS_MODELS", ""),  # 显式指定的模型根目录
        model_root or "",  # Rust 端实际下载目录（跨平台，由后端直接传入）
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "subtrans-models"),
        # macOS 没有 LOCALAPPDATA，dirs::data_local_dir() 落在 ~/Library/Application Support
        os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "subtrans-models",
        ),
    ]
    for root in roots:
        if not root:
            continue
        d = os.path.join(root, "faster-whisper-" + name)
        if os.path.isfile(os.path.join(d, "model.bin")):
            return d, True
    return name, False


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "small"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    compute = sys.argv[3] if len(sys.argv) > 3 else ("float16" if device == "cuda" else "int8")
    model_root = sys.argv[4] if len(sys.argv) > 4 else None

    # 本地有就用本地（local_files_only 避免联网 head 调用）；否则按模型名走（镜像）下载
    model_id, is_local = resolve_model(model_name, model_root)
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute, local_files_only=is_local)
    except Exception as e:
        emit({"ready": False, "error": str(e)})
        return
    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        if req.get("cmd") == "quit":
            break
        audio = req.get("audio")
        lang = req.get("language") or None
        separated = req.get("separated", False)  # 是否已经过人声分离
        vad_enabled = req.get("vad_enabled", False)  # 未分离音频是否也过滤音乐/静音
        try:
            segments, info = model.transcribe(
                audio,
                language=lang,
                beam_size=5,
                # VAD：已分离人声轨必开；未分离音频按用户开关决定（音乐多的视频建议开，
                # 纯唱歌/歌词视频可关，避免 Silero 把歌声当“非语音”切掉）
                vad_filter=separated or vad_enabled,
                condition_on_previous_text=False,  # 禁止幻觉传播
                # 0.4 为项目原值：轻声/带背景音乐的人声也能正常识别
                no_speech_threshold=0.4,
            )
            out = [
                {"start": float(s.start), "end": float(s.end), "text": s.text}
                for s in segments
            ]
            # 语言检测结果一并返回：前端可提示用户自动检测到了什么语言，
            # 发现误判时手动指定源语言即可修正
            emit(
                {
                    "segments": out,
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                }
            )
        except Exception as e:
            emit({"error": str(e)})


if __name__ == "__main__":
    main()
