#!/usr/bin/env python3
"""BS-RoFormer 常驻人声分离服务（JSON Lines 行协议）。

由 Rust 后端启动一次，把模型常驻内存，然后逐行处理分离请求：
  stdin  : 启动屏障 + 请求，每行一个 JSON：
           启动后先读一行 {"op": "load"}（Rust 完成 Job 内存硬限绑定后才会发送；
           收到 load 之前绝不加载模型，杜绝「spawn→取 PID→绑 Job」与
           模型加载之间的竞态窗口），随后是
           {"op": "separate", "request_id": "sep-0007",
            "input_path": ".../input_0007.wav", "output_dir": ".../output_0007"}
           或 {"op": "shutdown"}
  stdout : 每行一个 JSON：
           启动就绪 {"ready": true, "pid": <pid>}（load 且模型加载成功后输出）
           成功     {"request_id": ..., "ok": true, "vocals_path": "...", "elapsed_ms": ...}
           失败     {"request_id": ..., "ok": false, "error_code": "separation_failed"|"memory_error", "message": "..."}
           启动失败 {"ready": false, "error_code": "model_load_failed", "error": "..."}

stdout 只允许协议 JSON；日志与诊断全部写 stderr。
heavy imports（torch / audio-separator）延迟到 serve() 内，裸 Python 也能跑协议单测。

用法: python vocal_server.py <model_dir> <model_name> <device[cpu|cuda]>
"""

import gc
import json
import os
import re
import sys
import time

# stdin/stdout 强制 UTF-8：Rust 端按 UTF-8 写 JSON 请求，中文 Windows 上默认代码页
# 会把含中文的路径解成乱码（与 fw_server.py 相同的坑）。
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 国内直连 HuggingFace 的大文件 CDN 经常下不动 → 默认走镜像（用户若已设 HF_ENDPOINT 则尊重其设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def emit(stream, obj):
    stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    stream.flush()


def parse_request(line):
    try:
        value = json.loads(line)
    except Exception:
        return {"error_code": "bad_json"}
    if not isinstance(value, dict):
        return {"error_code": "invalid_request"}
    return value


def create_separator(model_dir, initial_output_dir, device):
    from audio_separator.separator import Separator

    return Separator(
        model_file_dir=model_dir,
        output_dir=initial_output_dir,
        output_format="WAV",
        output_single_stem="Vocals",
        use_soundfile=True,
        use_autocast=device == "cuda",
        # 依赖要求该值位于 (0, 1]；1.0 对合法 [-1, 1] 波形不做峰值衰减，
        # 避免默认 0.9 对不同分片分别缩放。0.0 非“关闭”，会被构造器拒绝。
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
    )


def is_memory_error(exc):
    """把 MemoryError 与已知分配器 OOM 消息映射为内存错误，供 Rust 降片重试分类。"""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "allocation failed",
            "not enough memory",
            "cuda error: memory allocation",
        )
    )


def separate_one(separator, request):
    request_id = request.get("request_id")
    input_path = request.get("input_path")
    output_dir = request.get("output_dir")
    if not request_id or not input_path or not output_dir:
        raise ValueError("request_id, input_path and output_dir are required")
    os.makedirs(output_dir, exist_ok=True)
    separator.output_dir = output_dir
    if separator.model_instance is not None:
        separator.model_instance.output_dir = output_dir
    started = time.perf_counter()
    try:
        files = separator.separate(input_path)
        named_vocals = next(
            (path for path in files if "vocals" in os.path.basename(path).lower()),
            None,
        )
        if named_vocals is None:
            raise RuntimeError("vocals output was not produced")
        vocals = (
            named_vocals if os.path.isabs(named_vocals) else os.path.join(output_dir, named_vocals)
        )
        if not os.path.isfile(vocals) or os.path.getsize(vocals) <= 44:
            raise RuntimeError("valid vocals output was not produced")
        return {
            "vocals_path": os.path.abspath(vocals),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    finally:
        # 注意：Separator.separate() 内部已在每次分离后调用
        # model_instance.clear_gpu_cache() / clear_file_specific_paths()（已核实
        # bundle 内安装版本），这里只清理本函数局部引用与解释器级缓存。
        gc.collect()
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and getattr(torch_mod, "cuda", None) is not None:
            try:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass


def serve(argv, stdin, stdout, stderr, separator_factory=create_separator):
    model_dir, model_name, device = argv
    if device == "cpu":
        # 与 Rust 端 demucs 回退行为对齐：强制 torch 不启用 CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # 启动屏障：Rust 端先 spawn 取 PID、绑定 Job 内存硬限，随后发 {"op":"load"}。
    # 收到 load 之前绝不加载模型——否则「spawn → 取 PID → 绑 Job」与模型加载
    # 之间存在竞态窗口，长视频模型加载期可能瞬时超限。
    first = stdin.readline()
    if not first:
        # stdin 被关闭：无人来发 load，直接退出
        return
    command = parse_request(first)
    if "error_code" in command or command.get("op") != "load":
        emit(
            stdout,
            {
                "ready": False,
                "error_code": "model_load_failed",
                "error": 'expected {"op": "load"} before model load',
            },
        )
        return
    try:
        separator = separator_factory(
            model_dir=model_dir,
            initial_output_dir=os.getcwd(),
            device=device,
        )
        separator.load_model(model_filename=model_name)
    except Exception as exc:
        emit(
            stdout,
            {
                "ready": False,
                "error_code": "model_load_failed",
                "error": str(exc),
            },
        )
        return
    emit(stdout, {"ready": True, "pid": os.getpid()})
    for line in stdin:
        request = parse_request(line)
        if "error_code" in request:
            emit(stdout, {"ok": False, **request})
            continue
        if request.get("op") == "shutdown":
            break
        request_id = request.get("request_id")
        try:
            result = separate_one(separator, request)
            emit(stdout, {"request_id": request_id, "ok": True, **result})
        except Exception as exc:
            print(f"separation failed: {exc}", file=stderr, flush=True)
            emit(
                stdout,
                {
                    "request_id": request_id,
                    "ok": False,
                    "error_code": "memory_error" if is_memory_error(exc) else "separation_failed",
                    "message": str(exc),
                },
            )


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    serve(sys.argv[1:4], sys.stdin, sys.stdout, sys.stderr)


if __name__ == "__main__":
    main()
