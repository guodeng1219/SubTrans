#!/usr/bin/env python3
"""demucs 启动屏障包装器（load 协议与 vocal_server.py 一致）。

Rust 端先 spawn 本脚本、取 PID、绑定 Job 内存硬限，再向 stdin 发送
{"op":"load"}；收到 load 之前绝不导入 demucs / 加载模型——否则
「spawn → 取 PID → 绑 Job」与模型加载之间存在竞态窗口，长视频场景
模型加载期可能瞬时超限。

用法: python demucs_barrier.py <model> <device> <out_dir> <audio>
stdin 第一行必须是 {"op":"load"}；模型加载发生在 demucs.separate.main() 内。
"""

import json
import sys


def wait_for_load(stdin):
    """等待启动屏障指令；收到合法 load 返回 True，EOF/坏 JSON/错误 op 返回 False。"""
    line = stdin.readline()
    if not line:
        return False
    try:
        command = json.loads(line)
    except Exception:
        return False
    return isinstance(command, dict) and command.get("op") == "load"


def main():
    if len(sys.argv) < 6:
        print("usage: demucs_barrier.py <model> <device> <out_dir> <audio>", file=sys.stderr)
        sys.exit(2)
    model, device, out_dir, audio = sys.argv[2:6]
    if not wait_for_load(sys.stdin):
        print('expected {"op": "load"} before model load', file=sys.stderr)
        sys.exit(2)
    # Job 硬限此刻已绑定：才允许导入 demucs 并加载模型（在 separate.main() 内）
    from demucs import separate

    sys.argv = ["demucs", "--two-stems", "vocals", "-n", model, "-d", device, "-o", out_dir, audio]
    separate.main()


if __name__ == "__main__":
    main()
