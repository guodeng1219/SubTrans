"""demucs_barrier.py 的协议单元测试（不导入 torch / demucs）。

运行（在仓库根目录，任选其一）：
    src-tauri/python-bundle/python.exe python/test_demucs_barrier.py  # 内嵌版 Python（._pth 隔离模式）
    python -m unittest python.test_demucs_barrier -v                 # 常规 Python
"""

import io
import os
import sys
import types
import unittest
from unittest import mock

# 内嵌版 Python 的 ._pth 隔离模式不把 cwd 放进 sys.path，
# 按本文件位置定位仓库根，保证两种解释器下都能 import python.demucs_barrier。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python import demucs_barrier


class DemucsBarrierTests(unittest.TestCase):
    def test_load_command_accepted(self):
        self.assertTrue(demucs_barrier.wait_for_load(io.StringIO('{"op": "load"}\n')))

    def test_wrong_op_rejected(self):
        self.assertFalse(demucs_barrier.wait_for_load(io.StringIO('{"op": "separate"}\n')))

    def test_empty_stdin_rejected(self):
        self.assertFalse(demucs_barrier.wait_for_load(io.StringIO("")))

    def test_bad_json_rejected(self):
        self.assertFalse(demucs_barrier.wait_for_load(io.StringIO("{bad json\n")))

    def test_non_object_rejected(self):
        self.assertFalse(demucs_barrier.wait_for_load(io.StringIO("[]\n")))

    def test_main_reconstructs_demucs_argv(self):
        # 用假 demucs 模块记录 main() 看到的 argv：覆盖屏障指令后的
        # argv 重组（防切片 off-by-one 类回归）
        calls = {}
        fake_separate = types.ModuleType("demucs.separate")

        def fake_main():
            calls["argv"] = list(sys.argv)

        fake_separate.main = fake_main
        fake_demucs = types.ModuleType("demucs")
        fake_demucs.separate = fake_separate
        with mock.patch.dict(
            sys.modules, {"demucs": fake_demucs, "demucs.separate": fake_separate}
        ), mock.patch.object(
            sys, "argv",
            ["demucs_barrier.py", "htdemucs", "cpu", r"C:\out", r"C:\audio.wav"],
        ), mock.patch.object(sys, "stdin", io.StringIO('{"op": "load"}\n')):
            demucs_barrier.main()
        self.assertEqual(
            calls["argv"],
            ["demucs", "--two-stems", "vocals", "-n", "htdemucs",
             "-d", "cpu", "-o", r"C:\out", r"C:\audio.wav"],
        )

    def test_main_short_argv_exits_usage(self):
        with mock.patch.object(sys, "argv", ["demucs_barrier.py", "htdemucs"]):
            with self.assertRaises(SystemExit) as ctx:
                demucs_barrier.main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

