"""demucs_barrier.py 的协议单元测试（不导入 torch / demucs）。

运行（在仓库根目录，任选其一）：
    src-tauri/python-bundle/python.exe python/test_demucs_barrier.py  # 内嵌版 Python（._pth 隔离模式）
    python -m unittest python.test_demucs_barrier -v                 # 常规 Python
"""

import io
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
