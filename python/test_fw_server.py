"""fw_server.py 的协议单元测试（不依赖 faster-whisper 安装）。

运行（在仓库根目录，任选其一）：
    src-tauri/python-bundle/python.exe python/test_fw_server.py   # 内嵌版 Python（._pth 隔离模式）
    python -m unittest python.test_fw_server -v                   # 常规 Python
"""

import os
import sys
import unittest

# 内嵌版 Python 的 ._pth 隔离模式不把 cwd 放进 sys.path，
# 按本文件位置定位仓库根，保证 `from python import fw_server` 两种解释器下都可用。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python import fw_server


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return [], object()


class ProfileProtocolTests(unittest.TestCase):
    def test_profile_prompt_and_beam_are_forwarded(self):
        model = FakeModel()
        fw_server._transcribe(
            model,
            "clip.wav",
            "en",
            False,
            False,
            "Pemberton",
            "British film dialogue",
            8,
        )
        self.assertEqual(model.kwargs["language"], "en")
        self.assertEqual(model.kwargs["initial_prompt"], "British film dialogue")
        self.assertEqual(model.kwargs["beam_size"], 8)
        self.assertEqual(model.kwargs["hotwords"], "Pemberton")

    def test_empty_new_fields_keep_legacy_defaults(self):
        model = FakeModel()
        fw_server._transcribe(model, "clip.wav", None, False, False, "", "", None)
        self.assertEqual(model.kwargs["beam_size"], 5)
        self.assertNotIn("initial_prompt", model.kwargs)


if __name__ == "__main__":
    unittest.main()
