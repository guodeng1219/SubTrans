"""vocal_server.py 的协议单元测试（不依赖 torch / audio-separator 安装）。

运行（在仓库根目录，任选其一）：
    src-tauri/python-bundle/python.exe python/test_vocal_server.py  # 内嵌版 Python（._pth 隔离模式）
    python -m unittest python.test_vocal_server -v                  # 常规 Python
"""

import io
import json
import os
import sys
import tempfile
import unittest

# 内嵌版 Python 的 ._pth 隔离模式不把 cwd 放进 sys.path，
# 按本文件位置定位仓库根，保证 `from python import vocal_server` 两种解释器下都可用。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python import vocal_server


class FakeModelInstance:
    def __init__(self):
        self.output_dir = None


class FakeSeparator:
    loads = 0
    calls = []

    def __init__(self, **kwargs):
        self.output_dir = kwargs["initial_output_dir"]
        self.model_instance = FakeModelInstance()

    def load_model(self, model_filename):
        type(self).loads += 1
        self.model_filename = model_filename

    def separate(self, audio_path):
        type(self).calls.append(audio_path)
        os.makedirs(self.output_dir, exist_ok=True)
        out = os.path.join(self.output_dir, "input_(Vocals).wav")
        with open(out, "wb") as f:
            f.write(b"RIFF" + b"x" * 64)
        return [os.path.basename(out)]


class VocalServerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="subtrans-vocal-test-")
        FakeSeparator.loads = 0
        FakeSeparator.calls = []

    def tearDown(self):
        FakeSeparator.loads = 0
        FakeSeparator.calls = []
        self.temp_dir.cleanup()

    def test_two_requests_load_model_once_and_return_matching_ids(self):
        requests = [
            {"op": "separate", "request_id": "a", "input_path": "a.wav",
             "output_dir": os.path.join(self.temp_dir.name, "out-a")},
            {"op": "separate", "request_id": "b", "input_path": "b.wav",
             "output_dir": os.path.join(self.temp_dir.name, "out-b")},
            {"op": "shutdown"},
        ]
        stdin = io.StringIO("\n".join(json.dumps(row) for row in requests) + "\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        vocal_server.serve(
            ["model-dir", "model.ckpt", "cpu"], stdin, stdout, stderr, FakeSeparator
        )
        rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertTrue(rows[0]["ready"])
        self.assertEqual([rows[1]["request_id"], rows[2]["request_id"]], ["a", "b"])
        self.assertEqual(FakeSeparator.loads, 1)

    def test_bad_json_and_non_object_requests_return_errors(self):
        self.assertEqual(vocal_server.parse_request("{"), {"error_code": "bad_json"})
        self.assertEqual(vocal_server.parse_request("[]"), {"error_code": "invalid_request"})

    def test_memory_failures_are_classified_for_rust_retry(self):
        self.assertTrue(vocal_server.is_memory_error(MemoryError("allocation failed")))
        self.assertTrue(vocal_server.is_memory_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(vocal_server.is_memory_error(RuntimeError("invalid wav")))


if __name__ == "__main__":
    unittest.main()
