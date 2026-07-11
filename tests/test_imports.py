import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SubmissionImportTests(unittest.TestCase):
    def test_packaged_submission_imports_in_isolated_process(self):
        result = subprocess.run(
            [sys.executable, "-c", "import main; print('imports ok')"],
            cwd=ROOT / "tests" / "test_sub",
            env={**os.environ, "PYTHONPATH": str(ROOT / "tests" / "test_sub")},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("imports ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
