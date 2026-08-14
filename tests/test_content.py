"""Content checks for mistakes that previously reached the published site."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContentTests(unittest.TestCase):
    def test_embedded_examples_are_synchronised(self):
        result = subprocess.run(
            [sys.executable, "hooks/sync_examples.py", "--check"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_current_labs_exclude_known_stale_content(self):
        current_pages = [
            ROOT / "src" / name
            for name in ("index.md", "lab1.md", "lab2.md", "lab3.md", "lab4.md", "lab5.md", "lab8a.md", "lab8b.md")
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in current_pages)
        for stale_text in (
            "isaac-sim.standalone.bat",
            "from_logits=True",
            "Robot Ant Prims (3D Differential Drive)",
            "repeat from fitnexx",
        ):
            self.assertNotIn(stale_text, combined)


if __name__ == "__main__":
    unittest.main()
