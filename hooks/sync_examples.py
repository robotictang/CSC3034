"""Keep the long Python examples embedded in lab pages synchronized.

Run ``python hooks/sync_examples.py`` after editing a mapped downloadable file,
or pass ``--check`` in CI to verify that no page contains a stale copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "src/lab1.md": "src/files/isaac_hexapod_drl.py",
    "src/lab2.md": "src/files/isaac_fuzzy_robot.py",
    "src/lab3.md": "src/files/isaac_ga_robot.py",
    "src/lab4.md": "src/files/isaac_pso_swarm.py",
    "src/lab5.md": "src/files/isaac_aco_route.py",
    "src/lab8a.md": "src/files/isaac_vision_classifier.py",
    "src/lab8b.md": "src/files/isaac_vision_detection.py",
}


def synchronized_page(markdown: str, source: str) -> str:
    """Replace the fenced block whose first line is the copyright marker."""
    lines = markdown.splitlines()
    source_lines = source.rstrip("\n").splitlines()

    for start in range(len(lines) - 1):
        if lines[start].startswith("```python") and lines[start + 1].startswith("# Copyright Author:"):
            for end in range(start + 2, len(lines)):
                if lines[end] == "```":
                    updated = lines[: start + 1] + source_lines + lines[end:]
                    return "\n".join(updated) + "\n"
            raise ValueError("embedded Python block has no closing fence")

    raise ValueError("embedded Python block with copyright marker was not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report stale pages without changing them")
    args = parser.parse_args()

    stale = []
    for page_name, source_name in EXAMPLES.items():
        page = ROOT / page_name
        source = ROOT / source_name
        current = page.read_text(encoding="utf-8")
        expected = synchronized_page(current, source.read_text(encoding="utf-8"))
        if current != expected:
            stale.append(page_name)
            if not args.check:
                page.write_text(expected, encoding="utf-8")

    if stale:
        action = "Stale embedded examples" if args.check else "Updated embedded examples"
        print(f"{action}: {', '.join(stale)}")
    return int(args.check and bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
