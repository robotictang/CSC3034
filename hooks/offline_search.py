"""Create the search index required when the generated site is opened locally."""

from pathlib import Path


def on_post_build(config, **kwargs):
    """Expose the JSON index as a script for Material's file:// search fallback."""
    search_dir = Path(config.site_dir) / "search"
    index = (search_dir / "search_index.json").read_text(encoding="utf-8")
    (search_dir / "search_index.js").write_text(
        f"var __index = {index};\n", encoding="utf-8"
    )
