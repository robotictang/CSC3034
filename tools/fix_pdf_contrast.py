#!/usr/bin/env python3
"""Darken the lecturer pack's pale orange/cyan text without changing its layout.

Only the known heading/emphasis colours are changed. White footer text, black
body text, images, diagrams, fonts, links and text positioning are preserved.
Run after sync_lecturer_materials.py if the original pack is reimported.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject, NumberObject, StreamObject


# RGB palettes verified against the light slide backgrounds. Both replacement
# colours exceed 4.5:1 against white; cyan is also readable on the cover image.
PALETTE = {
    (247, 148, 31): (145, 67, 0),
    (255, 172, 28): (145, 67, 0),
    (242, 107, 13): (145, 67, 0),
    (38, 168, 224): (0, 90, 125),
}


def replacement(operands, operator):
    if operator != b"rg":
        return None
    for original, updated in PALETTE.items():
        if all(abs(float(value) * 255 - channel) < 2 for value, channel in zip(operands, original)):
            return [FloatObject(channel / 255) for channel in updated]
    return None


def recolor_text(operations):
    """Wrap text-show operators with a fill change, restoring the original fill.

    Restoring explicitly (rather than q/Q) preserves the text-position advance
    and works inside BT/ET. Track saved graphics state and leave non-RGB text
    alone instead of guessing how a custom colour space should be interpreted.
    """
    fill = ([NumberObject(0)], b"g")
    stack = []
    updated = []
    changes = 0
    for operands, operator in operations:
        if operator == b"q":
            stack.append(fill)
        elif operator == b"Q" and stack:
            fill = stack.pop()
        elif operator in (b"g", b"rg", b"k", b"cs", b"sc", b"scn"):
            fill = (operands, operator)
        color = replacement(*fill) if operator in (b"Tj", b"TJ", b"'", b'"') else None
        if color is not None:
            updated.extend([(color, b"rg"), (operands, operator), fill])
            changes += 1
        else:
            updated.append((operands, operator))
    return updated, changes


def fix_pdf(path: Path) -> int:
    reader = PdfReader(path)
    writer = PdfWriter(clone_from=reader)
    changes = 0
    visited = set()

    def update_forms(resources):
        nonlocal changes
        for ref in (resources.get("/XObject") or {}).values():
            form = ref.get_object()
            if form.get("/Subtype") != "/Form" or id(form) in visited:
                continue
            visited.add(id(form))
            stream = ContentStream(form, writer)
            operations, count = recolor_text(stream.operations)
            if count:
                stream.operations = operations
                # Encoded streams may use filters that do not support set_data.
                # Store the edited bytes uncompressed and remove old filters.
                for key in ("/Filter", "/DecodeParms"):
                    form.pop(NameObject(key), None)
                StreamObject.set_data(form, stream.get_data())
                if hasattr(form, "decoded_self"):
                    form.decoded_self = None
                changes += count
            update_forms(form.get("/Resources") or {})

    for page in writer.pages:
        stream = page.get_contents()
        if stream is not None:
            operations, count = recolor_text(stream.operations)
            if count:
                stream.operations = operations
                page.replace_contents(stream)
                page.compress_content_streams()
                changes += count
        update_forms(page.get("/Resources") or {})

    if changes:
        # A temporary file beside the PDF keeps replacement on one filesystem.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pdf", delete=False) as output:
            temporary = Path(output.name)
        try:
            writer.write(temporary)
            temporary.chmod(path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    args = parser.parse_args()
    for path in sorted(args.materials.rglob("*.pdf")):
        print(f"{path.name}: {fix_pdf(path)} text runs improved", flush=True)


if __name__ == "__main__":
    main()
