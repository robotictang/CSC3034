#!/usr/bin/env python3
"""Synchronize the legacy lecturer pack with the current CSC3034 materials.

This utility deliberately keeps the original lecture pages intact. It adds a
dated course-update page to every legacy PDF, replaces the former lecturer name
on PDF title pages where it occurs, normalizes PDF/XLSX author metadata, updates
the legacy lab-solution entry points from ``src/files``, and copies the current
Isaac Sim examples into the appropriate lab folders.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject


AUTHOR = "Dr Tang Tiong Yew"
AUTHOR_OVERLAY_MARKER = f"{AUTHOR}; legacy references removed; v3"
BRIEFING_DELIVERY_MARKER = "CSC3034 briefing delivery details updated; v4"
SYNC_DATE = "14 August 2026"
OLD_AUTHOR_PATTERN = re.compile(r"Dr\.?\s+Richard\s+Wong(?:\s+Teck\s+Ken)?", re.I)
logging.getLogger("pypdf").setLevel(logging.ERROR)

SOLUTION_MAP = {
    "Lab Solutions/Lab1/Lab1.py": "lab1.py",
    "Lab Solutions/Lab2/train_fuzzy_system.py": "lab2_fuzzy.py",
    "Lab Solutions/Lab3/ga.py": "lab3_ga.py",
    "Lab Solutions/Lab4/pso.py": "lab4_pso.py",
    "Lab Solutions/Lab5/aco.py": "lab5_aco.py",
    "Lab Solutions/Lab6/ann.py": "ann_wine_mlp.py",
    "Lab Solutions/Lab7/lab7.py": "ann_hyperplane.py",
    "Lab Solutions/Lab7/vis.py": "vis.py",
}

ISAAC_MAP = {
    "Lab Solutions/Lab1/isaac_hexapod_drl.py": "isaac_hexapod_drl.py",
    "Lab Solutions/Lab2/isaac_fuzzy_robot.py": "isaac_fuzzy_robot.py",
    "Lab Solutions/Lab3/isaac_ga_robot.py": "isaac_ga_robot.py",
    "Lab Solutions/Lab4/isaac_pso_swarm.py": "isaac_pso_swarm.py",
    "Lab Solutions/Lab5/isaac_aco_route.py": "isaac_aco_route.py",
    "Lab Solutions/Lab8/isaac_vision_classifier.py": "isaac_vision_classifier.py",
    "Lab Solutions/Lab8/isaac_vision_detection.py": "isaac_vision_detection.py",
    "Lab Solutions/Lab8/lab8a_keras_cnn.py": "lab8a_keras_cnn.py",
    "Lab Solutions/Lab8/lab8b_keras_lstm.py": "lab8b_keras_lstm.py",
}

UPDATE_LINES = {
    "briefing": [
        "Current delivery: eight practicals scheduled from Week 2 to Week 12.",
        "Isaac Sim 6.0 uses workstation archives; Omniverse Launcher is retired.",
        "CPU/Matplotlib fallbacks remain available when an RTX workstation is unavailable.",
    ],
    "fuzzy": [
        "Current Lab 2 uses scikit-fuzzy to build and evaluate Mamdani control systems.",
        "The physical-AI extension applies fuzzy brake/throttle logic to a simulated robot.",
        "The updated solution checks optional dependencies and supports a standalone fallback.",
    ],
    "genetic": [
        "Current Lab 3 covers a complete GA workflow and continuous waypoint encoding.",
        "The Isaac Sim extension evolves collision-aware mobile-robot trajectories.",
        "A Matplotlib fallback demonstrates the optimization without simulator hardware.",
    ],
    "swarm": [
        "Current Labs 4 and 5 cover PSO target search and ACO route finding.",
        "Physical examples account for robot speed limits, obstacles, and waypoint motion.",
        "Current runnable scripts are included in the synchronized Lab Solutions folders.",
    ],
    "neural": [
        "Current Lab 6 trains and evaluates a scikit-learn MLP on the Wine dataset.",
        "Current Lab 7 visualizes neural-network hyperplanes on standardized Iris features.",
        "Current Lab 8 adds Keras CNN/LSTM examples and Isaac Sim vision demonstrations.",
    ],
    "recurrent": [
        "Current Lab 8b uses a Keras Embedding/LSTM pipeline for IMDb sentiment analysis.",
        "The practical covers preprocessing, training, evaluation, and prediction.",
        "Deep-learning requirements are separated from the core lab environment.",
    ],
    "metrics": [
        "Current classification examples report test accuracy, confusion matrices, and reports.",
        "Students should fit preprocessing only on training data and evaluate held-out data.",
        "The synchronized examples use reproducible train/test workflows where applicable.",
    ],
    "default": [
        "Current labs connect core CI algorithms with physical-AI simulation examples.",
        "Isaac Sim examples use its bundled Python 3.12 environment and current launchers.",
        "Every simulator-dependent example includes a CPU/Matplotlib learning fallback.",
    ],
}


def run(*args: str) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ps_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def update_lines_for(filename: str) -> list[str]:
    name = filename.lower()
    if "briefing" in name:
        key = "briefing"
    elif "fuzzy" in name:
        key = "fuzzy"
    elif "genetic" in name:
        key = "genetic"
    elif "swarm" in name:
        key = "swarm"
    elif "recurrent" in name:
        key = "recurrent"
    elif "neural" in name or "hebbian" in name or "mlnn" in name or "unsupervised" in name:
        key = "neural"
    elif "metric" in name:
        key = "metrics"
    else:
        key = "default"
    return UPDATE_LINES[key]


def ps_to_pdf(source: Path, destination: Path, width: float, height: float) -> None:
    run(
        "ps2pdf",
        "-dFIXEDMEDIA",
        f"-dDEVICEWIDTHPOINTS={width}",
        f"-dDEVICEHEIGHTPOINTS={height}",
        str(source),
        str(destination),
    )


def make_supplement(path: Path, title: str, lines: list[str]) -> None:
    content = [
        "%!PS-Adobe-3.0",
        "<< /PageSize [595 420] >> setpagedevice",
        "/Helvetica findfont 12 scalefont setfont",
        "0.05 0.20 0.38 setrgbcolor",
        "55 345 moveto (CSC3034 Current Course Update) show",
        "/Helvetica-Bold findfont 20 scalefont setfont",
        f"55 305 moveto ({ps_escape(title)}) show",
        "/Helvetica findfont 10 scalefont setfont",
        "0.10 0.10 0.10 setrgbcolor",
        f"55 276 moveto (Synchronized from the current CSC3034 course site on {SYNC_DATE}.) show",
    ]
    y = 232
    for line in lines:
        content.append(f"70 {y} moveto (\\267  {ps_escape(line)}) show")
        y -= 34
    content.extend(
        [
            "/Helvetica-Bold findfont 11 scalefont setfont",
            "0.05 0.20 0.38 setrgbcolor",
            f"55 62 moveto (Current materials author: {ps_escape(AUTHOR)}) show",
            "/Helvetica findfont 9 scalefont setfont",
            "55 40 moveto (See the synchronized lab notes and runnable examples for full instructions.) show",
            "showpage",
        ]
    )
    path.write_text("\n".join(content) + "\n", encoding="ascii")


def make_overlay(path: Path, box: tuple[float, float, float, float, float, float]) -> None:
    width, height, x_min, y_min, x_max, y_max = box
    left = x_min - 2
    bottom = height - y_max - 2
    rect_width = max(125, x_max - x_min + 12)
    rect_height = y_max - y_min + 5
    text_y = bottom + 3
    content = [
        "%!PS-Adobe-3.0",
        f"<< /PageSize [{width} {height}] >> setpagedevice",
        "0.91 0.95 0.97 setrgbcolor",
        f"{left:.2f} {bottom:.2f} {rect_width:.2f} {rect_height:.2f} rectfill",
        "0.05 0.20 0.38 setrgbcolor",
        "/Helvetica findfont 10 scalefont setfont",
        f"{x_min:.2f} {text_y:.2f} moveto ({ps_escape(AUTHOR)}) show",
        "showpage",
    ]
    path.write_text("\n".join(content) + "\n", encoding="ascii")


def make_briefing_contact_page(path: Path, width: float, height: float) -> None:
    content = [
        "%!PS-Adobe-3.0",
        f"<< /PageSize [{width} {height}] >> setpagedevice",
        "0.95 0.42 0.05 setrgbcolor",
        "/Helvetica-Bold findfont 18 scalefont setfont",
        "14 222 moveto (My info) show",
        "0.05 0.20 0.38 setrgbcolor",
        "/Helvetica-Bold findfont 14 scalefont setfont",
        f"160 154 moveto ({ps_escape(AUTHOR)}) show",
        "0.12 0.12 0.12 setrgbcolor",
        "/Helvetica findfont 11 scalefont setfont",
        "111 124 moveto (Current contact details are available on the CSC3034 course site.) show",
        "/Helvetica findfont 9 scalefont setfont",
        "170 98 moveto (Department of Computing and Information Systems) show",
        "0.05 0.20 0.38 setrgbcolor",
        "0 0 226 7 rectfill",
        "0.10 0.65 0.78 setrgbcolor",
        "226 0 228 7 rectfill",
        "showpage",
    ]
    path.write_text("\n".join(content) + "\n", encoding="ascii")


def update_briefing_delivery_details(page) -> bool:
    """Replace the legacy tutor and course-site text in the original slide stream."""
    contents = page.raw_get("/Contents").get_object()
    streams = list(contents) if isinstance(contents, ArrayObject) else [contents]
    for stream_ref in streams:
        stream = stream_ref.get_object()
        content = stream.get_data()
        if (
            b"https://ricwtk.github.io/ci-" not in content
            and b"https://robotictang.github.io/CSC3034/docs/" not in content
        ):
            continue
        content = content.replace(
            b"-358(Lim)-307(W)97(ei)-307(Lun)",
            b"-358(Dr)-307(Tang)-307(Tiong)-307(Yew)",
        )
        content = content.replace(
            b"(https://ricwtk.github.io/ci-)-55(labs)",
            b"()",
        )
        # The legacy slide font lacks several glyphs needed by the new URL.
        content = content.replace(b"(https://robotictang.github.io/CSC3034/docs/)", b"()")
        stream.set_data(content)
        # Keep only the edited original stream, discarding a prior overlay if present.
        page[NameObject("/Contents")] = ArrayObject([stream_ref])
        return True
    return False


def remove_briefing_legacy_link(page) -> None:
    """Remove the obsolete ci-labs hyperlink annotation from the briefing slide."""
    annotations = page.get("/Annots") or []
    retained = ArrayObject()
    for annotation_ref in annotations:
        annotation = annotation_ref.get_object()
        uri = str((annotation.get("/A") or {}).get("/URI", ""))
        if uri != "https://ricwtk.github.io/ci-labs":
            retained.append(annotation_ref)
    if retained:
        page[NameObject("/Annots")] = retained
    elif NameObject("/Annots") in page:
        del page[NameObject("/Annots")]


def make_briefing_url_overlay(path: Path, width: float, height: float) -> None:
    """Add the new course URL using a standard font with complete URL glyph support."""
    content = [
        "%!PS-Adobe-3.0",
        f"<< /PageSize [{width} {height}] >> setpagedevice",
        "0.12 0.12 0.12 setrgbcolor",
        "/Courier findfont 8 scalefont setfont",
        "174 90 moveto (https://robotictang.github.io/CSC3034/docs/) show",
        "showpage",
    ]
    path.write_text("\n".join(content) + "\n", encoding="ascii")

def author_boxes(pdf: Path) -> list[tuple[int, float, float, float, float, float, float]]:
    result = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = []
    page_pattern = re.compile(
        r'<page width="([0-9.]+)" height="([0-9.]+)">(.*?)</page>', re.S
    )
    block_pattern = re.compile(
        r'<block xMin="([0-9.]+)" yMin="([0-9.]+)" xMax="([0-9.]+)" yMax="([0-9.]+)">(.*?)</block>',
        re.S,
    )
    for page_index, page_match in enumerate(page_pattern.finditer(result.stdout)):
        width = float(page_match.group(1))
        height = float(page_match.group(2))
        for block in block_pattern.finditer(page_match.group(3)):
            words = [
                html.unescape(word).strip().rstrip(".").lower()
                for word in re.findall(r"<word[^>]*>(.*?)</word>", block.group(5), re.S)
            ]
            if words[:3] == ["dr", "richard", "wong"]:
                matches.append(
                    (
                        page_index,
                        width,
                        height,
                        float(block.group(1)),
                        float(block.group(2)),
                        float(block.group(3)),
                        float(block.group(4)),
                    )
                )
    return matches


def update_pdf(pdf: Path) -> None:
    existing = PdfReader(pdf)
    existing_metadata = existing.metadata or {}
    has_supplement = str(existing_metadata.get("/Subject", "")).startswith("Current CSC3034 update synchronized")
    has_all_overlays = existing_metadata.get("/CSC3034AuthorOverlay") == AUTHOR_OVERLAY_MARKER
    has_briefing_delivery_update = existing_metadata.get("/CSC3034BriefingDelivery") == BRIEFING_DELIVERY_MARKER
    needs_briefing_delivery_update = pdf.name == "00 Briefing.pdf" and not has_briefing_delivery_update
    if has_supplement and has_all_overlays and not needs_briefing_delivery_update:
        return
    with tempfile.TemporaryDirectory(prefix="csc3034-pdf-") as tmp_name:
        tmp = Path(tmp_name)
        supplement_ps = tmp / "update.ps"
        supplement_pdf = tmp / "update.pdf"
        output_pdf = tmp / "output.pdf"
        if not has_supplement:
            make_supplement(supplement_ps, pdf.stem, update_lines_for(pdf.name))
            ps_to_pdf(supplement_ps, supplement_pdf, 595, 420)

        reader = PdfReader(pdf)
        writer = PdfWriter()
        writer.append(reader)
        for page_index, width, height, x_min, y_min, x_max, y_max in author_boxes(pdf):
            box = (width, height, x_min, y_min, x_max, y_max)
            overlay_ps = tmp / f"overlay-{page_index}.ps"
            overlay_pdf = tmp / f"overlay-{page_index}.pdf"
            make_overlay(overlay_ps, box)
            ps_to_pdf(overlay_ps, overlay_pdf, width, height)
            writer.pages[page_index].merge_page(PdfReader(overlay_pdf).pages[0], over=True)
        if pdf.name == "00 Briefing.pdf" and len(writer.pages) >= 2:
            page = writer.pages[1]
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            contact_ps = tmp / "briefing-contact.ps"
            contact_pdf = tmp / "briefing-contact.pdf"
            make_briefing_contact_page(contact_ps, width, height)
            ps_to_pdf(contact_ps, contact_pdf, width, height)
            writer.remove_page(1)
            writer.insert_page(PdfReader(contact_pdf).pages[0], 1)
        if needs_briefing_delivery_update and len(writer.pages) >= 5:
            page = writer.pages[4]
            if update_briefing_delivery_details(page):
                delivery_ps = tmp / "briefing-delivery-url.ps"
                delivery_pdf = tmp / "briefing-delivery-url.pdf"
                make_briefing_url_overlay(
                    delivery_ps,
                    float(page.mediabox.width),
                    float(page.mediabox.height),
                )
                ps_to_pdf(
                    delivery_ps,
                    delivery_pdf,
                    float(page.mediabox.width),
                    float(page.mediabox.height),
                )
                page.merge_page(PdfReader(delivery_pdf).pages[0], over=True)
            remove_briefing_legacy_link(page)
        if not has_supplement:
            writer.append(PdfReader(supplement_pdf))
        metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items() if value is not None}
        metadata["/Author"] = AUTHOR
        metadata["/Subject"] = f"Current CSC3034 update synchronized {SYNC_DATE}"
        metadata["/CSC3034AuthorOverlay"] = AUTHOR_OVERLAY_MARKER
        if pdf.name == "00 Briefing.pdf":
            metadata["/CSC3034BriefingDelivery"] = BRIEFING_DELIVERY_MARKER
        writer.add_metadata(metadata)
        with output_pdf.open("wb") as stream:
            writer.write(stream)
        os.replace(output_pdf, pdf)


def update_xlsx(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="csc3034-xlsx-") as tmp_name:
        output = Path(tmp_name) / path.name
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(output, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename.endswith(".xml"):
                    text = data.decode("utf-8")
                    text = OLD_AUTHOR_PATTERN.sub(AUTHOR, text)
                    text = re.sub(r'(<x15ac:absPath\b[^>]*\burl=")[^"]*(")', r"\1\2", text)
                    data = text.encode("utf-8")
                if item.filename == "docProps/core.xml":
                    text = data.decode("utf-8")
                    text = re.sub(
                        r"(<dc:creator>).*?(</dc:creator>)",
                        rf"\1{html.escape(AUTHOR)}\2",
                        text,
                    )
                    text = re.sub(
                        r"(<cp:lastModifiedBy>).*?(</cp:lastModifiedBy>)",
                        rf"\1{html.escape(AUTHOR)}\2",
                        text,
                    )
                    data = text.encode("utf-8")
                target.writestr(item, data)
        os.replace(output, path)


def update_remaining_python_files(materials: Path, mapped: set[Path]) -> None:
    for path in materials.rglob("*.py"):
        if path in mapped:
            continue
        content = path.read_text(encoding="utf-8-sig")
        content = OLD_AUTHOR_PATTERN.sub(AUTHOR, content)
        content = re.sub(r"(?im)^\s*#?\s*@author:\s*(?:ricwtk|richardwtk)\s*$", f"@author: {AUTHOR}", content)
        if AUTHOR not in content[:500]:
            content = f"# Copyright Author: {AUTHOR}\n" + content
        path.write_text(content, encoding="utf-8", newline="")


def sync_code(repo: Path, materials: Path) -> None:
    source_dir = repo / "src" / "files"
    mapped: set[Path] = set()
    for relative, source_name in {**SOLUTION_MAP, **ISAAC_MAP}.items():
        destination = materials / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / source_name, destination)
        mapped.add(destination)
    update_remaining_python_files(materials, mapped)
    for cache_dir in materials.rglob("__pycache__"):
        shutil.rmtree(cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materials", type=Path, help="Legacy lecturer-materials directory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    materials = args.materials.resolve()
    sync_code(repo, materials)
    for pdf in sorted(materials.rglob("*.pdf")):
        update_pdf(pdf)
    for workbook in sorted(materials.rglob("*.xlsx")):
        update_xlsx(workbook)


if __name__ == "__main__":
    main()
