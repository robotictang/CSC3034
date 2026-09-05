"""Protect slide text and graphics when improving the lecturer PDF palette."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fix_pdf_contrast import PALETTE, recolor_text  # noqa: E402


class PdfContrastTests(unittest.TestCase):
    def test_only_text_fill_changes_and_graphics_state_is_restored(self):
        orange = ([247 / 255, 148 / 255, 31 / 255], b"rg")
        show = (["Heading"], b"Tj")
        original = [orange, ([], b"BT"), show, ([], b"ET"), ([], b"f")]
        updated, count = recolor_text(original)
        self.assertEqual(count, 1)
        self.assertEqual(updated[3], show)
        self.assertEqual(updated[4], orange)
        self.assertEqual(updated[-2:], original[-2:])
        self.assertEqual(recolor_text(updated), (updated, 0))

    def test_saved_fill_and_white_footer_are_preserved(self):
        orange = ([1, 172 / 255, 28 / 255], b"rg")
        original = [orange, ([], b"q"), ([1], b"g"), (["Footer"], b"Tj"),
                    ([], b"Q"), (["Title"], b"Tj")]
        updated, count = recolor_text(original)
        self.assertEqual(count, 1)
        self.assertEqual(updated[:5], original[:5])
        self.assertEqual(updated[-1], orange)

    def test_replacement_palette_is_readable_on_white(self):
        for rgb in PALETTE.values():
            linear = [v / 255 / 12.92 if v / 255 <= 0.04045
                      else ((v / 255 + 0.055) / 1.055) ** 2.4 for v in rgb]
            luminance = sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722)))
            self.assertGreaterEqual(1.05 / (luminance + 0.05), 4.5)


if __name__ == "__main__":
    unittest.main()
