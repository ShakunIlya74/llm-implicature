"""Export text + high-DPI page images from the G&S CogSci PDF in paper/.

Does not extract bar heights (those are only in figures). Run from repo root:

  PYTHONPATH=code python code/pipeline_v1/scripts/export_gs2013_pdf_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install PyMuPDF: pip install pymupdf") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="paper/Goodman_Stuhlmueller_2012_cogsci_implicature.pdf",
        help="Path to PDF relative to repo root",
    )
    parser.add_argument(
        "--out-dir",
        default="paper",
        help="Directory containing figures/ and extracted text",
    )
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument(
        "--pages",
        default="1-6",
        help="1-based page range, e.g. 1-6 or 2-4",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    pdf_path = root / args.pdf
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = root / args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def parse_pages(spec: str, n_pages: int) -> list[int]:
        if "-" in spec:
            a, b = spec.split("-", 1)
            lo, hi = int(a), int(b)
            return list(range(lo - 1, min(hi, n_pages)))
        return [int(spec) - 1]

    doc = fitz.open(str(pdf_path))
    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    full_text_parts: list[str] = []
    for i in range(doc.page_count):
        full_text_parts.append(doc[i].get_text())
    (out_dir / "Goodman_Stuhlmueller_2012_extracted_text.txt").write_text(
        "\n".join(full_text_parts), encoding="utf-8"
    )

    page_indices = parse_pages(args.pages, doc.page_count)
    for idx in page_indices:
        page = doc[idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_png = fig_dir / f"gs2012_page{idx + 1}_{int(args.dpi)}dpi.png"
        pix.save(str(out_png))
        print(f"Wrote {out_png.relative_to(root)} ({pix.width}x{pix.height})")

    doc.close()
    print(f"Wrote text -> {(out_dir / 'Goodman_Stuhlmueller_2012_extracted_text.txt').relative_to(root)}")


if __name__ == "__main__":
    main()
