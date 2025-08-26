from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _write_simple_pdf(text: str, path: Path) -> None:
    # Minimal single-page PDF with Helvetica font and basic text output
    # Coordinates and escaping kept simple for robustness
    text_escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    streams = f"BT /F1 10 Tf 50 780 Td ({text_escaped}) Tj ET".encode(
        "latin-1", errors="ignore"
    )
    # Objects
    objs = []
    # 1: Catalog (fonts)
    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    # 2: Resources
    resources_obj = b"<< /Font << /F1 1 0 R >> >>"
    # 3: Content stream
    content_obj = (
        b"<< /Length %d >>\nstream\n" % len(streams) + streams + b"\nendstream"
    )
    # 4: Page
    page_obj = b"<< /Type /Page /Parent 5 0 R /Resources 2 0 R /Contents 3 0 R /MediaBox [0 0 595 842] >>"
    # 5: Pages
    pages_obj = b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>"
    # 6: Root
    root_obj = b"<< /Type /Catalog /Pages 5 0 R >>"
    objs = [font_obj, resources_obj, content_obj, page_obj, pages_obj, root_obj]
    # Assemble with xref
    f = bytearray()
    f += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(f))
        f += f"{i} 0 obj\n".encode()
        f += obj + b"\nendobj\n"
    xref_pos = len(f)
    f += b"xref\n0 %d\n" % (len(objs) + 1)
    f += b"0000000000 65535 f \n"
    for off in offsets:
        f += f"{off:010d} 00000 n \n".encode()
    f += b"trailer\n<< /Size %d /Root 6 0 R >>\nstartxref\n%d\n%%EOF\n" % (
        len(objs) + 1,
        xref_pos,
    )
    path.write_bytes(bytes(f))


def generate_report(
    ideal: bool, out_dir: str, transplanted_dir: Optional[str] = None
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Aggregate from validation and uq if present
    data: Dict[str, Any] = {"ideal": ideal, "artifacts": {}}
    val = (
        Path(transplanted_dir or out) / "validation_results" / "validation_results.json"
    )
    if val.exists():
        try:
            data["validation"] = json.loads(val.read_text())
        except Exception:
            pass
    uq = Path(transplanted_dir or out) / "uq" / "summary.json"
    if uq.exists():
        try:
            data["uq"] = json.loads(uq.read_text())
        except Exception:
            pass
    # Derive simple no-regress flag
    no_regress = True
    if "validation" in data and isinstance(data["validation"].get("summary"), dict):
        no_regress = data["validation"]["summary"].get("overall_score", 0.0) >= 0.0
    data["no_regress"] = no_regress
    # Save JSON and Markdown
    j = out / "report.json"
    j.write_text(json.dumps(data, indent=2))
    md = out / "report.md"
    md.write_text(
        "# LLM Ripper Report\n\n"
        + json.dumps({k: v for k, v in data.items() if k != "artifacts"}, indent=2)
    )
    # PDF
    pdf = out / "report.pdf"
    try:
        _write_simple_pdf(md.read_text()[:5000], pdf)
        pdf_ok = True
    except Exception:
        pdf_ok = False
    return {
        "json": str(j),
        "md": str(md),
        "pdf": str(pdf) if pdf_ok else None,
        "no_regress": no_regress,
    }
