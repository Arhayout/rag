"""Download available official Adala PDFs listed in the source registry."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "moroccan_legal_sources.json"
DESTINATION = ROOT / "data" / "morocco"


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    downloaded = 0
    skipped: list[str] = []

    for item in registry["documents"]:
        for language in ("ar", "fr"):
            url = item.get(f"pdf_{language}")
            if not url:
                skipped.append(f"{item['id']} ({language}): no direct PDF URL registered")
                continue

            folder = DESTINATION / language
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{item['id']}.{language}.pdf"
            if target.exists() and target.stat().st_size > 0:
                print(f"[exists] {target.relative_to(ROOT)}")
                continue

            request = Request(url, headers={"User-Agent": "MoroccanLegalRAG/1.0"})
            try:
                with urlopen(request, timeout=45) as response:
                    content_type = response.headers.get("Content-Type", "")
                    payload = response.read()
                if not payload.startswith(b"%PDF-"):
                    raise ValueError(f"Response is not a PDF (Content-Type: {content_type})")
                target.write_bytes(payload)
                downloaded += 1
                print(f"[downloaded] {target.relative_to(ROOT)} ({len(payload):,} bytes)")
            except Exception as exc:  # keep other sources downloadable if one link fails
                print(f"[failed] {item['id']} ({language}): {exc}", file=sys.stderr)
                skipped.append(f"{item['id']} ({language}): download failed")

    print(f"\nDownloaded: {downloaded}")
    if skipped:
        print("Needs manual review:")
        for entry in skipped:
            print(f"- {entry}")
    print("\nCheck each document's consolidation date and amendments before legal use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
