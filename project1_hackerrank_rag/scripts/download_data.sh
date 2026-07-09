#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/HackerRank_2025_Developer_Skills_Report.pdf"
URL="https://pages.hackerrank.com/hubfs/PDFs/HackerRank%202025%20Developer%20Skills%20Report.pdf"

mkdir -p "$ROOT/data"
curl -fsSL "$URL" -o "$OUT"
echo "Downloaded: $OUT"