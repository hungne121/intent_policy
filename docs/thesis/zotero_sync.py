"""Lấy collection khóa luận từ Zotero (local API) ghi ra references.bib.

latexmkrc gọi script này trước mỗi lần build. Cần Zotero đang mở và đã bật
Settings → Advanced → "Allow other applications on this computer to communicate with Zotero".
Zotero tắt, lỗi mạng hoặc collection rỗng: giữ nguyên references.bib, build vẫn chạy.
Chạy tay: python3 zotero_sync.py
"""
import re
import sys
import urllib.request
from pathlib import Path

COLLECTION = "4RRF6AG7"  # key của collection khóa luận, xem bằng /api/users/0/collections
API = "http://localhost:23119/api/users/0"
OUT = Path(__file__).resolve().parent / "references.bib"
PAGE = 100
HEADER = ("% TỰ SINH từ Zotero bởi zotero_sync.py, mỗi lần build sẽ ghi đè.\n"
          "% Sửa tài liệu trong Zotero, không sửa file này.\n\n")


def fetch(start):
    url = f"{API}/collections/{COLLECTION}/items/top?format=biblatex&limit={PAGE}&start={start}"
    with urllib.request.urlopen(url, timeout=3) as r:
        return int(r.headers.get("Total-Results", 0)), r.read().decode("utf-8")


def main():
    try:
        total, text = fetch(0)
        parts = [text]
        for start in range(PAGE, total, PAGE):
            parts.append(fetch(start)[1])
    except Exception as e:
        print(f"zotero_sync: không lấy được từ Zotero ({e}); giữ nguyên {OUT.name}", file=sys.stderr)
        return
    # Trường file chứa đường dẫn PDF trên máy, không cần cho LaTeX.
    body = re.sub(r"^\s*file = \{.*\},\n", "", "\n".join(p.strip() for p in parts if p.strip()), flags=re.M)
    n = len(re.findall(r"^@", body, flags=re.M))  # PDF đứng riêng không có mục BibLaTeX
    if n == 0:
        print(f"zotero_sync: collection {COLLECTION} rỗng; giữ nguyên {OUT.name}", file=sys.stderr)
        return
    new = HEADER + body + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") == new:
        return
    OUT.write_text(new, encoding="utf-8")
    print(f"zotero_sync: ghi {n} mục vào {OUT.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
