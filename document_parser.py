"""文書パーサー: EDINET ダウンロード済みパッケージからテキスト抽出。

HTML（インラインXBRL）からセクション分割 + XBRL数値抽出。
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from schema import DocumentMeta, TextChunk

# 抽出対象のセクション見出しキーワード（優先度順）
SECTION_KEYWORDS = [
    "経営成績",
    "財政状態",
    "キャッシュ・フロー",
    "セグメント",
    "業績予想",
    "配当",
    "事業等のリスク",
    "経営方針",
    "経営上の重要な契約",
    "研究開発",
    "設備投資",
    "従業員",
    "役員",
    "コーポレート・ガバナンス",
    "サステナビリティ",
    "ESG",
    "連結経営成績",
    "連結財政状態",
    "今後の見通し",
]


def find_html_files(doc_dir: Path) -> list[Path]:
    """文書ディレクトリ内のHTML/HTMファイルを検出。"""
    files = []
    for ext in ("*.htm", "*.html", "*.xhtml"):
        files.extend(doc_dir.rglob(ext))
    # メインの開示文書を優先（ファイル名にhonbun/本文を含むもの）
    files.sort(key=lambda p: (0 if "honbun" in p.name.lower() or "本文" in p.name else 1, p.name))
    return files


def extract_text_from_html(html_path: Path) -> str:
    """HTMLファイルからテキストを抽出。"""
    content = html_path.read_bytes()
    # エンコーディング検出
    for enc in ("utf-8", "shift_jis", "cp932", "euc-jp"):
        try:
            text = content.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(text, "html.parser")

    # スクリプト・スタイルを除去
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def split_into_sections(full_text: str) -> list[tuple[str, str]]:
    """全文をセクション見出しで分割。

    Returns: [(section_name, section_text), ...]
    """
    sections: list[tuple[str, str]] = []

    # 見出しパターン: 数字+括弧+キーワード or キーワード単体で始まる行
    heading_pattern = re.compile(
        r"^[\s\d\(\)（）①-⑳]*("
        + "|".join(re.escape(kw) for kw in SECTION_KEYWORDS)
        + r")",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        # セクション分割できない場合は全文を1セクションとして返す
        return [("全文", full_text)]

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_name = m.group(1).strip()
        section_text = full_text[start:end].strip()
        if len(section_text) > 50:  # 短すぎるセクションを除外
            sections.append((section_name, section_text))

    return sections


def chunk_text(
    text: str,
    max_chars: int = 3000,
    overlap: int = 200,
) -> list[tuple[int, int, str]]:
    """テキストを重複付きチャンクに分割。

    Returns: [(char_start, char_end, chunk_text), ...]
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # 文末（。）で切る
        if end < len(text):
            last_period = text.rfind("。", start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 1
        chunks.append((start, end, text[start:end]))
        start = end - overlap if end < len(text) else end
    return chunks


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 30) -> str:
    """PDFファイルからテキストを抽出（PyMuPDF使用）。"""
    try:
        import pymupdf
    except ImportError:
        print(f"  pymupdf not available, skipping PDF: {pdf_path.name}")
        return ""

    try:
        doc = pymupdf.open(str(pdf_path))
        pages = min(len(doc), max_pages)
        text_parts = []
        for i in range(pages):
            page = doc[i]
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  PDF parse error {pdf_path.name}: {e}")
        return ""


def find_pdf_files(doc_dir: Path) -> list[Path]:
    """文書ディレクトリ内のPDFファイルを検出。"""
    files = list(doc_dir.rglob("*.pdf"))
    # response.bin もPDFの場合がある
    for bin_file in doc_dir.rglob("response.bin"):
        files.append(bin_file)
    return files


def parse_document(
    doc_dir: Path,
    doc_id: str = "",
    max_chunk_chars: int = 3000,
) -> list[TextChunk]:
    """文書ディレクトリからテキストチャンクを抽出。

    HTML → PDF の順で試行する。
    Returns: TextChunk のリスト（セクション情報 + 文字オフセット付き）
    """
    if not doc_id:
        doc_id = doc_dir.name

    all_chunks: list[TextChunk] = []

    # まずHTMLファイルを試行
    html_files = find_html_files(doc_dir)
    if html_files:
        for html_path in html_files[:5]:
            full_text = extract_text_from_html(html_path)
            if len(full_text) < 100:
                continue
            sections = split_into_sections(full_text)
            for section_name, section_text in sections:
                section_start = full_text.find(section_text[:50])
                if section_start < 0:
                    section_start = 0
                chunks = chunk_text(section_text, max_chars=max_chunk_chars)
                for chunk_start, chunk_end, chunk_text_content in chunks:
                    all_chunks.append(
                        TextChunk(
                            doc_id=doc_id,
                            section=section_name,
                            text=chunk_text_content,
                            char_start=section_start + chunk_start,
                            char_end=section_start + chunk_end,
                        )
                    )
        if all_chunks:
            return all_chunks

    # HTMLがない場合はPDFを試行
    pdf_files = find_pdf_files(doc_dir)
    if not pdf_files:
        print(f"  No HTML or PDF files found in {doc_dir}")
        return []

    for pdf_path in pdf_files[:2]:
        full_text = extract_text_from_pdf(pdf_path)
        if len(full_text) < 100:
            continue

        sections = split_into_sections(full_text)
        for section_name, section_text in sections:
            section_start = full_text.find(section_text[:50])
            if section_start < 0:
                section_start = 0
            chunks = chunk_text(section_text, max_chars=max_chunk_chars)
            for chunk_start, chunk_end, chunk_text_content in chunks:
                all_chunks.append(
                    TextChunk(
                        doc_id=doc_id,
                        section=section_name,
                        text=chunk_text_content,
                        char_start=section_start + chunk_start,
                        char_end=section_start + chunk_end,
                    )
                )

    return all_chunks


def extract_xbrl_numeric_facts(doc_dir: Path) -> list[dict]:
    """XBRLファイルから数値事実を抽出（簡易版）。

    lxml が利用可能な場合に機能する。利用不可の場合は空リストを返す。
    """
    try:
        from lxml import etree
    except ImportError:
        print("  lxml not available, skipping XBRL extraction")
        return []

    xbrl_files = list(doc_dir.rglob("*.xbrl")) + list(doc_dir.rglob("*.xml"))
    facts = []

    # 主要な財務指標のタクソノミ要素名（部分一致）
    target_elements = {
        "NetSales": "revenue_is",
        "OperatingIncome": "operating_profit_is",
        "OrdinaryIncome": "operating_profit_is",
        "NetIncome": "net_income_is",
        "ProfitLoss": "net_income_is",
        "TotalAssets": "total_assets_is",
        "NetAssets": "net_assets_is",
        "EarningsPerShare": "eps_is",
        "DividendPerShare": "dividend_is",
    }

    for xbrl_path in xbrl_files[:3]:
        try:
            tree = etree.parse(str(xbrl_path))
            root = tree.getroot()
            nsmap = root.nsmap

            for elem in root.iter():
                tag_local = etree.QName(elem.tag).localname if elem.tag else ""
                for target, relation in target_elements.items():
                    if target.lower() in tag_local.lower():
                        text_val = (elem.text or "").strip()
                        if text_val and text_val.replace("-", "").replace(".", "").isdigit():
                            context_ref = elem.get("contextRef", "")
                            unit_ref = elem.get("unitRef", "")
                            facts.append({
                                "element": tag_local,
                                "relation": relation,
                                "value": text_val,
                                "context": context_ref,
                                "unit": unit_ref,
                            })
        except Exception as e:
            print(f"  XBRL parse error {xbrl_path.name}: {e}")

    return facts
