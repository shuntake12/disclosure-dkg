"""EDINET API v2 クライアント: 文書一覧取得・ダウンロード。

EDINET API: https://disclosure.edinet-fsa.go.jp/api/v2
Subscription Key は無料登録で取得可能。
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import httpx
from tqdm import tqdm

from schema import DocumentMeta

BASE_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"
DATA_DIR = Path(__file__).parent / "data"

# 主要な文書種別コード
DOC_TYPE_CODES = {
    "120": "有価証券報告書",
    "130": "訂正有価証券報告書",
    "140": "四半期報告書",
    "160": "半期報告書",
    "350": "決算短信",
}


def _api_key() -> str:
    key = os.environ.get("EDINET_API_KEY", "")
    if not key:
        raise RuntimeError(
            "EDINET_API_KEY が設定されていません。"
            "https://disclosure.edinet-fsa.go.jp/ で無料登録し、"
            ".env に EDINET_API_KEY=... を追加してください。"
        )
    return key


def list_documents(
    target_date: str,
    doc_type_codes: list[str] | None = None,
) -> list[DocumentMeta]:
    """指定日の開示文書一覧を取得。

    Args:
        target_date: "YYYY-MM-DD"
        doc_type_codes: フィルタする文書種別コード（None=全件）
    """
    url = f"{BASE_URL}/documents.json"
    params = {
        "date": target_date,
        "type": 2,  # 2=メタデータ付き
        "Subscription-Key": _api_key(),
    }
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    docs = []
    for r in results:
        dtc = r.get("docTypeCode", "")
        if doc_type_codes and dtc not in doc_type_codes:
            continue
        docs.append(
            DocumentMeta(
                doc_id=r.get("docID", ""),
                edinet_code=r.get("edinetCode", ""),
                sec_code=r.get("secCode", "") or "",
                filer_name=r.get("filerName", ""),
                doc_type_code=dtc,
                doc_description=r.get("docDescription", ""),
                period_start=r.get("periodStart", "") or "",
                period_end=r.get("periodEnd", "") or "",
                submit_date=r.get("submitDateTime", "")[:10] if r.get("submitDateTime") else "",
            )
        )
    return docs


def list_documents_range(
    start_date: str,
    end_date: str,
    doc_type_codes: list[str] | None = None,
    sleep_sec: float = 0.5,
) -> list[DocumentMeta]:
    """日付範囲の文書一覧を取得。"""
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_docs: list[DocumentMeta] = []
    current = start
    days = (end - start).days + 1

    for _ in tqdm(range(days), desc="Fetching document list"):
        if current > end:
            break
        try:
            docs = list_documents(current.isoformat(), doc_type_codes)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  Warning: {current} failed: {e}")
        current += timedelta(days=1)
        time.sleep(sleep_sec)

    return all_docs


def download_document(doc_id: str, download_type: int = 2) -> Path:
    """文書パッケージをダウンロードして展開。

    Args:
        doc_id: EDINET 文書ID (例: "S100XXXX")
        download_type: 1=XBRL, 2=PDF, 3=代替書面, 4=英文XBRL, 5=CSV
                       ※2 が最も汎用的（PDF + 添付）

    Returns:
        展開先ディレクトリのパス
    """
    url = f"{BASE_URL}/documents/{doc_id}"
    params = {
        "type": download_type,
        "Subscription-Key": _api_key(),
    }

    doc_dir = DATA_DIR / doc_id
    if doc_dir.exists() and any(doc_dir.iterdir()):
        print(f"  Already downloaded: {doc_id}")
        return doc_dir

    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "zip" not in content_type and "octet-stream" not in content_type:
        print(f"  Warning: Unexpected content-type for {doc_id}: {content_type}")
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "response.bin").write_bytes(resp.content)
        return doc_dir

    doc_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(doc_dir)

    return doc_dir


def download_documents(
    docs: list[DocumentMeta],
    download_type: int = 2,
    sleep_sec: float = 1.0,
) -> list[Path]:
    """複数文書を一括ダウンロード。"""
    paths = []
    for doc in tqdm(docs, desc="Downloading documents"):
        try:
            p = download_document(doc.doc_id, download_type)
            paths.append(p)
        except Exception as e:
            print(f"  Error downloading {doc.doc_id}: {e}")
        time.sleep(sleep_sec)
    return paths
