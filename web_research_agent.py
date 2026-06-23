"""Web リサーチエージェント: 実際のニュース記事を取得して五つ組を抽出。

news_agent.py がFoundry APIで生成したニュースとは別に、
実際のWeb記事をフェッチして企業間関係を抽出する。

Usage:
  python web_research_agent.py --articles web_articles.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from news_agent import extract_from_article, save_news_extracted
from schema import ExtractedFact, FiscalPeriod

EXTRACTED_DIR = Path(__file__).parent / "extracted"


def process_web_articles(articles: list[dict]) -> list[ExtractedFact]:
    """Web記事リストから五つ組を抽出。

    articles format:
    [
      {
        "title": "記事タイトル",
        "url": "https://...",
        "text": "記事本文",
        "date": "2025-06-20",
        "company": "関連企業名"
      }
    ]
    """
    fiscal_period = FiscalPeriod(
        fiscal_year="FY2024",
        period_type="annual",
        disclosure_date="2025-06-20",
    )

    all_facts = []
    for i, article in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {article.get('title', 'N/A')[:60]}")
        try:
            facts = extract_from_article(
                text=article.get("text", ""),
                source_url=article.get("url", "web"),
                source_title=article.get("title", ""),
                fiscal_period=fiscal_period,
            )
            all_facts.extend(facts)
            print(f"    -> {len(facts)} facts")
        except Exception as e:
            print(f"    Error: {e}")

    return all_facts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Web記事リサーチエージェント")
    parser.add_argument("--articles", required=True, help="記事JSONファイル")
    args = parser.parse_args()

    articles_path = Path(args.articles)
    if not articles_path.exists():
        print(f"File not found: {articles_path}")
        return

    articles = json.loads(articles_path.read_text(encoding="utf-8"))
    print(f"Processing {len(articles)} web articles...")

    facts = process_web_articles(articles)
    if facts:
        save_news_extracted(facts, "web_research")
        print(f"\nTotal: {len(facts)} facts from {len(articles)} articles")


if __name__ == "__main__":
    main()
