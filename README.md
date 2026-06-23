# Disclosure DKG

日本の金融開示文書（EDINET）とニュースソースから **Dynamic Knowledge Graph (DKG)** を構築し、投資戦略分析を行うパイプライン。

## Architecture

[FinDKG](https://github.com/xiaohui-victor-li/FinDKG) の五つ組形式 `(subject, relation, object, entity_type, timestamp)` と [DyGFormer/DyGLib](https://github.com/yule-BUAA/DyGLib) の時系列エッジ表現を参考に設計。

```
EDINET API ──► PDF/XBRL解析 ──► LLM抽出 (Foundry gpt-5.4-nano) ──┐
                                                                    ├──► DKG構築 ──► 投資戦略分析 (gpt-5.4)
ニュースソース ──► LLM抽出 (Foundry gpt-5.4-nano) ─────────────────┘
```

### FinDKG からの設計採用

| FinDKG の設計 | 本プロジェクトでの適用 |
|---|---|
| 12 エンティティ型 + 15 関係型 | 8 エンティティ型 + 25 関係型（日本市場向けに拡張） |
| ICKG (Vicuna-7B) による抽出 | Azure AI Foundry (gpt-5.4-nano) による抽出 |
| WSJ 40万記事ベース | EDINET 開示文書 + ニュース記事のマルチソース |
| 週次ローリングスナップショット | FiscalPeriod ベースの時間管理 |

### DyGFormer からの設計参考

| DyGFormer の概念 | 将来の拡張方針 |
|---|---|
| (source_id, target_id, timestamp, features) 形式 | `graphs/dkg_triples.csv` で互換エクスポート |
| Neighbor co-occurrence encoding | 企業間の共起関係分析に応用予定 |
| パッチング技法 | 四半期単位のスナップショット分割に応用予定 |

## Quick Start

```bash
# 1. セットアップ
cp .env.example .env  # APIキーを設定
pip install -r requirements.txt

# 2. EDINET から文書収集
python pipeline.py collect --start 2025-06-16 --end 2025-06-20 --doc-types 120,140,350

# 3. 五つ組抽出（Foundry API）
python pipeline.py extract --all --fiscal-year FY2024

# 4. ニュースリサーチエージェント
python news_agent.py

# 5. DKG 構築 & 可視化
python pipeline.py build
python pipeline.py viz --csv

# 6. 投資戦略分析
python strategy_agent.py
python strategy_agent.py --company "トヨタ自動車"  # 企業別深掘り
```

## Components

| ファイル | 機能 |
|---------|------|
| `schema.py` | エンティティ型・関係型・五つ組スキーマ定義 |
| `edinet_client.py` | EDINET API v2 クライアント |
| `document_parser.py` | PDF/HTML → テキストチャンク変換 |
| `extractor.py` | Foundry API で五つ組抽出 |
| `kg_builder.py` | NetworkX ベースの DKG 構築 |
| `visualizer.py` | pyvis によるインタラクティブ可視化 |
| `pipeline.py` | CLI パイプライン (collect/extract/build/viz/query) |
| `news_agent.py` | ニュースリサーチエージェント |
| `strategy_agent.py` | 投資戦略分析エージェント |
| `web_research_agent.py` | Web 記事取得・抽出エージェント |

## Schema

### Entity Types (8種)

`company`, `person`, `segment`, `financial_metric`, `region`, `product`, `industry`, `event`

### Relation Types (25種)

**財務数値:** `revenue_is`, `operating_profit_is`, `net_income_is`, `eps_is`, `dividend_is`, `total_assets_is`, `net_assets_is`, `revenue_change`

**業績予想:** `forecast_revenue`, `forecast_profit`, `guidance_change`

**企業間関係:** `invests_in`, `partners_with`, `acquires`, `competes_with`, `supplies_to`

**組織:** `has_segment`, `segment_revenue`, `segment_profit`, `has_officer`, `appoints`

**その他:** `operates_in`, `belongs_to_industry`, `risk_factor`, `esg_initiative`, `launches`

## Data Sources

- **EDINET**: 金融庁の電子開示システム（有価証券報告書、四半期報告書、決算短信）
- **ニュース**: Foundry API 経由で主要日本企業のニュース情報を生成・抽出

## References

- [FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets](https://arxiv.org/abs/2407.10909)
- [DyGFormer: Towards Better Dynamic Graph Learning](https://arxiv.org/abs/2303.13047)
- [EDINET API v2](https://disclosure.edinet-fsa.go.jp/)
