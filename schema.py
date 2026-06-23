"""スキーマ定義: 決算短信・統合報告書 DKG の型体系。

FinDKG の五つ組 (s, r, o, type, t) を拡張し、
数値保持 (value + unit) と出典スパン (SourceSpan) を追加。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── エンティティ型 ─────────────────────────────────────────────


class EntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    SEGMENT = "segment"
    FINANCIAL_METRIC = "financial_metric"
    REGION = "region"
    PRODUCT = "product"
    INDUSTRY = "industry"
    EVENT = "event"
    # FinDKG 互換拡張 (ORG/GOV, ORG/REG, GPE, ECON_IND, FIN_INST, CONCEPT)
    GOVERNMENT = "government"
    REGULATOR = "regulator"
    GEO_POLITICAL = "geo_political"
    ECONOMIC_INDICATOR = "economic_indicator"
    FINANCIAL_INSTRUMENT = "financial_instrument"
    CONCEPT = "concept"


# ── 関係型 ─────────────────────────────────────────────────────


class RelationType(str, Enum):
    # 財務数値（数値保持）
    REVENUE_IS = "revenue_is"
    REVENUE_CHANGE = "revenue_change"
    OPERATING_PROFIT_IS = "operating_profit_is"
    NET_INCOME_IS = "net_income_is"
    EPS_IS = "eps_is"
    DIVIDEND_IS = "dividend_is"
    TOTAL_ASSETS_IS = "total_assets_is"
    NET_ASSETS_IS = "net_assets_is"

    # 業績予想
    FORECAST_REVENUE = "forecast_revenue"
    FORECAST_PROFIT = "forecast_profit"
    GUIDANCE_CHANGE = "guidance_change"

    # セグメント
    HAS_SEGMENT = "has_segment"
    SEGMENT_REVENUE = "segment_revenue"
    SEGMENT_PROFIT = "segment_profit"

    # 組織・人事
    HAS_OFFICER = "has_officer"
    APPOINTS = "appoints"

    # 事業関係
    INVESTS_IN = "invests_in"
    PARTNERS_WITH = "partners_with"
    ACQUIRES = "acquires"
    COMPETES_WITH = "competes_with"
    SUPPLIES_TO = "supplies_to"

    # 地域・業種
    OPERATES_IN = "operates_in"
    BELONGS_TO_INDUSTRY = "belongs_to_industry"

    # リスク・イベント
    RISK_FACTOR = "risk_factor"
    ESG_INITIATIVE = "esg_initiative"
    LAUNCHES = "launches"

    # 因果・影響（FinDKG Impact 系）
    IMPACTS = "impacts"
    POSITIVELY_IMPACTS = "positively_impacts"
    NEGATIVELY_IMPACTS = "negatively_impacts"
    ANNOUNCES = "announces"

    # その他
    OTHER = "other"


# ── 時間 ───────────────────────────────────────────────────────


class FiscalPeriod(BaseModel):
    fiscal_year: str  # "FY2024"
    period_type: str = "annual"  # annual / q1 / q2 / q3 / q4
    start_date: str | None = None  # "2024-04-01"
    end_date: str | None = None  # "2025-03-31"
    disclosure_date: str | None = None  # "2025-05-08"

    def to_unix_timestamp(self) -> float:
        """DyGFormer CTDG 対応: 連続タイムスタンプへ変換。

        disclosure_date があればそれを使用、なければ fiscal_year +
        period_type からデフォルト日を推定する（日本企業の標準的開示日）。
        """
        from datetime import datetime

        if self.disclosure_date:
            try:
                dt = datetime.strptime(self.disclosure_date[:10], "%Y-%m-%d")
                return dt.timestamp()
            except ValueError:
                pass

        # fiscal_year から年度を抽出 ("FY2024" → 2024)
        year = int("".join(c for c in self.fiscal_year if c.isdigit()) or "2024")
        # 日本企業の標準的開示日を推定
        month_map = {"annual": 5, "q1": 8, "q2": 11, "q3": 2, "q4": 5}
        month = month_map.get(self.period_type, 5)
        est_year = year + 1 if month <= 5 else year
        dt = datetime(est_year, month, 15)
        return dt.timestamp()


# ── エンティティ参照 ───────────────────────────────────────────


class EntityRef(BaseModel):
    name: str
    entity_type: EntityType
    identifiers: dict[str, str] = Field(default_factory=dict)
    # e.g. {"edinet_code": "E02144", "sec_code": "7203"}


# ── 五つ組（拡張版） ──────────────────────────────────────────


class Quintuple(BaseModel):
    subject: EntityRef
    relation: RelationType
    object_text: str  # 目的語のテキスト表現
    object_entity: EntityRef | None = None  # 目的語がエンティティの場合
    value: float | None = None  # 数値（FinDKG拡張）
    unit: str | None = None  # "百万円", "%", "円/株"
    timestamp: FiscalPeriod
    confidence: float = 1.0


# ── 出典 ───────────────────────────────────────────────────────


class SourceSpan(BaseModel):
    doc_id: str
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    text_snippet: str = ""


# ── 抽出事実 ───────────────────────────────────────────────────


class ExtractedFact(BaseModel):
    quintuple: Quintuple
    source: SourceSpan
    extraction_method: str = "llm_direct"  # llm_direct / xbrl_parsed


# ── 文書メタデータ ─────────────────────────────────────────────


class DocumentMeta(BaseModel):
    doc_id: str
    edinet_code: str = ""
    sec_code: str = ""
    filer_name: str = ""
    doc_type_code: str = ""
    doc_description: str = ""
    period_start: str = ""
    period_end: str = ""
    submit_date: str = ""


# ── テキストチャンク ───────────────────────────────────────────


class TextChunk(BaseModel):
    doc_id: str
    section: str
    text: str
    char_start: int = 0
    char_end: int = 0


# ── LLM 抽出用のスキーマヒント ─────────────────────────────────


ENTITY_TYPES_JA = {
    EntityType.COMPANY: "企業",
    EntityType.PERSON: "人物（役員等）",
    EntityType.SEGMENT: "事業セグメント",
    EntityType.FINANCIAL_METRIC: "財務指標",
    EntityType.REGION: "地域",
    EntityType.PRODUCT: "製品・サービス",
    EntityType.INDUSTRY: "業種",
    EntityType.EVENT: "イベント（M&A, 増配等）",
    EntityType.GOVERNMENT: "政府機関（経産省, 日銀等）",
    EntityType.REGULATOR: "規制機関（金融庁, SEC等）",
    EntityType.GEO_POLITICAL: "地政学的エンティティ（国, 経済圏）",
    EntityType.ECONOMIC_INDICATOR: "経済指標（CPI, 日銀短観等）",
    EntityType.FINANCIAL_INSTRUMENT: "金融商品（株式, 債券, デリバティブ）",
    EntityType.CONCEPT: "概念（AI, ESG, DX等）",
}

RELATION_TYPES_JA = {
    RelationType.REVENUE_IS: "売上高",
    RelationType.REVENUE_CHANGE: "売上高増減率",
    RelationType.OPERATING_PROFIT_IS: "営業利益",
    RelationType.NET_INCOME_IS: "純利益/当期利益",
    RelationType.EPS_IS: "1株当たり利益",
    RelationType.DIVIDEND_IS: "配当",
    RelationType.TOTAL_ASSETS_IS: "総資産",
    RelationType.NET_ASSETS_IS: "純資産",
    RelationType.FORECAST_REVENUE: "業績予想:売上",
    RelationType.FORECAST_PROFIT: "業績予想:利益",
    RelationType.GUIDANCE_CHANGE: "業績修正",
    RelationType.HAS_SEGMENT: "セグメント保有",
    RelationType.SEGMENT_REVENUE: "セグメント売上",
    RelationType.SEGMENT_PROFIT: "セグメント利益",
    RelationType.HAS_OFFICER: "役員",
    RelationType.APPOINTS: "就任",
    RelationType.INVESTS_IN: "投資",
    RelationType.PARTNERS_WITH: "提携",
    RelationType.ACQUIRES: "買収",
    RelationType.COMPETES_WITH: "競合",
    RelationType.SUPPLIES_TO: "供給",
    RelationType.OPERATES_IN: "事業展開地域",
    RelationType.BELONGS_TO_INDUSTRY: "業種所属",
    RelationType.RISK_FACTOR: "リスク要因",
    RelationType.ESG_INITIATIVE: "ESG施策",
    RelationType.LAUNCHES: "新規事業/製品投入",
    RelationType.IMPACTS: "影響",
    RelationType.POSITIVELY_IMPACTS: "正の影響",
    RelationType.NEGATIVELY_IMPACTS: "負の影響",
    RelationType.ANNOUNCES: "発表・公表",
    RelationType.OTHER: "その他",
}


def schema_hint_for_prompt() -> str:
    """LLM 抽出プロンプトに埋め込むスキーマ要約を生成。"""
    lines = ["## エンティティ型"]
    for et, ja in ENTITY_TYPES_JA.items():
        lines.append(f"- {et.value}: {ja}")
    lines.append("\n## 関係型")
    for rt, ja in RELATION_TYPES_JA.items():
        lines.append(f"- {rt.value}: {ja}")
    return "\n".join(lines)
