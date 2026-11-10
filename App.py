"""
Promotion Optimization System v4.0
RAG-Enhanced AI Platform for Retail Promotion Planning

Uplift numbers come from a LightGBM model trained on the dunnhumby / 84.51
"The Complete Journey" data (see ml/uplift_model_colab.ipynb). Departments with
too little promotion data fall back to clearly labelled rule-based values.

NUST University | Tauseef Iqbal | 2025-2026
"""

import os
import re
import glob
import json
import calendar
from datetime import datetime, timedelta
from typing import Optional

from openai import OpenAI
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Promotion Optimization System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Layout ── */
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* ── Page title ── */
    .page-title h1 { font-size: 1.65rem; font-weight: 700; color: #111827; margin: 0; padding: 0; }
    .page-title p  { color: #6B7280; margin: 0.3rem 0 1.4rem 0; font-size: 0.95rem; }

    /* ── KPI tiles ── */
    .kpi {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05); height: 100%;
    }
    .kpi .label { font-size: 0.74rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .kpi .value { font-size: 1.55rem; font-weight: 700; color: #111827; margin-top: 0.25rem; line-height: 1.2; }
    .kpi .sub   { font-size: 0.8rem; color: #6B7280; margin-top: 0.2rem; }
    .kpi .value.accent { color: #4F46E5; }

    /* ── Cards ── */
    .card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
        padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05); margin-bottom: 1rem;
    }
    .card-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
    .card-head h4 { margin: 0; font-size: 1.02rem; font-weight: 700; color: #111827; }
    .card .uplift { font-size: 1.9rem; font-weight: 800; color: #4F46E5; line-height: 1.1; }
    .card .uplift-sub { font-size: 0.82rem; color: #6B7280; margin-bottom: 0.7rem; }
    .card .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.2rem 1rem; font-size: 0.86rem; color: #374151; }
    .card .grid span { color: #6B7280; }
    .card .note { margin-top: 0.7rem; padding-top: 0.6rem; border-top: 1px solid #F3F4F6; font-size: 0.84rem; color: #4B5563; }
    .card .note p { margin: 0.15rem 0; }
    .muted { color: #6B7280; font-size: 0.86rem; }

    /* ── Badges ── */
    .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
    .badge.model { background: #DCFCE7; color: #166534; }
    .badge.rule  { background: #F3F4F6; color: #4B5563; }
    .badge.vh { background: #FEE2E2; color: #991B1B; }
    .badge.h  { background: #E0E7FF; color: #3730A3; }
    .badge.m  { background: #FEF3C7; color: #92400E; }
    .badge.l  { background: #F3F4F6; color: #4B5563; }

    /* ── Sidebar ── */
    .brand { font-size: 1.12rem; font-weight: 800; color: #111827; line-height: 1.25; }
    .brand span { color: #4F46E5; }
    .brand-sub { font-size: 0.8rem; color: #6B7280; margin: 0.2rem 0 1rem 0; }
    .status { font-size: 0.84rem; color: #374151; margin: 0.25rem 0; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .dot.ok { background: #22C55E; } .dot.off { background: #D1D5DB; }
    .side-label { font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin: 1rem 0 0.3rem 0; }
    .occ { font-size: 0.84rem; color: #374151; margin: 0.15rem 0; }
    .occ b { color: #111827; }

    /* ── Lists ── */
    .date-row { display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid #F3F4F6; font-size: 0.88rem; }
    .date-row:last-child { border-bottom: none; }
    .date-row .when { color: #6B7280; }

    /* ── Chat ── */
    .chat-wrapper { display: flex; flex-direction: column; padding-bottom: 0.5rem; }
    .chat-row { display: flex; align-items: flex-end; gap: 10px; margin-bottom: 14px; }
    .chat-row.user { flex-direction: row-reverse; }
    .avatar {
        width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 0.72rem; font-weight: 700; flex-shrink: 0; color: white;
    }
    .avatar.user { background: #4F46E5; }
    .avatar.bot  { background: #111827; }
    .bubble { max-width: 72%; padding: 0.7rem 1rem; border-radius: 14px; font-size: 0.9rem; line-height: 1.55; }
    .bubble.user { background: #4F46E5; color: white; border-bottom-right-radius: 4px; }
    .bubble.bot  { background: #FFFFFF; color: #1F2937; border: 1px solid #E5E7EB; border-bottom-left-radius: 4px; }
    .chat-meta { font-size: 0.72rem; color: #9CA3AF; margin-top: 4px; padding: 0 4px; }
    .chat-meta.user { text-align: right; }
    .provider-bar {
        display: flex; align-items: center; gap: 8px; padding: 0.5rem 0.9rem; background: #FFFFFF;
        border: 1px solid #E5E7EB; border-radius: 10px; margin-bottom: 1rem; font-size: 0.82rem; color: #4B5563;
    }
    .provider-dot { width: 8px; height: 8px; border-radius: 50%; background: #22C55E; flex-shrink: 0; }
    .provider-dot.off { background: #D1D5DB; }
    .empty-chat {
        text-align: center; padding: 2.5rem 1rem; color: #9CA3AF; font-size: 0.92rem; background: #FFFFFF;
        border: 1.5px dashed #E5E7EB; border-radius: 14px; margin-bottom: 1.2rem;
    }
    .empty-chat p { margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL RESULTS  (produced by ml/uplift_model_colab.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "uplift_results")


def load_model_results(
    model_dir: str = MODEL_DIR,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    """Read the uplift tables and metrics written by the training notebook.

    Each part is None when its file is missing or unreadable, so the app keeps
    working (with rule-based values) without the model files.
    """
    def _read_csv(name: str, index: str) -> Optional[pd.DataFrame]:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path).set_index(index)
        except Exception:
            return None

    metrics: Optional[dict] = None
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
        except Exception:
            metrics = None

    return (
        _read_csv("department_uplift.csv", "department"),
        _read_csv("category_uplift.csv", "product_category"),
        metrics,
    )


def load_accuracy_by_department(model_dir: str = MODEL_DIR) -> Optional[pd.DataFrame]:
    """Per-department forecast accuracy on the test weeks (written by newer notebook runs)."""
    path = os.path.join(model_dir, "accuracy_by_department.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path).set_index("department")
    except Exception:
        return None


def response_level(uplift: float) -> str:
    """Turn a model-estimated uplift into a response level."""
    if uplift >= 1.2:
        return "Very High"
    if uplift >= 0.9:
        return "High"
    if uplift >= 0.6:
        return "Medium"
    return "Low"


def is_model(info: dict) -> bool:
    return info.get("uplift_source") == "model"


def source_label(info: dict) -> str:
    return "Model-estimated" if is_model(info) else "Rule-based"


# ──────────────────────────────────────────────────────────────────────────────
# RAG KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────

class PromotionRAGSystem:
    """
    Lightweight Retrieval-Augmented Generation system for promotion queries.

    Builds a structured knowledge base by combining per-category statistics
    derived directly from the product CSV with curated business-rule insights.
    At query time, relevant documents are retrieved via a scored term-overlap
    algorithm and assembled into a context block that is injected into the
    LLM prompt — ensuring every AI response is grounded in real catalog data.
    """

    # Promotional and analytical vocabulary used to boost retrieval scores
    _PROMO_VOCAB: frozenset = frozenset({
        "promote", "promotion", "sale", "discount", "bogo", "bundle",
        "flash", "uplift", "roi", "recommend", "strategy", "best",
        "timing", "weekend", "holiday", "perishable", "category",
        "increase", "sales", "revenue", "basket", "traffic",
    })

    def __init__(self, items_df: pd.DataFrame, family_insights: dict):
        self.items_df = items_df
        self.family_insights = family_insights
        self._kb: dict[str, dict] = {}          # category → {text, stats, insight}
        self._doc_tokens: list[frozenset] = []   # pre-tokenised document token sets
        self._doc_keys: list[str] = []           # parallel list of category names
        self._build_knowledge_base()

    # ── build ────────────────────────────────────────────────────────────────

    def _build_knowledge_base(self) -> None:
        """
        For every product family present in the CSV, construct a rich text
        document that merges computed statistics with the business-rule
        insight dictionary.  Documents are also tokenised once here so
        retrieval is O(k·|V|) rather than repeating tokenisation at runtime.
        """
        for family in self.items_df["family"].unique():
            subset = self.items_df[self.items_df["family"] == family]
            total_skus       = len(subset)
            perishable_count = int(subset["perishable"].sum()) if "perishable" in subset.columns else 0
            class_count      = int(subset["class"].nunique()) if "class" in subset.columns else 0
            perishable_pct   = perishable_count / total_skus * 100 if total_skus else 0

            insight = self.family_insights.get(family, {
                "response": "Medium", "best_promo": "Percentage Off",
                "peak_days": "Weekend", "uplift": 0.20, "uplift_source": "rule-based",
                "timing_advantage": "Standard timing applies.",
                "customer_behavior": "Regular purchase pattern.",
                "strategic_value": "Standard category value.",
            })

            if is_model(insight):
                uplift_lines = (
                    f"Expected sales uplift (display + mailer): {insight['uplift'] * 100:.0f}% "
                    f"[model-estimated: LightGBM trained on dunnhumby promotion data"
                    + (f"; 90% range {insight['uplift_low'] * 100:.0f}-{insight['uplift_high'] * 100:.0f}%"
                       if "uplift_low" in insight else "")
                    + "]\n"
                    f"Uplift with in-store display only: {insight['uplift_display'] * 100:.0f}%\n"
                    f"Uplift with weekly mailer only: {insight['uplift_mailer'] * 100:.0f}%\n"
                )
            else:
                uplift_lines = (
                    f"Expected sales uplift: {insight['uplift'] * 100:.0f}% "
                    f"[rule-based assumption: too little promotion data for a model estimate]\n"
                )
            top = insight.get("top_categories") or []
            if top:
                uplift_lines += "Top product categories by model uplift: " + ", ".join(
                    f"{name} ({u * 100:.0f}%)" for name, u in top) + "\n"

            doc_text = (
                f"Department: {family}\n"
                f"Total SKUs in catalog: {total_skus}\n"
                f"Perishable SKUs: {perishable_count} ({perishable_pct:.1f}%)\n"
                f"Product categories: {class_count}\n"
                f"Customer response level: {insight['response']}\n"
                f"{uplift_lines}"
                f"Suggested promotion type (business rule): {insight['best_promo']}\n"
                f"Peak shopping period (business rule): {insight['peak_days']}\n"
                f"Timing advantage: {insight['timing_advantage']}\n"
                f"Customer behavior: {insight['customer_behavior']}\n"
                f"Strategic value: {insight['strategic_value']}"
            )

            self._kb[family] = {
                "text": doc_text,
                "stats": {
                    "total_skus": total_skus,
                    "perishable_count": perishable_count,
                    "class_count": class_count,
                },
                "insight": insight,
            }
            self._doc_tokens.append(frozenset(doc_text.lower().split()))
            self._doc_keys.append(family)

    # ── retrieval ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> frozenset:
        return frozenset(re.sub(r"[^\w\s]", "", text.lower()).split())

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Return the top-k most relevant category documents for a natural-language
        query.  Scoring combines:
          • raw term overlap between query and document tokens
          • 3× boost if any query term directly matches a category-name token
          • 0.5× boost per promotional vocabulary term in the query
        """
        q_tokens = self._tokenise(query)

        scores: list[tuple[int, float]] = []
        for i, doc_tokens in enumerate(self._doc_tokens):
            overlap         = len(q_tokens & doc_tokens)
            cat_tokens      = frozenset(self._doc_keys[i].lower().replace(",", " ").split())
            category_boost  = len(q_tokens & cat_tokens) * 3.0
            promo_boost     = len(q_tokens & self._PROMO_VOCAB) * 0.5
            scores.append((i, overlap + category_boost + promo_boost))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            {
                "category": self._doc_keys[i],
                "content": self._kb[self._doc_keys[i]]["text"],
                "score": score,
            }
            for i, score in scores[:top_k]
            if score > 0
        ]

        # Fallback: return highest-uplift categories when nothing matches
        if not results:
            fallback = sorted(
                self._kb.items(),
                key=lambda x: x[1]["insight"].get("uplift", 0),
                reverse=True,
            )[:top_k]
            results = [
                {"category": k, "content": v["text"], "score": 0.0}
                for k, v in fallback
            ]

        return results

    def build_context(self, query: str) -> str:
        """Assemble a formatted context block from retrieved documents."""
        retrieved = self.retrieve(query, top_k=4)
        lines = [
            "RETRIEVED CATALOG CONTEXT",
            f"Full catalog: {len(self.items_df):,} SKUs across "
            f"{self.items_df['family'].nunique()} departments.\n",
        ]
        for doc in retrieved:
            lines.append(f"[{doc['category']}]")
            lines.append(doc["content"])
            lines.append("")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CORE PROMOTION SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class PromotionOptimizationSystem:
    """
    Central engine for the Promotion Optimization System.

    Owns the business-rule knowledge base (family_insights, holiday calendar),
    loads and validates the product CSV, instantiates the RAG system once data
    is available, and exposes helpers used by the Streamlit rendering layer.
    """

    RESPONSE_COLORS = {
        "Very High": "#e74c3c",
        "High":      "#3498db",
        "Medium":    "#f39c12",
        "Low":       "#95a5a6",
    }

    def __init__(self):
        self.items_df:   Optional[pd.DataFrame]       = None
        self.rag_system: Optional[PromotionRAGSystem]  = None

        # Business rules per dunnhumby department. These texts are hand-written
        # guidance, NOT model output. Their "uplift" values are only a fallback for
        # departments where the model is missing or has too little promotion data;
        # _apply_model_uplift() overwrites them with model estimates where reliable.
        self.family_insights: dict = {
            "GROCERY": {
                "response": "Medium", "best_promo": "Multi-buy (e.g. 2 for $5)",
                "peak_days": "Weekend", "uplift": 0.25,
                "timing_advantage": "Big weekly stock-up trips happen on weekends.",
                "customer_behavior": "Shelf-stable staples; shoppers stock up when the price is right.",
                "strategic_value": "Largest department; drives basket size and store traffic.",
            },
            "DRUG GM": {
                "response": "Medium", "best_promo": "Bundle Deal",
                "peak_days": "Month-start", "uplift": 0.20,
                "timing_advantage": "Payday at the start of the month lifts non-food spending.",
                "customer_behavior": "Planned purchases of health and household items; brand loyal.",
                "strategic_value": "Higher-margin non-food items; good for bundles with groceries.",
            },
            "MEAT-PCKGD": {
                "response": "High", "best_promo": "Weekly Feature Price",
                "peak_days": "Weekend", "uplift": 0.30,
                "timing_advantage": "Weekend meal planning and grilling season raise demand.",
                "customer_behavior": "Bacon, sausage and lunch meat respond strongly to featured prices.",
                "strategic_value": "Frequent purchase that pulls shoppers into the store.",
            },
            "PRODUCE": {
                "response": "High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.30,
                "timing_advantage": "Short shelf life rewards quick sell-through before the weekend.",
                "customer_behavior": "Fresh-focused shoppers react to visible, seasonal deals.",
                "strategic_value": "Freshness image of the whole store; reduces waste.",
            },
            "DELI": {
                "response": "Medium", "best_promo": "Meal Deal Bundle",
                "peak_days": "Weekend", "uplift": 0.25,
                "timing_advantage": "Weekend parties and quick family meals.",
                "customer_behavior": "Convenience buying; impulse-friendly near the entrance.",
                "strategic_value": "High-margin fresh department with strong add-on sales.",
            },
            "MEAT": {
                "response": "High", "best_promo": "Weekly Feature Price",
                "peak_days": "Holiday", "uplift": 0.35,
                "timing_advantage": "Holidays and grilling weekends create big demand spikes.",
                "customer_behavior": "Big-ticket item; shoppers wait for featured prices.",
                "strategic_value": "Highest value per item; strong cross-sell with sides and drinks.",
            },
            "PASTRY": {
                "response": "Medium", "best_promo": "BOGO (Buy One Get One)",
                "peak_days": "Weekend", "uplift": 0.25,
                "timing_advantage": "Weekend breakfasts and celebrations.",
                "customer_behavior": "Impulse buy; freshness matters.",
                "strategic_value": "Perishable: promotions cut end-of-day waste.",
            },
            "NUTRITION": {
                "response": "Medium", "best_promo": "Percentage Off (15-25%)",
                "peak_days": "Month-start", "uplift": 0.20,
                "timing_advantage": "New-month health goals and payday.",
                "customer_behavior": "Health-focused, planned purchases; brand loyal.",
                "strategic_value": "Premium margins; attracts health-conscious shoppers.",
            },
            "SEAFOOD-PCKGD": {
                "response": "Medium", "best_promo": "Weekly Feature Price",
                "peak_days": "Weekly", "uplift": 0.20,
                "timing_advantage": "Steady weekly demand, higher during Lent.",
                "customer_behavior": "Planned purchase for specific meals.",
                "strategic_value": "Frozen and packaged: long shelf life allows longer promotions.",
            },
            "SEAFOOD": {
                "response": "Medium", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.20,
                "timing_advantage": "Fresh fish sells best near the weekend.",
                "customer_behavior": "Occasional, quality-driven purchase.",
                "strategic_value": "Very perishable: fast sell-through matters.",
            },
            "FLORAL": {
                "response": "Low", "best_promo": "Limited Time Offer",
                "peak_days": "Holiday", "uplift": 0.20,
                "timing_advantage": "Valentine's Day, Mother's Day and other gifting holidays.",
                "customer_behavior": "Gift and occasion purchases.",
                "strategic_value": "High margin on a few big days per year.",
            },
            "SPIRITS": {
                "response": "Medium", "best_promo": "Volume Discount",
                "peak_days": "Holiday", "uplift": 0.20,
                "timing_advantage": "Holiday parties and long weekends.",
                "customer_behavior": "Event-driven, multi-unit purchases.",
                "strategic_value": "High value per transaction.",
            },
            "COSMETICS": {
                "response": "Low", "best_promo": "Percentage Off (20-40%)",
                "peak_days": "Holiday", "uplift": 0.15,
                "timing_advantage": "Gifting season.",
                "customer_behavior": "Brand-driven, low purchase frequency.",
                "strategic_value": "Premium margins; small share of grocery baskets.",
            },
        }
        for insight in self.family_insights.values():
            insight["uplift_source"] = "rule-based"

        self.model_dept, self.model_cat, self.model_metrics = load_model_results()
        self.model_accuracy_dept = load_accuracy_by_department()
        self._apply_model_uplift()

        self.yearly_holidays: dict = {
            1:  [{"day":  1, "name": "New Year's Day"},       {"day": 15, "name": "Mid-January Sale"}],
            2:  [{"day": 14, "name": "Valentine's Day"},       {"day": 17, "name": "President's Day"}],
            3:  [{"day": 17, "name": "St. Patrick's Day"},     {"day": 20, "name": "Spring Equinox"}],
            4:  [{"day":  1, "name": "April Fool's Day"},      {"day": 22, "name": "Earth Day"}],
            5:  [{"day":  5, "name": "Cinco de Mayo"},         {"day": 26, "name": "Memorial Day"}],
            6:  [{"day": 15, "name": "Father's Day"},          {"day": 21, "name": "Summer Solstice"}],
            7:  [{"day":  4, "name": "Independence Day"},      {"day": 15, "name": "Mid-Summer Sale"}],
            8:  [{"day":  1, "name": "Back to School"},        {"day": 15, "name": "Mid-August Clearance"}],
            9:  [{"day":  1, "name": "Labor Day"},             {"day": 22, "name": "Fall Equinox"}],
            10: [{"day": 31, "name": "Halloween"},             {"day": 15, "name": "Fall Festival"}],
            11: [{"day": 11, "name": "Veterans Day"},          {"day": 27, "name": "Thanksgiving"}],
            12: [{"day": 25, "name": "Christmas"},             {"day": 31, "name": "New Year's Eve"}],
        }

        self.upcoming_occasions: list[dict] = self._generate_upcoming_occasions()

    # ── calendar generation ──────────────────────────────────────────────────

    def _generate_upcoming_occasions(self) -> list[dict]:
        today = datetime.now()
        occasions: list[dict] = []

        for i in range(90):
            future_date = today + timedelta(days=i)
            date_str    = future_date.strftime("%Y-%m-%d")
            day_name    = future_date.strftime("%A")

            if future_date.weekday() in (5, 6):
                occasions.append({
                    "date": date_str, "occasion": f"Weekend ({day_name})",
                    "type": "weekend", "priority": "high",
                })

            last_day = calendar.monthrange(future_date.year, future_date.month)[1]
            if future_date.day >= last_day - 2:
                occasions.append({
                    "date": date_str, "occasion": "Month-End Sale",
                    "type": "month_end", "priority": "medium",
                })

            if future_date.day <= 3:
                occasions.append({
                    "date": date_str, "occasion": "Month-Start (Payday)",
                    "type": "month_start", "priority": "high",
                })

            for holiday in self.yearly_holidays.get(future_date.month, []):
                if future_date.day == holiday["day"]:
                    occasions.append({
                        "date": date_str, "occasion": holiday["name"],
                        "type": "major_holiday", "priority": "very_high",
                    })

        # Deduplicate by date (keep first match per date)
        seen: set[str] = set()
        unique: list[dict] = []
        for occ in occasions:
            if occ["date"] not in seen:
                unique.append(occ)
                seen.add(occ["date"])

        return sorted(unique, key=lambda x: x["date"])[:30]

    # ── data loading ─────────────────────────────────────────────────────────

    # ── model results ────────────────────────────────────────────────────────

    def _apply_model_uplift(self) -> None:
        """Replace rule-based uplift with model estimates where the model is reliable."""
        if self.model_dept is None:
            return
        for dept, row in self.model_dept.iterrows():
            info = self.family_insights.setdefault(dept, self.get_default_insight())
            info["promo_weeks_in_training"] = int(row.get("promo_weeks_in_training", 0) or 0)
            if bool(row.get("low_confidence", True)):
                info["uplift_source"] = "rule-based"
                info["model_note"] = "too few promotion weeks in the data for a reliable model estimate"
                continue
            info.update(
                uplift=float(row["uplift_both"]),
                uplift_display=float(row["uplift_display"]),
                uplift_mailer=float(row["uplift_mailer"]),
                uplift_source="model",
                response=response_level(float(row["uplift_both"])),
            )
            if pd.notna(row.get("uplift_both_low")) and pd.notna(row.get("uplift_both_high")):
                info["uplift_low"] = float(row["uplift_both_low"])
                info["uplift_high"] = float(row["uplift_both_high"])

    def top_categories(self, department: str, n: int = 3) -> list[tuple[str, float]]:
        """Product categories in a department with the highest reliable model uplift."""
        if self.model_cat is None or self.items_df is None or "class" not in self.items_df.columns:
            return []
        cats = self.items_df.loc[self.items_df["family"] == department, "class"].dropna().unique()
        table = self.model_cat.reindex(cats).dropna(subset=["uplift_both"])
        if "low_confidence" in table.columns:
            table = table[~table["low_confidence"].astype(bool)]
        top = table["uplift_both"].sort_values(ascending=False).head(n)
        return [(str(name), float(u)) for name, u in top.items()]

    def model_summary(self) -> Optional[str]:
        """One-line description of the model's test performance, if available."""
        m = self.model_metrics
        if not m:
            return None
        try:
            s = m["scores_test"]
            weeks = m["data"]["test_weeks"]
            return (
                f"LightGBM trained on dunnhumby 'The Complete Journey' "
                f"({m['data']['products']:,} products). On unseen weeks {weeks[0]}-{weeks[1]} its error "
                f"(RMSLE {s['lightgbm']['rmsle']:.3f}) is {m['improvement_vs_baseline_rmsle']:.0%} lower "
                f"than a 4-week-average baseline ({s['baseline_4w_mean']['rmsle']:.3f})."
            )
        except (KeyError, TypeError):
            return None

    # ── data loading ─────────────────────────────────────────────────────────

    def load_data(self, df: pd.DataFrame) -> None:
        self.items_df        = df
        for dept, info in self.family_insights.items():
            info["top_categories"] = self.top_categories(dept)
        self.rag_system      = PromotionRAGSystem(df, self.family_insights)
        self.upcoming_occasions = self._generate_upcoming_occasions()

    def auto_load_csv(self) -> tuple[bool, str]:
        candidates = ["items.csv", "./items.csv", "./data/items.csv", "/content/items.csv"]
        candidates += glob.glob("*.csv") + glob.glob("/content/*.csv")

        for path in dict.fromkeys(candidates):   # preserve order, remove dupes
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
                if {"item_nbr", "family"}.issubset(df.columns):
                    self.load_data(df)
                    return True, (
                        f"Loaded {len(df):,} items across "
                        f"{df['family'].nunique()} departments from '{path}'."
                    )
            except Exception:
                continue

        return False, "No valid items.csv found. Please upload via the sidebar."

    # ── convenience helpers ──────────────────────────────────────────────────

    def get_ordered_families(self, limit: int) -> list[str]:
        """Departments with a reliable model estimate first (highest uplift first), then the rest."""
        if self.items_df is None:
            return []
        sizes = self.items_df["family"].value_counts()

        def sort_key(dept: str) -> tuple:
            info = self.family_insights.get(dept, {})
            return (not is_model(info), -info.get("uplift", 0.0) if is_model(info) else -sizes[dept])

        return sorted(sizes.index, key=sort_key)[:limit]

    def get_default_insight(self) -> dict:
        return {
            "response": "Medium", "best_promo": "Percentage Off (20-40%)",
            "peak_days": "Weekend", "uplift": 0.20, "uplift_source": "rule-based",
            "timing_advantage": "Weekends and paydays usually bring the most shoppers.",
            "customer_behavior": "Regular shoppers notice a clear price offer at the shelf.",
            "strategic_value": "A simple offer here keeps the department visible to shoppers.",
        }


# ──────────────────────────────────────────────────────────────────────────────
# AI CHATBOT  (RAG + Multi-Provider Fallback Chain)
#
# Provider order:  Groq (primary)  →  Gemini (secondary)
# If the active provider hits a quota or auth error, the chain automatically
# falls through to the next provider.  Both use the OpenAI-compatible SDK so
# the core prompt-building logic is shared and never duplicated.
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior retail promotion strategist. Answer every question using ONLY the
data in the RETRIEVED CATALOG CONTEXT provided. Never invent figures.

Response rules — follow these strictly:
- Maximum 120 words per answer.
- Lead with the direct answer or recommendation in the first sentence.
- Use plain numbered or bulleted lines when listing more than two items.
- You are talking to a shop owner: use plain, friendly words and no technical terms
  (no 'model', 'LightGBM', 'uplift', 'RMSLE'); say 'extra sales' instead of 'uplift'.
- Quote exact percentages and product counts from the context. If a number is a
  rule-based assumption, call it a rough estimate.
- No emojis. No filler phrases. No lengthy preambles.
- If the context does not cover the question, say so in one sentence and
  give one general best-practice principle.
"""

# Each provider entry: (display_name, base_url, model_id)
_PROVIDERS: list[tuple[str, str, str]] = [
    (
        "Groq",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
    ),
    (
        "Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash",
    ),
]


class ProviderFallbackChain:
    """
    Tries each configured provider in order.

    On rate-limit or authentication errors the chain advances to the next
    provider automatically.  A hard exception (network down, bad prompt, etc.)
    is re-raised immediately because retrying will not help in those cases.

    Usage
    -----
    chain = ProviderFallbackChain(groq_key="...", gemini_key="...")
    reply, used_provider = chain.complete(messages)
    """

    # HTTP status codes that justify falling through to the next provider
    _FALLTHROUGH_STATUS: frozenset[int] = frozenset({429, 401, 403})

    def __init__(self, groq_key: str, gemini_key: str) -> None:
        # Map provider name → API key (empty string = not configured)
        self._keys: dict[str, str] = {
            "Groq":   groq_key.strip(),
            "Gemini": gemini_key.strip(),
        }

    def _build_client(self, name: str, base_url: str) -> Optional[OpenAI]:
        key = self._keys.get(name, "")
        if not key:
            return None
        return OpenAI(api_key=key, base_url=base_url)

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> tuple[str, str]:
        """
        Returns (reply_text, provider_name_used).
        Raises RuntimeError if all providers fail.
        """
        last_error: str = ""

        for name, base_url, model in _PROVIDERS:
            client = self._build_client(name, base_url)
            if client is None:
                continue          # key not supplied — skip silently

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content, name

            except Exception as exc:
                # Inspect the raw exception string for HTTP status codes
                exc_str = str(exc)
                is_quota = any(str(s) in exc_str for s in self._FALLTHROUGH_STATUS)
                is_auth  = "authentication" in exc_str.lower() or "api key" in exc_str.lower()

                if is_quota or is_auth:
                    last_error = f"{name}: {exc_str[:120]}"
                    continue      # try next provider
                raise             # unrecoverable error — bubble up

        raise RuntimeError(
            f"All providers exhausted or unconfigured. Last error: {last_error}"
        )


def build_messages(
    query: str,
    rag_system: PromotionRAGSystem,
    history: list[dict],
) -> list[dict]:
    """
    Assemble the full OpenAI-format message list with injected RAG context.
    History is capped at the last 8 turns to stay within token budgets.
    """
    context            = rag_system.build_context(query)
    augmented_user_msg = f"{context}\n\nUSER QUESTION:\n{query}"

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history[-8:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": augmented_user_msg})

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

INDIGO, INDIGO_LIGHT, GREEN, AMBER, GREY = "#4F46E5", "#A5B4FC", "#22C55E", "#F59E0B", "#9CA3AF"

_CHART_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Inter, 'Segoe UI', sans-serif", size=12, color="#1F2937"),
    title=dict(font=dict(size=15, color="#111827"), x=0, xanchor="left"),
    margin=dict(l=10, r=10, t=56, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
    hoverlabel=dict(bgcolor="white"),
)

_RESPONSE_BADGE = {"Very High": "vh", "High": "h", "Medium": "m", "Low": "l"}


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, **_CHART_LAYOUT)
    fig.update_xaxes(gridcolor="#F3F4F6", zeroline=False, linecolor="#E5E7EB", automargin=True)
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False, linecolor="#E5E7EB", automargin=True)
    return fig


def show_fig(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def html(block: str) -> None:
    """Render an HTML block. Blank or indented lines would make Markdown print raw HTML, so strip them."""
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()), unsafe_allow_html=True)


def page_title(title: str, subtitle: str) -> None:
    html(f"<div class='page-title'><h1>{title}</h1><p>{subtitle}</p></div>")


def kpi(col, label: str, value: str, sub: str = "", accent: bool = False) -> None:
    with col:
        html(f"""<div class="kpi"><div class="label">{label}</div>
                 <div class="value{' accent' if accent else ''}">{value}</div>
                 <div class="sub">{sub}</div></div>""")


def fmt_uplift(x: float) -> str:
    return f"{x:+.0%}"


def range_text(info: dict) -> str:
    if "uplift_low" in info and "uplift_high" in info:
        return f"range {info['uplift_low']:.0%}–{info['uplift_high']:.0%}"
    return ""


def badges(info: dict) -> str:
    resp = info.get("response", "Medium")
    src = "<span class='badge model'>Model</span>" if is_model(info) else "<span class='badge rule'>Rule</span>"
    return f"<span class='badge {_RESPONSE_BADGE.get(resp, 'm')}'>{resp}</span> {src}"


# ──────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def chart_category_distribution(items_df: pd.DataFrame) -> go.Figure:
    counts = items_df["family"].value_counts().head(12).sort_values()
    fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                 title="Products per department", labels={"x": "Products", "y": ""})
    fig.update_traces(marker_color=INDIGO, hovertemplate="%{y}: %{x:,} products<extra></extra>")
    return style_fig(fig, 430)


def chart_roi_matrix(items_df: pd.DataFrame, family_insights: dict) -> go.Figure:
    counts = items_df["family"].value_counts().head(12)
    rows = []
    for family in counts.index:
        info = family_insights.get(family, {"uplift": 0.20})
        rows.append({"Department": family, "Uplift (%)": round(info["uplift"] * 100, 1),
                     "Products": int(counts[family]), "Source": source_label(info)})
    fig = px.scatter(pd.DataFrame(rows), x="Products", y="Uplift (%)", size="Products", color="Source",
                     hover_data=["Department"], log_x=True, size_max=38,
                     color_discrete_map={"Model-estimated": INDIGO, "Rule-based": GREY},
                     title="Uplift vs. catalog depth")
    return style_fig(fig, 430)


def chart_opportunity_calendar(upcoming: list[dict]) -> go.Figure:
    priority_score = {"very_high": 4, "high": 3, "medium": 2, "low": 1}
    df = pd.DataFrame([{
        "Date": occ["date"], "Occasion": occ["occasion"],
        "Priority": occ.get("priority", "medium").replace("_", " ").title(),
        "Score": priority_score.get(occ.get("priority", "medium"), 2),
    } for occ in upcoming[:21]])
    fig = px.bar(df, x="Date", y="Score", color="Priority", hover_data=["Occasion"],
                 title="Promotion opportunities, next 3 weeks", labels={"Score": "Priority"},
                 color_discrete_map={"Very High": INDIGO, "High": INDIGO_LIGHT, "Medium": AMBER, "Low": GREY})
    fig.update_xaxes(tickangle=-45)
    return style_fig(fig, 380)


def chart_department_uplift(system: "PromotionOptimizationSystem") -> Optional[go.Figure]:
    rows = [(d, i) for d, i in system.family_insights.items()
            if is_model(i) and system.items_df is not None and d in set(system.items_df["family"])]
    if not rows:
        return None
    rows.sort(key=lambda r: r[1]["uplift"])
    names = [d for d, _ in rows]
    fig = go.Figure()
    fig.add_bar(y=names, x=[i["uplift_display"] * 100 for _, i in rows], orientation="h",
                name="Display only", marker_color=INDIGO_LIGHT)
    fig.add_bar(y=names, x=[i["uplift_mailer"] * 100 for _, i in rows], orientation="h",
                name="Mailer only", marker_color=GREEN)
    both = dict(y=names, x=[i["uplift"] * 100 for _, i in rows], orientation="h",
                name="Display + mailer", marker_color=INDIGO)
    if all("uplift_low" in i for _, i in rows):
        both["error_x"] = dict(type="data", symmetric=False, color="#6B7280", thickness=1.2,
                               array=[(i["uplift_high"] - i["uplift"]) * 100 for _, i in rows],
                               arrayminus=[(i["uplift"] - i["uplift_low"]) * 100 for _, i in rows])
    fig.add_bar(**both)
    fig.update_layout(barmode="group", title="Model-estimated uplift by department (%)", bargap=0.25)
    return style_fig(fig, 470)


def chart_accuracy_levels(levels: dict) -> go.Figure:
    names = {"product_week": "Per product, per week", "department_week": "Per department, per week",
             "total_week": "Whole store, per week"}
    keys = [k for k in names if k in levels]
    fig = go.Figure()
    fig.add_bar(x=[names[k] for k in keys], y=[levels[k]["model"]["accuracy"] * 100 for k in keys],
                name="LightGBM model", marker_color=INDIGO,
                text=[f"{levels[k]['model']['accuracy']:.0%}" for k in keys], textposition="outside")
    fig.add_bar(x=[names[k] for k in keys], y=[levels[k]["baseline"]["accuracy"] * 100 for k in keys],
                name="Simple guess (4-week average)", marker_color=INDIGO_LIGHT,
                text=[f"{levels[k]['baseline']['accuracy']:.0%}" for k in keys], textposition="outside")
    fig.update_layout(barmode="group", title="Forecast accuracy on unseen weeks (1 − WAPE)",
                      yaxis=dict(range=[0, 110], title="%"))
    return style_fig(fig, 380)


# ──────────────────────────────────────────────────────────────────────────────
# SHOP-OWNER WORDING
# ──────────────────────────────────────────────────────────────────────────────

FRIENDLY_NAMES = {
    "GROCERY": "Grocery", "DRUG GM": "Health & Household", "MEAT-PCKGD": "Packaged Meat",
    "PRODUCE": "Fruit & Vegetables", "DELI": "Deli", "MEAT": "Fresh Meat", "PASTRY": "Bakery",
    "NUTRITION": "Health Foods", "SEAFOOD-PCKGD": "Packaged Seafood", "SEAFOOD": "Fresh Seafood",
    "SALAD BAR": "Salad Bar", "FLORAL": "Flowers", "SPIRITS": "Spirits", "COSMETICS": "Cosmetics",
    "COUPON": "Coupons", "CHEF SHOPPE": "Ready Meals", "TRAVEL & LEISURE": "Travel & Leisure",
    "AUTOMOTIVE": "Car Care", "RESTAURANT": "Restaurant",
}


def nice(dept: str) -> str:
    """Readable department name for shop owners."""
    return FRIENDLY_NAMES.get(dept, str(dept).replace("-", " ").title())


def boost_text(uplift: float) -> str:
    return f"+{uplift:.0%}"


def times_text(uplift: float) -> str:
    return f"about {1 + uplift:.1f}× normal sales"


def how_to_promote(info: dict) -> str:
    if not is_model(info):
        return "Weekly flyer, or a display stand near the entrance"
    d, m = info["uplift_display"], info["uplift_mailer"]
    if m >= 2 * d:
        return "Put it in the weekly flyer. A display stand in the shop adds a little more."
    if d >= 2 * m:
        return "Put it on a display stand in the shop."
    return "Use both: the weekly flyer and a display stand in the shop."


def days_until(date_str: str) -> int:
    return max(0, (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days)


def when_text(days: int) -> str:
    return "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")


def nice_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%a %d %b")


def promotable_departments(system: "PromotionOptimizationSystem") -> list[str]:
    return system.get_ordered_families(10_000) if system.items_df is not None else []


def chart_boost(system: "PromotionOptimizationSystem", depts: list[str]) -> go.Figure:
    infos = [system.family_insights.get(d, system.get_default_insight()) for d in depts]
    rows = [(nice(d), i["uplift"], is_model(i)) for d, i in zip(depts, infos)]
    rows.reverse()
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows], x=[r[1] * 100 for r in rows], orientation="h",
        marker_color=[INDIGO if r[2] else "#D1D5DB" for r in rows],
        text=[boost_text(r[1]) for r in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text} more sales<extra></extra>",
    ))
    fig.update_layout(title="Expected extra sales when promoted", xaxis=dict(title="", ticksuffix="%"))
    return style_fig(fig, 60 + 46 * len(rows))


# ──────────────────────────────────────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────────────────────────────────────

def render_home(system: PromotionOptimizationSystem) -> None:
    page_title("This week's promotion plan", "What to put on offer, how, and when.")
    if system.items_df is None:
        st.warning("Upload your product list in the sidebar to get started.")
        return

    depts = promotable_departments(system)
    ins = system.family_insights
    best = depts[0] if depts else None
    occ = system.upcoming_occasions[0] if system.upcoming_occasions else None
    ou = (system.model_metrics or {}).get("overall_uplift")

    c = st.columns(3)
    if best:
        info = ins.get(best, system.get_default_insight())
        kpi(c[0], "Best department to promote", nice(best), f"sales can rise {boost_text(info['uplift'])}", accent=True)
    if ou and ou["mailer"] >= ou["display"]:
        kpi(c[1], "Best way to promote", "Weekly flyer",
            f"{boost_text(ou['mailer'])} on average · display stand {boost_text(ou['display'])}")
    elif ou:
        kpi(c[1], "Best way to promote", "Display stand",
            f"{boost_text(ou['display'])} on average · flyer {boost_text(ou['mailer'])}")
    else:
        kpi(c[1], "Best way to promote", "Flyer + display", "use both for the biggest effect")
    if occ:
        kpi(c[2], "Next promotion day", occ["occasion"], f"{nice_date(occ['date'])} · {when_text(days_until(occ['date']))}")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        show_fig(chart_boost(system, depts[:6]))
        if any(not is_model(ins.get(d, {})) for d in depts[:6]):
            st.caption("Grey bars are rough estimates: there were too few past promotions in that department to be sure.")
    with right:
        rows = ""
        for i, d in enumerate(depts[:3]):
            info = ins.get(d, system.get_default_insight())
            top = info.get("top_categories") or []
            items = ", ".join(n.title() for n, _ in top[:2]) or "best sellers"
            if not is_model(info):
                items += " · rough estimate"
            rows += (f"<div class='date-row'><span><b>{i + 1}. {nice(d)}</b><br>"
                     f"<span class='when'>{items}</span></span><span>{boost_text(info['uplift'])}</span></div>")
        html(f"<div class='card'><div class='card-head'><h4>Top 3 picks</h4></div>{rows}</div>")
        rows = ""
        for o in system.upcoming_occasions[:5]:
            rows += (f"<div class='date-row'><span>{o['occasion']}</span>"
                     f"<span class='when'>{nice_date(o['date'])}</span></div>")
        html(f"<div class='card'><div class='card-head'><h4>Coming up</h4></div>{rows}</div>")


def render_suggestions(system: PromotionOptimizationSystem) -> None:
    page_title("What to promote", "Choose how many suggestions you want, then press the button.")
    if system.items_df is None:
        st.warning("Upload your product list in the sidebar to get suggestions.")
        return

    depts = promotable_departments(system)
    ss = st.session_state
    with st.form("suggestions_form", border=False):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        n = c1.number_input("How many suggestions?", min_value=1, max_value=len(depts),
                            value=min(ss.get("rec_n", 3), len(depts)), step=1)
        submitted = c2.form_submit_button("Show suggestions", type="primary", width="stretch")
    if submitted:
        ss.rec_n = int(n)
    if "rec_n" not in ss:
        st.info("Pick a number and press **Show suggestions**.")
        return

    cols = st.columns(2, gap="medium")
    for i, family in enumerate(depts[: ss.rec_n]):
        subset = system.items_df[system.items_df["family"] == family]
        info = system.family_insights.get(family, system.get_default_insight())
        top = info.get("top_categories") or []
        if top and "class" in subset.columns and (subset["class"] == top[0][0]).any():
            subset = subset[subset["class"] == top[0][0]]
        item = subset.sample(1, random_state=i).iloc[0]
        occ = system.upcoming_occasions[i % len(system.upcoming_occasions)]
        category = str(item["class"]).strip() if "class" in item.index else ""
        has_name = bool(category) and not category.replace(".", "").isdigit()   # some lists only have number codes
        items = ", ".join(n.title() for n, _ in top) if top else (
            category.title() if has_name else "Best sellers in this department")
        sure = "" if is_model(info) else " <span class='badge rule'>Rough estimate</span>"
        with cols[i % 2]:
            html(f"""
            <div class="card">
              <div class="card-head"><h4>{i + 1}. {nice(family)}</h4><div>{sure}</div></div>
              <div class="uplift">{boost_text(info['uplift'])}</div>
              <div class="uplift-sub">expected extra sales: {times_text(info['uplift'])}</div>
              <div class="grid">
                <div><span>Put on offer</span><br>{items}</div>
                <div><span>When</span><br>{nice_date(occ['date'])} · {occ['occasion']}</div>
                <div><span>Offer idea</span><br>{info['best_promo']}</div>
                <div><span>Example item</span><br>#{item['item_nbr']}</div>
              </div>
              <div class="note"><p><b>How:</b> {how_to_promote(info)}</p><p>{info['timing_advantage']}</p></div>
            </div>""")
    if any(not is_model(system.family_insights.get(d, {})) for d in depts[: ss.rec_n]):
        st.caption("Rough estimate = there were too few past promotions in this department to be sure.")


def render_products(system: PromotionOptimizationSystem) -> None:
    page_title("My products", "Your product list and the promotion days coming up.")
    if system.items_df is None:
        st.warning("Upload your product list in the sidebar.")
        return

    df = system.items_df
    perishable = int(df["perishable"].sum()) if "perishable" in df.columns else 0
    c = st.columns(3)
    kpi(c[0], "Products", f"{len(df):,}")
    kpi(c[1], "Departments", f"{df['family'].nunique()}")
    kpi(c[2], "Fresh products", f"{perishable:,}", f"{perishable / len(df):.0%} of your list")

    st.write("")
    counts = df["family"].value_counts().head(12).sort_values()
    fig = px.bar(x=counts.values, y=[nice(d) for d in counts.index], orientation="h",
                 title="Products per department", labels={"x": "Products", "y": ""})
    fig.update_traces(marker_color=INDIGO, hovertemplate="%{y}: %{x:,} products<extra></extra>")
    left, right = st.columns(2, gap="medium")
    with left:
        show_fig(style_fig(fig, 430))
    with right:
        show_fig(chart_opportunity_calendar(system.upcoming_occasions))

    st.markdown("#### Every department at a glance")
    rows = []
    for family in promotable_departments(system):
        info = system.family_insights.get(family, system.get_default_insight())
        rows.append({
            "Department": nice(family),
            "Products": int((df["family"] == family).sum()),
            "Expected extra sales": boost_text(info["uplift"]) + ("" if is_model(info) else " (rough)"),
            "How to promote": how_to_promote(info),
            "Offer idea": info.get("best_promo", ""),
            "Best time": info.get("peak_days", ""),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_model(system: PromotionOptimizationSystem) -> None:
    page_title("Model details", "Technical page (only visible with ?details=1): how the uplift model works and how accurate it is.")
    m = system.model_metrics
    if not m:
        st.info("No model results found in models/uplift_results/. Run ml/uplift_model_colab.ipynb on Google Colab "
                "and unzip the results there.")
        return
    d = m.get("data", {})
    tw = d.get("test_weeks", ["?", "?"])

    c = st.columns(4)
    kpi(c[0], "Model", "LightGBM", "gradient-boosted decision trees")
    kpi(c[1], "Products", f"{d.get('products', 0):,}", "product × week rows")
    kpi(c[2], "Training rows", f"{d.get('rows', {}).get('train', 0):,}",
        f"weeks {d.get('train_weeks', ['?'])[0]}–{d.get('train_weeks', ['?', '?'])[1]} of 2017")
    kpi(c[3], "Test weeks", f"{tw[0]}–{tw[1]}", "never seen during training")

    st.markdown("#### What it does")
    st.markdown(
        "- **Data:** one year (2017) of shopping by 2,469 households at a US grocery chain "
        "(dunnhumby / 84.51° *The Complete Journey*), including which products were **on in-store display** "
        "or **in the weekly mailer**.\n"
        "- **Learns:** *recent sales + promotion this week + holidays + product type → units sold that week.*\n"
        "- **Uplift:** the same week is predicted twice, with the promotion **on** and **off**; the difference is the uplift.\n"
        "- **Honest test:** trained on earlier weeks and scored on the last weeks of the year, which it never saw."
    )

    st.markdown("#### How accurate is it?")
    levels = m.get("accuracy_levels")
    if levels:
        left, right = st.columns([3, 2], gap="large")
        with left:
            show_fig(chart_accuracy_levels(levels))
        with right:
            pw, dw = levels["product_week"], levels["department_week"]
            html(f"""<div class="card"><div class="card-head"><h4>In plain words</h4></div>
                <p class="muted"><b>Whole departments:</b> about <b>{dw['model']['accuracy']:.0%}</b> accurate per week,
                on par with a simple 4-week average ({dw['baseline']['accuracy']:.0%}).</p>
                <p class="muted"><b>Single products:</b> {pw['model']['accuracy']:.0%} vs {pw['baseline']['accuracy']:.0%}
                for the simple guess. Lower, because one product sells only 0–3 units a week in this household panel,
                but here the model is clearly better, and it's best in promotion weeks.</p>
                <p class="muted"><b>So:</b> trust it to <b>rank</b> departments and channels, not to promise exact sales numbers.</p>
                </div>""")
        st.caption(f"Accuracy = 1 − WAPE (total absolute error ÷ total units). Predictions are corrected by one factor "
                   f"measured on the validation weeks ({m.get('bias_correction_factors', {}).get('model', 1):.2f}× for the "
                   f"model), because converting back from the log scale makes them too low. Uplift is a ratio, so this "
                   f"correction doesn't change it.")
        if system.model_accuracy_dept is not None:
            with st.expander("Accuracy per department"):
                a = system.model_accuracy_dept.reset_index()
                a = a.rename(columns={"department": "Department", "test_units": "Units in test weeks",
                                      "accuracy_model": "Model accuracy", "accuracy_baseline": "Simple guess accuracy"})
                for col in ["Model accuracy", "Simple guess accuracy"]:
                    a[col] = (a[col] * 100).round(1)
                st.dataframe(a, width="stretch", hide_index=True)
                st.caption("Small departments have few units, so their accuracy swings a lot.")
    else:
        s = m.get("scores_test", {})
        st.write(f"Error (RMSLE) on unseen weeks: model **{s.get('lightgbm', {}).get('rmsle', float('nan')):.3f}** vs "
                 f"simple guess **{s.get('baseline_4w_mean', {}).get('rmsle', float('nan')):.3f}** (lower is better). "
                 "Re-run the notebook to see accuracy per product, department and store.")

    fig = chart_department_uplift(system)
    if fig is not None:
        st.markdown("#### Uplift by channel")
        show_fig(fig)

    st.markdown("#### What the model relies on")
    fig_dir = os.path.join(MODEL_DIR, "figures")
    imgs = [(os.path.join(fig_dir, f), cap) for f, cap in [
        ("03_feature_importance.png", "Recent sales averages matter most, then product category, then mailer and display."),
        ("04_pred_vs_actual.png", "Predicted vs actual (log scale) on the test weeks: points near the red line are good."),
    ] if os.path.exists(os.path.join(fig_dir, f))]
    if imgs:
        cols = st.columns(len(imgs), gap="large")
        for col, (path, cap) in zip(cols, imgs):
            with col:
                st.image(path, caption=cap, width="stretch")

    if system.model_dept is not None:
        st.markdown("#### Uplift estimates per department")
        t = system.model_dept.reset_index()
        out = pd.DataFrame({
            "Department": t["department"],
            "Display + mailer %": (t["uplift_both"] * 100).round(0),
            "Display only %": (t["uplift_display"] * 100).round(0),
            "Mailer only %": (t["uplift_mailer"] * 100).round(0),
            "Promo weeks in training": t.get("promo_weeks_in_training"),
            "Confidence": np.where(t["low_confidence"].astype(bool), "Low (not used)", "OK"),
        })
        if "uplift_both_low" in t.columns:
            out.insert(2, "90% range", [f"{lo:.0%}–{hi:.0%}" for lo, hi in zip(t["uplift_both_low"], t["uplift_both_high"])])
        st.dataframe(out.sort_values(["Confidence", "Display + mailer %"], ascending=[False, False]),
                     width="stretch", hide_index=True)

    st.markdown("#### Limitations")
    st.markdown(
        "- **Not proven cause and effect:** the retailer chose which products to promote, so uplift is an estimate.\n"
        "- **Household panel:** 2,469 households, not all store traffic; single-product numbers are noisy.\n"
        "- **One year of data:** Thanksgiving and Christmas fall only in the test weeks, so the model never learned them.\n"
        "- **Small departments** (cosmetics, spirits, floral…) have too few promotions and use rule-based values."
    )
    with st.expander("How to retrain"):
        st.markdown(
            "1. Open Google Colab → **File → Upload notebook** → `ml/uplift_model_colab.ipynb`.\n"
            "2. **Runtime → Run all** (CPU is fine, a few minutes).\n"
            "3. Unzip the downloaded `uplift_results.zip` into `models/uplift_results/` and restart the app."
        )


def _chat_bubble(role: str, text: str, provider: str = "", ts: str = "") -> str:
    """Return an HTML chat bubble for user or assistant turns."""
    import html as _html
    is_user = role == "user"
    cls = "user" if is_user else "bot"
    initials = "You" if is_user else "AI"
    meta = ts if is_user else ""
    safe = _html.escape(text).replace("\n", "<br>")
    return (f"<div class='chat-row {cls}'><div class='avatar {cls}'>{initials}</div>"
            f"<div><div class='bubble {cls}'>{safe}</div><div class='chat-meta {cls}'>{meta}</div></div></div>")


def server_key(name: str) -> str:
    """API key configured by the app owner (Streamlit secrets or environment variable)."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, "")).strip()


def render_chatbot(system: PromotionOptimizationSystem) -> None:
    page_title("Ask the assistant", "Ask anything about your promotions, in your own words.")
    if system.items_df is None:
        st.warning("Upload your product list in the sidebar first.")
        return

    ss = st.session_state
    groq_key, gemini_key = server_key("GROQ_API_KEY"), server_key("GEMINI_API_KEY")
    if not groq_key and not gemini_key:
        groq_key, gemini_key = ss.get("_groq_key", ""), ss.get("_gemini_key", "")
        with st.expander("Set up the assistant (one time)", expanded=not (groq_key or gemini_key),
                         icon=":material/key:"):
            st.caption("Paste a free Groq and/or Gemini API key. To set it up once for everyone, put "
                       "GROQ_API_KEY / GEMINI_API_KEY in .streamlit/secrets.toml.")
            k1, k2 = st.columns(2)
            ss["_groq_key"] = groq_key = k1.text_input("Groq API key", value=groq_key, type="password",
                                                       placeholder="gsk_...", help="Free at console.groq.com")
            ss["_gemini_key"] = gemini_key = k2.text_input("Gemini API key", value=gemini_key, type="password",
                                                           placeholder="AIza...", help="Free at aistudio.google.com")
        if not groq_key and not gemini_key:
            st.info("The assistant is not set up yet. Add a key above.")
            return

    ss.setdefault("chat_history", [])
    chain = ProviderFallbackChain(groq_key=groq_key, gemini_key=gemini_key)
    history = ss.chat_history
    if not history:
        html("""<div class="empty-chat"><p>Ask me anything about your promotions.</p>
                <p>For example: "What should I put on offer this weekend?"</p></div>""")
    else:
        bubbles = "".join(_chat_bubble("user", t["user"], ts=t.get("ts", ""))
                          + _chat_bubble("assistant", t["assistant"]) for t in history)
        st.markdown(f"<div class='chat-wrapper'>{bubbles}</div>", unsafe_allow_html=True)

    q = st.columns(4)
    quick_q: Optional[str] = None
    if q[0].button("What to promote?", width="stretch"):
        quick_q = "Which departments should I put on offer this week and why?"
    if q[1].button("Flyer or display?", width="stretch"):
        quick_q = "Should I use the weekly flyer or a display stand, and for which departments?"
    if q[2].button("Weekend plan", width="stretch"):
        quick_q = "Give me a simple weekend promotion plan for my top three departments."
    if q[3].button("Best items", width="stretch"):
        quick_q = "Which kinds of products sell much more when they are on offer?"

    query = quick_q or st.chat_input("Type your question...")
    if query:
        ts_now = datetime.now().strftime("%H:%M")
        with st.spinner("Thinking..."):
            try:
                messages = build_messages(query, system.rag_system, history)
                reply, prov = chain.complete(messages, max_tokens=260, temperature=0.35)
                ss.chat_history.append({"user": query, "assistant": reply, "provider": prov, "ts": ts_now})
            except Exception:
                st.error("The assistant couldn't answer right now. Please try again in a minute.")
                return
        st.rerun()

    if history and st.button("Clear chat", icon=":material/delete:"):
        ss.chat_history = []
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

SHOP_PAGES = {
    "Home":              (":material/home:",            render_home),
    "What to promote":   (":material/sell:",            render_suggestions),
    "My products":       (":material/inventory_2:",     render_products),
    "Ask the assistant": (":material/forum:",           render_chatbot),
}
DETAIL_PAGES = {"Model details": (":material/model_training:", render_model)}


def details_mode() -> bool:
    """The technical page is hidden from shop owners; open the app with ?details=1 to show it."""
    return st.query_params.get("details") == "1"


def handle_upload(system: PromotionOptimizationSystem) -> None:
    ss = st.session_state
    with st.expander("Upload a new product list"):
        uploaded = st.file_uploader("Product list (CSV)", type=["csv"], label_visibility="collapsed")
        st.caption("A CSV with at least the columns item_nbr and family (department).")
    # A file stays in the uploader across reruns: load each file only once
    if uploaded is None or ss.get("_loaded_file") == uploaded.file_id:
        return
    ss["_loaded_file"] = uploaded.file_id
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return
    missing = {"item_nbr", "family"} - set(df.columns)
    if missing:
        st.error(f"This file is missing the column(s): {', '.join(sorted(missing))}.")
        return
    system.load_data(df)
    ss.load_status = (True, f"Loaded {len(df):,} products.")
    ss.pop("rec_n", None)
    st.toast(f"Loaded {len(df):,} products in {df['family'].nunique()} departments.", icon=":material/check_circle:")


def render_sidebar(system: PromotionOptimizationSystem) -> str:
    pages = {**SHOP_PAGES, **(DETAIL_PAGES if details_mode() else {})}
    if st.session_state.get("page") not in pages:
        st.session_state["page"] = list(pages)[0]
    with st.sidebar:
        html("""<div class="brand">Promo <span>Planner</span></div>
                <div class="brand-sub">What to put on offer, how, and when</div>""")
        page = st.radio("Page", list(pages), format_func=lambda p: f"{pages[p][0]} {p}",
                        key="page", label_visibility="collapsed")

        html("<div class='side-label'>Your shop</div>")
        handle_upload(system)
        if system.items_df is not None:
            html(f"<div class='status'><span class='dot ok'></span>{len(system.items_df):,} products in "
                 f"{system.items_df['family'].nunique()} departments</div>")
        else:
            ok, msg = st.session_state.load_status
            st.warning(msg)

        if system.items_df is not None:
            html("<div class='side-label'>Coming up</div>")
            rows = "".join(f"<div class='occ'><b>{o['occasion']}</b> · {when_text(days_until(o['date']))}</div>"
                           for o in system.upcoming_occasions[:4])
            html(rows)
    return page if page in pages else list(pages)[0]


def main() -> None:
    if "promo_system" not in st.session_state:
        sys_obj = PromotionOptimizationSystem()
        ok, msg = sys_obj.auto_load_csv()
        st.session_state.promo_system = sys_obj
        st.session_state.load_status = (ok, msg)

    system: PromotionOptimizationSystem = st.session_state.promo_system
    page = render_sidebar(system)
    {**SHOP_PAGES, **DETAIL_PAGES}[page][1](system)


if __name__ == "__main__":
    main()
