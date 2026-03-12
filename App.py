"""
Promotion Optimization System v3.0
RAG-Enhanced AI Platform for Retail Promotion Planning

NUST University | Tauseef Iqbal | 2025
"""

import os
import re
import glob
import calendar
from datetime import datetime, timedelta
from typing import Optional

from openai import OpenAI
import numpy as np
import pandas as pd
import plotly.express as px
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
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    .rec-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .rec-card h4 {
        margin: 0 0 0.9rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        font-size: 1.05rem;
    }
    .rec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }
    .rec-grid p { margin: 3px 0; font-size: 0.9rem; }
    .analysis-box {
        background: rgba(0,0,0,0.22);
        border-radius: 6px;
        padding: 0.9rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
    .analysis-box p { margin: 3px 0; }
    .badge {
        display: inline-block;
        padding: 1px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .uplift-badge {
        background: rgba(255,255,255,0.18);
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }

    /* ── Chat interface ── */
    .chat-wrapper { display: flex; flex-direction: column; gap: 0; padding-bottom: 0.5rem; }
    .chat-row { display: flex; align-items: flex-end; gap: 10px; margin-bottom: 14px; }
    .chat-row.user { flex-direction: row-reverse; }
    .chat-row.bot  { flex-direction: row; }

    .avatar {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
        color: white; letter-spacing: 0.02em;
    }
    .avatar.user { background: #2a5298; }
    .avatar.bot  { background: #1a1a2e; }

    .bubble {
        max-width: 72%; padding: 0.7rem 1rem;
        border-radius: 14px; font-size: 0.9rem; line-height: 1.55;
    }
    .bubble.user {
        background: #2a5298; color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble.bot {
        background: #f4f6fb; color: #1a1a2e;
        border: 1px solid #e2e6f0; border-bottom-left-radius: 4px;
    }
    .bubble.bot strong { color: #1e3c72; }

    .chat-meta { font-size: 0.72rem; color: #aab0c0; margin-top: 4px; padding: 0 4px; }
    .chat-meta.user { text-align: right; }

    .provider-bar {
        display: flex; align-items: center; gap: 8px;
        padding: 0.45rem 0.9rem;
        background: #f4f6fb; border: 1px solid #e2e6f0;
        border-radius: 8px; margin-bottom: 1rem;
        font-size: 0.82rem; color: #555;
    }
    .provider-dot { width: 8px; height: 8px; border-radius: 50%; background: #27ae60; flex-shrink: 0; }
    .provider-dot.off { background: #ccc; }

    .empty-chat {
        text-align: center; padding: 2.5rem 1rem;
        color: #9aa0b0; font-size: 0.92rem;
        border: 1.5px dashed #dde1eb; border-radius: 12px; margin-bottom: 1.2rem;
    }
    .empty-chat p { margin: 0.3rem 0; }
    .empty-chat .hint { font-size: 0.82rem; margin-top: 0.6rem; color: #b0b8c8; }
</style>
""", unsafe_allow_html=True)


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
                "peak_days": "Weekend", "uplift": 0.25,
                "timing_advantage": "Standard timing applies.",
                "customer_behavior": "Regular purchase pattern.",
                "strategic_value": "Standard category value.",
            })

            doc_text = (
                f"Category: {family}\n"
                f"Total SKUs in catalog: {total_skus}\n"
                f"Perishable SKUs: {perishable_count} ({perishable_pct:.1f}%)\n"
                f"Distinct product classes: {class_count}\n"
                f"Customer response level: {insight['response']}\n"
                f"Best promotion type: {insight['best_promo']}\n"
                f"Peak shopping period: {insight['peak_days']}\n"
                f"Expected sales uplift: {insight['uplift'] * 100:.0f}%\n"
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
            f"{self.items_df['family'].nunique()} categories.\n",
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

    # Priority order used when selecting families for recommendations
    PRIORITY_FAMILIES = [
        "BEVERAGES", "MEATS", "GROCERY I", "PRODUCE", "BEAUTY",
        "DAIRY", "BREAD/BAKERY", "DELI", "POULTRY", "EGGS",
        "CLEANING", "PERSONAL CARE", "FROZEN FOODS", "HOME CARE",
        "LIQUOR,WINE,BEER",
    ]

    RESPONSE_COLORS = {
        "Very High": "#e74c3c",
        "High":      "#3498db",
        "Medium":    "#f39c12",
        "Low":       "#95a5a6",
    }

    def __init__(self):
        self.items_df:   Optional[pd.DataFrame]       = None
        self.rag_system: Optional[PromotionRAGSystem]  = None

        self.family_insights: dict = {
            "GROCERY I": {
                "response": "High", "best_promo": "BOGO (Buy One Get One)",
                "peak_days": "Weekend", "uplift": 0.35,
                "timing_advantage": "Weekend shopping creates bulk buying opportunities.",
                "customer_behavior": "Value-driven bulk purchases increase basket size.",
                "strategic_value": "Strong repeat purchase driver with high basket impact.",
            },
            "CLEANING": {
                "response": "Medium", "best_promo": "Volume Discount",
                "peak_days": "Month-end", "uplift": 0.25,
                "timing_advantage": "Month-end creates a stock-up mentality.",
                "customer_behavior": "Planned purchases align with household budgets.",
                "strategic_value": "Inventory clearance opportunity with stable margins.",
            },
            "BEVERAGES": {
                "response": "Very High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.45,
                "timing_advantage": "Weekend urgency amplifies impulse purchasing.",
                "customer_behavior": "Social occasion and entertainment purchases peak on weekends.",
                "strategic_value": "Strong traffic driver that lifts total basket size.",
            },
            "DAIRY": {
                "response": "Medium", "best_promo": "Percentage Off (20-40%)",
                "peak_days": "Weekly", "uplift": 0.20,
                "timing_advantage": "Weekly replenishment cycle drives consistent visits.",
                "customer_behavior": "Freshness priority leads to frequent return purchases.",
                "strategic_value": "Essential staple category that anchors weekly store visits.",
            },
            "BREAD/BAKERY": {
                "response": "High", "best_promo": "BOGO (Buy One Get One)",
                "peak_days": "Daily", "uplift": 0.30,
                "timing_advantage": "Daily consumption frequency sustains demand.",
                "customer_behavior": "Convenience-focused daily purchase habits.",
                "strategic_value": "Perishable urgency creates natural promotional momentum.",
            },
            "PERSONAL CARE": {
                "response": "Medium", "best_promo": "Bundle Deal",
                "peak_days": "Month-start", "uplift": 0.22,
                "timing_advantage": "Payday spending peaks in the first three days of the month.",
                "customer_behavior": "Strong brand loyalty correlates with monthly payday cycles.",
                "strategic_value": "High-margin premium positioning opportunity.",
            },
            "MEATS": {
                "response": "Very High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.50,
                "timing_advantage": "Weekend family meal planning drives premium category demand.",
                "customer_behavior": "Weekend grilling and family dining create strong pull.",
                "strategic_value": "Highest per-unit basket value with strong cross-sell potential.",
            },
            "AUTOMOTIVE": {
                "response": "Low", "best_promo": "Limited Time Offer",
                "peak_days": "Seasonal", "uplift": 0.15,
                "timing_advantage": "Seasonal maintenance cycles trigger infrequent but targeted demand.",
                "customer_behavior": "Need-based purchasing with low price sensitivity.",
                "strategic_value": "Margin protection through limited discount depth.",
            },
            "HOME CARE": {
                "response": "Medium", "best_promo": "Volume Discount",
                "peak_days": "Spring", "uplift": 0.23,
                "timing_advantage": "Spring cleaning season amplifies category relevance.",
                "customer_behavior": "Seasonal bulk-buying behavior with multi-unit purchases.",
                "strategic_value": "Extended shelf life enables longer promotional windows.",
            },
            "PRODUCE": {
                "response": "High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Daily", "uplift": 0.40,
                "timing_advantage": "Short shelf life creates natural urgency for rapid sell-through.",
                "customer_behavior": "Health-conscious shoppers prioritise freshness and respond to flash events.",
                "strategic_value": "Perishable nature generates organic scarcity-driven demand.",
            },
            "FROZEN FOODS": {
                "response": "Medium", "best_promo": "Bundle Deal",
                "peak_days": "Month-end", "uplift": 0.25,
                "timing_advantage": "Month-end meal planning drives multi-unit convenience purchases.",
                "customer_behavior": "Convenience-oriented shoppers plan multiple meals in a single trip.",
                "strategic_value": "Long shelf life supports extended promotional periods.",
            },
            "BEAUTY": {
                "response": "High", "best_promo": "Percentage Off (20-40%)",
                "peak_days": "Holiday", "uplift": 0.35,
                "timing_advantage": "Holiday gifting occasions create emotional purchasing peaks.",
                "customer_behavior": "Gift-driven and self-reward purchasing behavior.",
                "strategic_value": "Premium margins with strong loyalty development potential.",
            },
            "LIQUOR,WINE,BEER": {
                "response": "High", "best_promo": "Volume Discount",
                "peak_days": "Weekend", "uplift": 0.38,
                "timing_advantage": "Weekend social events and party planning align with purchase timing.",
                "customer_behavior": "Event-driven purchasing with multi-unit basket composition.",
                "strategic_value": "High absolute value per transaction with strong trade-up potential.",
            },
            "EGGS": {
                "response": "Medium", "best_promo": "Percentage Off (20-40%)",
                "peak_days": "Weekly", "uplift": 0.20,
                "timing_advantage": "Consistent weekly replenishment need across all customer segments.",
                "customer_behavior": "Essential protein staple with predictable replenishment cycles.",
                "strategic_value": "Traffic driver; discounts generate footfall across all demographics.",
            },
            "DELI": {
                "response": "High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.32,
                "timing_advantage": "Weekend family meal preparation drives fresh deli demand.",
                "customer_behavior": "Convenience-led meal-solution purchasing behavior.",
                "strategic_value": "High-margin fresh department anchor with strong attach rate.",
            },
            "POULTRY": {
                "response": "Very High", "best_promo": "Flash Sale (24-48 hours)",
                "peak_days": "Weekend", "uplift": 0.45,
                "timing_advantage": "Weekend meal planning generates the strongest weekly demand spike.",
                "customer_behavior": "Core weekly protein purchase tied to family meal planning routines.",
                "strategic_value": "Essential protein driver with significant basket-size multiplier.",
            },
        }

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

    def load_data(self, df: pd.DataFrame) -> None:
        self.items_df        = df
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
                        f"{df['family'].nunique()} categories from '{path}'."
                    )
            except Exception:
                continue

        return False, "No valid items.csv found. Please upload via the sidebar."

    # ── convenience helpers ──────────────────────────────────────────────────

    def get_ordered_families(self, limit: int) -> list[str]:
        """Return up to `limit` category names ordered by business priority."""
        if self.items_df is None:
            return []
        available = set(self.items_df["family"].unique())
        ordered   = [f for f in self.PRIORITY_FAMILIES if f in available]
        for f in available:
            if f not in ordered:
                ordered.append(f)
        return ordered[:limit]

    def get_default_insight(self) -> dict:
        return {
            "response": "Medium", "best_promo": "Percentage Off (20-40%)",
            "peak_days": "Weekend", "uplift": 0.25,
            "timing_advantage": "Standard timing strategy applies.",
            "customer_behavior": "Standard purchase pattern observed.",
            "strategic_value": "Standard category value.",
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
- Quote exact uplift percentages and SKU counts from the context.
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
# CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font_size=15,
    font=dict(size=11),
)


def chart_category_distribution(items_df: pd.DataFrame) -> px.bar:
    counts = items_df["family"].value_counts().head(12)
    fig = px.bar(
        x=counts.values, y=counts.index,
        orientation="h",
        title="Product Portfolio — SKU Distribution by Category",
        labels={"x": "SKU Count", "y": "Category"},
        color=counts.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=480, showlegend=False, **_CHART_LAYOUT)
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    return fig


def chart_roi_matrix(items_df: pd.DataFrame, family_insights: dict) -> px.scatter:
    counts = items_df["family"].value_counts().head(12)
    rows = []
    for family in counts.index:
        info = family_insights.get(family, {"uplift": 0.25, "response": "Medium"})
        rows.append({
            "Category":       family,
            "Expected Uplift (%)": info["uplift"] * 100,
            "SKU Count":      int(counts[family]),
            "Response Level": info.get("response", "Medium"),
        })
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="SKU Count", y="Expected Uplift (%)",
        size="SKU Count", color="Response Level",
        hover_data=["Category"],
        title="Promotion ROI Matrix — Uplift vs. Catalog Depth",
        color_discrete_map={
            "Very High": "#e74c3c",
            "High":      "#3498db",
            "Medium":    "#f39c12",
            "Low":       "#95a5a6",
        },
    )
    fig.update_layout(height=480, **_CHART_LAYOUT)
    return fig


def chart_opportunity_calendar(upcoming: list[dict]) -> px.bar:
    priority_score = {"very_high": 4, "high": 3, "medium": 2, "low": 1}
    rows = [
        {
            "Date":     occ["date"],
            "Occasion": occ["occasion"],
            "Priority": occ.get("priority", "medium").replace("_", " ").title(),
            "Score":    priority_score.get(occ.get("priority", "medium"), 2),
        }
        for occ in upcoming[:21]
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Date", y="Score",
        color="Priority",
        hover_data=["Occasion"],
        title="3-Week Promotion Opportunity Calendar",
        labels={"Score": "Priority Score"},
        color_discrete_map={
            "Very High": "#e74c3c",
            "High":      "#3498db",
            "Medium":    "#f39c12",
            "Low":       "#95a5a6",
        },
    )
    fig.update_layout(
        height=420,
        xaxis=dict(tickangle=-45),
        **_CHART_LAYOUT,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDERING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def render_recommendations(system: PromotionOptimizationSystem, num_recs: int) -> None:
    if system.items_df is None:
        st.warning("Load items.csv to generate recommendations.")
        return

    families = system.get_ordered_families(num_recs)

    for i, family in enumerate(families):
        subset = system.items_df[system.items_df["family"] == family]
        if subset.empty:
            continue

        item    = subset.sample(1, random_state=i).iloc[0]
        info    = system.family_insights.get(family, system.get_default_insight())
        occ     = system.upcoming_occasions[i % len(system.upcoming_occasions)]
        occ_dt  = datetime.strptime(occ["date"], "%Y-%m-%d")
        days_to = max(0, (occ_dt - datetime.now()).days)

        r_color = system.RESPONSE_COLORS.get(info.get("response", "Medium"), "#3498db")

        st.markdown(f"""
        <div class="rec-card">
            <h4>Recommendation {i + 1} &mdash; {family}</h4>
            <div class="rec-grid">
                <div>
                    <p><strong>Item Number:</strong> {item['item_nbr']}</p>
                    <p><strong>Category:</strong> {family}</p>
                    <p><strong>Promotion Type:</strong> {info['best_promo']}</p>
                    <p><strong>Peak Period:</strong> {info['peak_days']}</p>
                </div>
                <div>
                    <p><strong>Target Date:</strong> {occ['date']}</p>
                    <p><strong>Occasion:</strong> {occ['occasion']}</p>
                    <p><strong>Days Until:</strong> {days_to}</p>
                    <p><strong>Predicted Uplift:</strong>
                        <span class="uplift-badge">{info['uplift']:.0%}</span>
                    </p>
                </div>
            </div>
            <div class="analysis-box">
                <p><strong>Strategic Analysis</strong></p>
                <p>Response rate:
                    <span class="badge" style="background:{r_color};">{info['response']}</span>
                </p>
                <p>Timing: {info['timing_advantage']}</p>
                <p>Behavior: {info['customer_behavior']}</p>
                <p>Value driver: {info['strategic_value']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_analytics(system: PromotionOptimizationSystem) -> None:
    if system.items_df is None:
        st.warning("Load items.csv to view analytics.")
        return

    df         = system.items_df
    total      = len(df)
    n_cats     = df["family"].nunique()
    perishable = int(df["perishable"].sum()) if "perishable" in df.columns else 0

    counts = df["family"].value_counts().head(12)
    avg_uplift = np.mean([
        system.family_insights.get(f, {}).get("uplift", 0.25)
        for f in counts.index
    ]) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total SKUs",         f"{total:,}")
    c2.metric("Categories",          n_cats)
    c3.metric("Perishable SKUs",    f"{perishable:,}")
    c4.metric("Avg Expected Uplift", f"{avg_uplift:.1f}%")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_category_distribution(df),        use_container_width=True)
    with col_r:
        st.plotly_chart(chart_roi_matrix(df, system.family_insights), use_container_width=True)

    st.plotly_chart(
        chart_opportunity_calendar(system.upcoming_occasions),
        use_container_width=True,
    )

    # Opportunities table
    st.subheader("Ranked Promotion Opportunities")
    rows = []
    for family in counts.index:
        info = system.family_insights.get(family, {})
        rows.append({
            "Category":       family,
            "SKUs":           int(counts[family]),
            "Response":       info.get("response", "Medium"),
            "Best Promotion": info.get("best_promo", "Percentage Off"),
            "Peak Period":    info.get("peak_days", "Weekend"),
            "Uplift":         f"{info.get('uplift', 0.25):.0%}",
        })
    tbl = pd.DataFrame(rows).sort_values("Uplift", ascending=False)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


def _chat_bubble(role: str, text: str, provider: str = "", ts: str = "") -> str:
    """Return an HTML chat bubble for user or assistant turns."""
    is_user  = role == "user"
    cls      = "user" if is_user else "bot"
    initials = "You" if is_user else "AI"
    meta     = ts if is_user else (f"via {provider}" if provider else "")

    # Escape special HTML characters in the message text
    import html as _html
    safe = _html.escape(text).replace("\n", "<br>")

    meta_block = f"<div class='chat-meta {cls}'>{meta}</div>" if meta else ""
    return f"""
    <div class="chat-row {cls}">
        <div class="avatar {cls}">{initials}</div>
        <div>
            <div class="bubble {cls}">{safe}</div>
            {meta_block}
        </div>
    </div>"""


def render_chatbot(
    system: PromotionOptimizationSystem,
    groq_key: str,
    gemini_key: str,
) -> None:
    if system.items_df is None:
        st.warning("Load items.csv to use the AI assistant.")
        return

    if not groq_key and not gemini_key:
        st.info(
            "Enter at least one API key in the sidebar. "
            "Groq and Gemini are both free — adding both enables automatic failover."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict] = []
    if "last_provider" not in st.session_state:
        st.session_state.last_provider: str = ""

    chain = ProviderFallbackChain(groq_key=groq_key, gemini_key=gemini_key)

    # ── Provider status bar ──────────────────────────────────────────────────
    groq_dot   = "provider-dot"      if groq_key   else "provider-dot off"
    gemini_dot = "provider-dot"      if gemini_key else "provider-dot off"
    active_txt = " &rarr; ".join(
        (["Groq"] if groq_key else []) + (["Gemini"] if gemini_key else [])
    )
    st.markdown(f"""
    <div class="provider-bar">
        <span class="{groq_dot}"></span> Groq
        &nbsp;&nbsp;
        <span class="{gemini_dot}"></span> Gemini
        &nbsp;&nbsp;&mdash;&nbsp;&nbsp;
        Active chain: <strong>{active_txt}</strong>
        &nbsp;&nbsp;&mdash;&nbsp;&nbsp;
        Failover is automatic
    </div>
    """, unsafe_allow_html=True)

    # ── Conversation window ──────────────────────────────────────────────────
    history = st.session_state.chat_history

    if not history:
        st.markdown("""
        <div class="empty-chat">
            <p>No messages yet.</p>
            <p class="hint">Use the quick-question buttons below or type your own query.</p>
        </div>""", unsafe_allow_html=True)
    else:
        bubbles_html = "<div class='chat-wrapper'>"
        for turn in history:
            ts = turn.get("ts", "")
            bubbles_html += _chat_bubble("user",      turn["user"],      ts=ts)
            bubbles_html += _chat_bubble("assistant", turn["assistant"],
                                         provider=turn.get("provider", ""), ts=ts)
        bubbles_html += "</div>"
        st.markdown(bubbles_html, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #eef0f5;margin:0.4rem 0 1rem 0'>",
                unsafe_allow_html=True)

    # ── Quick-question buttons ───────────────────────────────────────────────
    st.caption("Quick questions")
    c1, c2, c3, c4 = st.columns(4)
    quick_q: Optional[str] = None
    if c1.button("Top categories",        use_container_width=True):
        quick_q = "Which categories should I prioritise for promotions this week and why?"
    if c2.button("Highest uplift",         use_container_width=True):
        quick_q = "Which categories have the highest expected uplift and what promotion type achieves it?"
    if c3.button("Weekend plan",           use_container_width=True):
        quick_q = "Give me a weekend promotion plan for the top three categories."
    if c4.button("Month-end strategy",     use_container_width=True):
        quick_q = "What is the best promotion strategy for the end-of-month period?"

    # ── Chat input ───────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about promotions, timing, ROI, category strategy...")
    query = quick_q or user_input

    if query:
        ts_now = datetime.now().strftime("%H:%M")

        # Show the user bubble immediately
        st.markdown(
            "<div class='chat-wrapper'>"
            + _chat_bubble("user", query, ts=ts_now)
            + "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner(""):
            try:
                messages      = build_messages(query, system.rag_system, history)
                reply, prov   = chain.complete(messages, max_tokens=220, temperature=0.35)

                st.markdown(
                    "<div class='chat-wrapper'>"
                    + _chat_bubble("assistant", reply, provider=prov, ts=ts_now)
                    + "</div>",
                    unsafe_allow_html=True,
                )

                st.session_state.chat_history.append({
                    "user":     query,
                    "assistant": reply,
                    "provider": prov,
                    "ts":       ts_now,
                })
                st.session_state.last_provider = prov

            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

        st.rerun()

    # ── Footer controls ──────────────────────────────────────────────────────
    if history:
        col_l, col_r = st.columns([5, 1])
        with col_r:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history  = []
                st.session_state.last_provider = ""
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Initialise system once per session
    if "promo_system" not in st.session_state:
        sys_obj = PromotionOptimizationSystem()
        ok, msg = sys_obj.auto_load_csv()
        st.session_state.promo_system   = sys_obj
        st.session_state.load_status    = (ok, msg)
        st.session_state.recs_generated = False

    system: PromotionOptimizationSystem = st.session_state.promo_system

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>Promotion Optimization System</h1>
        <p>AI-powered retail promotion planning &mdash; NUST University &mdash; Tauseef Iqbal</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        st.subheader("API Keys")
        st.caption(
            "Add both keys for automatic failover. "
            "If Groq hits its daily limit, Gemini takes over — and vice versa."
        )

        groq_key = st.text_input(
            "Groq API Key  (primary — fastest)",
            type="password",
            placeholder="gsk_...",
            help="Free at console.groq.com — no credit card needed.",
        )
        gemini_key = st.text_input(
            "Gemini API Key  (secondary — fallback)",
            type="password",
            placeholder="AIza...",
            help="Free at aistudio.google.com — no credit card needed.",
        )

        st.divider()
        st.subheader("Data Source")

        ok, msg = st.session_state.load_status
        (st.success if ok else st.warning)(msg)

        uploaded = st.file_uploader("Upload items.csv", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if {"item_nbr", "family"}.issubset(df.columns):
                    system.load_data(df)
                    st.session_state.load_status = (
                        True,
                        f"Loaded {len(df):,} items across {df['family'].nunique()} categories.",
                    )
                    st.session_state.recs_generated = False
                    st.rerun()
                else:
                    st.error("CSV must contain columns: item_nbr, family.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if system.items_df is not None:
            st.divider()
            st.subheader("Dataset Summary")
            st.write(f"Total items: **{len(system.items_df):,}**")
            st.write(f"Categories: **{system.items_df['family'].nunique()}**")
            if "perishable" in system.items_df.columns:
                pct = system.items_df["perishable"].mean() * 100
                st.write(f"Perishable share: **{pct:.1f}%**")

            st.divider()
            st.subheader("Upcoming Opportunities")
            for occ in system.upcoming_occasions[:6]:
                dt   = datetime.strptime(occ["date"], "%Y-%m-%d")
                days = max(0, (dt - datetime.now()).days)
                st.write(f"**{occ['occasion']}** — {days}d away")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_recs, tab_analytics, tab_chat = st.tabs([
        "Recommendations",
        "Analytics",
        "AI Assistant",
    ])

    with tab_recs:
        st.subheader("Promotion Recommendations")
        col_s, col_b = st.columns([3, 1])
        with col_s:
            num_recs = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)
        with col_b:
            if st.button("Generate", type="primary", use_container_width=True):
                st.session_state.recs_generated = True

        if st.session_state.recs_generated:
            render_recommendations(system, num_recs)
        else:
            st.info("Click Generate to produce data-driven promotion recommendations.")

    with tab_analytics:
        st.subheader("Analytics Dashboard")
        render_analytics(system)

    with tab_chat:
        st.subheader("Promotion Assistant")
        render_chatbot(system, groq_key, gemini_key)


if __name__ == "__main__":
    main()
