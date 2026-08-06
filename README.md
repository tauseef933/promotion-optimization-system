# Promotion Optimization System

AI-powered retail promotion planning platform developed at **NUST University** by **Tauseef Iqbal** (Department of Computer Science, 2025).

The system analyzes a product catalog, recommends category-level promotion strategies, visualizes expected uplift and timing opportunities, and answers planning questions through a RAG-grounded AI assistant.

---

## Overview

Retailers often choose promotions from habit rather than catalog structure, perishability, or calendar timing. This project turns a product CSV into actionable promotion guidance:

- **Recommendations** — priority categories with suggested promo types (BOGO, flash sale, bundle, volume discount, etc.), peak timing, and expected sales uplift
- **Analytics** — interactive charts for SKU distribution, ROI/uplift matrix, and an opportunity calendar
- **AI Assistant** — natural-language Q&A grounded in catalog statistics via lightweight retrieval-augmented generation (RAG), with multi-provider LLM failover

---

## Features

| Module | Description |
|--------|-------------|
| Smart recommendations | Category-wise strategies ordered by business priority, with response level, promo type, peak days, and uplift |
| Analytics dashboard | Plotly charts for portfolio mix, uplift vs. catalog depth, and upcoming weekends / paydays / holidays |
| RAG knowledge base | Per-category documents built from SKU counts, perishability, and curated business-rule insights |
| Multi-provider AI chat | Groq (`llama-3.3-70b-versatile`) primary → Gemini (`gemini-2.0-flash`) fallback on quota/auth errors |
| Flexible data loading | Auto-loads `items.csv` from the project root, or accepts upload via the sidebar |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| UI | [Streamlit](https://streamlit.io/) |
| Data | Pandas, NumPy |
| Visualization | Plotly Express |
| LLM clients | OpenAI-compatible SDK (Groq & Gemini endpoints) |
| Exploration notebook | Jupyter / Google Colab (`promotion_optimization_updated.ipynb`, earlier Gradio prototype) |

---

## Project Structure

```
promotion-optimization-system/
├── App.py                                      # Streamlit application (v3.0)
├── items.csv                                   # Sample product catalog
├── requirements.txt                            # Python dependencies
├── promotion_optimization_updated.ipynb        # Earlier Gradio / Colab prototype
├── LICENSE                                     # MIT
└── README.md
```

---

## Dataset

Sample catalog: **4,100 SKUs** across **33 product families**.

| Column | Description |
|--------|-------------|
| `item_nbr` | Unique item identifier |
| `family` | Product category (e.g. `GROCERY I`, `BEVERAGES`, `MEATS`) |
| `class` | Product class code |
| `perishable` | `1` if perishable, otherwise `0` |

Required columns for upload: `item_nbr`, `family`.

---

## Architecture

```
items.csv
    │
    ▼
PromotionOptimizationSystem
    ├── Category insights & holiday / occasion calendar
    ├── Recommendation engine (priority families + uplift rules)
    └── PromotionRAGSystem (per-category knowledge docs)
            │
            ▼
    Streamlit UI
    ├── Recommendations tab
    ├── Analytics tab (Plotly)
    └── AI Assistant tab
            │
            ▼
    ProviderFallbackChain
    Groq (primary) → Gemini (fallback)
```

Retrieval scores term overlap between the user query and category documents, with boosts for category-name matches and promotion vocabulary. Retrieved context is injected into the LLM prompt so answers stay tied to catalog data.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Optional (for AI Assistant): free API keys from [Groq Console](https://console.groq.com) and/or [Google AI Studio](https://aistudio.google.com)

### Installation

```bash
git clone https://github.com/tauseef933/promotion-optimization-system.git
cd promotion-optimization-system
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the application

```bash
streamlit run App.py
```

The app opens in your browser (typically `http://localhost:8501`).

1. Confirm `items.csv` loaded in the sidebar, or upload a compatible CSV.
2. Open **Recommendations** → set count → **Generate**.
3. Explore **Analytics** for distribution, ROI matrix, and calendar views.
4. In **AI Assistant**, paste a Groq and/or Gemini key, then ask promotion questions (e.g. highest-uplift categories, best timing for perishables).

Recommendations and analytics work without API keys. The chat tab requires at least one key.

---

## Example Questions for the AI Assistant

- Which categories have the highest expected uplift?
- What promotion type works best for MEATS?
- How should we promote perishable categories this weekend?
- Compare BEVERAGES and PRODUCE for a flash sale.

---

## Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
openai>=1.30.0
```

See [`requirements.txt`](requirements.txt).

---

## Future Enhancements

- Connect live sales and inventory APIs
- Replace rule-based uplift with trained predictive models
- Multi-store / regional analytics
- Cloud deployment (Streamlit Community Cloud, Hugging Face Spaces, or similar)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Developed at **NUST University** as an AI-driven business analytics initiative.

**Developer:** Tauseef Iqbal  
**Repository:** [github.com/tauseef933/promotion-optimization-system](https://github.com/tauseef933/promotion-optimization-system)
