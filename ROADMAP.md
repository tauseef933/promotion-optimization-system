# Promotion Uplift Model: Roadmap & Progress Tracker

> This is the single place to see **what is done, what is in progress and what is next**.
> It is updated after every step. Last updated: **2026-11-10**

---

## 1. Why we are doing this

The app shows "Predicted Uplift", but those numbers are **typed by hand** in `family_insights` (`App.py`).
No model was ever trained. We will **train a real model on real supermarket data**, measure how good it is,
and put its numbers into the app. Then the project (and the CV) are honest and explainable.

## 2. The idea in simple words

| Term | Simple meaning |
|---|---|
| **Training** | Start with an empty model and show it many examples (inputs + the correct answer) until it learns the pattern. **This is what we do here.** |
| **Fine-tuning** | Take a model that is *already trained* (e.g. Llama, BERT) and teach it a bit more on your data. That's for a later project (Roman Urdu LLM). |
| **Dataset** | dunnhumby / 84.51° **"The Complete Journey"**: one year (2017) of shopping by 2,469 households at a US grocery chain. For every product, store and week it also says whether the product was **on an in-store display** and/or **in the weekly mailer**. |
| **Model: LightGBM** | Builds hundreds of small yes/no decision trees; each new tree fixes the mistakes of the previous ones. Best choice for tables of numbers, runs on a CPU. |
| **Features** | The inputs the model looks at: recent sales, display/mailer this week, holidays, department… |
| **Target** | What the model predicts: units of a product sold in a week (as `log(1 + units)`, so best-sellers don't dominate). |
| **Uplift** | Ask the trained model twice, with the promotion ON and OFF. The difference is the promotion's effect. (ML name: "S-learner".) |
| **Train / validation / test** | Learn on weeks 9–38, use weeks 39–44 to decide when to stop, and score on weeks 45–52, which the model **never saw**. |
| **Baseline** | A simple guess ("average of the last 4 weeks"). The model must beat it, or it's useless. |
| **RMSLE / MAE / WAPE** | Error scores. **Lower = better.** RMSLE = error on the log scale; MAE = average error in units; WAPE = total error ÷ total sales. |
| **Early stopping** | Stop adding trees once the validation score stops improving, so the model doesn't just memorise (overfitting). |
| **Data leakage** | The model accidentally "seeing the future". We only use earlier weeks for sales features. |

**Honest limitation:** the retailer didn't choose promotions at random (they promote items that already sell well),
so the uplift is a **model estimate**, not proven cause and effect. We say this openly.

## 3. Status board

**Legend:** ✅ done · 🔄 in progress · ⏳ next / waiting on you · ⬜ not started

> **Where things run:** training runs on **Google Colab** (free). Your PC only runs the Streamlit app.
> Everything for Colab is in **one notebook: `ml/uplift_model_colab.ipynb`**.

### Phase 0: Setup
- ✅ Create branch `feature/uplift-model`
- ✅ Create this `ROADMAP.md`
- ✅ Tried Kaggle Favorita data → blocked: Kaggle identity verification failed → switched to Colab + dunnhumby

### Phases 1–5: one Colab notebook (`ml/uplift_model_colab.ipynb`)
- ✅ Written: download data → clean → explore → features → baseline → LightGBM → uplift
- ✅ Test run on the PC with the real data (about 1 minute), see section 6
- ✅ **Run on Colab** by you; results unzipped into `models/uplift_results/`
- ✅ Colab numbers match the PC test run exactly (same scores, same 559 trees): the pipeline is reproducible

What the notebook does, step by step:
1. Downloads 3 files (37 MB) from the public `completejourney` R package on GitHub
2. Removes fuel/misc (their "quantity" is gallons) and adds up all stores → units per **product per week**
3. Keeps products sold in ≥ 20 weeks before the test period (~8,000 products)
4. Promotion features: share of stores with the product **on display** / **in mailer** that week
5. Charts: weekly units, raw promo lift per department
6. Features: past sales (earlier weeks only), promo history, holiday weeks, department / category / private label
7. Baseline (4-week average) vs LightGBM, scored on weeks 45–52
8. Uplift per department and per category for **display only / mailer only / both**
9. Saves `department_uplift.csv`, `category_uplift.csv`, `metrics.json`, `lgb_model.txt` and charts as a zip

### Phase 6: Plug into the app ✅
- ✅ **Decided:** the app switches to the dunnhumby catalog (section 5)
- ✅ **Catalog:** `items.csv` = the 8,741 modelled dunnhumby products (`item_nbr` = product id, `family` = department,
  `class` = product category, `perishable`, `brand`). The notebook now exports it as `product_catalog.csv`.
- ✅ **Model numbers in the app:** `App.py` reads `models/uplift_results/` on start
  - Cards show **Model-estimated Uplift (display + mailer)** plus display-only and mailer-only
  - Response level is computed from the model uplift (≥120% Very High, ≥90% High, ≥60% Medium)
  - Departments are ranked by model uplift; each card features a product from the department's best-responding category
  - `low_confidence` departments show a **Rule-based** value with the reason
- ✅ **Rule-based text** rewritten for dunnhumby departments and labelled "(rule)" in the app
- ✅ **AI assistant (RAG):** documents include model uplift, source label and top categories; the prompt tells the AI to say whether a number is model-estimated or rule-based
- ✅ Analytics: model performance line, ranked table with Display / Mailer / Source columns
- ✅ Holiday calendar kept (already US holidays, matching this US retailer)
- ✅ README rewritten: Uplift Model section (data, method, results, limitations, how to retrain)
- ✅ Checked in the browser: works **with** the model files (9 departments model-estimated) and **without** them (clear "No model results" note, labelled rule-based values)
- ✅ Fixed along the way: chart text was invisible in dark mode; raw HTML showed on cards without a category line
- ⚠️ Not tested: a live AI chat (needs your Groq/Gemini key). Its input context was checked and contains the model numbers.

### Phase 7: Finish
- ✅ Final results table (section 6)
- ✅ CV bullet + interview talking points (section 8)
- ⬜ Commit on `feature/uplift-model`, merge into `main`, push to GitHub (only after your OK)

### Phase 8: UI redesign + clearer accuracy
- ✅ **Accuracy at 3 levels** in the notebook (per product, per department, whole store), model vs simple guess → `accuracy_levels` in `metrics.json`, `accuracy_by_department.csv`, chart `06_accuracy_levels.png`
- ✅ **Found and fixed a bias:** converted back from log, the model's totals were **26% too low**, which made department-level accuracy look bad (73.5%).
  One correction factor measured on the validation weeks (1.28×) fixes it → **91% department-level accuracy**. Same treatment for the baseline (fair comparison). Uplift numbers unchanged (ratios).
- ✅ **Uplift ranges** (90%, from resampling products 300×) in `department_uplift.csv` / `category_uplift.csv`
- ✅ Notebook fix: each chart is closed after showing it (locally, the feature-importance chart had been drawn on top of the previous chart)
- ✅ **New look:** light, clean dashboard (`.streamlit/config.toml`: indigo accent, white cards), always light even on dark-mode PCs
- ✅ **Sidebar = navigation + status** (catalog / model / AI connected), CSV upload folded away, upcoming dates
- ✅ **New Overview page:** KPI tiles, uplift-by-department chart with ranges, "In short" summary, next promotion dates
- ✅ **New Model page:** what the model does, accuracy chart + plain-words explanation, per-department accuracy, feature importance, uplift table with ranges, limitations, how to retrain
- ✅ Recommendations as white cards in 2 columns with Model/Rule and response badges and ranges; Analytics restyled; API keys moved into the AI Assistant page
- ✅ Tested all 5 pages × 3 situations (new files, old files, no model files) with Streamlit's test runner: 0 errors
### Phase 9: Shop-owner version
- ✅ **Bug fixed: CSV upload hung forever.** The uploaded file stays in the upload box, so every reload loaded it again and restarted the page in an endless loop. Now each file is loaded once and the page updates immediately (tested with a 500-product upload)
- ✅ **"What to promote":** "How many suggestions?" box + **Show suggestions** button (choose 3 → exactly 3 cards)
- ✅ **Plain words for shop owners:** "Expected extra sales +118% (about 2.2× normal sales)", "Put it in the weekly flyer", readable department names (MEAT-PCKGD → Packaged Meat), "Rough estimate" instead of "rule-based"
- ✅ **Model information removed from the customer view.** The technical **Model details** page only appears when the app is opened with `?details=1` (e.g. `http://localhost:8501/?details=1`), for you and interviews
- ✅ Pages: Home · What to promote · My products · Ask the assistant (+ Model details in details mode)
- ✅ AI keys can be set once by the app owner in `.streamlit/secrets.toml` (`GROQ_API_KEY`, `GEMINI_API_KEY`) so shop owners never see a key box; the AI is told to answer in plain shop language
- ✅ Tested: all pages in shop mode, details mode and without model files: 0 errors, no technical words on shop pages
- ⏳ **YOU: re-run the notebook on Colab** (section 4) and unzip into `models/uplift_results/`. The files there now come from my PC run with the same code and the same numbers, but your own Colab run is the official one
- ⏳ **YOU:** look at the app (`streamlit run App.py`), then OK the commit

## 4. How to (re)run the notebook on Colab

1. Go to https://colab.research.google.com and sign in with your Google account.
2. **File → Upload notebook** → choose `D:\repo\promotion-optimization-system\ml\uplift_model_colab.ipynb`.
3. **Runtime → Run all**. The default CPU runtime is fine (no GPU needed). Takes about 5–10 minutes.
   If Colab warns "This notebook was not authored by Google", click **Run anyway**.
4. At the end your browser downloads **`uplift_results.zip`**. (If it doesn't, open the 📁 Files panel on the left
   and download `uplift_results.zip` by hand.)
5. Unzip it into `D:\repo\promotion-optimization-system\models\` and tell me "results are ready".

Read the charts and printed tables as you go; they're what you'll explain in interviews.
If a cell fails, copy the red error text and send it to me.

## 5. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-10 | LightGBM, not a transformer | Sales data is a table of numbers; tree models are the best fit and run on a CPU |
| 2026-11-10 | Uplift via S-learner (predict promo ON vs OFF) | Simplest honest method; easy to explain |
| 2026-11-10 | Switched from Kaggle Favorita to dunnhumby on Colab | Kaggle identity verification failed; dunnhumby data is public on GitHub and has real display/mailer promotion records |
| 2026-11-10 | Model at product × week level | Per-store sales are too thin (median 1 unit per store-product-week) |
| 2026-11-10 | Promotion = display + mailer, not price discount | Discounts are only visible when something was bought (that would leak the answer); display/mailer are recorded every week |
| 2026-11-10 | Product filter uses only pre-test weeks | Choosing products with test-week data would be cheating |
| 2026-11-10 | App switches to the dunnhumby catalog | Every number in the app then comes from the same retailer the model learned from; mapping onto Favorita (Ecuador) families would mix two retailers |
| 2026-11-10 | Bias correction factor from validation weeks | Log-trained predictions were 26% too low in total; one factor fixes aggregates, doesn't touch test data, doesn't change uplift |
| 2026-11-10 | Light dashboard theme, sidebar navigation, Overview + Model pages | Your choice; makes the model's evidence visible for demos and interviews |
| 2026-11-10 | Kept results in `models/uplift_results/`; model file and zip not in git | The CSV/JSON/charts are small and useful on GitHub; `lgb_model.txt` (3.7 MB) can be rebuilt from the notebook |

## 6. Results

Confirmed on Google Colab (2026-11-10): 8,741 products, 262k training rows, 70k test rows, 559 trees.
Files: `models/uplift_results/` (`metrics.json`, `department_uplift.csv`, `category_uplift.csv`, `figures/`).

| Model (test weeks 45–52) | RMSLE ↓ | MAE (units) ↓ | WAPE ↓ |
|---|---|---|---|
| Baseline (4-week average) | 0.638 | 1.94 | 61.5% |
| LightGBM | **0.587** | **1.82** | **57.7%** |

- LightGBM error is **8.0% lower** than the baseline; it helps most in promotion weeks (0.676 → 0.603).

**Forecast accuracy (1 − WAPE) on test weeks, after bias correction:**

| Level | LightGBM | Simple guess |
|---|---|---|
| Per product, per week | **43.2%** | 36.4% |
| Per department, per week | 91.2% | 91.9% |
| Whole store, per week | 92.4% | 93.9% |

Honest reading: the model clearly beats the simple guess **per product** and in **promotion weeks**. For whole departments both are about 91–92%,
so the model's value is the **promotion effect** (uplift), not a better total forecast. Use it to rank departments and channels, not to promise exact sales.
- Overall model-estimated uplift at typical store coverage: **display only +15%**, **mailer only +80%**, **both +92%**.
- Largest reliable departments: MEAT-PCKGD +121% (both), PRODUCE +118%, DRUG GM +107%, GROCERY +82%, DELI +80%.
- Small departments (cosmetics, spirits, seafood…) are marked `low_confidence`.
- Limitation: Thanksgiving and Christmas fall only in the test weeks, so the model never saw them in training.
- Feature importance: 13- and 8-week sales averages matter most, then product category, then **mailer share** and **display share**. Promotions really drive part of the prediction.

## 7. File map

| File | Purpose |
|---|---|
| `ROADMAP.md` | This tracker |
| `ml/uplift_model_colab.ipynb` | The whole training pipeline; runs on Google Colab |
| `models/` | Results from Colab: `department_uplift.csv`, `category_uplift.csv`, `metrics.json`, `figures/` (`lgb_model.txt` is not stored in git) |
| `App.py` | Streamlit app; reads the model's uplift from `models/uplift_results/` |
| `items.csv` | Product catalog: 8,741 dunnhumby products (same file as `models/uplift_results/product_catalog.csv`) |
| `README.md` | Project overview, model results, how to run and retrain |

## 8. For your CV and interviews

**CV bullet (Key Projects), every number from this project:**
> **Promotion Uplift Model & Planner** (Python, LightGBM, Streamlit, RAG): Trained a LightGBM model on one year of real grocery data
> (dunnhumby, 8.7K products) to estimate how much in-store displays and mailers lift sales. It beat a moving-average baseline by 8% on
> unseen weeks (RMSLE 0.587 vs 0.638) and estimated +80% uplift from mailers vs +15% from displays. Integrated the results into a
> Streamlit planner with a RAG assistant (Groq → Gemini failover).

Keep the internship bullet as it is now (what you really built in 2025). This is a separate project line, rebuilt in 2026.

**Interview talking points: be ready to explain these in your own words**
1. **Why LightGBM?** Table data with numbers and categories; tree boosting is the standard, strongest choice there, and fast on a CPU.
2. **Why product × week?** Per-store sales were too thin (median 1 unit); aggregating gives a usable signal.
3. **How did you avoid leakage?** Sales features only use earlier weeks; products were selected using pre-test weeks only; time-based split.
4. **Why display/mailer and not discounts?** Discounts are only visible when something was bought, so they leak the answer.
5. **How is uplift computed?** S-learner: same row predicted with promotion ON vs OFF; the ratio of summed units.
6. **Is it causal?** No. Promotions weren't random, so it's an estimate. Next step: double machine learning or an A/B test.
7. **What did the model rely on?** Recent 13/8-week averages most, then product category, then mailer and display share.
8. **How accurate is it?** Per product 43% vs 36% for a simple guess; per department ~91%, on par with the guess. I found that log-trained predictions were 26% too low in total and fixed it with a validation-based correction. The model's value is the promotion effect, not the total forecast.
9. **What would you improve?** More history (only 1 year; holidays only in the test weeks), price effects, store-level data, causal methods.
