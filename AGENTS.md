# AGENTS.md - AlphaStream, FINS3645 Part B

These are my working instructions for the AI coding assistant on the Part B
build (Funds, Sentiment & App). The full rules are in `PROJECT_BRIEF.md`; the
data traps are in `context/DATA_GUIDE.md`; read both before writing code. 

## What this project is

AlphaStream is my prototype FinTech app: several systematically managed funds
built from optimal portfolios (equity-only, crypto-only, and combined),
evaluated with strict out-of-sample backtests, plus a standalone news-sentiment
index across the equity sectors, all surfaced in a deployed Streamlit app.

- Data: 50 US large-caps across 10 sectors (5 each), 10 cryptos, and daily news
  headlines for the 50 equities, 2020-2023. Load ONLY through `src/data_access.py`
  (provided, never edit it). Raw data is never committed - the app and scripts
  read derived artifacts under `results/`.
- Part A lives in `../z5590113_projectA`. Never open or modify that folder from
  here; reuse its cleaned outputs only by reading them, never by editing them.
- Repo interpreter (run everything through it):
  - Windows: `..\..\.venv\Scripts\python.exe`
  - macOS/Linux: `../../.venv/bin/python`

## Folder layout

- `src/` - importable package: `data_access.py` (provided, frozen), `etl.py`,
  `portfolios.py`, `sentiment.py`, `fusion.py`, `factsheet.py`, `figures.py`,
  `design.py` (the AlphaStream design system).
- `scripts/` - `run_part_b.py` (one reproducible end-to-end build),
  `check_handin.py` (provided verifier - do not edit it).
- `results/data/`, `results/tables/`, `results/figures/` - committed, app-readable
  artifacts. Exact required names: `results/data/fund_returns.csv`,
  `results/data/fund_weights.csv`, `results/data/sector_sentiment_index.csv`,
  `results/tables/performance_metrics.csv`.
- `streamlit_app.py` at the project root - the deployed app entrypoint.
- `report/` - Word-first report (`report.docx` is the source; export `report.pdf`).
- `ai/` - prompt logs and AI-use notes.
- `context/` - provided docs, do not edit.

## Non-negotiable rules

1. **No look-ahead, ever.** A decision on day t uses only data through day t-1.
   - A decision date is valid only if at least `MIN_TRAINING` (504) prior rows
     exist in the returns history.
   - Weights are solved from `history.loc[returns.index < d]` only.
   - Sentiment signals are shifted by `lag >= 1` trading day before they touch a
     portfolio.
2. **Compute returns within each panel first** (equity on its ~252-day calendar,
   crypto on its own 365-day calendar), then left-merge crypto onto the equity
   trading calendar. Never merge price levels first and difference afterwards.
3. **Crypto sample ends 2023-12-31** - drop the 10 stray 2024-01-01 rows.
4. **Keep and document outliers**, never silently delete them.
5. **The app reads precomputed artifacts only.** The deployed app must not import
   nltk, run VADER, or re-run an optimiser at request time. nltk stays in
   `requirements-dev.txt` only.
6. **Per-name caps are per family**: 10% equity (50 names), 20% crypto (10 names,
   so at least 5 names held), 10% combined (60 names). A flat 10% cap collapses
   the crypto methods to equal weight - do not do that.
7. **Methods must genuinely differ.** Sanity-check that solver output actually
   changes across methods (the brief warns that optimisers can silently stall).
8. **All figures, tables, and app UI use the AlphaStream design system** in
   `src/design.py` (navy #0B1E36, cyan #17BEBB, amber #F0A500, coral #E4572E,
   teal, violet). No matplotlib defaults.
9. **Exhibits are self-contained**: caption, labelled axes, units, sample period,
   built via `src/design.py:caption()`.
10. **Honest results.** Negative results are fine and must be explained, not
    hidden. The fusion funds are a mixed bag: the sentiment momentum fund
    underperforms its equal-weight base, and the sentiment tilt improves on its
    max-Sharpe base but stays negative.
11. **Numbers must trace to data or a re-runnable computation.** Never invent a
    statistic, citation, or source (see `context/verify_ai_output.md`). If a
    claim cannot be verified, say so.

## Verification before I accept any change

- Tests: repo interpreter `-m pytest -q tests`
- Full build: repo interpreter `scripts/run_part_b.py`
- App smoke test: `..\..\.venv\Scripts\python.exe -c "import ast; ast.parse(open('streamlit_app.py', encoding='utf-8').read())"`, then
  `..\..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py`
- Pre-hand-in gate: repo interpreter `scripts/check_handin.py` - fix every
  `[FAIL]`; `[WARN]` items are reminders.

## How I work with you

- Ask for the smallest change that satisfies the task; do not refactor unrelated
  code.
- Do not add comments to code unless I ask; name things to be self-explanatory.
- Show your working for any number you produce; if you are unsure about a
  financial convention, ask before coding it in.
- When you propose a financial method or claim, flag what could be wrong rather
  than stating it confidently.
