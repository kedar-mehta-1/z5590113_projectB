# Prompt Log - App: Streamlit Investor Journey and Verification Pass

## Prompt 1: Results-First Streamlit App

**My prompt:**
"Build streamlit_app.py with five tabs (home, funds, sentiment, allocate,
data). It must read results/ CSVs and figures only - never re-run an optimiser
or import nltk. Use src/design.py's palette. Then run pytest and
scripts/check_handin.py and fix everything."

**AI output (key parts):**
```python
# AI's first version imported the scorer at the top of the module:
from nltk.sentiment.vader_sentiment import SentimentIntensityAnalyzer
```

**What was wrong:**
1. The nltk import and live VADER scoring would fail or be very slow on
   Streamlit Community Cloud, and it duplicated work the build already did.
   The brief's common-mistakes list calls this out explicitly.
2. The sentiment tab recomputed the neutrality numbers from raw headlines at
   request time, so the app could not start without the full data download on
   the free tier.

**My correction:**
- All scoring, optimisation, and backtesting run once in scripts/run_part_b.py
  and write CSVs and figures to results/. The deployed app only reads
  results/data/, results/tables/, and results/figures/.
- nltk and the optimiser imports live in src/ and requirements-dev.txt, never
  in the app import path; requirements.txt keeps the runtime slim.
- The sentiment tab reads the precomputed neutrality figure and the
  sector_sentiment_index.csv series; the numbers in the UI cards are the same
  values computed in the build and verified against the artifact.

**Why the correction matters:** The results-first architecture is what keeps the
app on the free tier and makes the repo reproducible from a clean checkout. As a
bonus it removes any code path for live optimisation or masked look-ahead at
request time.

---

## Prompt 2: Pre-Hand-In Cleanup and a Formatting Bug

**My prompt:**
"Clean up the code for the hand-in: fix long lines, unused imports, and unused
variables, and make sure the report tables format correctly."

**AI output (key parts):**
```python
# AI had written code with long lines, unused imports, and unused variables,
# plus a table cell that crashed on integer formatting.
```

**What was wrong:** Long lines and unused imports/variables across src/,
scripts/run_part_b.py, streamlit_app.py, and tests/test_core.py; and a report
table that formatted an integer count with an f-string expecting a float.

**My correction:**
- Cleaned up the long lines and removed the unused imports and variables, and
  fixed the integer-formatting bug in the report builder.
- Re-ran pytest (19 passed), the full scripts/run_part_b.py build (approx.
  15 s), an AST parse of streamlit_app.py, and scripts/check_handin.py
  (23 checks passed, no FAIL).
- Cleaned __pycache__ and .pyc before the hand-in zip.

**Why the correction matters:** The cleanup keeps the code readable and the
report tables reliable. check_handin.py is the pre-submission gate the brief
specifies, and passing it with zero FAIL is the concrete definition of "ready
to zip and deploy".
