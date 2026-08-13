# AlphaStream - FINS3645 Part B (z5590113_projectB)

Part B: funds, sentiment, and the app (DFF Stations 3-4). This folder is also the
public GitHub repository; the app entrypoint is streamlit_app.py at the root.

AlphaStream is a prototype FinTech app offering 17 systematically managed funds
(equity-only, crypto-only, combined, and sentiment-augmented), each backtested
strictly out of sample, plus a finance-aware news-sentiment index (FinSent) and
an investor allocation tool.

## How to run

    python -m pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py            # reproduces your results into results/
    streamlit run streamlit_app.py          # runs the app locally

Load raw data through src/data_access.py (see context/DATA_GUIDE.md); never commit
raw data. The deployed app, by contrast, reads your precomputed artifacts from
results/ - those ARE committed.

## Innovations (30% of Part B marks)

Innovation is the most heavily weighted Part B criterion, and the brief is
explicit that going beyond a short AI prompt is the bar. Every extension below
is implemented, reproduced by `scripts/run_part_b.py`, and evidenced in the
report and the app's Innovation tab:

- **FinSent lexicon** - ~80 finance words and ~65 phrases extending VADER; the
  false-neutral fraction falls from 48.1% to 29.1% (report Section 3.1, Figure 5).
- **17-fund shelf, five methods** - equity, crypto, and combined funds adding
  Minimum CVaR (tail risk) and a build-time check that methods genuinely differ
  (Section 1.2, Table 2).
- **Turnover as a cost proxy** - monthly one-way turnover reported and
  interpreted for every fund (Section 1.3, Table 2).
- **Fusion designs** - a constraint-preserving sentiment tilt and a long-short
  sentiment-momentum fund, both look-ahead-safe; the honest negative result is
  kept and explained (Section 4, Table 6, Figures 7-8).
- **AlphaStream design system** - an original palette and figure language shared
  across figures, app, and report (Section 5.2; src/design.py).
- **Results-first thin app** - the deployed app reads precomputed results/ and
  never imports nltk or re-runs an optimiser at request time (Section 5.3).

See Section 4 of report/report.docx (Table 5) for the full innovation map with
evidence pointers, and ai/log_innovation.md for how the portfolio was designed
and validated with AI.

## What is here

- streamlit_app.py    the app entrypoint (repo root)
- .streamlit/         app config
- PROJECT_BRIEF.md    the full assignment brief for your course (read this first)
- src/                your code (data_access is provided; portfolios/sentiment/fusion are yours)
- scripts/            runnable scripts that reproduce your results
- results/            your outputs: figures in results/figures/, tables in results/tables/, app data artifacts in results/data/
- context/            provided data guide and project context (do not edit)
- report/             your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/                 your prompt logs and AI notes
- requirements-dev.txt build/repro-only deps (nltk); keep them out of the deployed app
- AGENTS.md           your own AI agent instruction file (graded part of the AI-workflow pack)

## Deploy + hand in

This folder is its own GitHub repo, independent of fins-agent. Your AI agent can run
the check and push the repo; the browser deploy is yours (it needs your login). See
PROJECT_BRIEF.md Appendix D. In short:

    python scripts/check_handin.py        # your agent can run this
    # commit your precomputed app artifacts under results/ (the app reads them)
    # git init in this folder, then push the contents to a NEW private GitHub repo

Then YOU connect the repo on share.streamlit.io (entrypoint streamlit_app.py). At
hand-in, make the repo PUBLIC, submit the live URL + repo link, and also zip this
whole folder and upload the zip to Moodle.
