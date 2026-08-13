# AI_NOTES - how I directed and checked the AI (Part B)

This is my own account of how I used the AI assistant on the Part B build, what
it got right, and more importantly what it got wrong and how I caught it.
The prompt-by-prompt detail is in the task logs in this folder
(`ai/log_etl.md`, `ai/log_funds.md`, `ai/log_sentiment.md`, `ai/log_fusion.md`,
`ai/log_app.md`, `ai/log_innovation.md`) and the agent instructions I actually work
from are in `AGENTS.md` 

## Division of labour

I kept the assistant in an execution and drafting role and kept the financial
judgement to myself. Concretely: the assistant drafted the ETL, the backtest
loop, the VADER extension, the fusion transforms, the fact-sheet helpers, and
the app; I specified the method choices and the constraints, and I rewrote or rejected
the parts that got the finance wrong. I did
not accept any code without running it and reading the output.

## Where the AI helped most

- Volume: generating a coherent first cut of the whole pipeline in one session
  - ETL, walk-forward backtest, sentiment, fusion, app - was far faster than
    writing it by hand, and it kept the design system consistent across figures,
    tables, and app UI.
- The FinSent lexicon: the assistant proposed finance phrases and scores that I
  then sanity-checked and curated myself against real headlines. That gave me a
  starting lexicon to correct rather than a blank page.
- Repetitive refactors: the code-cleanup pass (long lines, unused imports, and
  unused variables) across five files was mechanical work the assistant did
  quickly and I spot-checked.

## Where the AI was wrong, and how I caught it

1. **Merging before differencing (ETL).** The first ETL merged equity and crypto
   price levels and then computed returns - the exact "spurious returns" trap in
   the data guide. I caught it because the combined panel had weekend-only crypto
   moves that could not exist in a real fund. Fix: returns computed within each
   panel first, then left-merged.
2. **Look-ahead in the backtest.** The first backtest estimated from
   `returns.index <= d` (and one version from the full panel) and used a
   whole-sample mean. I caught it by checking that the "estimation window" ended
   at the decision date instead of the day before. Fix: `history.loc[returns.index < d]`
   everywhere, and a minimum 504-row window before a date is eligible.
3. **The solver silently did nothing.** Risk parity came back numerically equal
   to equal weight and a flat 10% cap collapsed the crypto funds into one. The
   brief explicitly warns about this. I caught it by diffing the weight vectors
   across methods. Fix: risk-contribution-share formulation + per-family caps +
   a build-time assertion that methods differ.
4. **A no-op "extension".** The first FinSent patch edited VADER module globals
   that NLTK 3.8.1 no longer reads; the extension changed nothing. I caught it
   because the neutral fraction was identical before and after. Fix: patch
   `analyzer.constants`, add phrase head-words, then compare neutral fractions
   (~55% -> ~30%) as a guard.
5. **The app would not have deployed.** The first app imported nltk and
   re-scored headlines at request time. I caught it by reading the brief's
   common-mistakes list and checking the import graph. Fix: the app reads only
   precomputed `results/` artifacts.
6. **Bad CAGR.** The assistant computed CAGR as `mean * 252`, which ignores
   compounding. I caught it by reproducing the number by hand for one fund. Fix:
   total return over actual elapsed years.

## The honest-negative decision

The fusion results are a genuine mixed bag. The sentiment tilt improves on its
max-Sharpe base in the OOS sample (Sharpe -0.154 vs -0.226, CAGR -2.85% vs
-4.14%) but is still negative, while the momentum fund underperforms its
equal-weight base (Sharpe -0.094 vs 0.338, CAGR -0.72% vs +5.97%). I kept
these results and wrote them up as findings (noisy headline signal, monthly
rebalance too slow for a daily sentiment signal, weak signal persistence)
rather than tuning the fusion until it "won". That decision is itself part of
what the agent and I reviewed together: the marks are for evidenced original
work, not for a winning number.

## Innovations and how I validated them

Part B's most heavily weighted criterion (30%) rewards going beyond a short AI
prompt. I directed the build toward a portfolio of extensions and, for each one,
required an evidence artifact rather than a claim. `ai/log_innovation.md` records
the design and presentation work; the short version:

- **FinSent lexicon** - I asked the assistant to propose finance words and
  phrases, then curated the ~80 words and ~65 phrases myself and validated them
  against a hand-scored sample of 50 real headlines. The evidence is the neutral
  fraction: 48.1% under plain VADER, 29.1% under FinSent (Figure 5).
- **17-fund shelf** - beyond the required two-method combined fund, five methods
  across three families plus two sentiment funds, with a build-time assertion
  that the weight vectors genuinely differ (the brief warns that optimisers can
  silently stall).
- **Turnover as a cost proxy** - every fund reports average monthly one-way
  turnover, and Section 6.3 turns it into a concrete recommendation.
- **Fusion designs** - the tilt and momentum funds, validated for look-ahead
  safety, with the honest negative result reported rather than tuned.
- **Design system and thin app** - original palette and figure language
  (src/design.py), and an app that reads precomputed results only.

Each of these is mapped to its evidence in report Section 4, Table 5, and
surfaced in the app's Innovation tab, so a marker can find the innovation case
in any surface they open.
