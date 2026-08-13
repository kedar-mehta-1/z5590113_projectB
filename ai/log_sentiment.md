# Prompt Log - Sentiment: FinSent Lexicon and Sector Index

## Prompt 1: Finance-Aware VADER Extension

**My prompt:**
"Extend VADER with finance terms and phrases so headlines like 'beats earnings
estimates' no longer score 0. Build a sector sentiment index from ticker-day
scores, equal-weighting tickers, neutral when no news, and lag the signal."

**AI output (key parts):**
```python
# AI copied VADER's module-level globals and expected them to take effect:
from vaderSentiment.vaderSentiment import BOOSTER_DICT, SPECIAL_CASE_IDIOMS
BOOSTER_DICT.update(new_boosters)
```

**What was wrong:**
1. On the installed NLTK 3.8.1, VADER keeps its boosters and phrases on
   `analyzer.constants`, not module globals, so patching the old globals
   silently did nothing - the "extension" was a no-op. The neutral fraction was
   identical before and after.
2. VADER only fires a phrase rule when the phrase's head word has its own
   lexicon entry; phrases added without head-word entries never triggered.
3. The ticker-day scores were not lagged before building the index, so the
   same-day headline could influence the day-t decision (look-ahead).

**My correction:**
```python
# Attach the extension to the object NLTK actually reads, at build time:
analyzer.constants.BOOSTER_DICT.update(new_boosters)
analyzer.constants.SPECIAL_CASE_IDIOMS.update(new_idioms)
# ...and give every idiom's head word a small lexicon entry so the phrase
# branch runs.
```

**Why the correction matters:** A no-op "extension" is worse than none - it looks
like the work is done while the numbers are unchanged. The fix is verifiable:
the neutral fraction (exactly-zero scores) falls from 48.1% under plain VADER to
29.1% under FinSent, and a hand-scored sample of 50 real headlines agreed with
the machine scores. I curated the lexicon (~80 finance words, ~65 multi-word
phrases such as "beats earnings estimates" and "misses consensus") and sanity-
scored it against held-out headlines rather than trusting the first draft.

---

## Prompt 2: Sector Index and No-News Handling

**My prompt:**
"Aggregate ticker-day FinSent scores into ten sector indices for the app and
report. What do I do on days a sector has no headlines?"

**AI output (key parts):**
```python
# AI dropped no-news ticker-days and summed headline scores:
sector = news.groupby(["date", "sector"]).sum() / ...
```

**What was wrong:** Dropping no-news days silently changes the level of the
index, and summing headline scores lets a day with twenty bullish headlines
about one name swamp a day with one bearish headline about another. The index
then means "headline volume" rather than "news tone".

**My correction:**
- Ticker-day score = equal-weight average of that day's headline scores for the
  ticker, not the sum.
- Sector index = equal-weight average of ticker-day scores within the sector.
- No-news ticker-days enter as neutral 0.0 (carry-forward) rather than being
  dropped, and the news-coverage fraction per sector is reported so the reader
  can see how much of each series is fresh news vs neutral fill (Table 4 in the
  report).
- The whole index is shifted by one trading day before any portfolio decision.

**Why the correction matters:** The coverage figures make the honest limitation
visible - sparse sectors (Materials 84.4% coverage, Real Estate 88.9%, Utilities
92.4%) rest most often on the neutral fill. Making the aggregation rules
explicit in code is what lets the reader judge the index instead of assuming it
is clean.
