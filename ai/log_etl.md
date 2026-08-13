# Prompt Log - Station 1: ETL and Calendar Alignment

## Prompt 1: ETL Module

**My prompt:**
"Write src/etl.py that loads equity prices, crypto prices, and news headlines
through src/data_access.py. For each panel: check for missing dates, remove
duplicate ticker-date pairs, and flag outlier returns. For crypto: remove the
stray 2024-01-01 rows. For headlines: remove exact duplicates. Return an
integrity audit I can drop into the report, and compute daily returns."

**AI output (key parts):**
```python
# AI initially produced for headlines:
dup_mask = df.duplicated(subset=["ticker", "date"], keep="first")

# and for the combined panel:
combined_prices = pd.merge(eq_prices, cr_prices, on="date")
combined_returns = combined_prices.pct_change()
```

**What was wrong:**
1. The headline duplicate check used ticker+date, which flags every legitimate
   multiple-headline day (many tickers have 3-5 headlines a day). The data
   guide requires exact duplicates on ticker+date+title.
2. The AI merged equity and crypto price levels and only then computed returns.
   This is the exact "spurious returns" trap in the data guide: weekend-only
   crypto moves would leak into the combined panel as one-day returns a real
   equity-day fund could never trade on.
3. The stray 2024-01-01 crypto rows and the genuine outlier returns were to be
   deleted silently, which loses information the audit is meant to quantify.

**My correction:**
```python
# Dedup headlines on exact duplicates only:
df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")

# Compute returns within each panel first, then align calendars:
eq_returns = daily_returns(eq_prices)        # on the ~252-day equity calendar
cr_returns = crypto_daily_returns(cr_prices) # on the 365-day crypto calendar
cr_aligned = cr_returns.reindex(eq_returns.index)  # left-merge returns, not levels

# Keep outliers, count them, drop strays as a counted audit line.
```

**Why the correction matters:** Returns must be computed within each panel on its
own calendar and then aligned, never the reverse. The headline dedup key is a
common mistake flagged in the brief; ticker+date alone would remove thousands of
legitimate rows. The audit needs to count what was dropped (2,847 duplicate
headlines, 10 stray crypto rows, 384 outlier returns kept) so every number in
the report traces to an explicit decision.

---

## Prompt 2: News-to-Trading-Day Alignment

**My prompt:**
"Headlines are timestamped in timezone-aware UTC but the price calendar is
timezone-naive. How do I map each headline to the correct trading day so
sentiment never looks ahead?"

**AI output (key parts):**
```python
# AI suggested simple truncation:
news["date"] = news["date"].dt.date
```

**What was wrong:** Naive truncation leaves Saturday/Sunday headlines stranded
with no matching price row, and mixing tz-aware and tz-naive comparisons raises
or silently misaligns. The guide requires normalising the timezone first and
aligning each headline to an equity trading day.

**My correction:**
- Normalised all headline datetimes to tz-naive before any alignment.
- Mapped each headline to its equity trading day with a search on the trading
  index: the same day when it is a trading day, otherwise the next trading day.
- Preserved the raw headline text untouched (no stopword stripping or stemming)
  because VADER and its FinSent extension depend on original word order,
  punctuation, and intensity signals.

**Why the correction matters:** A headline dated Saturday applies to Monday's
bar, and with a one-day signal lag it becomes usable for Tuesday's decision. Any
misalignment here corrupts the sentiment index and the fusion funds in a way
that is invisible at the aggregate level.
