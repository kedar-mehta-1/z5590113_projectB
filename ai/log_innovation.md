# Prompt Log - Innovation: Extension Portfolio and Evidence

## Prompt 1: Proposing the Extension Portfolio

**My prompt:**
"Part B's innovation criterion is 30% of the marks and rewards going beyond what
a short AI prompt would produce. Propose a portfolio of extensions for the
project - new funds, new uses of the news data, custom sentiment tooling, a
design system, evaluation methods - and for each one say how I would demonstrate
it with evidence, and what could go wrong."

**AI output (key parts):**
```text
The AI proposed: (1) an LSTM-based sentiment model, (2) an XGBoost return
predictor, (3) "add more funds", (4) a dark-mode theme. It listed them as a
checklist with no evidence plan and no failure analysis, and suggested an LSTM
"will perform better than VADER".
```

**What was wrong:**
1. The suggestions were shallow bolt-ons rather than integrated extensions.
   An LSTM on 146k short headlines cannot be shown to beat a curated lexicon
   within the coursework data and would be mostly a claim, not evidence.
2. "Add more funds" was unspecified - more of the same methods is not an
   innovation, and the brief says a wide shelf only earns marks if it is
   argued and evidenced.
3. The AI treated the portfolio as a checklist, which is exactly the
   "proposed more than shown" failure mode in the marking rubric.
4. The AI offered no honest-negative framing - it implied every extension
   should "improve" results, which conflicts with the brief's explicit
   statement that a careful extension with a negative result still earns the
   band.

**My correction:**
- Chose extensions that each produce a verifiable artifact: the FinSent
  lexicon (neutral fraction before/after), the 17-fund shelf (weight vectors
  differ across methods), turnover as a cost proxy (a column in every metrics
  table), the fusion designs (before-vs-after table and figures), the design
  system (every figure + the app), and the results-first thin app.
- Required an evidence pointer (figure, table, or app tab) for each extension
  before I would accept it.
- Framed the fusion explicitly as a test that might fail, and committed to
  reporting the negative result.

**Why the correction matters:** The marks are for evidenced original work, not
for a list of ideas or a winning number. An extension that cannot be shown
with a figure or table cannot be marked, and a shallow list of popular methods
reads as prompt-and-paste output. Tying each extension to an artifact is what
makes the innovation case auditable.

---

## Prompt 2: Evidence Artifacts for the Innovation Case

**My prompt:**
"The innovation criterion is graded across the whole project, not in one place.
The report prose is my own writing. Your job is to produce the code and
artifacts - an evidence-map table, the figures, and app surfaces - so a marker
finds the evidence in whatever surface they open first."

**AI output (key parts):**
```text
The AI answered with prose instead of artifacts: it proposed writing a long
paragraph about innovation for the report introduction and a plain text list
for the README. It produced no table, no figure, no app surface, and no
checklist item.
```

**What was wrong:**
1. It produced claims, not code. A paragraph names the extensions but shows
   nothing; the rubric rewards extensions shown with evidence.
2. It treated the report as the only surface; the app and README are equally
   likely to be opened first by a marker.
3. There was no cross-referencing discipline (each extension -> its evidence
   exhibit), so a reader could not verify a claim without hunting for it.

**My correction:**
- I wrote the Section 4 narrative myself. I directed the AI to generate the
  evidence-map table (Table 5): one row per extension, what it adds beyond the
  baseline, and the exact exhibit that evidences it (figure or table).
- Added an Innovation tab to the app that reads precomputed figures and shows
  the lexicon comparison, the fund shelf, the design system, the thin-app
  architecture, and the honest negative result.
- Added an Innovations section to the README pointing at the artifacts and this log + an AI_NOTES
  section, so the case is stated consistently on every graded surface.

**Why the correction matters:** A marker who opens any one surface - report,
app, README, or the AI pack - should find the innovation case backed by a
generated artifact, not assembled from claims. Table 5 is the single map: each
extension names the exhibit that proves it, and the app tab repeats the same
figures, so the report and the product cannot drift apart.
