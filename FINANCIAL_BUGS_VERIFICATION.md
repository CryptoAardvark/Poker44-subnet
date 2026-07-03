# Poker44 Subnet — Financial Bug Verification (executed, not just read)

**Method:** every claim below was checked by **running the actual repository code**
(`poker44.score.scoring.reward`, `poker44.validator.forward._compute_windowed_rewards`,
`_select_weight_targets`, `poker44.base.utils.weight_utils.convert_weights_and_uids_for_emit`)
against constructed inputs, then reading the real outputs. Where my earlier written review
asserted behavior I had not executed, the correction is called out explicitly.

This document is the evidence log behind `FINANCIAL_BUGS_REPORT.md`.

---

## Summary table

| # | Claim | Verdict | Notes |
|---|-------|---------|-------|
| 1 | Offline miner keeps a positive reward from its frozen buffer | **CONFIRMED** | reward stays 1.0 with no new response |
| 1b | Stale-but-strong offline miner beats a fresh-but-weak online miner (wins 100%) | **CONFIRMED** | requires a proper noisy test; see correction below |
| 1c | Exploit only fires when current-cycle window ≤ stored history length | **CONFIRMED (nuance)** | if this cycle has more chunks than stored, reward is zeroed |
| 1d | Exploit requires the miner to have been genuinely good before | **CONFIRMED (precondition)** | a weak stale miner loses to a strong fresh one |
| 2 | Prediction/label buffers are never trimmed (unbounded + enables permanent staleness) | **CONFIRMED** | only read-slicing; no del/clear/maxlen |
| 3 | NaN/Inf scores crash `reward()` → whole-cycle abort (DoS), not a payout capture | **CONFIRMED** | corrects earlier "captures 100%" framing |
| 3b | `_compute_windowed_rewards` has no per-miner guard → one poisoned buffer starves all miners that cycle | **CONFIRMED** | honest miner never scored |
| 3c | Selection guard `winner_reward <= 0.0` is latently NaN-unsafe | **CONFIRMED (latent)** | order-dependent; not reachable via scores today |
| 4 | Out-of-range **finite** scores (e.g. 1e9) inflate the reward | **REFUTED** | AP is rank-based; magnitude gives no advantage |
| 5 | uint16 weight conversion truncates (floors); tiny weights → 0 | **CONFIRMED** | should round, not floor |
| 6 | Winner-take-all ties broken by lowest UID (incumbent bias) | **CONFIRMED** | deterministic low-UID win |
| 7 | Fallback `set_weights` uses the EMA, so it is NOT winner-take-all | **CONFIRMED** | weight spread across recent winners |

---

## Detailed evidence

### Claim 1 — Offline miner keeps a positive reward (CONFIRMED)
Fed one miner (UID 42) a full window of good predictions, then called
`_compute_windowed_rewards` for a cycle in which it did **not** respond (buffer untouched).

```
offline miner 42 reward (no new response) = 1.0000  ap=1.000
```

The reward is recomputed from the frozen buffer with no recency check. Confirmed.

### Claim 1b — Stale-but-strong beats fresh-but-weak (CONFIRMED — with a correction to my own test)
**Correction:** my first attempt used perfectly-separable synthetic data, so *both* miners
scored AP = 1.0 (a tie won by the lower UID). That was a bad test, not a refutation — average
precision is **rank-based**, so any cleanly separable input yields 1.0. Re-run with realistic
noisy predictions (seeded), where the offline miner had skill 0.9 last cycle and the online
miner has skill 0.3 now:

```
rewards: {42: 1.0, 7: 0.7714}
selection: [0, 42] weights [0.0, 1.0]
VERDICT: STALE OFFLINE MINER 42 WINS 100%  <-- Bug #1 confirmed
```

The offline miner takes the entire payout over a genuinely-online competitor. Confirmed.

### Claim 1c — Window-size dependency (CONFIRMED nuance)
The exploit is gated by `if len(pred_buf) < window: reward = 0`. When the current cycle has
**more** chunks than the miner's stored history, the stale reward is zeroed:

```
reward when window(100) > stored(80) = 0.0000  (guarded to 0? True)
```

So the exploit only fires while the current-cycle chunk count is ≤ the miner's stored history
length. Chunk counts are backend-controlled and fairly stable (~80–100), so this holds in
practice, but it is a real precondition worth stating.

### Claim 1d — Requires prior good performance (CONFIRMED precondition)
Reversed the roles — weak stale miner (skill 0.3) vs strong fresh miner (skill 0.9):

```
rewards: {42: 0.8254, 7: 1.0}
selection: [0, 7]
VERDICT: weak stale miner CANNOT win -> exploit requires prior good performance
```

Confirms the earlier answer: Bug #1 is *"a past winner keeps winning after going offline,"*
not *"a loser wins by disappearing."* The frozen reward is only dangerous if it was high.

### Claim 2 — Buffers never trimmed (CONFIRMED)
Static scan of `forward.py`: the only writes to `prediction_buffer`/`label_buffer` are
`setdefault(uid, []).extend(...)`. There is no `del`, `.clear()`, slice-assignment, or
`maxlen` anywhere. Reads use `pred_buf[-window:]` (a copy, non-destructive). So the buffers
grow for the life of the process, which is also what makes the stale reward permanent and the
NaN poison persistent.

### Claim 3 — NaN/Inf crash the reward path (CONFIRMED; corrects earlier framing)
```
NaN : ValueError: Input contains NaN.
Inf : ValueError: Input contains infinity or a value too large ...
-Inf: ValueError: Input contains infinity or a value too large ...
```

`sklearn.average_precision_score` rejects non-finite input, so a NaN reward is **never
produced**. My original report called this "a miner captures 100% of emissions" — that was
**wrong** and has been corrected. The real effect is a crash.

### Claim 3b — One poisoned buffer aborts the whole cycle (CONFIRMED)
Two miners in one cycle: UID 7 with a NaN buffer, UID 3 honest.

```
=> whole-cycle abort reproduced: ValueError: Input contains NaN.
   (honest miner 3 never gets scored because miner 7 crashed the loop)
```

`_compute_windowed_rewards` has no per-miner try/except; the exception propagates to the
top-level `forward()` handler, which swallows it. No weights are set that cycle. Because the
buffer is never cleared (Claim 2), this recurs every cycle that UID is evaluated → persistent
denial-of-service from a single malformed submission.

### Claim 3c — Selection guard is latently NaN-unsafe (CONFIRMED latent)
```
select {7: nan, 3: 0.85} -> winner [0, 7]     # NaN entry takes the winner slot
select {3: 0.85, 7: nan} -> winner [0, 3]     # order-dependent
```

`float("nan") <= 0.0` is `False`, so a NaN reward would bypass the burn gate and can win
depending on dict/sort order. **Not reachable through miner scores today** (sklearn crashes
first), so it is a latent defect — fix defensively with `np.isfinite`.

### Claim 4 — Out-of-range finite scores do NOT inflate reward (REFUTED)
A tempting variant attack — send `1e9` instead of `1.0` — was tested:

```
reward(huge finite, correct ranking) = 1.0000
reward(perfect calibrated)           = 1.0000
```

Average precision and recall@FPR are **rank-based**, so absurd magnitudes give no advantage
over honest calibration. This attack does **not** work; do not spend effort defending against
it beyond the [0,1] clamp already recommended.

### Claim 5 — uint16 conversion truncates (CONFIRMED)
```
input weights: [0.99998, 1e-05, 1e-05]
uint16 out    : [65533, 0, 0]
0.4999975*65535 = 32767.336 -> uint16 32767 (floor, not round)
```

`(weights * 65535).astype(np.uint16)` floors. Weights below `1/65535 ≈ 1.5e-5` become exactly
0. Harmless under winner-take-all (winner = 1.0 → 65533+), but a real rounding loss for any
distributed/EMA vector. Use round-half-up before casting.

### Claim 6 — Tie broken by lowest UID (CONFIRMED)
```
tie among {9,50,200} -> winner [0, 9]   (x3, deterministic)
```

Equal rewards always route the full payout to the lowest UID — a persistent bias toward
earlier-registered miners.

### Claim 7 — Fallback weights are EMA, not winner-take-all (CONFIRMED)
Replaying the real `update_scores` EMA (alpha 0.05) with winners `[2,3,2,4,2]`:

```
fallback on-chain weights = [0.0, 0.0, 0.6005, 0.1895, 0.2100, 0.0]
=> spread across 3 UIDs, NOT a single winner
```

When the backend vector is unavailable, `set_weights` emits `self.scores / norm` — a soft
distribution over recent winners, not the single-winner allocation the code advertises. The
two payout models disagree and should be reconciled.

---

## What changed versus my earlier (read-only) review

| Item | Earlier (asserted) | After execution |
|------|--------------------|-----------------|
| Bug #3 effect | "miner captures 100% of emissions" | **crash / denial-of-service**; NaN reward never produced |
| Bug #1b test | claimed stale wins | **true**, but my first test was degenerate (AP=1.0 tie); re-proved with noisy data |
| Out-of-range finite scores | implied a manipulation vector | **refuted** — rank-based scoring ignores magnitude |
| NaN selection-guard bypass | presented as reachable | **latent only** — not reachable via scores today |

The two recommended fixes are unchanged and cover everything confirmed above:
1. **Reject non-finite and out-of-range scores at ingestion**, clamp to [0,1] — fixes the DoS
   (Claim 3/3b), the poison persistence, and the (harmless-but-untidy) out-of-range input.
2. **Gate the reward on a fresh current-cycle response and expire the buffer** — fixes Claim 1
   (all sub-parts) and Claim 2.
3. **Defensive extras:** `np.isfinite` in the selection guard (Claim 3c); round instead of
   floor in uint16 conversion (Claim 5); reconsider the tie-break (Claim 6) and the
   EMA-vs-winner-take-all inconsistency (Claim 7).

*All outputs above are reproducible from a clean checkout with the project dependencies
installed; the snippets used are self-contained and require no chain access.*
