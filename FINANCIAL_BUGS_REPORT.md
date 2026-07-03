# Poker44 Subnet — Financial Bug Report

**Scope:** Emission/payout correctness in the validator reward pipeline.
**Files:** `poker44/validator/forward.py`, `poker44/score/scoring.py`, `poker44/base/validator.py`
**Audience:** Engineering manager / subnet owner
**Severity:** High — Bug #1 sends 100% of a validator's miner emissions to the wrong UID; Bug #3 crashes the validator's reward computation (denial-of-service), and chained with Bug #1's never-cleared buffer becomes a persistent stall of all scoring/payouts.

> **Correction note (post-verification):** An earlier draft described Bug #3 as a miner
> "capturing 100% of emissions" with NaN scores. On execution that is **not** what happens —
> sklearn rejects NaN/Inf and raises `ValueError` *before* any reward is produced, so the real
> effect is a crash/denial-of-service, not a payout capture. Section 2 has been corrected. The
> recommended fixes are unchanged.

---

## 0. Background: how "weights" become money

In Bittensor, a validator does not pay miners directly. Each validator periodically
writes a **weight vector** on-chain (one weight per miner UID). The chain then splits the
subnet's TAO **emissions** among miners in proportion to the stake-weighted consensus of
those weights. So:

> **The weight vector a validator sets is the payout instruction. If the wrong UID has the
> highest weight, the wrong miner gets paid real TAO.**

This subnet runs **winner-take-all**: each cycle the validator computes a reward per miner,
picks the single highest-reward miner, and assigns it (effectively) 100% of the weight.
See `_select_weight_targets` in `poker44/validator/forward.py`:

```python
def _select_weight_targets(reward_map):
    ...
    sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
    winner_uid, winner_reward = sorted_rewards[0]      # highest reward wins everything
    if winner_reward <= 0.0:
        return [UID_ZERO], np.asarray([1.0])           # nobody positive -> burn to UID 0
    ...
    return [winner_uid], np.asarray([1.0])             # winner gets 100%
```

Because it is winner-take-all, **any error in the reward number for even one miner can flip
who receives the entire payout.** That is what makes the two bugs below financially serious
rather than cosmetic.

Both bugs live in the reward-computation path that drives this selection. This path is the
authoritative payout logic whenever the central backend weight vector is unavailable (the
documented fallback path), and it is the logic the whole scoring/leaderboard system is built
around.

---

## 1. BUG #1 — A miner that stops responding keeps winning 100% of emissions

### One-sentence summary
The validator scores each miner from its **last stored response**, with no check that the
miner actually replied recently, so a miner that answers well once and then goes offline
keeps being scored as if it were still performing — and under winner-take-all can keep
collecting the full payout indefinitely.

### The mechanism, step by step

**Step 1 — When a miner does not respond, its data is left frozen, not cleared.**
In `_run_forward_cycle` (`forward.py`), the per-miner response loop skips non-responders
without touching their prediction/label history:

```python
for uid, resp in zip(miner_uids, synapse_responses):
    if resp is None:                                  # miner offline / timed out
        bt.logging.debug(f"Miner {uid} returned None response")
        response_metadata[uid] = {"coverage_rate": 0.0, "latency_seconds": None}
        continue                                      # <-- prediction_buffer[uid] UNCHANGED
```

`prediction_buffer[uid]` and `label_buffer[uid]` are only appended to when the miner **does**
respond (later in the same loop). For a non-responder they keep whatever was stored the last
time it answered.

**Step 2 — The reward is computed from that frozen history, with no recency check.**
`_compute_windowed_rewards` (`forward.py`) reads the buffers and scores the last `window`
entries:

```python
window = current_sample_count            # = number of chunks in THIS cycle (~80 by default)
...
pred_buf  = validator.prediction_buffer.get(uid, [])
label_buf = validator.label_buffer.get(uid, [])
...
if len(pred_buf) < window or len(label_buf) < window:
    rewards.append(0.0)                  # only zeroed if there isn't even one full window
    continue

preds_window  = np.asarray(pred_buf[-window:], dtype=float)     # <-- last good response
labels_window = np.asarray(label_buf[-window:], dtype=bool)
rew, metric = reward(preds_window, labels_window)               # <-- full reward, stale data
rewards.append(rew)
```

Key point: the only thing that produces a `0.0` reward is **not having a full window of
history at all**. Once a miner has responded enough times to fill one window (~80 chunks —
typically a single good cycle), `pred_buf[-window:]` returns that same good response **forever**,
and the miner earns a full reward on every future cycle whether or not it ever replies again.
`coverage_rate` (which *does* track whether the miner responded) is computed and stored but is
**never used** to reduce the reward.

**Step 3 — Winner-take-all pays the stale winner.**
`_select_weight_targets` picks the maximum reward across all candidate miners and assigns it
100% of the weight. A permanently-offline miner with a stale high reward can be that maximum,
so it wins the full emission.

### Worked example (what you can tell your manager)

1. Cycle 10: Miner **A** (UID 42) submits an excellent response. Its buffer now holds ~80
   strong predictions; its reward is, say, **0.91**.
2. Miner A's operator turns the machine off. It never responds again.
3. Cycles 11, 12, 13, … : every honest, online miner is re-scored on fresh (harder / noisier)
   data and lands around **0.6–0.8**. Miner A is not re-scored on anything new — the code
   re-reads its frozen cycle-10 buffer and reports **0.91** again.
4. Winner-take-all compares 0.91 (offline A) vs 0.6–0.8 (online miners) → **offline Miner A
   wins and receives 100% of this validator's miner emissions**, cycle after cycle, until the
   validator process restarts or A's history finally scrolls out of the window (which, because
   only responders append to the buffer, may be *never*).

**Financial impact:** real TAO emissions are paid to a miner that is doing no work, and
simultaneously withheld from miners that are actually online and performing. This is a direct
misallocation of subnet incentive.

### How to prove it (repro without a live chain)
The reward path is pure Python and unit-testable. Pseudocode of the proof:

```python
# 1. Simulate one good response from UID 42, then simulate it going offline.
validator.prediction_buffer = {42: good_preds}      # ~80 strong predictions
validator.label_buffer      = {42: labels}
validator.current_eval_sample_count = len(labels)

# 2. Recompute rewards for a later cycle in which UID 42 did NOT respond.
rewards, _ = _compute_windowed_rewards(validator, miner_uids=[42])

# 3. Observe: rewards[0] is still the full ~0.91 -- NOT 0 -- despite no new response.
assert rewards[0] > 0.9        # BUG: offline miner still scored as a top performer
```

There is no branch in `_compute_windowed_rewards` that consults "did UID 42 respond this
cycle?", so this assertion holds. This is the proof the bug is real, not theoretical.

### Recommended fix
Tie the reward to a **fresh response in the current cycle**. Minimal options:

- **Gate on coverage:** if the miner's current-cycle `coverage_rate` is 0 (it did not
  respond), force its reward to 0 for this cycle instead of scoring stale history.
- **Or expire the buffer:** clear / zero-pad `prediction_buffer[uid]` and `label_buffer[uid]`
  when a miner returns `None` or an incomplete response, so old data cannot dominate the window.

Either fix ensures emissions only flow to miners that are actually producing responses now.

---

## 2. BUG #3 — Malformed (NaN/Inf) miner scores crash the reward cycle, and chained with Bug #1 cause a persistent scoring stall

### One-sentence summary
Miner-supplied scores are converted with `float()` but never checked for being finite, so a
malicious miner can return `NaN` (or `Infinity`); those values reach `sklearn`, which raises a
`ValueError`, **crashing the entire reward cycle so no miner is scored or paid** — and because
Bug #1 never clears the buffer, one such submission keeps re-crashing the validator on every
future cycle that UID is evaluated, even after the attacker goes offline.

### What actually happens (verified by running the code)

**Step 1 — Scores are accepted without a finiteness/range check.**
In `_run_forward_cycle` (`forward.py`):

```python
scores_f = [float(s) for s in scores]      # <-- accepts "NaN", "Infinity", -5, 10, ...
if len(scores_f) != len(chunks):
    ... discard ...
```

The **only** validation is the *count* of scores. `float("nan")`, `float("inf")`, and
arbitrary out-of-[0,1] values all pass and are stored in `prediction_buffer`. The reference
miner clamps its own scores to [0,1], but nothing in the protocol or the validator *requires*
a miner to — a custom/malicious miner can send anything.

**Step 2 — The reward computation crashes on the non-finite values.**
`reward()` in `poker44/score/scoring.py` calls `sklearn`'s `average_precision_score`, which
**explicitly rejects non-finite input**. Running it:

```
NaN preds, mixed labels : RAISED ValueError: Input contains NaN.
Inf preds, mixed labels : RAISED ValueError: Input contains infinity or a value too large...
```

So a NaN reward is **never produced** — sklearn raises first.

**Step 3 — The crash aborts the whole cycle; no weights are set.**
The `ValueError` is raised inside `_compute_windowed_rewards`, which has **no per-miner
try/except** around `reward()`. It propagates up to the top-level `forward()` wrapper, which
merely logs and swallows it:

```python
async def forward(validator):
    try:
        await _run_forward_cycle(validator)   # <-- raises here when it hits the poisoned buffer
    except Exception:
        ... log ...                            # swallowed; cycle ends with NO scoring, NO weights
```

So one miner's malformed scores stop **all** miners in that cycle from being scored or rewarded.
This is a **denial-of-service**, not a payout capture.

**Step 4 — Chained with Bug #1, the DoS becomes persistent.**
Bug #1's root defect is that the buffer is **never cleared**. The NaN values stay frozen in the
attacker's `prediction_buffer` forever. Every subsequent cycle where that UID is included,
`_compute_windowed_rewards` re-reads the NaN buffer → `reward()` raises again → the cycle aborts
again. The attacker therefore submits NaN **once**, disappears completely, and the validator's
scoring keeps crashing on every cycle that UID is sampled — until the process is restarted.

### Worked example (what you can tell your manager)

1. Attacker registers a miner and returns `[NaN, NaN, …]` once (correct count, so it isn't
   discarded on the length check).
2. The validator stores those NaNs in the attacker's buffer.
3. That cycle — and every later cycle the attacker's UID is evaluated — `reward()` raises
   `ValueError`, the whole cycle aborts, and **no miner receives a score or a weight update**.
4. The attacker goes fully offline. Because the buffer is never cleared (Bug #1), the poison
   persists and keeps crashing the reward cycle indefinitely.

**Impact:** this does **not** let the attacker win emissions. Instead it lets a single cheap
miner **stall the validator's entire local scoring/payout pipeline** — starving *every* honest
miner of the emissions that a functioning cycle would have distributed — for the cost of one
malformed response.

### Related latent defect (fix defensively, not currently reachable via scores)
The winner-selection guard is also weak on its own:

```python
if winner_reward <= 0.0:                 # float("nan") <= 0.0 is False -> NaN would slip through
    return [UID_ZERO], np.asarray([1.0])
```

Verified in isolation — if a NaN reward *did* reach selection, it would win instead of being
burned:

```
select {7: nan, 3: 0.85} -> [0, 7]       # the NaN entry, not honest UID 3, takes the winner slot
```

Today this path is **not reachable through miner scores** (sklearn crashes before a NaN reward
is ever produced), so it is a latent bug rather than an active payout-theft vector. It should
still be hardened, because any future code path that introduced a NaN reward without going
through sklearn would turn it into a real payout-capture bug.

### How to prove it (repro, no chain needed)

```python
import numpy as np
from poker44.score.scoring import reward

# Fact A -- malformed scores crash the reward computation (the DoS):
try:
    reward(np.array([float("nan")]*80), np.array([i % 2 for i in range(80)], dtype=bool))
except ValueError as e:
    print("crashed as expected:", e)      # -> "Input contains NaN."

# Fact B -- the selection guard is latently weak (would matter if a NaN reward existed):
assert (float("nan") <= 0.0) is False     # NaN bypasses the "<= 0.0" burn gate
```

Fact A demonstrates the actual (DoS) behavior. Fact B documents the latent guard weakness. The
DoS becomes *persistent* only because Bug #1 never clears the buffer — which is why the two
bugs are best fixed together.

### Recommended fix
Reject non-finite and out-of-range scores at the ingestion point, before they ever enter the
buffers:

```python
import math
scores_f = []
for s in scores:
    try:
        v = float(s)
    except (TypeError, ValueError):
        v = None
    if v is None or not math.isfinite(v):     # drop NaN / +-Inf / non-numeric
        # treat the whole response as invalid -> 0 coverage, no reward this cycle
        break
    scores_f.append(min(1.0, max(0.0, v)))    # clamp to the valid [0, 1] score range
```

And as defense-in-depth, harden the selection guard so it only accepts a finite, positive
winner:

```python
if not np.isfinite(winner_reward) or winner_reward <= 0.0:
    return [UID_ZERO], np.asarray([1.0])
```

---

## 3. Why these two are the priority

- Both are on the **miner → validator** boundary, which is adversarial by design (miners are
  financially motivated to game it).
- Both corrupt the payout in complementary ways — Bug #1 **misdirects** the entire
  winner-take-all payout to a dead-but-formerly-good miner; Bug #3 **halts** payouts entirely
  by crashing the scoring cycle, and (because Bug #1 never clears the buffer) keeps crashing it.
- They share a root cause and a fix surface: unvalidated miner output that is buffered and
  never expired. Fixing them together closes both the misdirection and the persistent DoS.
- Both fixes are **small, local, and unit-testable** with no chain interaction required, so
  they can ship with regression tests that prove the fix:
  - `test_offline_miner_earns_zero_reward` — a miner with stale history but no current-cycle
    response scores 0.
  - `test_nan_scores_do_not_crash_cycle` — a NaN/Inf-returning miner is rejected at ingestion;
    the reward cycle completes for all other miners instead of raising.

## 4. Suggested regression tests (acceptance criteria)

| Test | Setup | Expected result after fix |
|------|-------|---------------------------|
| Offline miner | UID has a full window of old good predictions, no response this cycle | reward == 0.0; it cannot be the winner |
| NaN scores | Miner returns `[NaN, …]` (correct count) | response rejected at ingestion; **cycle does not crash**; other miners still scored; NaN miner reward == 0.0 |
| Inf scores | Miner returns `[inf, …]` | response rejected at ingestion; **cycle does not crash**; other miners still scored; Inf miner reward == 0.0 |
| Persistent-DoS (chain) | UID submitted NaN once, then goes offline | poisoned buffer is cleared/expired; later cycles do not crash; scoring continues for everyone |
| Selection guard | reward_map contains a NaN value directly | `np.isfinite` guard sends it to the burn path (UID 0), never the winner slot |
| Out-of-range | Miner returns `[5.0, -2.0, …]` | scores clamped to [0,1] (or rejected); no reward inflation |
| Honest baseline | Miner returns valid [0,1] scores, responds every cycle | scored normally; unaffected by the fix |

---

*Prepared as part of a code review of the Poker44 subnet validator reward pipeline. Line
references are to the repository state at review time; both behaviors were confirmed by
reading the source and tracing the reward → selection → weight path.*
