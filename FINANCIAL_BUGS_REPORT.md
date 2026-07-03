# Poker44 Subnet — Financial Bug Report

**Scope:** Emission/payout correctness in the validator reward pipeline.
**Files:** `poker44/validator/forward.py`, `poker44/score/scoring.py`, `poker44/base/validator.py`
**Audience:** Engineering manager / subnet owner
**Severity of both issues:** High — each one can send 100% of a validator's miner emissions to the wrong UID.

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

## 2. BUG #3 — A miner can seize 100% of emissions by returning NaN / Infinity scores

### One-sentence summary
Miner-supplied scores are converted with `float()` but never checked for being finite or
in-range, so a malicious miner can return `NaN` (or `Infinity`), which slips past the
"positive reward" guard and can win winner-take-all — letting a miner **pay itself** with
garbage output instead of real detection work.

### The mechanism, step by step

**Step 1 — Scores are accepted without a finiteness/range check.**
In `_run_forward_cycle` (`forward.py`):

```python
scores_f = [float(s) for s in scores]      # <-- accepts "NaN", "Infinity", -5, 10, ...
if len(scores_f) != len(chunks):
    ... discard ...
```

The **only** validation is the *count* of scores. `float("nan")`, `float("inf")`, and
arbitrary out-of-[0,1] values all pass. The reference miner clamps its own scores to [0,1],
but nothing in the protocol or the validator *requires* a miner to — a custom/malicious miner
can send anything.

**Step 2 — NaN propagates into the reward.**
Those values go into `prediction_buffer` and are scored by `reward()` in
`poker44/score/scoring.py`, which calls `sklearn`'s `average_precision_score`. With `NaN`
predictions the resulting reward is `NaN` (NaN arithmetic stays NaN).

**Step 3 — The "must be positive" guard does not stop NaN.**
`_select_weight_targets` (`forward.py`):

```python
sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
winner_uid, winner_reward = sorted_rewards[0]

if winner_reward <= 0.0:                    # <-- intended to reject "no good miner"
    return [UID_ZERO], np.asarray([1.0])    #     burn to UID 0
...
return [winner_uid], np.asarray([1.0])      # winner gets 100%
```

The problem is IEEE-754 semantics:

- `float("nan") <= 0.0` evaluates to **`False`**, so a NaN reward **passes** the guard that
  was supposed to catch "nobody earned a positive score."
- `sorted()` on a list containing NaN is **order-undefined** — NaN comparisons all return
  False, so a NaN-reward entry can end up at position `[0]` (the winner slot) depending on
  input order.
- For `Infinity` scores it is even more direct: `inf` ranks strictly highest in the
  `average_precision_score` ordering, producing the top reward legitimately-looking.

Net effect: a miner submitting `NaN`/`Inf` scores can be selected as the winner and assigned
**100% of the weight**, i.e. the full emission — without doing any real detection.

### Worked example (what you can tell your manager)

1. Attacker registers a miner. Instead of detecting bots, it returns `[NaN, NaN, …]` (one per
   chunk, correct count so it isn't discarded).
2. The validator accepts the scores (only the count is checked), stores them, and computes a
   `NaN` reward for the attacker.
3. In selection, `NaN <= 0.0` is `False`, so the "no positive miner → burn" safety net does
   **not** trigger, and the attacker's entry can sort into the winner slot.
4. The attacker is assigned 100% of the weight and **collects the emission that should have
   gone to the best honest miner.**

**Financial impact:** an attacker games the payout with trivially cheap garbage output; honest
miners are starved. This is a self-dealing / incentive-theft vulnerability, not just a crash
risk.

### How to prove it (repro, no chain needed)
Two independent, checkable facts establish the bug:

```python
# Fact A -- the guard is broken:
assert (float("nan") <= 0.0) is False        # NaN bypasses the "<= 0.0" burn gate

# Fact B -- end to end, a NaN score yields a NaN reward that reaches selection:
uids, weights = _select_weight_targets({7: float("nan"), 3: 0.5})
# Depending on dict/sort order, UID 7 (the NaN miner) can occupy the winner slot
# instead of the honest UID 3, and it is NOT burned to UID 0.
```

Fact A is language-level and indisputable. Fact B demonstrates the NaN reaching the payout
decision. Together they prove a miner controls whether it wins by choosing its output format.

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
- Both can redirect **the entire winner-take-all payout** to the wrong UID — Bug #1 by
  accident (a dead miner), Bug #3 by deliberate attack (a NaN miner).
- Both fixes are **small, local, and unit-testable** with no chain interaction required, so
  they can ship with regression tests that prove the fix:
  - `test_offline_miner_earns_zero_reward` — a miner with stale history but no current-cycle
    response scores 0.
  - `test_nan_score_cannot_win` — a NaN/Inf-returning miner is rejected and cannot be the
    winner; the burn path triggers instead.

## 4. Suggested regression tests (acceptance criteria)

| Test | Setup | Expected result after fix |
|------|-------|---------------------------|
| Offline miner | UID has a full window of old good predictions, no response this cycle | reward == 0.0; it cannot be the winner |
| NaN scores | Miner returns `[NaN, …]` (correct count) | response rejected; reward == 0.0; not selected as winner |
| Inf scores | Miner returns `[inf, …]` | response rejected; reward == 0.0; not selected as winner |
| Out-of-range | Miner returns `[5.0, -2.0, …]` | scores clamped to [0,1] (or rejected); no reward inflation |
| Honest baseline | Miner returns valid [0,1] scores, responds every cycle | scored normally; unaffected by the fix |

---

*Prepared as part of a code review of the Poker44 subnet validator reward pipeline. Line
references are to the repository state at review time; both behaviors were confirmed by
reading the source and tracing the reward → selection → weight path.*
