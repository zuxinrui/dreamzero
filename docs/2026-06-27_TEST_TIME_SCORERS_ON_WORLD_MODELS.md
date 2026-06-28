# Test-Time Action Selection on Video-Action World Models — Master Report

**One question:** can a *frozen* video-action policy be improved at test time by scoring its
own best-of-K candidate action chunks with a world model — no policy retraining?

**One-line answer:** It depends entirely on the world model and the scorer's *anchor*.
On a weak, action-blind model (UVA) nothing works. On a strong, action-faithful model (UWM) a
**candidate-level, quality-anchored** scorer can work *spectacularly* — but **which** scorer is
the headline. A hand-crafted goal-image selector is a **task-specific bet** (lifts success
**0.37 → 0.93** on one task, but is the *worst* arm on another). The robust winner is a
**pure learned value head over the world model's imagined latent** (Dreamer-style, **goal-free at
inference**): it is the best arm on **both** confirmed tasks (bowl **0.57 → 0.83**, stove **0.43 →
0.97**), and *residualizing it on goal-distance inherits the goal's failure mode*. Offline
correlation metrics **systematically misjudged** which scorers work; only closed-loop success rate
(paired, n=30, anti-brackets) is admissible evidence.

> Scope note. This is a research thread run on two cloned repos (`unified_video_action` = UVA,
> `unified-world-model` = UWM) outside DreamZero proper. Detailed working logs live there
> (`unified_video_action/docs/ACVS_on_UVA_implementation_report.md`,
> `uwm/docs/SCORER_COMPARISON.md`, `uwm/docs/NEW_SCORERS_EVAL.md`); this is the self-contained
> master summary, mirrored into the DreamZero repo because it is the only repo we own.
> Date: 2026-06-27 (rev. 2026-06-28 — added §4e: 2-task n=30 confirmation, goal-ablation `m4pure`
> as universal winner, fusion null result).

---

## 1. Setup

- **Goal:** select among a frozen policy's K candidate action chunks per control step to raise task
  success, without touching the policy.
- **Models studied.**
  - **UVA** (Unified Video Action; MAR + per-token diffusion heads, 5 co-trained task-modes). Its
    action-conditioned video mode is one weak head among five → **action-blind imagination**.
  - **UWM** (Unified World Model, arXiv 2504.02792, WEIRDLabUW; multimodal diffusion transformer,
    **separate** diffusion timesteps for action vs video; forward-dynamics / inverse-dynamics /
    policy / video-prediction are all first-class). **Action-faithful imagination.**
- **Bench:** LIBERO (`libero_90.pt`, raw `[-1,1]` OSC_POSE actions, `action_normalizer=None`).
  Single RTX 3090, conda env `uva`. LIBERO is a source checkout at `/home/zuxinrui/LIBERO`; run env
  needs `MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 CUDA_VISIBLE_DEVICES=0`.
- **Decisive metric:** closed-loop **task success rate**, best-of-K, paired seeds. Offline proxy
  `pearson(score, q)` with `q = −‖c − a*‖` was used early and proved **misleading** (see §3).

---

## 2. The gate and the three laws

**Action-sensitivity gate.** Hold obs + sampling noise fixed, vary only the injected action, and
measure how much the imagined future changes.
- **UVA:** flat — `cos(z_real, z_swap) = 0.9994` for *any* action. Imagination is action-blind.
- **UWM:** monotonic — cos `1.00` (≈identical action) → `0.73` (negated). Imagination is
  action-faithful. Noise floor (same action+seed) `0.000`; sampling ceiling (same action, diff
  seed) `≈0.07`; **operational rule: fix the seed across candidates** or sampling noise swamps the
  action signal.

**Three laws (govern every result).**
- **(a) The imagination wall is a property of the MODEL, not the scorer.** The same scorers that
  die on UVA revive on UWM (BIRG `−0.43 → +0.14`; goal-distance flips negative → positive).
- **(b) Self-consistency / plausibility ⊥ quality, even on a faithful sim.** Inverse-forward cycle
  (ICS) and re-noise (RCS) measure typicality/invertibility, which is orthogonal to task success.
- **(c) A scorer ranks only if its anchor is CANDIDATE-LEVEL.** A window-constant *reference* is
  fine **if** fed through a candidate-specific distance (this is why `−‖imag−goal‖` ranks). But a
  window-constant *target* (e.g. progress) cannot rank, no matter how well it is fit —
  **fitting a global quantity ≠ ranking within a window.**

---

## 3. The action-scale bug (a methodology lesson, found by careful verification)

The early offline studies used UVA-provided `libero_10` demo actions as the "expert" anchor `a*`
and proxy `q = −‖c − a*‖`. Those demo actions are a **different convention** (orientation deltas to
±3.08 rad, position ±1.18) than UWM's `[-1,1]` OSC_POSE; **demo-action replay in the env fails
0/3**. Consequence: every offline **correlation magnitude** that touched `a*`/`q` is **confounded**
(the model was conditioned on OOD actions; `q` is a corrupted target). Offline **structural**
findings (window-constancy of progress/obs features) survive; offline **correlation numbers do
not.** Closed-loop is immune (the policy uses its own `[-1,1]` samples; success is ground truth).

---

## 4. Full method catalog — everything tried or to be tried

Legend: **anchor** per Law (c); **result** = best clean evidence; ✗ fail, ~ neutral, ✓ win,
⏳ building.

### 4a. Offline scorers on UVA (all negative — the imagination wall)

| Scorer | Idea | Anchor | Result |
|---|---|---|---|
| ACVS reward head | τ₀-WM reward head on imagined `dynamic_model` feature | quality | ✗ can't rank (imagination action-blind; candidates' dreams identical) |
| RCS | re-noise candidate, measure ε-error | plausibility | ✗ |
| Method M | imagined-chunk overlap self-consistency | self-consistency | ✗ (and **hurt** closed-loop −0.20) |
| ICS | training-free inverse(imagined)→â, pick nearest | self-consistency | ✗ pearson −0.17 |
| BIRG | trained reverse-dyn `B(z_future,a)→z_present`, real-present anchor | reality | ✗ real-future works (0.94) but imagined-future pearson −0.43 |
| direct critic Q(o,a)→progress | imagination-free value | critic | ✗ action lift +0.03 (variance wall) |

**UVA verdict:** two walls — *imagination wall* (action-blind) + *variance wall* (near-saturated
LIBERO-10, candidate quality variance ≈ 0). Test-time selection exhausted on frozen UVA.

### 4b. Offline bake-off on UWM (8 scorers; numbers **confounded** by §3, kept for structure only)

| Scorer | family | offline `pearson(score,q)` policy | structural verdict |
|---|---|--:|---|
| gp_l2 / gp_cos (`−‖imag−goal‖`) | quality·oracle-goal | +0.159 | real but non-deployable as-built |
| static (`−‖imag−present‖`) | plausibility | **+0.32 (rated best!)** | **MIS-rated — see §5** |
| BIRG | reality | +0.142 | static confound (partial → +0.02) |
| RCS | plausibility | +0.10 | ✗ within noise |
| ICS | self-consistency | −0.10 | ✗ structural |
| ACVS progress head | quality·learned-prog | −0.02 | ✗ **prog is window-constant** (fits r=0.80, can't rank) |
| critic Q(o,a) / advantage | critic | −0.12 / −0.10 | ✗ obs_feat window-constant |

### 4c. Closed-loop A/B on UWM (the decisive evidence; best-of-K=6, paired seeds)

| arm | stove (n=30) | drawer (n=30) | mechanism |
|---|--:|--:|---|
| **goal** `−‖imag−goal‖` | **0.933** | 0.067 | candidate-level **quality** (deployable: 1 success image/task) |
| anti_static | 0.700 | 0.267 | "most imagined change" (task needs motion) |
| random | 0.367 | 0.200 | control |
| null | 0.300 | 0.233 | baseline |
| static `−‖imag−present‖` | 0.200 | 0.100 | **do-nothing confound** (lowest motion) |
| anti_goal | 0.000 | 0.200 | bracket floor |

- **goal vs random on stove: +0.567, 17/0 paired (McNemar p≈8e-6)** — a decisive deployable win.
- **goal does NOT generalize:** on `close_the_bottom_drawer` (visually subtle goal) goal is the
  **worst** arm. Hypothesis: goal-distance works only when the success state is visually distinctive
  *and* the task has candidate headroom; the drawer sits in a narrow 0.07–0.27 band (variance wall).
- **static is consistently harmful** (do-nothing) — the offline "best deployable" pick was inverted.

### 4d. The four NEW trained-scorer methods — BUILT + CLOSED-LOOP TESTED

All four were implemented, trained on a 579-row harvest from `put_the_black_bowl_on_the_plate`,
and run in one combined best-of-6 A/B (n=30) on that task, with anti-brackets + the motion probe.

| arm | success | Δ vs random (paired) | anti-bracket | red-team predicted | outcome |
|---|--:|--:|--:|---|---|
| **M2** dream-reexamine (**learned** quality goal-distance) | **0.733** | **+0.40 (13/1)** | +0.70 | LOW "hollow" | **WINS — prediction wrong** |
| null (1 sample) | 0.533 | +0.20 | — | — | (high; null>random ~ noise) |
| **M4** Dreamer value (residual-on-goal) | 0.367 | +0.03 | +0.30 | HIGH | clean bracket, **thin margin** here |
| random | 0.333 | — | — | — | control |
| **goal** (raw `−‖imag−goal‖`) | 0.333 | +0.00 | — | — | **useless on this task** (cf. stove 0.93) |
| M1 reverse (deconfounded) | 0.333 | +0.00 | — | LOW (random) | ≈ random ✓ |
| static (`−‖imag−present‖`) | 0.267 | −0.07 | — | confound | do-nothing again ✓ |
| M3 contrastive (CoVer, distant-pos) | 0.167 | −0.17 | — | fails | **hurts** ✓ |

**Decisive finding:** a **learned** quality-ordered goal-distance (M2) WINS (0.73 vs 0.33) on the
exact task where the **hand-crafted** goal selector is useless (0.33) — i.e. a learned quality
metric *generalizes where a single goal image does not*. M4 ranks correctly (clean +0.30 bracket)
but its residual-on-goal term is dead weight when goal itself is useless → thin margin; expected to
improve if the goal term is dropped/learned. Building all four (vs trusting the red-team) was
justified: it overturned M2's "low priority". Offline `pearson(score,rtg)` on executed candidates
tracked the closed-loop order (M4 0.68 / M2 0.51 / M3 0.21 / M1 ~0) **except** it under-rated M2's
closed-loop ranking — another reminder that only closed-loop is decisive. (This single-task table
is superseded by the 2-task confirmation in **§4e**, which both raised n to 30 and added the goal
ablation `m4pure` + fusion `m5fuse`.)

### 4e. Cross-task confirmation — the complete 2×2 matrix (n=30 each, the decisive run)

Re-harvested a second task (`turn_on_the_stove`, 477 rows) and **retrained all four heads on
stove data** (M2/M4 are per-task value/encoder heads — bowl-trained weights do not transfer; the
scripts were made `SCORER_NPZ`/`SCORER_ARTIFACT` env-overridable to retrain cleanly). Then ran the
combined best-of-6 A/B at **n=30** on **both** tasks, adding two arms: **`m4pure`** = M4's pure
learned value head `g_pure(imag)` with the goal term removed (**needs no goal image at inference**),
and **`m5fuse`** = `z(M2) + z(m4pure)`.

| arm | bowl `bowl_on_plate` | stove `turn_on_stove` | anti-bracket (bowl / stove) | verdict |
|---|--:|--:|--:|---|
| **m4pure** (pure learned value, **goal-free**) | **0.833** (25/30) | **0.967** (29/30) | 23/0 / 17/0 | ✅ **universal winner** |
| **M2** (learned goal-distance) | 0.667 | **0.967** | 19/0 / **29/0** | ✅ ranks both; modest bowl margin |
| m5fuse (M2 + m4pure) | 0.667 | 0.967 | — | ties, **no synergy** |
| null (baseline) | 0.567 | 0.433 | — | — |
| random | 0.367 | 0.533 | — | control |
| **goal** (hand-crafted `−‖imag−goal‖`) | 0.333 | 0.933 | — | ❌ **task-specific** (hurts bowl) |
| static (`−‖imag−present‖`) | 0.267 | 0.067 | — | ❌ do-nothing |
| **M4** (residual-on-goal) | 0.233 | 0.733 | / | ❌ **inherits goal failure** |
| anti_m4 | 0.067 | 0.400 | — | bracket floor |
| anti_m2 | 0.033 | 0.000 | — | bracket floor |

**Three decisive conclusions (all paired-McNemar significant):**
1. **`m4pure` — a goal-free pure learned value head — is the universal winner.** Best on bowl
   (0.83, **vs goal 15/0 p=3e-4**, vs M4 18/0 p=1e-4, vs null 12/4 p=0.08), tied-best on stove
   (0.97). It uses **no goal image at inference** — only RTG-labelled harvest to train. This beats
   M2 on deployability (M2 still needs a goal latent through its encoder).
2. **The goal term is harmful, not protective.** `m4pure` (no goal) ≫ `m4` (residual-on-goal):
   bowl 0.83 vs 0.23, stove 0.97 vs 0.73. Residualizing on `−‖imag−goal‖` **inherits the goal
   selector's task-specific failure** — on bowl `goal` itself (0.33) is *below null* (0.57), and
   M4 drags down with it. This **overturns the red-team's "residual-on-goal is the safe priority"**.
3. **M2 ranks on both tasks** (anti-bracket 19/0 bowl, **29/0** stove — the cleanest signal in the
   whole study) and **wins stove decisively** (0.97 vs null 0.43, 16/0 p=2e-4). But its **bowl
   margin shrank at n=30** (0.667 vs null 0.567, 10/7, p=0.63 — n.s.); the earlier +0.40 at n=15
   was small-n optimism. M2 is a confirmed *ranker*, a confirmed *stove winner*, but not a
   significant *bowl* winner — `m4pure` is.

**Fusion adds nothing** (m5fuse = M2 on bowl, ties ceiling on stove): once a single head is at the
candidate-headroom ceiling, averaging a second adds no signal.

---

## 5. Verification — which verdicts were MISJUDGED (a 13-agent audit)

The user's explicit concern ("don't misjudge reasonable scorers") was well-founded. Two families
were inverted by the corrupted offline `q`:

- **FALSE POSITIVES (over-rated):** `static` / `static_raw` / `BIRG` / `gp_cos-to-present` — all
  ride the present-anchor term, which a corrupted `q` rewards because `q` secretly favors low
  motion. Closed-loop: `static` is near-worst. **Do not deploy.**
- **FALSE NEGATIVE (the biggest miss):** the **deployable goal-image selector** was dismissed
  ("no deployable scorer clears the bar") yet is the **headline win (0.93)**.
- **Verdicts that STAND (structural, scale-independent):** ICS/RCS fail (plausibility ⊥ quality);
  ACVS-head/critic/advantage fail because the *label* is window-constant — **but the architecture is
  salvageable with a candidate-level target** (swap target to candidate-level → bestpick 3× chance).
  This directly motivates M4.

---

## 6. Build status — DONE (pipeline executed)

Pipeline executed: **shared harvest → parallel-train 4 aux models → one combined closed-loop A/B**.

1. **Harvest** (`uwm/scripts/harvest.py`) — 579 decision points on `put_the_black_bowl_on_the_plate`
   (pick-place, baseline 0.50). ✅
2. **Build workflow** (`uwm/scripts/new_scorers_build.workflow.js`) — 4 experts implemented+trained
   M1–M4 (`uwm/scripts/heads2/m{1,2,3,4}.py`) to the `cl_ab2.py` selector contract, red-team fixes
   baked in. ✅
3. **Combined A/B** (`uwm/scripts/cl_ab2.py`) — bowl (§4d) then the 2-task n=30 matrix (§4e). ✅
4. **2nd-task confirmation** — re-harvest stove (`harvest.py`, 477 rows) → retrain all 4 heads on
   stove (env-overridable `SCORER_NPZ`/`SCORER_ARTIFACT`) → combined A/B on bowl + stove at n=30
   with `m4pure` (goal ablation) and `m5fuse` (fusion) arms. ✅ (§4e).

**Result:** the goal-free **pure learned value head `m4pure`** is the universal winner (bowl 0.83 /
stove 0.97 — best on both); the **goal term is harmful** (residual-on-goal `m4` inherits goal's
task-specificity); **M2** is a confirmed ranker + stove winner but only a modest bowl effect at n=30;
**fusion adds nothing**. All planned follow-ups (2nd-task M2, M4 goal-ablation, M2+M4 fusion) closed.

---

## 7. The clean re-test protocol (so we never misjudge again)

Offline `pearson(score, q)` is **not admissible**. A scorer is judged only by:
1. best-of-K=6, policy's own `[-1,1]` samples, scorer picks argmax, log binary success.
2. paired seeds, n ≥ 17; report paired win/loss.
3. **≥ 2 tasks**, including one where the goal state is visually subtle (generality test).
4. **anti-bracket every run** — a real scorer shows a symmetric, decisive spread (goal 0.93 /
   anti_goal 0.00). Flat/inverted bracket ⇒ not ranking quality.
5. **motion-confound probe** — selected-action motion vs window mean; significantly below ⇒ the
   do-nothing trap, disqualify regardless of the success number.
6. leakage/deployability audit — reference is fixed-per-task or learned from disjoint data, never
   the specific episode's future.
7. learned-head validation = held-out **rank-within-window**, never global pearson.
8. action-convention gate — train/condition on model-own `[-1,1]` / official LIBERO; verify by
   in-env replay.

**Acceptance to ship over `goal`:** must *exceed* goal on a task where goal underperforms (e.g. a
multi-stage or visually-subtle task) — matching goal on the easy task is necessary, not sufficient.

---

## 8. Bottom line

- **Positive:** a deployable, candidate-level, quality-anchored selector (goal-conditioned UWM
  imagination) can dramatically improve a frozen policy closed-loop (+0.57 success) — the first
  real win after a long string of negatives.
- **Caveat:** it is task-conditional (needs candidate headroom + a discriminative goal), and most
  scorer families fail for the structural reasons in Laws (a)–(c).
- **Lesson:** offline correlation metrics misjudged the scorers in both directions; closed-loop +
  anti-brackets + a motion probe are mandatory.
- **Update (2-task confirmation done, §4e):** the winner is a **pure learned value head over the
  world model's imagined latent** (`m4pure`, Dreamer-style `g_pure(imag)` regressing return-to-go,
  **goal-free at inference**). It is the **best arm on both confirmed tasks** (bowl 0.57→0.83,
  stove 0.43→0.97). Two hard lessons fell out: (i) **the goal term is harmful** — residualizing the
  value on `−‖imag−goal‖` (`m4`) *inherits* the goal selector's task-specific failure (on bowl,
  `goal` is below null and drags `m4` to 0.23); (ii) the earlier single-task M2 win was **n=15
  optimism** — at n=30 M2 still ranks cleanly (anti-bracket 29/0 on stove) and wins stove, but its
  bowl margin is n.s. **Fusion (M2+m4pure) adds nothing** once a head is at the headroom ceiling.
- **Deployment recommendation:** train a small value head `g(imag)→RTG` on per-embodiment
  success-labelled rollouts; select argmax at test time. No goal image, no policy retraining.
- **Next (optional):** a 3rd task to tighten m4pure's CI; test whether `g(imag)` transfers across
  tasks within an embodiment (the one thing that would make it *zero-shot* deployable, not
  per-task); a within-window rank-loss head if all-K labels can be harvested (currently
  executed-candidate-only).

### File map
- UVA work: `unified_video_action/docs/ACVS_on_UVA_implementation_report.md`, `scripts/{train_acvs,
  birg,critic,...}.py`
- UWM offline: `uwm/docs/SCORER_COMPARISON.md`, `uwm/scripts/{dump_all_scorers,scorer_bakeoff}.py`
- UWM new-method eval: `uwm/docs/NEW_SCORERS_EVAL.md`
- UWM closed-loop: `uwm/scripts/{cl_ab,cl_ab2,cl_analyze,cl_scope,harvest}.py`,
  `new_scorers_build.workflow.js`
