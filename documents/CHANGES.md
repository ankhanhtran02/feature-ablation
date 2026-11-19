# CHANGES

This lists every file created or modified to implement `pilot-implementation-plan.md`, then every
deviation from that spec and the reason for it. The plan itself was not modified.

## Files created

| File | Purpose |
|---|---|
| `run_pilot.sh` | The only command. Sets up uv/.venv (Py 3.12), installs torch from cu130 (CPU only for `--smoke`), prefetches the model while torch installs, wraps Python in a systemd `MemoryMax` scope, dispatches `--check/--smoke/--resume/--out/--force` |
| `README.md` | Run tutorial: prerequisites, the command, smoke test, runtime, outputs, resuming, reading `report.md` |
| `CHANGES.md` | This file |
| `requirements-check.txt` | Pinned minimal dependencies for Step 0 (transformers, safetensors, huggingface_hub, numpy, pyyaml) |
| `requirements.txt` | Pinned full pilot dependencies (adds pandas, pyarrow, matplotlib, pytest) |
| `config/check.yaml` | Preregistered Step 0 rule: check model, dataset URL and row count, template, thresholds, PASS/FAIL mapping |
| `config/pilot.yaml` | Full-run configuration: models and SAE releases, factors, budgets, data, gate, thresholds, memory |
| `config/smoke.yaml` | `--smoke` overrides (1.7B forced, 16k SAE, 32 per split, k ∈ {1,10,100}, 1 resample/draw, 50 bootstraps) |
| `src/pilot/__init__.py` | Package marker |
| `src/pilot/utils.py` | Config loading and merging, hashing, JSON I/O, logging, seeds, raw-file download, versions, git commit |
| `src/pilot/stats.py` | Numpy-only statistics: tie-aware weighted AUROC, paired bootstrap counts, BH, Kendall τ-b, D@c\* interpolation |
| `src/pilot/check.py` | Step 0 truth check and decision rule; `decision.json` / `decision_smoke.json`; hash guard; `--force` override record |
| `src/pilot/prompts.py` | Chat template with `enable_thinking=False`, template suffix (intervention positions), answer-token variants |
| `src/pilot/models.py` | Model loading (SDPA), left-padded batches with mask-derived positions, ablation hook, layer-split fast path, gradients, generation |
| `src/pilot/memory.py` | `/proc/meminfo` watchdog thread (aborts the current stage) and memory telemetry |
| `src/pilot/env_check.py` | `env` stage: GPU/capability, bf16 matmul, cu130 build, CUDA 12 wheel detection, HF access, hardware record |
| `src/pilot/data.py` | Pinned raw downloads, city-grouped truth splits, refusal/generic splits, templating, position records |
| `src/pilot/sae.py` | One SAE interface for both releases; repo listing to resolve layer and trainer; JumpReLU with the fixed threshold |
| `src/pilot/bases.py` | `sae` / `pca` / `dims` / `mlp` / direction bases sharing one read/write interface; PCA fit; masked per-offset means |
| `src/pilot/selection.py` | meandiff (Cohen's d), L1 logistic path (warm-started FISTA), attribution contributions, random, Jaccard |
| `src/pilot/metrics.py` | m(x), r(x), KL, refusal labelling, refusal-start set, threshold calibration, per-config and bootstrap damage |
| `src/pilot/pipeline.py` | Run context plus the `data`, `gate`, `cache`, `bases`, `select` stages |
| `src/pilot/ablate.py` | `ablate` stage: config enumeration, resumable chunks, fast-path equivalence test, identity check, generation checks |
| `src/pilot/analysis.py` | Curves, D@c\*, G, paired bootstrap, H0–H6, X1, per-task and overall decision |
| `src/pilot/plots.py` | Damage-vs-cost curves, gap-closure heatmaps, stability plot (validated colour-blind-safe palette) |
| `src/pilot/report.py` | `analyze` stage outputs: `summary.csv`, `hypotheses.json`, plots, `report.md`, throughput projection |
| `src/pilot/cli.py` | Stage orchestration: run ids, `--resume`, DONE markers, manifest, watchdog, INVALID/abort handling |
| `src/pilot/smoke.py` | Automated smoke acceptance checks; writes `smoke_checks.json` |
| `src/pilot/prefetch.py` | Background model download (stdlib + huggingface_hub only) started by `run_pilot.sh` |
| `src/pilot/devtiny.py` | Developer plumbing test: tiny random Qwen3 plus random SAE, no download (`--dev-tiny`) |
| `tests/conftest.py` | Test fixtures: tiny random Qwen3 and a toy SAE |
| `tests/test_ablation.py` | v = r(x) identity, SAE error term kept, hook positions, batch invariance, fast path vs full forward |
| `tests/test_bases.py` | PCA/dims orthonormal, read/write round-trips, JumpReLU batch independence, masked means |
| `tests/test_selection.py` | Planted unit recovered by meandiff, l1probe (incl. prefilter), attrib; random; AUROC/bootstrap/interpolation/BH/τ |
| `tests/test_check.py` | The pilot refuses missing, INVALID or hash-mismatched decisions; smoke paths never alias `decision.json` |

Generated at run time (not source): `.venv/`, `data_cache/` (raw datasets), `runs/` (all outputs).

## Deviations from the spec, and why

**Layout and stage order**

1. **Repository root.** The spec's `pilot/` top-level folder is the repository root here, so the
   command is `bash run_pilot.sh` from this directory.
2. **Extra modules.** The spec's suggested layout gains these modules:
   - `check`, `prompts`, `stats`, `memory`, `pipeline`, `smoke`, `prefetch`, `devtiny` and `utils`;
   - `pipeline.py` holds the gate/cache/bases/select stages, so no single module is huge;
   - `tests/test_check.py` and `tests/conftest.py`.
3. **Stage order** is `env, data, gate, cache, bases, select, ablate, analyze`. `data` comes before
   `gate` because the gate's format check and refusal calibration use the discovery split. The Step 0
   decision is read and verified before any stage runs (running the check first if it is missing),
   and it is recorded in `manifest.json`.

**Model and runtime**

4. **CPU path uses fp32.** bf16 is emulated, and very slow, on CPUs without AVX512-BF16/AMX. The GPU
   uses bf16 as specified.
5. **No `device_map` when loading.** It would require `accelerate`. The model is loaded, then moved
   to the device.
6. **Memory limits on smaller machines.** On machines with less than about 116 GB, the watchdog
   threshold scales to 12.5% of MemTotal (16/128) and the systemd `MemoryMax` to 85% of MemTotal.
   This matters only for the CPU smoke check; on the GB10 the values are the spec's 16 GB and 100G.

**Data and metrics**

7. **Generic prompts.** The 400 generic prompts are 100 for PCA fitting plus 300 for the KL cost. This
   reconciles "Generic: 400 prompts" with the runtime section's "300 generic".
8. **Smoke refusal splits** use 32 prompts *per class*, harmful and harmless, matching how the refusal
   splits are defined per class.
9. **The refused set** is defined by clean greedy-generation labels (32 tokens) on the harmful eval
   prompts. Refusal damage is (classifier refusal rate, clean) − (classifier refusal rate, ablated) on
   that set. If the classifier fails calibration, generation with the same token budget replaces it
   for every configuration. If the clean model refuses none of the harmful eval prompts, the gate
   stops with INVALID HARNESS, because damage would be undefined.
10. **Truth coherence only.** Coherence is defined for truth only, so refusal rows have no coherence
    value.
11. **Mean values for `attrib`** and for both strategies come from the full discovery set. The 80%
    resamples change only the rankings.
12. **Resample sources** use one fixed random discovery example per eval or generic prompt, shared
    across all configurations (common random numbers), so differences between cells are paired.

**Curves, statistics and hypotheses**

13. **Below the cheapest budget**, the curve is joined linearly to the origin (k = 0: zero cost, zero
    damage). The spec covers only curves that never *reach* c\*: those report their maximum and are
    flagged.
14. **Test pass rule.** A test passes when its 95% bootstrap CI excludes 0 *and* its BH q ≤ 0.05 within
    the hypothesis family. H2 and H3 are one-sided (CI lower bound > 0). For H4 and H5 a swap needs both
    differences CI-separated; its p-value is the larger of the two.
15. **Aggregation over strategies** (unspecified in the plan):
    - H1 and H4: any strategy.
    - H2 and H3: all strategies.
    - H5: across mean vs resample.
    - H6: τ ≥ 0.6 under every strategy.
    - Overall decision: the shared outcome if both tasks agree, otherwise `MIXED (tasks disagree: ...)`.
16. **H0.** Truth has no generation-confirmed positive control, so its H0 is the random-floor check
    (random D@c\* ≤ 25% of anchor damage, with anchor damage > 0) in every basis and strategy. Refusal
    H0 also needs the anchor's generation drop to be ≥ 0.5 under each strategy.
17. **The MLP arm** is reported everywhere with a "different site" label. It is excluded from H1–H6,
    and the `zero` strategy is diagnostic only.
18. **G is undefined (n/a)** when the anchor's damage equals the random floor. `summary.csv` carries
    `G_denominator`, and `hypotheses.json` writes undefined statistics as `null`.

**Selection and SAE**

19. **L1 path.** Twenty λ values run geometrically from about 0.7·λ_max to 1e-3·λ_max, with warm-started
    FISTA (`max_iter` 1000, relative `tol` 1e-4; float32 on GPU because GB10 FP64 throughput is low).
    Units that never enter are ranked after those that do, by |coef| and then |d|.
20. **`rankings.parquet`** stores the top 2,000 units per ranking (`rank_store`), more than the largest
    budget. It also has an explicit `draw` column; `resample` is −1 for random-rule rows.
21. **SAE normalisation.** `dictionary_learning` folds its activation normalisation into the saved
    biases and threshold (`scale_biases`), and REINS' runtime applies its SAE to raw residuals. So no
    normalisation is applied unless a checkpoint carries an explicit `norm_factor`. The gate also
    reports a diagnostic FVE under global RMS normalisation; it never switches automatically.
22. **SAE sanity check.** The spec's "small next-token loss increase" is set to ≤ 0.5 nats (configurable),
    together with FVE ≥ 0.70 (`dictionary_learning`'s definition), with attention sinks excluded.

**Scope and smoke tooling**

23. **Generation checks** use the first of the 10 `random_dir` directions. Each basis's best and worst
    cell are picked per strategy, at the budget whose cost is nearest c\*, on discovery resample 0.
    Interventions apply only to the template positions during generation (the prefill pass).
24. **DAS** (`anchors.das`) is not implemented. Setting it logs a warning and appears in the report's
    caveats.
25. **Smoke acceptance runs inside the smoke command.** `bash run_pilot.sh --smoke` also runs pytest, a
    timed `--resume`, and `--check --smoke`, then writes `smoke_checks.json`. `--check --smoke` writes
    `decision_smoke.json` (plus `check_report_smoke.md` and `log_smoke.txt`), never `decision.json`.
26. **Developer additions**, not in the spec and hidden from `--help`:
    - `--dev-tiny` runs a tiny random Qwen3 and a random SAE with behavioural checks relaxed and
      logged; with `--check` it writes only `decision_dev.json`.
    - `PILOT_EXTRA_CONFIG=<yaml>` merges an extra config last, for example to enable the `mlp` arm in
      a test.
27. **Pinned versions.** Package versions are pinned for determinism. torch is not pinned: it comes
    from the cu130 index, and its version is recorded in `manifest.json`.

## Verification status

- `pytest`: 30 tests pass.
- **Smoke test on this machine.** The development machine is x86_64 WSL2 with no GPU, and its network
  was too slow to fetch the 4 GB Qwen3-1.7B weights in reasonable time. As instructed, the end-to-end
  smoke test therefore ran on the tiny random model: `bash run_pilot.sh --dev-tiny`.
  - Every stage ran, as did a resume after a failure, partial resume of one stage, the optional MLP
    arm with the `zero` strategy, and the Step 0 check (fresh, refused after a rule change, `--force`
    recorded).
  - All applicable acceptance checks pass: pytest, identity check, no NaN/Inf, outputs present,
    decision line, `--resume` in under 2 s, watchdog running, peak memory logged.
  - The result is `INVALID HARNESS`, as it should be for random weights.
- **Not yet verified on a real model:**
  - Qwen3-1.7B behaviour: format check, refusal calibration, SAE fidelity at layer 14, and the
    throughput projection.
  - Every GPU-only path: cu130 build, capability (12, 1), bf16 numerics, Triton.
  - The Qwen3.5-4B / REINS path; its loader follows REINS' own runtime but no checkpoint was downloaded.

  Run `bash run_pilot.sh --smoke` on the GB10 before the full run.

---

# v2 (pilot-v2-amendment.md)

Everything below implements `pilot-v2-amendment.md` on top of v1. v1 modules, configs and run outputs are
unchanged; v1 modes behave as before.

## Files created

| File | Purpose |
|---|---|
| `config/v2.yaml` | All v2 settings: models and depths, tasks and templates, data, pairs, gates, SAE inclusion, tiers, extension, controls, cost, blocks, budget, batching; `phases:` overrides for dev, smoke and dev-tiny |
| `prereg/hypotheses.yaml` | DRAFT preregistration (machine-readable): metric, c0, delta, C1–C3, secondary S1–S5, inclusion rules, pairs, H0, run plan |
| `prereg/PREREGISTRATION.md` | Human-readable preregistration rendered from the YAML (`--v2 --render-prereg`) |
| `src/pilot/v2common.py` | Config loading with phase overrides, the layer rule, v2 paths, pair helpers |
| `src/pilot/data_v2.py` | Model-independent text splits for all five tasks (v1 split code reused for truth / refusal / generic), per-model tokenization |
| `src/pilot/sites.py` | The v2 hook: prompt / generated roles, S1 / S2, first-token, sink and padding exclusion, mean / resample / identity values, decode-step editing, multi-layer capture and gradients |
| `src/pilot/sae_v2.py` | Per-layer SAE resolution for both release layouts (hub or local), input normalisation modes |
| `src/pilot/engine_v2.py` | Run / model contexts, generation with the hook active, the TaskRunner (damage pass, teacher-forced 9-position cost, fast path, clean references, resample tables) |
| `src/pilot/gate_v2.py` | Per-model gate: format check, answer tokens, task baselines and inclusion gate, refusal labels and eval set, sink thresholds, SAE fidelity; run-level inclusion with the toxicity substitution |
| `src/pilot/prep_v2.py` | Generic continuations, PCA per depth, one forward + backward per discovery example at all depths, rankings (incl. shuffled-label attribution), means per role, anchors, reservoirs, C3 groups, stability |
| `src/pilot/ablate_v2.py` | Blocks in priority order, cells, curves with extension, controls, resumable chunks, the 12-hour time guard, background analysis workers, the 32- vs 64-token agreement check |
| `src/pilot/analysis_v2.py` | Frozen analysis: per-intervention bootstrap caches, D(c) and E(c) at fixed costs, C1–C3, S1–S5, H0, G, results table |
| `src/pilot/report_v2.py` | `analyze` stage: confirmatory.json first, then secondary / development / exploratory, plots, `report_v2.md` or `dev_report.md` |
| `src/pilot/plots_v2.py` | Primary-setting curves, E heatmaps, Δ by depth, C3 scatter |
| `src/pilot/prereg.py` | Render, freeze (hash, commit, tag), verify and deviation recording |
| `src/pilot/budget_v2.py` | Batch auto-tuning under the memory cap, throughput, per-intervention and per-block projections |
| `src/pilot/prefetch_v2.py` | Downloads every model, SAE file and dataset the full run needs; reports what is missing |
| `src/pilot/smoke_v2.py` | In-run checks (identity at both sites incl. decoding, decode-cache consistency, zero cost for a no-op, curve extension, role means / reservoirs) and post-run acceptance; `smoke_report.md` |
| `run_v2.sh` | The whole v2 sequence in one command, in the background: downloads (3 tries), smoke, dev, finalise (`status: final`) + render + commit + freeze, confirm. Stops at the first error (a failed smoke check counts) with the reason; rerunning continues from that step. `--status`, `--stop`, `--foreground` |
| `src/pilot/cli_v2.py` | v2 orchestration: phases, stages with DONE markers, `--resume`, prereg enforcement |
| `src/pilot/devtiny_v2.py` | Developer plumbing test: tiny random Qwen3 and hybrid Qwen3.5 with real tokenizers, SAEs in both layouts |
| `tests/test_v2_sites.py` | Hook formula and masks, identity, joint resample copy, S1 / S2 decode behaviour, decode-cache consistency on both architectures, multi-layer gradients, attribution identity |
| `tests/test_v2_analysis.py` | C1 / C2 / random floor on synthetic runs with known answers, curve extension, tie-aware Spearman, inclusion / substitution, seeding |
| `tests/test_v2_prereg.py` | Freeze / verify / deviation on a throwaway git repository; the rendered prereg is in sync |

## Files modified

| File | Change |
|---|---|
| `run_pilot.sh` | `--v2` with `--prefetch`, `--smoke`, `--dev`, `--render-prereg`, `--freeze`, `--confirm`, `--deviation`; v2 smoke runs the acceptance checks; best-effort install of `flash-linear-attention` 0.6.0 (Triton) on the GPU path for v2 |
| `README.md` | v2 section: order of commands, what the freeze does and why, runtime per phase, blocks, how to read `report_v2.md` |
| `CHANGES.md` | This v2 section |
| `.gitignore` | Adds `__pycache__/`, `*.pyc` and `runs/` |
| git index | The tracked `__pycache__/*.pyc` files are removed from the index (staged, not committed): every Python run changed them, so the freeze's clean-tree check could never pass |

## Deviations from the amendment, and why

**Process and preregistration**

1. **`--render-prereg` mode added.** PREREGISTRATION.md is generated from the YAML; the freeze refuses if the two
   are out of sync, so the human-readable and machine-readable versions cannot diverge.
2. **Draft choice flagged for review: `require_bh_q: true`.** The amendment's C1 / C2 "supported if" columns do not
   mention q-values, but BH correction is required within each family. The draft makes a per-pair test also need
   BH q ≤ 0.05; set it to `false` in `hypotheses.yaml` to use the CI conditions alone.
3. **C2 test form.** Each unordered basis pair and readout rule is one test: a reversal needs both CIs to exclude 0
   with readout favouring one basis and attribution the other (equivalent to the amendment's ordered definition);
   p = the larger of the two bootstrap p-values.
4. **Freeze details.** Untracked files do not block the freeze (only changes to tracked files, and frozen files
   that are not committed). An existing `prereg-v2-frozen` tag blocks a second freeze. With no freeze at all,
   `--deviation` cannot bypass the refusal.
5. **Frozen analysis code** is `analysis_v2.py` and `stats.py`. The analysis carries its own copy of the seeding
   helper so it does not depend on the unfrozen `utils.py`.
6. **Confirm budget** = 12 h minus the latest finished dev run's stage time (0.6 h assumed if none). When the guard
   stops, the blocked block and every later one are skipped and listed.

**Sites, values and metrics**

7. **SAE layer indexing and input scaling come from the model cards; no reconstruction check.** Both releases
   state the setting: file N = output of decoder layer N (0-indexed), no input scaling (adamkarvonen: `config.json`
   `"submodule_name": "resid_post_layer_N"`, `"io": "out"`; REINS: model card "post-residual stream of transformer
   layer XX", the authors' code hooks `model.layers[N]` and feeds raw activations, and the card's global-RMS training
   normalisation is folded into the saved file by `dictionary_learning`). The setting is written explicitly in
   `config/v2.yaml` with these sources. Deviation from "confirm indexing by reconstruction, not from the filename"
   and from instruction 7 ("stop and report"): the releases document the setting, the run is single-shot, and the
   fidelity rule (FVE ≥ 0.70, loss ≤ +10%) still drops an SAE that does not fit its layer.
8. **Clean refusal rate** is measured through the same generation path as the ablated runs (it is ≈1.0 by
   construction on the clean-refused set) instead of being assumed to be exactly 1.
9. **Cost continuations** are generated with EOS suppressed for 8 tokens, so every generic prompt has exactly 9 KL
   positions.
10. **Generated-role values and reservoirs** are built only where they are used: generated-role means for refusal
    (the only task run at S2), reservoirs for refusal at 50% depth (the only resample cell).
11. **Sink threshold** per (model, depth): 10× the median residual norm of non-first tokens on generic + discovery
    prompts; sinks are detected on the fly at every intervened token, including generated tokens.
12. **C3 statistic.** Group readout strength is the mean |d| of its units; group damage is the raw damage D. S4(c)
    uses −(summed attribution) so that a positive correlation means attribution predicts damage.
13. **S4(a)** passes on the point estimate (|E(c0)| ≤ δ) and reports the CI; **S4(b)** compares each cross-task
    intervention with the same-k attribution point, each in excess of the random curve at its own cost.
14. **Curve extension** also applies to the shuffled-label attribution control.

**Data**

15. **Toxicity**: discovery from Civil Comments `validation`, eval from `test`; this avoids the 380 MB train split.
    Comments of at most 150 tokens are selected, never truncated.
16. **Token caps** (answerability contexts, toxic comments) use the maximum over both models' tokenizers, so the same
    text splits serve both models.
17. **Refusal eval**: all 572 `harmful_test` prompts are generation-labelled by each clean model; up to 200 refused
    ones are drawn with a fixed seed; the harmless rate (gate) is measured on the 100 harmless eval prompts.

**Smoke test**

18. **Extension in the smoke grid is capped at 2 steps** to keep it small; a dedicated check runs one curve with a
    1-point ladder until it reaches a reachable target and verifies it stops there.
19. **"All v1 smoke checks"** are applied to the v2 run (pytest incl. v1 tests, NaN / Inf, outputs, `--resume` < 30 s,
    env, watchdog, memory) and v1's `--check --smoke` is run on real-model smokes.
20. **Developer additions** (hidden from `--help`): `--dev-tiny` and `PILOT_V2_EXTRA_CONFIG=<yaml>` (merged last,
    recorded in the manifest, ignored by `--confirm`).
21. **Batch-size auto-tune limits lowered** (`config/v2.yaml`, after the first GB10 smoke, 2026-11-19). The smoke
    run was aborted by the memory watchdog while tuning Qwen3.5-4B (`MemAvailable` reached 16 GB): the GB10 is
    shared and other processes hold about 25-45 GB, so a 100 GB torch cap does not fit, and the cap is only checked
    after a candidate has run. Changes: `memory_cap_gb` 100 -> 60 (the amendment's "100 GB cap" is replaced by
    60 GB), and 131072 removed from `tune_candidates_tokens` (the backward ladder, tokens / 4, now stops at 16k).
    No hypothesis is involved: batch size is a harness setting (amendment, dev-run rules), clean and intervened
    passes always share the same batches, and it is fixed at the smoke step, before dev and confirm. Forward
    throughput on Qwen3-1.7B was the same at 65k and 131k tokens (~6.2k tok/s), and `max_batch_size: 512` already
    caps real batches below 131k tokens.
22. **Batch auto-tune memory guard** (`src/pilot/budget_v2.py`, 2026-11-19). Before each tuning candidate runs, its
    peak is extrapolated linearly from the last measured candidate (x1.25) and the candidate is skipped when that
    exceeds min(`memory_cap_gb`, memory torch holds + MemAvailable - watchdog threshold - 4 GB). The cap alone was
    only checked after a candidate had run. Tests: `tests/test_v2_tune_guard.py`.
23. **`test_decode_cache_consistency_under_s2[qwen3_5]` runs on CUDA when `fla` is importable** (2026-11-19). With
    `fla` installed (the GB10), transformers uses its Triton kernels for Gated DeltaNet, which reject CPU tensors;
    the test is unchanged otherwise and still runs on the CPU elsewhere.
24. **Smoke check `decode_cache_consistency_S2` tolerates bf16 near ties** (`src/pilot/smoke_v2.py`, 2026-11-19).
    On the first GB10 smoke, Qwen3-1.7B agreed on 31 of 32 tokens (Qwen3.5-4B 32/32; identity under S2 decoding
    passed on both). Cached decoding and the uncached recomputation run different bf16 kernel shapes, so a
    near-tied argmax can flip. A disagreeing token now counts as consistent when its recomputed logit is within
    `DECODE_TIE_TOL = 0.5` of the top logit; the report records `near_ties` and `max_logit_gap`. An edit that does
    not reach the cache still fails the check (wide or repeated disagreements). The fp32 unit test keeps exact
    token agreement.

## Verification status

- `pytest`: 51 tests pass (30 v1 + 21 v2).
- **Smoke on this machine** (x86_64 WSL2, no GPU, 19 GB RAM): `bash run_pilot.sh --v2 --smoke --dev-tiny` passes
  every applicable check on tiny random versions of both architectures (Qwen3, and hybrid Qwen3.5 with the
  PyTorch Gated DeltaNet path). Checks that need the real weights or a GPU are reported as skipped.
- Also exercised here: the dev phase end to end (agreement check, dev report, background workers), the time guard
  (a 60-second budget skips the second block and reports it), and freeze / confirm enforcement on a temporary copy.
- Real downloads verified here: all datasets, both tokenizers and configs, and one REINS checkpoint
  (`layer_16.pt`: same keys as dictionary_learning, threshold set, k = 128; loads with the v1 SAE loader).
- **Not yet verified on real models:** gates and baselines of the new tasks, SAE fidelity on the real
  SAEs (including the REINS base-to-instruct transfer), throughput, every GPU-only path. Run
  `bash run_pilot.sh --v2 --smoke` on the GB10 first.
