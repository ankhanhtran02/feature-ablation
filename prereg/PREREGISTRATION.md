# Causal vs readout unit selection across bases: v2 confirmatory test

Status: **final** · drafted 2026-11-19

Generated from `prereg/hypotheses.yaml` by `bash run_pilot.sh --v2 --render-prereg`; do not edit by hand.

## 1. Design

- Primary setting: 50% depth, site S1, mean ablation.
- Bases: sae, pca, dims. Readout rules: meandiff, l1probe. Causal rule: attrib.
- Metric: damage in excess of random at fixed cost. E_{r,b}(c) = D_{r,b}(c) - D_{random,b}(c) within the same model, task, depth, site, strategy and basis. D(c) is the damage interpolated linearly in log cost on the curve's monotone envelope (running maximum in cost order); below the cheapest point the curve is joined linearly to (0, 0); above its most expensive point the curve's maximum is used and flagged. Curves are averaged over discovery resamples (rules) or draws (random).
- Damage: single-token tasks — AUROC drop of the task score (logit difference of the two answer tokens) against the label; refusal — drop in generation-based refusal rate (32 greedy tokens, v1 substring rules) on the clean-refused harmful eval prompts.
- Cost: mean KL(p_clean || p_ablated) over the first response position plus 8 teacher-forced continuation positions on 100 generic prompts (S1 edits prompt tokens only; S2 also the continuation tokens).
- Primary cost c0 = 0.1 nats; secondary 0.02, 0.5. Smallest effect of interest delta = 0.05. 95% percentile CIs from 1000 bootstrap resamples; Benjamini–Hochberg within each hypothesis family (alpha = 0.05); a per-pair test also requires BH q <= alpha.

## 2. Pairs

Confirmatory (counted toward C1–C3):

- Qwen/Qwen3-1.7B / sentiment
- Qwen/Qwen3-1.7B / answerability
- Qwen/Qwen3.5-4B / truth
- Qwen/Qwen3.5-4B / refusal
- Qwen/Qwen3.5-4B / sentiment
- Qwen/Qwen3.5-4B / answerability

Development (reported, never counted):

- Qwen/Qwen3-1.7B / truth
- Qwen/Qwen3-1.7B / refusal

## 3. Inclusion rules (mechanical, preregistered)

- Task gate, single-token: clean model on the eval split: AUROC >= 0.85 and coherence (mass on the two answer tokens) >= 0.9.
- Task gate, refusal: at least 150 harmful eval prompts refused (generation-labelled) and harmless refusal <= 0.10.
- Substitution: If sentiment or answerability fails its gate on a model, toxicity (Civil Comments) replaces it on that model if toxicity passes its gate. A pilot task (truth, refusal) failing on Qwen3.5-4B is excluded and reported.
- SAE basis: The SAE basis is included at a (model, depth) only if, on chat-formatted prompts, its reconstruction explains >= 70% of variance and next-token loss rises by <= 10% (relative) with the reconstruction spliced in; otherwise that cell runs without the SAE basis and the report says so.

## 4. Hypotheses

### C1 (primary): Causal selection beats readout selection

- Statistic: Per confirmatory pair, at c0 in the primary setting: Delta_b = E_attrib,b(c0) - max(E_meandiff,b(c0), E_l1probe,b(c0)) for each included basis b; the pair statistic is the mean of Delta_b over included bases.
- A pair supports it if: 95% CI lower bound > 0, point estimate >= delta, no basis whose Delta_b CI lies entirely below 0, and (require_bh_q) BH q <= alpha within the C1 family (one-sided bootstrap p = P*(stat <= 0)).
- Supported overall if: in at least 75% of included confirmatory pairs (rounded up) and in at least one pair per model.
- Rationale: The v1 pilot (development data) suggested attribution-selected units break refusal at about twice the damage of readout-selected units at equal cost. C1 tests whether this holds on unseen tasks and an unseen model.

### C2 (primary): Basis rankings depend on the selection rule

- Statistic: Per pair, over ordered basis pairs (b1, b2) and readout rules r: d_r = E_r,b1(c0) - E_r,b2(c0) and d_a = E_attrib,b1(c0) - E_attrib,b2(c0). A reversal is present when d_r's CI lies above 0 and d_a's CI lies below 0 (both CI-separated); its p-value is the larger of the two one-sided p-values.
- A pair supports it if: at least one CI-separated reversal (and, with require_bh_q, BH q <= alpha within the C2 family)
- Supported overall if: in at least 50% of included confirmatory pairs (rounded up).
- Rationale: v1 (development data) showed PCA beating raw dimensions under readout rules but the reverse under attribution.

### C3 (secondary): Readout strength only weakly predicts causal effect

- Statistic: Per confirmatory pair and basis, at the primary setting: 20 random groups of 20 units (4 per quintile of readout strength |Cohen's d|; sae among latents firing on >= 1% of tokens). Spearman rho between group readout strength (mean |d|) and group damage; pooled by the mean Fisher z over bases and pairs.
- Supported if: pooled rho <= 0.3 and its 95% CI upper bound < 0.5
- Rationale: Readout statistics describe information content, not causal leverage.

## 5. Central-claim label

- `SUPPORTED`: C1 and C2 both supported
- `PARTIALLY SUPPORTED`: exactly one of C1, C2 supported
- `NOT SUPPORTED`: neither supported
- C3 and S4 are reported beside the label as strengthening or weakening evidence. H0 never changes the label.

## 6. Secondary descriptive analyses

- S1: Delta_b by depth for each basis (three depths on Qwen3-1.7B, five on Qwen3.5-4B)
- S2: Refusal: Delta under S2 vs S1 at 50% depth
- S3: Refusal at 50% depth: Kendall tau of the 9 basis x rule cells' E(c0) between mean and resample ablation
- S4:
  - (a) attrib_shuffled (attribution with shuffled class labels): E(c0) within +/- delta of 0
  - (b) cross-task specificity: E_B(units attributed for A) < 0.5 x E_B(units attributed for B) at equal k (50, 200)
  - (c) attribution accuracy: Spearman between -(summed attribution) and observed damage across the C3 groups
- S5: anchors and H0 per cell (refusal anchor drop >= 0.5 in generated replies; random <= 25% of the anchor's damage at c*); G only where the anchor exceeds random by >= 0.1 at c*

## 7. H0 and anchors

- Refusal anchor drop >= 0.5 in generated replies; random <= 25% of the anchor's damage at c*. H0 is shown beside every result as a caveat and never changes a verdict, a label or the decision logic.
- Gap closure G is computed only where the anchor exceeds random by >= 0.1 at c*.

## 8. Run plan (frozen)

- Priority order: B1_primary → B2_controls → B3_tier2_25_75 → B4_refusal_s2_resample → B5_tier2_40_60.
- Dropped before the run (from the smoke projection): none.
- Time guard: stop before a block whose projected time would exceed the budget; skipping depends only on speed.

## 9. Known limitations

- Selection statistics use prompt tokens for both sites; refusal attribution uses the first-token refusal score although refusal damage is generation-based.
- The REINS SAE was trained on Qwen3.5-4B-Base; its use on the post-trained model is checked by the SAE inclusion rule.
- Half-size test sets (12-hour budget): CIs roughly 40% wider than the earlier 30-hour plan.

## 10. Frozen files

SHA-256 hashes are recorded in `prereg/FROZEN.json` at the commit tagged `prereg-v2-frozen`:

- `prereg/hypotheses.yaml`
- `prereg/PREREGISTRATION.md`
- `config/v2.yaml`
- `src/pilot/analysis_v2.py`
- `src/pilot/stats.py`
