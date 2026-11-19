"""Batch auto-tuning, throughput measurement and run-time projection (smoke) against the 12-hour budget.

Tuning (per model, in the smoke run): the largest forward batch (padded tokens), backward batch and generation
batch whose peak memory stays under the cap, with measured throughput. Results go to runs/v2/tuning.json, which
--dev / --confirm read. Projection: seconds per intervention per (model, task) from the measured throughput and
the full-run set sizes, then per block in priority order; blocks past the budget are listed (the user decides
what to drop before freezing; the drop list is frozen in prereg/hypotheses.yaml).
"""

from __future__ import annotations

import math
import resource
import time

import numpy as np
import torch

from . import models as M
from .utils import LOG, fmt_secs, read_json, write_json
from .v2common import TUNING_PATH, layer_for, load_v2_config, models_in


def _peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _MemGuard:
    """Decides, before a tuning candidate runs, whether its predicted peak fits.

    The fixed cap alone is checked only after a candidate has run, so one oversized candidate can take a shared
    unified-memory machine (GB10) below the watchdog threshold. Here the next peak is extrapolated linearly in
    batch size from the last measured candidate (peak memory above the resident baseline grows ~linearly), times
    a safety factor, and compared with the tighter of the configured cap and what the machine can still give:
    memory torch already holds + MemAvailable - watchdog threshold - margin.
    """

    def __init__(self, cap_gb: float, watchdog_gb: float, margin_gb: float = 4.0, safety: float = 1.25):
        self.cap, self.watchdog, self.margin, self.safety = cap_gb, watchdog_gb, margin_gb, safety
        self.base = 0.0

    @staticmethod
    def _held_gb() -> float:
        return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    def start_ladder(self):
        _reset()
        self.base = self._held_gb()                       # weights and other resident tensors

    def budget_gb(self) -> float:
        from .memory import mem_available_gb

        avail = mem_available_gb()
        if avail <= 0:                                    # no /proc/meminfo (not Linux): fixed cap only
            return self.cap
        return min(self.cap, self._held_gb() + avail - self.watchdog - self.margin)

    def check(self, prev: dict | None, size_key: str, size: int, what: str) -> dict | None:
        """None when the candidate may run; otherwise a 'skipped' record for the tuning file."""
        if prev is None or not prev.get(size_key):
            return None
        pred = self.base + max(prev["peak_gb"] - self.base, 0.0) * size / prev[size_key] * self.safety
        budget = self.budget_gb()
        if pred <= budget:
            return None
        LOG.info("tune %s: skip %s=%d (predicted peak %.1f GB > budget %.1f GB)", what, size_key, size, pred, budget)
        return {size_key: int(size), "skipped": True, "predicted_gb": round(pred, 2), "budget_gb": round(budget, 2)}


def _batch_of(mc, tokens: int, src: list[list[int]]):
    lens = [len(t) for t in src]
    L = int(np.median(lens))
    n = max(1, tokens // L)
    reps = [src[i % len(src)][:L] if len(src[i % len(src)]) >= L else src[i % len(src)] for i in range(n)]
    return M.make_batches(reps, mc.tok.pad_token_id, mc.run.device, 10**6, 10**9)[0]


def tune_model(mc) -> dict:
    cfg = mc.cfg
    bc = cfg["batching"]
    cap = float(bc["memory_cap_gb"])
    src = [list(x) for x in mc.tokenized("answerability", "eval").input_ids] if "answerability" in _tasks(cfg) else \
        [list(x) for x in mc.tokenized("truth", "eval").input_ids]
    model = mc.model
    out = {"device": str(mc.run.device), "memory_cap_gb": cap, "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "forward": [], "backward": [], "generate": []}
    from .memory import effective_threshold_gb

    guard = _MemGuard(cap, effective_threshold_gb(float(cfg["memory"]["watchdog_min_available_gb"]))[0])
    # forward (scoring / cost passes)
    best = None
    guard.start_ladder()
    for tok_n in bc["tune_candidates_tokens"]:
        b = _batch_of(mc, int(tok_n), src)
        skip = guard.check(best, "tokens", b.n_tokens, "forward")
        if skip:
            out["forward"].append(skip)
            break
        _reset()
        try:
            with torch.inference_mode():
                M.last_logits(model, b)
                _sync()
                t0 = time.time()
                for _ in range(2):
                    M.last_logits(model, b)
                _sync()
                dt = (time.time() - t0) / 2
        except torch.OutOfMemoryError:
            out["forward"].append({"tokens": int(tok_n), "oom": True})
            break
        peak = _peak_gb()
        rec = {"tokens": int(b.n_tokens), "batch": int(len(b.idx)), "seconds": dt, "tokens_per_s": b.n_tokens / dt, "peak_gb": peak}
        out["forward"].append(rec)
        if peak > 0.9 * cap:
            break
        best = rec
    out["max_batch_tokens"] = int(max(best["tokens"], 1024)) if best else int(bc["max_batch_tokens"])
    out["fwd_tokens_per_s"] = best["tokens_per_s"] if best else None
    # backward (attribution passes): gradient w.r.t. the lowest depth's layer output
    Ls = sorted({layer_for(d, mc.n_layers) for d in mc.depths()})
    from .sites import GradCapture

    bestg = None
    guard.start_ladder()
    for tok_n in [x // 4 for x in bc["tune_candidates_tokens"]]:
        b = _batch_of(mc, int(tok_n), src)
        skip = guard.check(bestg, "tokens", b.n_tokens, "backward")
        if skip:
            out["backward"].append(skip)
            break
        _reset()
        try:
            gc = GradCapture(mc.layers, Ls)
            try:
                t0 = time.time()
                with torch.enable_grad():
                    M.last_logits(model, b)[:, 0].sum().backward()
                _sync()
                dt = time.time() - t0
            finally:
                gc.remove()
        except torch.OutOfMemoryError:
            out["backward"].append({"tokens": int(tok_n), "oom": True})
            break
        peak = _peak_gb()
        rec = {"tokens": int(b.n_tokens), "seconds": dt, "tokens_per_s": b.n_tokens / dt, "peak_gb": peak}
        out["backward"].append(rec)
        if peak > 0.9 * cap:
            break
        bestg = rec
    out["grad_batch_tokens"] = int(max(bestg["tokens"], 512)) if bestg else int(bc["grad_batch_tokens"])
    out["grad_tokens_per_s"] = bestg["tokens_per_s"] if bestg else None
    # generation (refusal damage): time per call with 4 and 16 new tokens -> fixed + per-step cost
    gsrc = [list(x) for x in mc.tokenized("refusal", "eval_harmless").input_ids] if "refusal" in _tasks(cfg) else src
    bestgen = None
    guard.start_ladder()
    for bs in bc["tune_candidates_gen"]:
        skip = guard.check(bestgen, "batch", int(bs), "generation")
        if skip:
            out["generate"].append(skip)
            break
        b = M.make_batches([gsrc[i % len(gsrc)] for i in range(int(bs))], mc.tok.pad_token_id, mc.run.device, 10**6, 10**9)[0]
        _reset()
        try:
            ts = {}
            with torch.inference_mode():
                for new in (4, 16):
                    t0 = time.time()
                    model.generate(input_ids=b.input_ids, attention_mask=b.attention_mask, max_new_tokens=new,
                                   min_new_tokens=new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=mc.tok.pad_token_id)
                    _sync()
                    ts[new] = time.time() - t0
        except torch.OutOfMemoryError:
            out["generate"].append({"batch": int(bs), "oom": True})
            break
        peak = _peak_gb()
        step = max((ts[16] - ts[4]) / 12, 1e-6)
        rec = {"batch": int(bs), "step_seconds": step, "fixed_seconds": max(ts[4] - 4 * step, 0.0), "peak_gb": peak,
               "mean_prompt_tokens": float(np.mean([len(x) for x in gsrc]))}
        out["generate"].append(rec)
        if peak > 0.9 * cap:
            break
        bestgen = rec
    out["gen_batch_size"] = int(bestgen["batch"]) if bestgen else int(bc["gen_batch_size"])
    out["gen"] = bestgen
    LOG.info("%s tuned: forward %d tokens/batch (%.0f tok/s), backward %d tokens/batch, generation batch %d "
             "(%.3f s/step)", mc.slug, out["max_batch_tokens"], out["fwd_tokens_per_s"] or 0, out["grad_batch_tokens"],
             out["gen_batch_size"], (bestgen or {}).get("step_seconds", float("nan")))
    return out


def _tasks(cfg):
    from .v2common import tasks_in

    return tasks_in(cfg)


def save_tuning(key: str, rec: dict):
    data = read_json(TUNING_PATH) if TUNING_PATH.exists() else {}
    data[key] = rec
    write_json(TUNING_PATH, data)


# ============================================================================= projection

def _task_sizes(cfg: dict) -> dict:
    d = cfg["data"]
    return {"truth": int(d["truth"]["n_eval"]), "sentiment": int(d["sentiment"]["n_eval"]),
            "answerability": int(d["answerability"]["n_eval"]), "toxicity": int(d["toxicity"]["n_eval"]),
            "refusal": int(d["refusal"]["n_eval_harmful"]) + int(d["refusal"]["n_eval_harmless"])}


def sec_per_intervention(t: dict, cfg: dict, task: str, mean_tokens: dict, frac: float) -> float:
    """t: tuning record; frac: share of layers run by the fast path (1 - L/n)."""
    n_eval = _task_sizes(cfg)[task]
    tps = float(t["fwd_tokens_per_s"])
    cost_tok = int(cfg["cost"]["n_generic"]) * (mean_tokens.get("generic", 30) + int(cfg["cost"]["continuation_tokens"]))
    s = cost_tok * frac / tps
    if task == "refusal":
        g = t["gen"]
        nb = math.ceil(n_eval / int(t["gen_batch_size"]))
        s += nb * (g["fixed_seconds"] + int(cfg["refusal_generation"]["max_new_tokens"]) * g["step_seconds"])
    else:
        s += n_eval * mean_tokens.get(task, 40) * frac / tps
    return s + float(cfg["budget"]["overhead_s_per_intervention"])


def projected_seconds(cfg: dict, model: str, task: str) -> float | None:
    if not TUNING_PATH.exists():
        return None
    t = read_json(TUNING_PATH).get(("dev_tiny:" if cfg.get("dev_tiny") else "") + model)
    if not t or not t.get("fwd_tokens_per_s") or not t.get("mean_tokens"):
        return None
    return sec_per_intervention(t, cfg, task, t["mean_tokens"], 1.0 - float(cfg["primary_depth"]))


def project(dev_tiny: bool) -> dict:
    """Projected --dev and --confirm times per block from runs/v2/tuning.json (full-run configs)."""
    from .ablate_v2 import n_interventions

    data = read_json(TUNING_PATH) if TUNING_PATH.exists() else {}
    out = {}
    for phase in ("dev", "confirm"):
        cfg = load_v2_config(phase, dev_tiny=dev_tiny)
        rows, total = [], 0.0
        prep_total = 0.0
        for m in models_in(cfg):
            t = data.get(("dev_tiny:" if dev_tiny else "") + m)
            if not t or not t.get("fwd_tokens_per_s"):
                continue
            prep_total += _prep_seconds(t, cfg, m)
        rows.append({"block": "prep (gates, caching, PCA, selection)", "seconds": prep_total})
        total += prep_total
        pairs_by_model = {}
        for role in ("confirmatory", "development"):
            for m, task in cfg["pairs"][role]:
                if m in models_in(cfg) and task in (cfg.get("tasks_only") or [task]):
                    pairs_by_model.setdefault(m, []).append((task, role))
        prim = float(cfg["primary_depth"])
        for blk in cfg["blocks"]:
            s = 0.0
            n = 0.0
            for m, plist in pairs_by_model.items():
                t = data.get(("dev_tiny:" if dev_tiny else "") + m)
                if not t or not t.get("fwd_tokens_per_s"):
                    continue
                n_layers = int(t["n_layers"])
                mds = [float(x) for x in cfg["models"][m]["depths"]]
                if cfg.get("depths_only"):
                    mds = [d for d in mds if any(abs(d - float(o)) < 1e-9 for o in cfg["depths_only"])]
                cells = []
                for task, role in plist:
                    if blk["cells"] in ("primary", "controls") and prim in mds:
                        cells.append((task, prim, "tier1", "controls" if blk["cells"] == "controls" else "grid"))
                    elif blk["cells"] == "tier2":
                        cells += [(task, d, "tier2", "grid") for d in blk.get("depths", []) if d in mds and d != prim]
                    elif blk["cells"] == "refusal_extra" and task == "refusal" and prim in mds:
                        cells += [(task, prim, "tier1", "grid")] * 2
                for task, d, tier, kind in cells:
                    frac = 1.0 - layer_for(d, n_layers) / n_layers
                    spi = sec_per_intervention(t, cfg, task, t["mean_tokens"], frac)
                    ni = _n_int(cfg, tier, kind)
                    s += ni * spi
                    n += ni
            rows.append({"block": blk["id"], "seconds": s, "interventions": n})
            total += s
        out[phase] = {"blocks": rows, "total_seconds": total}
    cfgc = load_v2_config("confirm", dev_tiny=dev_tiny)
    budget = float(cfgc["budget"]["total_hours"]) * 3600
    dev_s = out["dev"]["total_seconds"]
    remaining = budget - dev_s
    acc, fits = 0.0, []
    for r in out["confirm"]["blocks"]:
        acc += r["seconds"]
        r["cumulative_seconds"] = acc
        r["fits"] = acc <= remaining
        if not r["fits"]:
            fits.append(r["block"])
    out["budget_seconds"] = budget
    out["confirm_budget_seconds"] = remaining
    out["blocks_not_fitting"] = fits
    return out


def _n_int(cfg, tier, kind) -> float:
    t = cfg["tiers"][tier]
    lad = len(t["budgets"])
    ext = 1.0 + float(cfg["budget"]["extension_factor"])
    if kind == "controls":
        c3 = cfg["controls"]["c3"]
        return 3 * (lad * ext + len(cfg["controls"]["cross_task_k"]) + int(c3["n_quintiles"]) * int(c3["groups_per_quintile"]))
    return 3 * (3 * int(t["discovery_resamples"]) + int(t["random_draws"])) * lad * ext + 1 + int(t["random_dir"])


def _prep_seconds(t: dict, cfg: dict, m: str) -> float:
    """Gate + prep: baselines, refusal generation (calibration, eval pool, discovery replies), one forward +
    backward over every discovery set, PCA forward."""
    from .v2common import tasks_in

    tps, gtps = float(t["fwd_tokens_per_s"]), float(t.get("grad_tokens_per_s") or t["fwd_tokens_per_s"] / 3)
    mt = t["mean_tokens"]
    d = cfg["data"]
    disc = {"truth": 1196, "refusal": 560, "sentiment": int(d["sentiment"]["n_discovery"]),
            "answerability": int(d["answerability"]["n_discovery"]), "toxicity": int(d["toxicity"]["n_discovery"])}
    s = 0.0
    for task in tasks_in(cfg):
        s += disc[task] * mt.get(task, 40) / gtps                       # attribution pass (all depths)
        s += (disc[task] + _task_sizes(cfg).get(task, 500)) * mt.get(task, 40) / tps * 3   # baselines, PCA, fidelity
    g = t["gen"]
    n_gen = 2 * int(cfg["tasks"]["refusal"]["calib_n"]) + int(d["refusal"]["eval_harmful_pool"]) + \
        int(d["refusal"]["n_eval_harmless"]) + 560
    s += math.ceil(n_gen / int(t["gen_batch_size"])) * (g["fixed_seconds"] + 32 * g["step_seconds"])
    return s + 120.0                                                    # model load, tokenization, L1 paths


def projection_md(p: dict) -> list[str]:
    L = ["| phase | block | interventions | projected time | cumulative | fits in budget |", "|---|---|---|---|---|---|"]
    for r in p["dev"]["blocks"]:
        L.append(f"| dev | {r['block']} | {r.get('interventions', 0):.0f} | {fmt_secs(r['seconds'])} | | |")
    for r in p["confirm"]["blocks"]:
        L.append(f"| confirm | {r['block']} | {r.get('interventions', 0):.0f} | {fmt_secs(r['seconds'])} | "
                 f"{fmt_secs(r['cumulative_seconds'])} | {'yes' if r['fits'] else '**NO**'} |")
    L += ["", f"Dev total {fmt_secs(p['dev']['total_seconds'])}; confirm total {fmt_secs(p['confirm']['total_seconds'])}; "
              f"budget {fmt_secs(p['budget_seconds'])} (confirm gets {fmt_secs(p['confirm_budget_seconds'])} after the dev run)."]
    if p["blocks_not_fitting"]:
        L.append(f"**Blocks that would not fit: {', '.join(p['blocks_not_fitting'])}.** Decide what to drop or shrink and "
                 "list it under run_plan.dropped_blocks in prereg/hypotheses.yaml before freezing.")
    else:
        L.append("All blocks fit in the budget.")
    return L
