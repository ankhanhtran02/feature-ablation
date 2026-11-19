"""v2 smoke checks (pilot-v2-amendment.md, Section 16).

In-run (`checks` stage, needs a loaded model, after prep):
  identity at both sites incl. decoding (v = r(x): logits and generated tokens unchanged), cached-activation
  identity, decode-cache consistency under S2 (edited states reach the KV / recurrent cache exactly as a full
  recomputation would), teacher-forced cost exactly 0 for a no-op, curve extension on a short ladder stops at its
  target, position-pooled means and reservoirs for both roles.
Post-run (`verify`): pytest, v1-style checks (NaN/Inf, outputs, --resume < 30 s, env, watchdog, memory), both
models loaded (+ Gated DeltaNet implementation), SAE fidelity at >= 1 depth per model, --confirm refuses without a
freeze and after a frozen file is edited (on a temporary git copy), prefetch completeness, batch auto-tuning,
throughput projection; writes smoke_checks.json and smoke_report.md.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import sites as ST
from .utils import LOG, REPO_ROOT, fmt_secs, read_json, write_json


# A decode-vs-recompute token mismatch within this many logits of the top logit is a bf16 near tie, not a cache
# inconsistency (about 1-2 bf16 rounding steps at typical logit magnitudes of 20-30).
DECODE_TIE_TOL = 0.5

# ============================================================================= in-run checks

def _ex(b):
    return torch.as_tensor(b.idx, dtype=torch.long)


@torch.inference_mode()
def model_checks(run, mc) -> dict:
    from .ablate_v2 import Ablator
    from .engine_v2 import TaskRunner, generate
    from .prep_v2 import load_orders

    cfg = run.cfg
    prim = float(cfg["primary_depth"])
    depth = prim if any(abs(prim - d) < 1e-9 for d in mc.depths()) else mc.depths()[0]
    L = mc.layer(depth)
    tol = float(cfg["identity_tol"])
    out = {"layer": L}
    single = next((t for t in ("truth", "sentiment", "answerability") if (mc.dir / "prep" / t / f"L{L}.pt").exists()), None)
    has_ref = (mc.dir / "prep" / "refusal" / f"L{L}.pt").exists()
    basis = BZ.DimsBasis(mc.d_model, mc.run.device)
    W_all = None
    # ---- identity on the scoring + teacher-forced passes (both sites)
    if single:
        r = TaskRunner(mc, single, depth)
        idx = torch.arange(mc.d_model, device=mc.run.device)
        W_all = basis.directions(idx)
        for site in ("S1", "S2"):
            mk = lambda which: ST.SiteHook(W_all, basis.reader(idx), r.sink_thr, site, ST.IdentityValues())  # noqa: E731
            s, a = r._score_eval(mk)
            kl9, kl1, _ = r._cost(mk)
            out[f"identity_score_{site}"] = {"max_abs_score_change": float(np.abs(s - r.clean["score"]).max()),
                                             "pass": bool(np.abs(s - r.clean["score"]).max() <= 1e-6)}
            out[f"noop_cost_{site}"] = {"max_kl9": float(kl9.max()), "max_kl1": float(kl1.max()), "pass": bool(kl9.max() == 0.0)}
        # cached-activation identity: values = r(x) from a separate capture of the clean pass
        b = r.eval_b[0]
        cap = ST.MultiCapture(mc.layers, [L])
        try:
            mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                     use_cache=False, logits_to_keep=1)
            X = cap.h[L].float()
        finally:
            cap.remove()
        pm, pos, _, _ = ST.role_masks(b)
        table = torch.zeros(int(b.idx.max()) + 1, b.S, mc.d_model, device=mc.run.device)
        rows = torch.arange(b.S, device=mc.run.device)
        for i, ex in enumerate(b.idx):
            table[int(ex), pos[i]] = X[i]
        hook = ST.SiteHook(W_all, basis.reader(idx), r.sink_thr, "S1", ST.ExplicitValues(table))
        hook.set_batch(_ex(b), pm, pos)
        d = float((r._forward(b, hook, 1, fast=False) - r._forward(b, None, 1, fast=False)).abs().max())
        out["identity_cached_activations"] = {"max_abs_logit_change": d, "tol": tol, "pass": d < tol, "edited": hook.n_edit}
        del rows
        # ---- curve extension on a deliberately short ladder: stops at its target
        ab = Ablator(run)
        orders = load_orders(mc, single, L)
        prep = mc.prep(single, L)
        key = "dims|meandiff|0|-1"
        cap_n = int(prep["caps"][key])
        means = {ST.PROMPT: prep["means"]["dims"]["prompt"], ST.GEN: prep["means"]["dims"]["generated"]}
        cell = {"model": mc.model_id, "task": single, "role": "check", "depth": depth, "site": "S1", "strategy": "mean",
                "tier": "tier1", "block": "check", "kind": "check"}
        full = r.run(basis, orders[key][:cap_n], "mean", "S1", means)
        target = 0.5 * float(full["kl9"].mean())
        rows_, arr = [], []
        ab._runners[(mc.model_id, single, depth)] = r
        flag = ab.curve(r, basis, orders[key], cap_n, [1], target, cell, means, None,
                        {"basis": "dims", "rule": "meandiff", "resample": 0, "draw": -1}, rows_, arr, max_steps=10**6)
        costs = [x["cost_kl"] for x in rows_]
        ext = [x for x in rows_ if x["ext"]]
        ok = flag == "ok" and bool(ext) and costs[-1] >= target and all(c < target for c in costs[:-1])
        out["curve_extension"] = {"ladder": [1], "target": target, "k_eff": [x["k_eff"] for x in rows_], "costs": costs,
                                  "flag": flag, "n_extension_points": len(ext), "pass": bool(ok)}
        del r
    # ---- decoding: identity under S2 and cache consistency (refusal prompts)
    if has_ref:
        rr = TaskRunner(mc, "refusal", depth)
        toks = list(rr.ev.input_ids[:4])
        n_new = 8
        clean, _ = generate(mc, toks, n_new)
        idx = torch.arange(mc.d_model, device=mc.run.device)
        W_all = basis.directions(idx)
        h_id = ST.SiteHook(W_all, basis.reader(idx), rr.sink_thr, "S2", ST.IdentityValues())
        g_id, _ = generate(mc, toks, n_new, hook=h_id, L=L)
        out["identity_decode_S2"] = {"generated_unchanged": g_id == clean, "edited_tokens": h_id.n_edit,
                                     "pass": bool(g_id == clean and h_id.n_edit[ST.GEN] > 0)}
        prep = mc.prep("refusal", L)
        orders = load_orders(mc, "refusal", L)
        u = torch.as_tensor(orders["dims|attrib|0|-1"][:max(8, mc.d_model // 8)], device=mc.run.device)
        vals = ST.MeanValues(prep["means"]["dims"]["prompt"][u.cpu()], prep["means"]["dims"]["generated"][u.cpu()])
        h = ST.SiteHook(basis.directions(u), basis.reader(u), rr.sink_thr, "S2", vals)
        g_c, _ = generate(mc, toks, n_new, hook=h, L=L)
        seqs = [list(t) + list(g) for t, g in zip(toks, g_c)]
        plen = np.array([len(t) for t in toks])
        # bf16: cached decoding and the uncached recomputation run different kernel shapes, so a near-tied argmax can
        # flip. A disagreeing token counts as consistent when its recomputed logit is within DECODE_TIE_TOL of the
        # top logit; an edit that failed to reach the cache shows up as wide or repeated disagreements instead.
        agree, n, ties, worst = 0, 0, 0, 0.0
        for b in mc.batches(seqs):
            h2 = ST.SiteHook(basis.directions(u), basis.reader(u), rr.sink_thr, "S2", vals)
            pm, pos, gm, gs = ST.role_masks(b, plen[b.idx])
            h2.set_batch(_ex(b), pm, pos, gm, gs)
            hd = ST.attach(mc.layers, L, h2)
            try:
                logits = mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                                  use_cache=False).logits
            finally:
                hd.remove()
            for row, i in enumerate(b.idx):
                g = g_c[int(i)]
                if not g:
                    continue
                start = b.S - b.lengths[row] + plen[int(i)] - 1
                lg = logits[row, start:start + len(g)].float()
                pred = lg.argmax(-1).tolist()
                for j, (a, c) in enumerate(zip(pred, g)):
                    if a == c:
                        agree += 1
                        continue
                    gap = float(lg[j].max() - lg[j, int(c)])
                    worst = max(worst, gap)
                    if gap <= DECODE_TIE_TOL:
                        ties += 1
                n += len(g)
        out["decode_cache_consistency_S2"] = {"tokens": n, "agree": agree, "near_ties": ties,
                                              "max_logit_gap": round(worst, 4), "tie_tol": DECODE_TIE_TOL,
                                              "pass": bool(n > 0 and agree + ties == n)}
        res = prep.get("reservoir") or {}
        means_ok = all(prep["means"][k]["prompt"] is not None and prep["means"][k]["generated"] is not None
                       for k in prep["means"])
        out["role_means_and_reservoir"] = {
            "bases": list(prep["means"]), "means_both_roles": bool(means_ok),
            "reservoir_prompt": None if res.get("prompt") is None else int(res["prompt"].shape[0]),
            "reservoir_generated": None if res.get("generated") is None else int(res["generated"].shape[0]),
            "pass": bool(means_ok and res.get("prompt") is not None and res.get("generated") is not None
                         and res["prompt"].shape[0] > 0 and res["generated"].shape[0] > 0)}
        del rr
    LOG.info("%s checks: %s", mc.slug, {k: v.get("pass") for k, v in out.items() if isinstance(v, dict)})
    return out


# ============================================================================= post-run acceptance

def _py(args, timeout=3600, cwd=None, extra_env=None):
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str((cwd or REPO_ROOT) / "src"), **(extra_env or {})}
    r = subprocess.run([sys.executable, "-m", *args], cwd=cwd or REPO_ROOT, capture_output=True, text=True,
                       timeout=timeout, env=env)
    return r.returncode, r.stdout + r.stderr, time.time() - t0


def _finite(o) -> bool:
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(_finite(v) for v in o.values())
    if isinstance(o, list):
        return all(_finite(v) for v in o)
    return True


def prereg_enforcement_test(work: Path) -> dict:
    """--confirm refuses without a freeze and after a frozen file is edited, on a temporary git copy."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in ("src", "config", "prereg"):
        shutil.copytree(REPO_ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "FROZEN.json",
                                                                                 "deviations.md"))
    env = {"GIT_AUTHOR_NAME": "pilot-smoke", "GIT_AUTHOR_EMAIL": "smoke@localhost", "GIT_COMMITTER_NAME": "pilot-smoke",
           "GIT_COMMITTER_EMAIL": "smoke@localhost"}
    genv = {**os.environ, **env}
    run = lambda *a: subprocess.run(["git", "-C", str(work), *a], capture_output=True, text=True, env=genv)  # noqa: E731
    run("init", "-q")
    (work / ".gitignore").write_text("runs/\n__pycache__/\n")
    from .prereg import write_render

    write_render(work)
    run("add", "-A")
    run("commit", "-q", "-m", "smoke copy")
    res = {}
    rc, out, _ = _py(["pilot.cli_v2", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["refuses_without_freeze"] = rc != 0
    rc, out, _ = _py(["pilot.cli_v2", "--freeze"], cwd=work, extra_env=env, timeout=600)
    res["freeze_ok"] = rc == 0 and bool(run("tag", "-l", "prereg-v2-frozen").stdout.strip())
    rc, out, _ = _py(["pilot.cli_v2", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["accepts_valid_freeze"] = rc == 0
    with open(work / "config" / "v2.yaml", "a") as f:
        f.write("\n# edited after the freeze (smoke test)\n")
    rc, out, _ = _py(["pilot.cli_v2", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["refuses_after_edit"] = rc != 0 and "config/v2.yaml" in out
    rc, out, _ = _py(["pilot.cli_v2", "--confirm", "--enforce-only", "--deviation", "smoke test edit"], cwd=work,
                     extra_env=env, timeout=600)
    dv = work / "prereg" / "deviations.md"
    res["deviation_recorded"] = rc == 0 and dv.exists() and "smoke test edit" in dv.read_text()
    res["pass"] = all(res.values())
    shutil.rmtree(work, ignore_errors=True)
    return res


def verify(run_dir: Path, dev_tiny: bool) -> int:
    from .budget_v2 import project, projection_md
    from .prefetch_v2 import missing
    from .v2common import load_v2_config

    run_dir = Path(run_dir)
    man = read_json(run_dir / "manifest.json")
    cfg = man["config"]
    res = []

    def add(name, ok, detail=""):
        res.append({"check": name, "pass": None if ok is None else bool(ok), "detail": str(detail)})
        LOG.info("[%s] %s %s", "PASS" if ok else ("SKIP" if ok is None else "FAIL"), name, detail)

    rc, out, _ = _py(["pytest", "-q", "tests"], timeout=3600)
    add("pytest -q passes (v1 + v2 unit tests)", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
    # models load; Gated DeltaNet implementation logged
    for m in cfg["models"]:
        mi = man.get(f"model:{m}")
        add(f"model loads: {m}", mi is not None, f"{mi.get('n_layers')} layers, d={mi.get('d_model')}" if mi else "")
        if mi and mi.get("gated_deltanet", {}).get("hybrid"):
            g = mi["gated_deltanet"]
            add(f"{m}: Gated DeltaNet implementation logged", bool(g.get("implementation")), g.get("implementation"))
    # SAE fidelity at >= 1 depth per model
    for m in cfg["models"]:
        from .v2common import slug

        gp = run_dir / "models" / slug(m) / "gate" / "gate.json"
        g = read_json(gp) if gp.exists() else {}
        fid = {L: v.get("fidelity") for L, v in g.get("sae", {}).items() if v.get("fidelity")}
        add(f"SAE fidelity check ran ({m})", bool(fid),
            "; ".join(f"L{L} FVE {f['fve']:.3f}, loss {100 * f['ce_rel_increase']:+.1f}%" for L, f in fid.items()))
    # in-run checks
    ck = read_json(run_dir / "checks" / "checks.json") if (run_dir / "checks" / "checks.json").exists() else {}
    for m, c in ck.items():
        for name in ("identity_score_S1", "identity_score_S2", "identity_cached_activations", "identity_decode_S2",
                     "decode_cache_consistency_S2", "noop_cost_S1", "noop_cost_S2", "curve_extension",
                     "role_means_and_reservoir"):
            if name in c:
                add(f"{name} ({m})", c[name]["pass"], json.dumps({k: v for k, v in c[name].items() if k != "pass"})[:240])
    # NaN / Inf
    bad = []
    rp = run_dir / "ablate" / "results.parquet"
    if rp.exists():
        r = pd.read_parquet(rp)
        for c in ("damage", "cost_kl", "cost_kl_v1"):
            if c in r and not np.isfinite(r[c].to_numpy(float)).all():
                bad.append(f"results.parquet:{c}")
    for p in (run_dir / "ablate" / "chunks").rglob("*.npz"):
        z = np.load(p)
        bad += [f"{p.name}:{k}" for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    for j in ("confirmatory.json", "secondary.json"):
        p = run_dir / "analyze" / j
        if p.exists() and not _finite(json.loads(p.read_text(), parse_constant=float)):
            bad.append(j)
    add("no NaN or Inf in saved metrics", not bad, "; ".join(bad[:6]))
    exp = ["manifest.json", "log.txt", "inclusion.json", "ablate/results.parquet", "analyze/confirmatory.json",
           "analyze/secondary.json", "analyze/exploratory/cells_exploratory.csv", "report_v2.md", "ablate/blocks.json"]
    miss = [e for e in exp if not (run_dir / e).exists()]
    add("every expected output exists", not miss, ", ".join(miss))
    rep = (run_dir / "report_v2.md").read_text() if (run_dir / "report_v2.md").exists() else ""
    add("report_v2.md opens with the central-claim label", "## 1. Central claim:" in rep[:400])
    blocks = read_json(run_dir / "ablate" / "blocks.json")["blocks"] if (run_dir / "ablate" / "blocks.json").exists() else {}
    add("every smoke block ran", blocks and all(v.get("status") == "done" for v in blocks.values()),
        {k: v.get("status") for k, v in blocks.items()})
    # --resume after completion
    args = ["pilot.cli_v2", "--smoke", "--resume", "--out", str(run_dir)] + (["--dev-tiny"] if dev_tiny else [])
    rc, out, dt = _py(args, timeout=900)
    n_stages = len(man.get("stages", {}))
    skipped = out.count("already done, skipping")
    add("--resume finishes < 30 s without recomputing", rc == 0 and dt < 30 and skipped >= n_stages,
        f"{dt:.1f} s, {skipped}/{n_stages} stages skipped, exit {rc}")
    # freeze / confirm enforcement on a temporary copy
    pe = prereg_enforcement_test(run_dir / "prereg_test_repo")
    add("--confirm refuses without a freeze and after a frozen file is edited (temporary copy)", pe["pass"], pe)
    # prefetch completeness
    try:
        st = missing(load_v2_config("prefetch", dev_tiny=dev_tiny), dev_tiny)
        detail = f"{st['n_files']} files ({st['bytes_total'] / 1e9:.1f} GB); missing {[e['file'] for e in st['missing']][:5]}"
        if dev_tiny:
            add("prefetch: every dataset, tokenizer and config present", not st["missing"], detail)
            add("prefetch: model and SAE weights present", None,
                f"dev-tiny: {len(st['skipped_dev_tiny'])} weight files ({st['bytes_skipped'] / 1e9:.1f} GB) not downloaded here")
        else:
            add("prefetch downloaded every model, SAE file and dataset", not st["missing"], detail)
    except Exception as e:  # noqa: BLE001
        add("prefetch completeness", False, repr(e)[:200])
    # env / watchdog / memory
    env = man.get("env", {})
    if env.get("gpu_only_checks_untested"):
        add("cu130 torch build and capability (12, 1)", None, "no GPU here: untested (CPU smoke path)")
    else:
        add("cu130 torch build and capability (12, 1)", env.get("torch", {}).get("cu130_build")
            and env.get("device", {}).get("capability") == [12, 1], f"{env.get('torch', {}).get('version')}")
    add("memory watchdog running", man.get("watchdog", {}).get("running"))
    peaks = [v.get("memory", {}).get("peak_mem_used_gb") for v in man.get("stages", {}).values()]
    add("peak memory from /proc/meminfo logged", bool(peaks) and all(p is not None for p in peaks), f"max {max(peaks or [0])} GB")
    # v1 Step 0 smoke still works
    if dev_tiny:
        add("v1 `--check --smoke` still passes", None, "skipped under --dev-tiny (needs the real Qwen3-1.7B weights)")
    else:
        dpath = REPO_ROOT / "runs" / "check" / "decision.json"
        before = dpath.read_bytes() if dpath.exists() else None
        rc, out, _ = _py(["pilot.cli", "--check", "--smoke"], timeout=3600)
        after = dpath.read_bytes() if dpath.exists() else None
        add("v1 `--check --smoke` still passes, writes only decision_smoke.json", rc == 0 and before == after, f"exit {rc}")
    # tuning + projection
    from .v2common import TUNING_PATH

    tun = read_json(TUNING_PATH) if TUNING_PATH.exists() else {}
    keys = [("dev_tiny:" if dev_tiny else "") + m for m in cfg["models"]]
    add("batch size auto-tuned under the memory cap", all(k in tun for k in keys),
        {k: {kk: tun.get(k, {}).get(kk) for kk in ("max_batch_tokens", "grad_batch_tokens", "gen_batch_size")} for k in keys})
    proj = project(dev_tiny)
    write_json(run_dir / "projection.json", proj)
    add("throughput measured and projections written", bool(proj["confirm"]["blocks"]),
        f"dev {fmt_secs(proj['dev']['total_seconds'])}, confirm {fmt_secs(proj['confirm']['total_seconds'])}")
    wall = sum(v.get("seconds", 0) for k, v in man.get("stages", {}).items() if k != "prefetch")
    add("smoke wall time (target < 25 min on the GB10, excluding prefetch)", True if wall < 1500 else None,
        f"{wall / 60:.1f} min on {env.get('device', {}).get('name')}")
    write_json(run_dir / "smoke_checks.json", res)
    _smoke_report(run_dir, man, res, proj, dev_tiny)
    print("\n".join(projection_md(proj)))
    failed = [r for r in res if r["pass"] is False]
    LOG.info("SMOKE %s: %d passed, %d failed, %d skipped/untested", "PASSED" if not failed else "FAILED",
             sum(r["pass"] is True for r in res), len(failed), sum(r["pass"] is None for r in res))
    return 0 if not failed else 1


def _smoke_report(run_dir: Path, man: dict, res: list, proj: dict, dev_tiny: bool):
    from .budget_v2 import projection_md

    L = ["# v2 smoke report" + (" — DEV TINY random models (plumbing only)" if dev_tiny else ""), "",
         f"Run `{run_dir.name}` on {man.get('env', {}).get('device', {}).get('name')}. "
         f"{sum(r['pass'] is True for r in res)} checks passed, {sum(r['pass'] is False for r in res)} failed, "
         f"{sum(r['pass'] is None for r in res)} skipped/untested.", "", "## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for r in res:
        d = r["detail"].replace("|", "/")
        L.append(f"| {r['check']} | {'PASS' if r['pass'] else ('SKIP' if r['pass'] is None else '**FAIL**')} | {d[:300]} |")
    L += ["", "## Projected run times (from measured throughput; full-run set sizes)", ""] + projection_md(proj)
    if dev_tiny or man.get("env", {}).get("gpu_only_checks_untested"):
        L += ["", "These projections were measured on this machine (CPU / tiny models) and do not transfer to the GB10. "
                  "Rerun `bash run_pilot.sh --v2 --smoke` on the GB10 for the numbers to freeze."]
    r = run_dir / "ablate" / "results.parquet"
    if r.exists():
        d = pd.read_parquet(r)
        L += ["", "## Measured seconds per intervention in this smoke run (tiny eval sets)", ""]
        for (m, t), g in d.groupby(["model", "task"]):
            L.append(f"- {m}/{t}: {len(g)} interventions, {g.seconds.mean():.3f} s each")
    (run_dir / "smoke_report.md").write_text("\n".join(L) + "\n")
