"""Batch auto-tune memory guard: a candidate is skipped before it runs when its extrapolated peak exceeds the
tighter of the configured cap and the memory the machine can still give above the watchdog threshold."""

from pilot import budget_v2 as BV
from pilot import memory as MEM


def _guard(monkeypatch, avail_gb, held_gb=10.0, cap=60.0):
    monkeypatch.setattr(MEM, "mem_available_gb", lambda: avail_gb)
    monkeypatch.setattr(BV._MemGuard, "_held_gb", staticmethod(lambda: held_gb))
    g = BV._MemGuard(cap, watchdog_gb=16.0, margin_gb=4.0, safety=1.25)
    g.base = held_gb
    return g


def test_first_candidate_always_runs(monkeypatch):
    assert _guard(monkeypatch, 100.0).check(None, "tokens", 8192, "forward") is None


def test_extrapolation_skips_oversized_candidate(monkeypatch):
    # Qwen3-1.7B backward on the GB10: 37.4 GB at 15.7k tokens -> 64 GB measured at 31.4k
    g = _guard(monkeypatch, avail_gb=100.0)
    prev = {"tokens": 15696, "peak_gb": 37.4}
    rec = g.check(prev, "tokens", 31408, "backward")
    assert rec is not None and rec["skipped"] and rec["predicted_gb"] > 60.0


def test_small_step_runs(monkeypatch):
    g = _guard(monkeypatch, avail_gb=100.0)
    assert g.check({"tokens": 8000, "peak_gb": 14.0}, "tokens", 16000, "forward") is None   # 10 + 4*2*1.25 = 20


def test_machine_headroom_tighter_than_cap(monkeypatch):
    # cap 60 GB, but only 30 GB available: budget = 10 held + 30 - 16 - 4 = 20 GB
    g = _guard(monkeypatch, avail_gb=30.0)
    assert g.budget_gb() == 20.0
    rec = g.check({"batch": 64, "peak_gb": 14.0}, "batch", 128, "generation")      # 10 + 4*2*1.25 = 20 -> fits
    assert rec is None
    rec = g.check({"batch": 128, "peak_gb": 18.0}, "batch", 256, "generation")     # 10 + 8*2*1.25 = 30 -> skip
    assert rec is not None and rec["budget_gb"] == 20.0


def test_no_meminfo_falls_back_to_cap(monkeypatch):
    assert _guard(monkeypatch, avail_gb=0.0).budget_gb() == 60.0
