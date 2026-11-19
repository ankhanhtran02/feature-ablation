"""v2 intervention sites: formula and masks, identity, resample joint copy, S1/S2 decode behaviour, decode-cache
consistency on tiny random Qwen3 and Qwen3.5 models, multi-layer gradients, and the attribution identity."""

import numpy as np
import pytest
import torch

from pilot import bases as BZ
from pilot import models as M
from pilot import sites as ST


def _hook(basis, idx, values, site="S1", thr=1e9):
    idx = torch.as_tensor(idx)
    return ST.SiteHook(basis.directions(idx), basis.reader(idx), thr, site, values)


def test_formula_masks_padding_first_token_and_sinks():
    torch.manual_seed(0)
    B, S, d = 3, 6, 8
    h = torch.randn(B, S, d)
    h[1, 3] *= 100.0                                        # an attention sink inside a real prompt
    batch = M.Batch(np.arange(B), torch.zeros(B, S, dtype=torch.long), torch.ones(B, S, dtype=torch.long),
                    torch.zeros(B, S, dtype=torch.long), np.array([6, 4, 2]))
    pm, pos, _, _ = ST.role_masks(batch)
    basis = BZ.DimsBasis(d, "cpu")
    v = torch.full((d,), 0.5)
    hook = _hook(basis, np.arange(d), ST.MeanValues(v), thr=float(h[0].norm(dim=-1).median() * 10))
    hook.set_batch(torch.arange(B), pm, pos)
    out = hook(None, None, h)
    for b, n in enumerate([6, 4, 2]):
        for t in range(S):
            real = t >= S - n
            first = t == S - n
            sink = (b, t) == (1, 3)
            if real and not first and not sink:
                assert torch.allclose(out[b, t], v.expand(d)), (b, t)
            else:
                assert torch.equal(out[b, t], h[b, t]), (b, t)
    assert hook.n_sink[ST.PROMPT] == 1


def test_identity_values_exact_for_every_basis(toy_sae):
    torch.manual_seed(1)
    h = torch.randn(2, 5, 16) * 3
    batch = M.Batch(np.arange(2), torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long),
                    torch.zeros(2, 5, dtype=torch.long), np.array([5, 5]))
    pm, pos, _, _ = ST.role_masks(batch)
    X = torch.randn(100, 16)
    for basis in (BZ.DimsBasis(16, "cpu"), BZ.PCABasis(*BZ.fit_pca(X)), BZ.SAEBasis(toy_sae)):
        hook = _hook(basis, np.arange(min(10, basis.n_units)), ST.IdentityValues())
        hook.set_batch(torch.arange(2), pm, pos)
        out = hook(None, None, h)
        assert torch.allclose(out, h, atol=1e-5), basis.key


def test_resample_copies_all_units_jointly_from_one_source_token():
    torch.manual_seed(2)
    d, k = 6, 4
    src = torch.randn(50, k)
    assign = torch.randint(0, 50, (3, 10))
    vals = ST.ResampleValues(src, assign)
    ex = torch.tensor([0, 0, 2])
    pos = torch.tensor([1, 4, 7])
    v = vals.get(ST.PROMPT, ex, pos, torch.zeros(3, k))
    for i in range(3):
        assert torch.equal(v[i], src[assign[ex[i], pos[i]]])


def test_role_masks_with_teacher_forced_continuation():
    B, S = 2, 7
    batch = M.Batch(np.arange(B), torch.zeros(B, S, dtype=torch.long), torch.ones(B, S, dtype=torch.long),
                    torch.zeros(B, S, dtype=torch.long), np.array([7, 5]))
    pm, pos, gm, gs = ST.role_masks(batch, np.array([4, 3]))
    # row 0: tokens 0..6, prompt 0..3 (first excluded), continuation 4..6
    assert pm[0].tolist() == [False, True, True, True, False, False, False]
    assert gm[0].tolist() == [False, False, False, False, True, True, True]
    assert gs[0, 4:].tolist() == [0, 1, 2]
    # row 1: two pads, prompt tokens at cols 2..4 (first excluded), continuation cols 5..6
    assert pm[1].tolist() == [False, False, False, True, True, False, False]
    assert gm[1].tolist() == [False, False, False, False, False, True, True]


def test_s1_passes_decode_steps_and_s2_edits_them():
    d = 4
    basis = BZ.DimsBasis(d, "cpu")
    batch = M.Batch(np.arange(2), torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long),
                    torch.zeros(2, 3, dtype=torch.long), np.array([3, 3]))
    pm, pos, _, _ = ST.role_masks(batch)
    x1 = torch.randn(2, 1, d)
    for site in ("S1", "S2"):
        hook = _hook(basis, np.arange(d), ST.MeanValues(torch.zeros(d), torch.ones(d)), site=site)
        hook.set_batch(torch.arange(2), pm, pos, generating=True)
        hook(None, None, torch.randn(2, 3, d))                      # prefill
        out = hook(None, None, x1)                                  # decode step 0
        if site == "S1":
            assert torch.equal(out, x1)
        else:
            assert torch.allclose(out, torch.ones_like(x1))
            assert hook.n_edit[ST.GEN] == 2
    with pytest.raises(RuntimeError):
        hook(None, None, torch.randn(2, 2, d))                      # unexpected length


def _device(arch) -> str:
    """Qwen3.5's Gated DeltaNet uses the fla Triton kernels whenever fla is importable, and those run only on CUDA:
    on such a machine (the GB10) the tiny Qwen3.5 runs on the GPU; everywhere else on the CPU."""
    if arch == "qwen3_5" and torch.cuda.is_available():
        import importlib.util

        if importlib.util.find_spec("fla") is not None:
            return "cuda"
    return "cpu"


def _tiny(arch):
    if arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
        torch.manual_seed(0)
        return Qwen3ForCausalLM(cfg).eval()
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, linear_num_value_heads=4, linear_num_key_heads=2,
                            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                            max_position_embeddings=256, layer_types=["linear_attention"] * 3 + ["full_attention"])
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval()


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_5"])
def test_decode_cache_consistency_under_s2(arch):
    """Edits at decode steps flow into the KV / recurrent cache exactly as a full recomputation would."""
    dev = _device(arch)
    model = _tiny(arch).to(dev)
    model.requires_grad_(False)
    layers = M.decoder_layers(model)
    L = 1
    g = torch.Generator().manual_seed(5)
    toks = [torch.randint(3, 97, (n,), generator=g).tolist() for n in (7, 11, 9)]
    batch = M.make_batches(toks, 0, dev, 16, 10**6)[0]
    basis = BZ.DimsBasis(32, dev)
    units = np.arange(0, 32, 3)
    vals = ST.MeanValues(torch.randn(len(units), generator=g).to(dev), (torch.randn(len(units), generator=g) * 3).to(dev))
    n_new = 6
    hook = _hook(basis, units, vals, site="S2")
    pm, pos, _, _ = ST.role_masks(batch)
    hook.set_batch(torch.as_tensor(batch.idx), pm, pos, generating=True)
    hd = ST.attach(layers, L, hook)
    try:
        with torch.inference_mode():
            gen = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, max_new_tokens=n_new,
                                 min_new_tokens=n_new, do_sample=False, pad_token_id=0)
    finally:
        hd.remove()
    new = gen[:, batch.S:]
    assert hook.n_edit[ST.GEN] == len(toks) * (n_new - 1)
    seqs = [toks[int(i)] + new[r].tolist() for r, i in enumerate(batch.idx)]
    plen = np.array([len(toks[int(i)]) for i in batch.idx])
    b2 = M.make_batches(seqs, 0, dev, 16, 10**6)[0]
    plen2 = np.array([len(toks[int(i)]) for i in np.asarray(batch.idx)[b2.idx]])
    h2 = _hook(basis, units, vals, site="S2")
    pm2, pos2, gm2, gs2 = ST.role_masks(b2, plen2)
    h2.set_batch(torch.as_tensor(b2.idx), pm2, pos2, gm2, gs2)
    hd = ST.attach(layers, L, h2)
    try:
        with torch.inference_mode():
            logits = model(input_ids=b2.input_ids, attention_mask=b2.attention_mask, position_ids=b2.position_ids,
                           use_cache=False).logits
    finally:
        hd.remove()
    for row in range(len(b2.idx)):
        start = b2.S - b2.lengths[row] + plen2[row] - 1
        pred = logits[row, start:start + n_new].argmax(-1)
        assert pred.tolist() == seqs[int(b2.idx[row])][plen2[row]:], (arch, row)
    del plen


def test_multilayer_gradients_are_total_derivatives(tiny_qwen):
    layers = M.decoder_layers(tiny_qwen)
    batch = M.make_batches([[5, 6, 7, 8, 9], [10, 11, 12]], 0, "cpu", 8, 10**6)[0]

    def grads(Ls):
        gc = ST.GradCapture(layers, Ls)
        try:
            with torch.enable_grad():
                M.last_logits(tiny_qwen, batch)[:, 3].sum().backward()
        finally:
            gc.remove()
        return {L: gc.g[L].clone() for L in Ls}
    both = grads([0, 2])
    assert torch.allclose(both[0], grads([0])[0], atol=1e-5)
    assert torch.allclose(both[2], grads([2])[2], atol=1e-5)


def test_attribution_is_linear_in_the_mean_value(toy_sae):
    """sum_t (v - r_t) * (w . g_t) == v * G - RG, the form the prep pass accumulates."""
    torch.manual_seed(3)
    x, g = torch.randn(7, 16) * 2, torch.randn(7, 16)
    basis = BZ.SAEBasis(toy_sae)
    v = torch.rand(basis.n_units)
    r = basis.read(x)
    gw = g @ toy_sae.directions().T
    explicit = ((v - r) * gw).sum(0)
    assert torch.allclose(v * gw.sum(0) - (r * gw).sum(0), explicit, atol=1e-5)
