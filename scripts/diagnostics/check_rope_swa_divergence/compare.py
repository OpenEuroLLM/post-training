"""Compare HF (stock, buggy) and optionally HF (patched) against vLLM (correct).

If only --hf and --vllm are provided: 2-way comparison (V1: bug fires?).
If --hf-patched is also provided: 3-way (V2: does the patch close the gap?).
"""
import argparse
from pathlib import Path

import numpy as np


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def pair_stats(hf_logits, vl, input_ids, label):
    """Return per-position diff stats between an HF run and the vLLM reference."""
    seq_len = len(input_ids)
    hf_logprobs = log_softmax(hf_logits, axis=-1)
    hf_top1 = np.argmax(hf_logits, axis=-1)
    vl_top1 = vl["top1_token"]
    vl_actual = vl["actual_logprob"]

    matched = 0
    total = 0
    deltas = []
    for i in range(1, seq_len):
        v_top = int(vl_top1[i])
        if v_top < 0:
            continue
        total += 1
        if int(hf_top1[i - 1]) == v_top:
            matched += 1
        v_lp = float(vl_actual[i])
        if not np.isnan(v_lp):
            h_lp = float(hf_logprobs[i - 1, input_ids[i]])
            deltas.append((i, h_lp, v_lp, int(hf_top1[i - 1]), v_top))

    diffs = np.array([h - v for _, h, v, _, _ in deltas])
    abs_diffs = np.abs(diffs)

    print(f"  {label}:")
    print(f"    top-1 agree:  {matched}/{total} = {100 * matched / total:.2f}%")
    print(f"    N positions:  {len(diffs)}")
    print(f"    mean |Δlp|:   {abs_diffs.mean():.4f} nats")
    print(f"    median |Δlp|: {np.median(abs_diffs):.4f} nats")
    print(f"    P95 |Δlp|:    {np.percentile(abs_diffs, 95):.4f} nats")
    print(f"    max |Δlp|:    {abs_diffs.max():.4f} nats")
    print(f"    signed mean:  {diffs.mean():+.4f} nats  (HF - vLLM)")

    return {
        "label": label,
        "match_rate": matched / total if total else 0.0,
        "mean_abs": float(abs_diffs.mean()),
        "median_abs": float(np.median(abs_diffs)),
        "p95_abs": float(np.percentile(abs_diffs, 95)),
        "max_abs": float(abs_diffs.max()),
        "signed_mean": float(diffs.mean()),
        "deltas": deltas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf",
                    default=str(Path(__file__).parent / "run" / "hf_logits.npz"))
    ap.add_argument("--hf-patched",
                    default=str(Path(__file__).parent / "run" / "hf_logits_patched.npz"),
                    help="Optional: HF forward with patch_olmo3_swa_rope installed.")
    ap.add_argument("--vllm",
                    default=str(Path(__file__).parent / "run" / "vllm_logprobs.npz"))
    args = ap.parse_args()

    hf = np.load(args.hf)
    vl = np.load(args.vllm)
    assert np.array_equal(hf["input_ids"], vl["input_ids"]), \
        "input_ids differ between HF and vLLM npz — diagnostic premise broken."

    input_ids = hf["input_ids"]
    seq_len = len(input_ids)

    print("=" * 72)
    print(f"Sequence length: {seq_len}  (positions 1..{seq_len - 1} compared)")
    print("=" * 72)

    print()
    print("Stock HF (single model-level rotary_emb) vs vLLM (per-layer):")
    res_stock = pair_stats(hf["logits"], vl, input_ids, "HF stock  vs vLLM")

    res_patched = None
    if Path(args.hf_patched).exists():
        hfp = np.load(args.hf_patched)
        assert np.array_equal(hfp["input_ids"], vl["input_ids"]), \
            "input_ids differ in patched HF npz."
        print()
        print("Patched HF (per-layer rotary_emb, SWA vanilla) vs vLLM:")
        res_patched = pair_stats(hfp["logits"], vl, input_ids, "HF patched vs vLLM")

    # Position-bin breakdown for the stock comparison (where the bug fires).
    print()
    print(f"Stock HF vs vLLM, divergence by position bin:")
    bins = [(0, 64), (64, 256), (256, 512), (512, 1024), (1024, seq_len)]
    print(f"  bin            N    mean|Δlp|   max|Δlp|   top1 agree   |Δlp|>0.5")
    for lo, hi in bins:
        sub = [d for d in res_stock["deltas"] if lo <= d[0] < hi]
        if not sub:
            continue
        sub_diffs = np.array([h - v for _, h, v, _, _ in sub])
        sub_abs = np.abs(sub_diffs)
        n_agree = sum(1 for _, _, _, ht, vt in sub if ht == vt)
        n_hi = int((sub_abs > 0.5).sum())
        print(f"  [{lo:>4}, {hi:>5}):  {len(sub):>4}  "
              f"{sub_abs.mean():>7.3f}    {sub_abs.max():>7.3f}    "
              f"{100 * n_agree / len(sub):>5.1f}%      "
              f"{n_hi:>3}/{len(sub)} ({100 * n_hi / len(sub):.1f}%)")

    # Speed comparison: did the patch slow down the forward?
    print()
    print("Forward-pass timing (single forward, same input, same GPU):")
    def _print_timing(npz, label):
        if "timings_ms" not in npz.files:
            print(f"  {label}: no timings recorded")
            return None
        t = npz["timings_ms"]
        if len(t) == 0:
            print(f"  {label}: timings array empty")
            return None
        med = float(np.median(t))
        print(f"  {label:30s}  median={med:6.2f} ms  "
              f"min={t.min():6.2f}  mean={t.mean():6.2f}  "
              f"p90={np.percentile(t, 90):6.2f}  (n={len(t)})")
        return med

    med_stock = _print_timing(hf, "HF stock")
    med_patched = None
    if res_patched is not None:
        hfp = np.load(args.hf_patched)
        med_patched = _print_timing(hfp, "HF patched (per-layer RoPE)")
    if med_stock and med_patched:
        rel = (med_patched - med_stock) / med_stock * 100.0
        sign = "+" if rel >= 0 else ""
        print(f"  → patch overhead: {sign}{rel:.2f}%  "
              f"({med_patched - med_stock:+.2f} ms / forward)")

    print()
    print("=" * 72)
    # Verdict
    if res_patched is None:
        if res_stock["mean_abs"] > 0.2 or res_stock["max_abs"] > 2.0:
            print(">>> VERDICT (V1): meaningful divergence between stock HF and vLLM.")
            print(">>>   The RoPE-on-SWA-layers bug fires.")
        elif res_stock["mean_abs"] > 0.05:
            print(">>> VERDICT (V1): marginal divergence — above kernel noise floor.")
            print(">>>   Bug may be quiet on this short input; try --max-tokens 4096.")
        else:
            print(">>> VERDICT (V1): divergence within kernel-impl noise floor.")
            print(">>>   Bug not firing on this input.")
        print(">>>   Run with --apply-patch to verify the proposed fix (V2).")
    else:
        # V2: does the patch close the gap?
        # The patch should make patched-HF and vLLM agree to within kernel noise.
        # If patched is meaningfully closer to vLLM than stock is, the patch is doing
        # the right thing.
        delta_match = res_patched["match_rate"] - res_stock["match_rate"]
        delta_mean = res_stock["mean_abs"] - res_patched["mean_abs"]
        print(">>> VERDICT (V2): patch effect on HF↔vLLM divergence")
        print(f">>>   top-1 agree:  {100*res_stock['match_rate']:.2f}%  →  "
              f"{100*res_patched['match_rate']:.2f}%  (Δ {100*delta_match:+.2f} pp)")
        print(f">>>   mean |Δlp|:   {res_stock['mean_abs']:.4f}  →  "
              f"{res_patched['mean_abs']:.4f}  (Δ {-delta_mean:+.4f})")
        print(f">>>   max  |Δlp|:   {res_stock['max_abs']:.4f}  →  "
              f"{res_patched['max_abs']:.4f}")
        if res_patched["mean_abs"] < 0.01 and res_patched["max_abs"] < 0.5:
            print(">>>   ✓ Patched HF and vLLM agree to within kernel noise.")
            print(">>>     The patch is correct — SWA layers receive vanilla RoPE")
            print(">>>     matching vLLM's reference. Ready for a training A/B.")
        elif res_patched["mean_abs"] < res_stock["mean_abs"] / 2:
            print(">>>   ⚠ Patched HF is closer to vLLM than stock, but residual")
            print(">>>     divergence remains (possibly kernel-impl differences).")
        else:
            print(">>>   ✗ Patch did not meaningfully close the HF↔vLLM gap.")
            print(">>>     The fix may be incorrect or incomplete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
