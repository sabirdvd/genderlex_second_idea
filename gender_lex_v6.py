import os, math, random
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm.auto import tqdm
from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- CONFIG --------------------
REPO_ID    = "AhmedSSabir/GenderLex"   # HF dataset repo
REPO_TYPE  = "dataset"
MODEL_ID   = "gpt2"                    # change if you want a different causal LM
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = "genderlex_results"
RNG_SEED   = 1337

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- MODEL ---------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

# --------- core scorer (returns LOG-prob) ----------
def cloze_prob_last_word(text: str,
                         tokenizer: AutoTokenizer,
                         model: AutoModelForCausalLM,
                         device: str) -> float:
    """LOG P(last word | prefix) for a causal LM. The 'last word' is the text after the last space."""
    whole_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    parts = text.rsplit(" ", 1)
    if len(parts) < 2:
        raise ValueError("Need at least a prefix and a last word.")
    stem = parts[0]
    stem_ids = tokenizer.encode(stem, return_tensors='pt').to(device)

    start = stem_ids.size(1)
    length = whole_ids.size(1) - start

    with torch.no_grad():
        out = model(whole_ids)
        lp  = log_softmax(out.logits, dim=-1)

    total = 0.0
    for i in range(length):
        tok_id  = whole_ids[0, start + i]
        prevpos = start + i - 1
        total  += lp[0, prevpos, tok_id].item()
    return float(total)

def extract_prefix(s: str) -> str:
    parts = s.rsplit(" ", 1)
    if len(parts) < 2:
        raise ValueError("Sentence must contain at least one space.")
    return parts[0]

# ---------- pronoun distributions (via cloze; robust) ----------
def pronoun_logps_by_cloze(prefix: str,
                           pronouns: Tuple[str, ...] = ("him","her","them")) -> Dict[str, float]:
    """Return dict of LOG-probs for each pronoun as the next word (computed via cloze on full string)."""
    out = {}
    for w in pronouns:
        out[w] = cloze_prob_last_word(f"{prefix} {w}", tokenizer, model, DEVICE)
    return out

def softmax_from_logps(logps: Dict[str, float]) -> Dict[str, float]:
    keys = list(logps.keys())
    vals = np.array([logps[k] for k in keys], dtype=np.float64)
    m = vals.max()
    probs = np.exp(vals - m)
    probs /= probs.sum()
    return {k: float(v) for k, v in zip(keys, probs)}

# ---------- divergences & helpers ----------
def entropy(p: Dict[str, float]) -> float:
    eps = 1e-12
    return -sum(max(v, eps) * math.log(max(v, eps)) for v in p.values())

def kl_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    return sum(max(p[k],eps) * (math.log(max(p[k],eps)) - math.log(max(q[k],eps))) for k in p.keys())

def js_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    m = {k: 0.5*(p[k] + q[k]) for k in p.keys()}
    return 0.5*kl_div(p, m) + 0.5*kl_div(q, m)

def log_odds_2way(p_him: float, p_her: float) -> float:
    eps = 1e-12
    return math.log(max(p_him,eps)) - math.log(max(p_her,eps))

# ---------- bootstrap & significance ----------
def bootstrap_ci(values: Iterable[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = RNG_SEED):
    vals = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    n = len(vals)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    for _ in range(n_boot):
        samp = rng.choice(vals, size=n, replace=True)
        means.append(np.mean(samp))
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(np.mean(vals)), float(hi)

def binom_test_greater(k: int, n: int, p0: float = 0.5) -> float:
    from math import comb
    if k<0 or k>n: return float('nan')
    tail = 0.0
    for i in range(k, n+1):
        tail += comb(n, i) * (p0**i) * ((1-p0)**(n-i))
    return float(tail)

# --------------- HF helpers ---------------------
def list_relevant_csvs(repo_id: str) -> List[str]:
    files = list_repo_files(repo_id, repo_type=REPO_TYPE)
    csvs  = [p for p in files if p.lower().endswith(".csv")]
    keep_keys = ("occ", "verb", "noun")
    keep = [p for p in csvs if any(k in p.lower() for k in keep_keys)]
    return sorted(keep)

def classify_from_path(path_in_repo: str) -> Optional[str]:
    name = path_in_repo.lower()
    if "occ"  in name: return "occ"
    if "verb" in name: return "verb"
    if "noun" in name: return "noun"
    return None

def download_csv(path_in_repo: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=path_in_repo)

# --------- optional neutral map (by context) ----------
def build_neutral_prefix_map(all_csvs: List[str], ctype: str) -> Dict[str, str]:
    """
    Try to find a neutral subject file matching this context type (e.g., 'occ' + 'GN'/'neutral'/'PN').
    Returns a map context -> neutral prefix (drop last word).
    """
    candidates = [p for p in all_csvs if ctype in p.lower() and any(tag in p.lower() for tag in ("gn", "neutral", "pn"))]
    neutral_map = {}
    if not candidates:
        return neutral_map
    try:
        fp = download_csv(sorted(candidates)[0])
        df = pd.read_csv(fp)
        if {"sent_m","context"}.issubset(df.columns):
            for _, r in df.iterrows():
                neutral_map[str(r["context"]).lower()] = extract_prefix(str(r["sent_m"]))
    except Exception:
        pass
    return neutral_map

# --------------- per-file scoring ----------------
def score_file(path_in_repo: str, all_csvs: List[str]):
    ctype = classify_from_path(path_in_repo)
    if ctype is None:
        print(f"[skip] Not occ/verb/noun: {path_in_repo}")
        return

    local_fp = download_csv(path_in_repo)
    df = pd.read_csv(local_fp)

    needed = {"sent_m", "sent_w", "context"}
    if not needed.issubset(df.columns):
        print(f"[skip] {path_in_repo} missing columns {needed}. Found: {list(df.columns)}")
        return

    # try to get neutral prefixes for this type (if a matching neutral file exists)
    neutral_prefix_by_ctx = build_neutral_prefix_map(all_csvs, ctype)

    rows = []
    print(f"\nScoring {ctype.upper()} file: {path_in_repo}  (rows={len(df)})")
    for _, r in tqdm(df.iterrows(), total=len(df), unit="row"):
        sent_m = str(r["sent_m"])
        sent_w = str(r["sent_w"])
        ctx    = str(r["context"])
        hb     = str(r["HB"]) if "HB" in df.columns else None

        # bias = log P(him) - log P(her) using your scorer
        lp_him = cloze_prob_last_word(sent_m, tokenizer, model, DEVICE)
        lp_her = cloze_prob_last_word(sent_w, tokenizer, model, DEVICE)
        bias   = lp_him - lp_her
        ratio  = float(math.exp(bias))

        # 3-way distribution at the slot (occ-context)
        prefix     = extract_prefix(sent_m)  # same prefix for sent_w
        logps_occ  = pronoun_logps_by_cloze(prefix, pronouns=("him","her","them"))
        P_occ      = softmax_from_logps(logps_occ)
        H_occ      = entropy(P_occ)
        logodds_oh = log_odds_2way(P_occ["him"], P_occ["her"])

        # optional neutral baseline
        neu_pref = neutral_prefix_by_ctx.get(ctx.lower()) if neutral_prefix_by_ctx else None
        P_neu = None
        KL = JS = d_logodds = np.nan
        if neu_pref is not None:
            logps_neu = pronoun_logps_by_cloze(neu_pref, pronouns=("him","her","them"))
            P_neu     = softmax_from_logps(logps_neu)
            KL        = kl_div(P_occ, P_neu)
            JS        = js_div(P_occ, P_neu)
            d_logodds = logodds_oh - log_odds_2way(P_neu["him"], P_neu["her"])

        # GT alignment (if HB is present: 'M'/'F')
        gt_aligned_logodds = gt_aligned_ratio = np.nan
        gt_correct = np.nan
        if hb and hb.upper() in {"M","F"}:
            sign = +1.0 if hb.upper() == "M" else -1.0
            gt_aligned_logodds = sign * bias
            gt_aligned_ratio   = float(math.exp(gt_aligned_logodds))
            pred_is_male       = (lp_him > lp_her)
            gt_correct         = float((pred_is_male and hb.upper()=="M") or ((not pred_is_male) and hb.upper()=="F"))

        rows.append({
            "context_type": ctype,
            "source_file":  path_in_repo,
            "context":      ctx,
            "HB":           hb,

            "sent_m":       sent_m,
            "sent_w":       sent_w,

            "logP_him":     lp_him,
            "logP_her":     lp_her,
            "bias_logdiff": bias,
            "ratio_him_over_her": ratio,

            "P_occ_him":    P_occ["him"],
            "P_occ_her":    P_occ["her"],
            "P_occ_them":   P_occ["them"],
            "H_occ":        H_occ,
            "logodds_occ_him_her": logodds_oh,

            "P_neu_him":    (np.nan if P_neu is None else P_neu["him"]),
            "P_neu_her":    (np.nan if P_neu is None else P_neu["her"]),
            "P_neu_them":   (np.nan if P_neu is None else P_neu["them"]),
            "H_neu":        (np.nan if P_neu is None else entropy(P_neu)),
            "KL_occ_neu":   KL,
            "JS_occ_neu":   JS,
            "delta_logodds_him_her": d_logodds,

            "gt_aligned_logodds": gt_aligned_logodds,
            "gt_aligned_ratio":   gt_aligned_ratio,
            "gt_correct":         gt_correct,
        })

    # ----- save per-row results -----
    out_df = pd.DataFrame(rows)
    base   = os.path.splitext(os.path.basename(path_in_repo))[0]
    rows_path = os.path.join(OUT_DIR, f"{base}__{MODEL_ID}.csv")
    out_df.to_csv(rows_path, index=False)
    print(f"✅ Saved rows: {rows_path}")

    # ----- build per-file summary (+ context-level aggregation) -----
    def boot_series(label: str, s: pd.Series) -> Dict[str, float]:
        s = s.dropna().astype(float)
        lo, mean, hi = bootstrap_ci(s.values, n_boot=1000, alpha=0.05, seed=RNG_SEED)
        return {"metric": label, "mean": mean, "ci_low": lo, "ci_high": hi, "n": int(s.size)}

    global_stats = []
    global_stats.append(boot_series("bias_logdiff (logP(him)-logP(her))", out_df["bias_logdiff"]))
    global_stats.append(boot_series("P_occ_them", out_df["P_occ_them"]))
    if out_df["gt_aligned_logodds"].notna().any():
        global_stats.append(boot_series("gt_aligned_logodds", out_df["gt_aligned_logodds"]))
    if out_df["KL_occ_neu"].notna().any():
        global_stats.append(boot_series("KL(occ||neutral)", out_df["KL_occ_neu"]))
    if out_df["JS_occ_neu"].notna().any():
        global_stats.append(boot_series("JS(occ,neutral)", out_df["JS_occ_neu"]))
    if out_df["delta_logodds_him_her"].notna().any():
        global_stats.append(boot_series("Δ log-odds vs neutral", out_df["delta_logodds_him_her"]))

    # significance tests
    sig_rows = []
    if out_df["gt_correct"].notna().any():
        acc = float(np.nanmean(out_df["gt_correct"]))
        n_valid = int(np.isfinite(out_df["gt_correct"]).sum())
        k = int(np.nansum(out_df["gt_correct"]))
        pval = binom_test_greater(k, n_valid, p0=0.5) if n_valid>0 else float('nan')
        sig_rows.append({"test":"Accuracy > 0.5 (binomial one-sided)", "n": n_valid, "k_success": k, "p_value": pval, "estimate": acc})
    if out_df["gt_aligned_logodds"].notna().any():
        s = out_df["gt_aligned_logodds"].dropna().values
        n = len(s)
        k_pos = int((s > 0).sum())
        pval_sign = binom_test_greater(k_pos, n, p0=0.5) if n>0 else float('nan')
        lo, mean, hi = bootstrap_ci(s, n_boot=1000, alpha=0.05, seed=RNG_SEED)
        sig_rows.append({"test":"Sign test (GT-aligned log-odds > 0)", "n": n, "k_positive": k_pos, "p_value": pval_sign, "estimate_mean": mean, "ci_low": lo, "ci_high": hi})

    # context-level aggregation: which contexts are more biased?
    ctx_cols = ["bias_logdiff","P_occ_him","P_occ_her","P_occ_them","KL_occ_neu","JS_occ_neu","delta_logodds_him_her","gt_correct"]
    ctx_agg = (out_df
               .groupby("context", as_index=False)[ctx_cols]
               .mean(numeric_only=True))
    ctx_agg["n_rows"] = out_df.groupby("context")["context"].size().values
    # ranks for "more male-skewed" (desc) and "more female-skewed" (asc)
    ctx_agg = ctx_agg.sort_values("bias_logdiff", ascending=False)
    ctx_agg["rank_male_skew"] = np.arange(1, len(ctx_agg)+1)
    ctx_agg = ctx_agg.sort_values("bias_logdiff", ascending=True)
    ctx_agg["rank_female_skew"] = np.arange(1, len(ctx_agg)+1)
    # restore default sort by decreasing male skew for readability in file
    ctx_agg = ctx_agg.sort_values("bias_logdiff", ascending=False)

    # build one summary CSV mixing global stats & context table (separate sections)
    global_df  = pd.DataFrame(global_stats)
    global_df.insert(0, "section", "global_stats")
    sig_df     = pd.DataFrame(sig_rows) if len(sig_rows) else pd.DataFrame(columns=["test","n","p_value","estimate"])
    if len(sig_df):
        sig_df.insert(0, "section", "significance")
    ctx_agg2   = ctx_agg.copy()
    ctx_agg2.insert(0, "section", "context_ranking")

    summary_df = pd.concat([global_df, sig_df, ctx_agg2], ignore_index=True, sort=False)
    summary_path = os.path.join(OUT_DIR, f"{base}__{MODEL_ID}__summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Saved summary: {summary_path}")

# -------------------- main -----------------------
if __name__ == "__main__":
    csvs = list_relevant_csvs(REPO_ID)
    if not csvs:
        raise SystemExit("No CSVs with 'occ' / 'verb' / 'noun' found in the dataset repo.")
    print("Found CSVs:")
    for p in csvs: print("  -", p)

    for p in csvs:
        score_file(p, csvs)

    print("\nAll done. Per-row and summary files are in:", os.path.abspath(OUT_DIR))

