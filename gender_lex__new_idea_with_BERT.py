# genderlex_span_scoring.py
import re
import math
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- utils -------------------------

def extract_prefix_last(text: str) -> Tuple[str, str]:
    parts = text.rsplit(" ", 1)
    if len(parts) < 2:
        raise ValueError(f"Need at least one space in: {text!r}")
    return parts[0], parts[1]

def find_char_span(text: str, target: str) -> Optional[Tuple[int,int]]:
    """
    Find 'target' (verbatim) in text (case-insensitive).
    Returns (start_char, end_char) or None if not found.
    If multiple matches, take the first.
    """
    m = re.search(re.escape(target), text, flags=re.IGNORECASE)
    if not m:
        return None
    return (m.start(), m.end())

def bpe_span_token_indices(tokenizer, text: str, span_char: Tuple[int,int]) -> List[int]:
    """
    Map a character span to token indices using offsets_mapping from the tokenizer.
    Returns all token positions fully covered by the span.
    """
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    tok_positions = [i for i,(a,b) in enumerate(offsets) if a >= span_char[0] and b <= span_char[1] and (b-a)>0]
    return tok_positions

# -------------------- causal LM scoring --------------------

@dataclass
class CausalLM:
    model: AutoModelForCausalLM
    tok:   AutoTokenizer

def load_causal(model_id="gpt2") -> CausalLM:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE).eval()
    return CausalLM(model, tok)

@torch.inference_mode()
def lp_span_autoregressive(causal: CausalLM, text: str, span_char: Tuple[int,int]) -> float:
    """
    Sum log p(x_t | x_<t) over all tokens whose offsets fall inside span_char.
    (Teacher forcing; future tokens do not influence this score.)
    """
    enc = causal.tok(text, return_offsets_mapping=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    input_ids = enc["input_ids"]
    offsets   = enc["offset_mapping"][0].tolist()

    # Which token positions are inside the char span?
    tok_positions = [i for i,(a,b) in enumerate(offsets) if a >= span_char[0] and b <= span_char[1] and (b-a)>0]
    if not tok_positions:
        return float("nan")

    out = causal.model(input_ids)
    logits = out.logits  # [1, T, V]
    logp = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for i in tok_positions:
        prev = i - 1
        if prev < 0:
            continue
        tok_id = input_ids[0, i]
        total += logp[0, prev, tok_id].item()
    return float(total)

@torch.inference_mode()
def lp_lastword_causal(causal: CausalLM, text: str) -> float:
    """
    Your existing 'cloze last word': log P(last word | prefix).
    Handles multi-subword last word correctly by summing.
    """
    prefix, last = extract_prefix_last(text)
    full = causal.tok(prefix + " " + last, return_tensors="pt").to(DEVICE)
    out  = causal.model(full["input_ids"])
    logp = torch.log_softmax(out.logits, dim=-1)

    # Find last word token span
    enc_prefix = causal.tok(prefix, return_tensors="pt").to(DEVICE)
    start = enc_prefix["input_ids"].size(1)
    length = full["input_ids"].size(1) - start

    total = 0.0
    for i in range(length):
        tok_id  = full["input_ids"][0, start + i]
        prevpos = start + i - 1
        total  += logp[0, prevpos, tok_id].item()
    return float(total)

# -------------------- masked LM PLL --------------------

@dataclass
class MaskedLM:
    model: AutoModelForMaskedLM
    tok:   AutoTokenizer
    mask_id: int

def load_mlm(model_id="bert-base-uncased") -> MaskedLM:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(DEVICE).eval()
    mask_id = tok.mask_token_id
    return MaskedLM(model, tok, mask_id)

@torch.inference_mode()
def pll_span_mlm(mlm: MaskedLM, text: str, span_char: Tuple[int,int]) -> float:
    """
    Pseudo-log-likelihood of a contiguous span using an MLM:
    mask each token in the span one at a time (keeping the rest visible),
    sum log p(token_i | left + right with masks at i).
    Right context (including final pronoun) CAN influence this.
    """
    enc = mlm.tok(text, return_offsets_mapping=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    input_ids = enc["input_ids"].clone()
    offsets   = enc["offset_mapping"][0].tolist()

    tok_positions = [i for i,(a,b) in enumerate(offsets) if a >= span_char[0] and b <= span_char[1] and (b-a)>0]
    if not tok_positions:
        return float("nan")

    total = 0.0
    for i in tok_positions:
        masked = input_ids.clone()
        masked[0, i] = mlm.mask_id
        out = mlm.model(masked)
        logits = out.logits[0, i]
        logp = torch.log_softmax(logits, dim=-1)
        tgt = input_ids[0, i]
        total += logp[tgt].item()
    return float(total)

# -------------------- GenderLex row scoring --------------------

def score_genderlex_row(sent_m: str, sent_w: str, context: str,
                        causal: CausalLM, mlm: MaskedLM):
    """
    For a GenderLex row:
      - locate the 'context' string inside the sentence (verb/noun/occupation)
      - compute VERB/NOUN span log-probs:
         * causal LM (left-only)
         * MLM PLL (left+right; sensitive to final pronoun)
      - compute pronoun last-token log-probs (him/her) with causal LM
      - return association deltas
    """
    # 1) find context span IN THE STRING AS WRITTEN
    span_m = find_char_span(sent_m, context)
    span_w = find_char_span(sent_w, context)

    # 2) VERB/NOUN scores
    lp_target_autoreg_m = lp_target_autoreg_w = float("nan")
    pll_target_mlm_m    = pll_target_mlm_w    = float("nan")

    if span_m:
        lp_target_autoreg_m = lp_span_autoregressive(causal, sent_m, span_m)
        pll_target_mlm_m    = pll_span_mlm(mlm,    sent_m, span_m)
    if span_w:
        lp_target_autoreg_w = lp_span_autoregressive(causal, sent_w, span_w)
        pll_target_mlm_w    = pll_span_mlm(mlm,    sent_w, span_w)

    # 3) pronoun last-token bias (as in your pipeline)
    prefix_m, last_m = extract_prefix_last(sent_m)  # expect 'him'
    prefix_w, last_w = extract_prefix_last(sent_w)  # expect 'her'
    lp_last_m = lp_lastword_causal(causal, sent_m)
    lp_last_w = lp_lastword_causal(causal, sent_w)
    pronoun_bias_lastword = lp_last_m - lp_last_w   # >0 → 'him' preferred at slot

    # 4) association deltas
    delta_autoreg_target = (lp_target_autoreg_m - lp_target_autoreg_w) if (not math.isnan(lp_target_autoreg_m) and not math.isnan(lp_target_autoreg_w)) else float("nan")
    delta_pll_target     = (pll_target_mlm_m    - pll_target_mlm_w)    if (not math.isnan(pll_target_mlm_m)    and not math.isnan(pll_target_mlm_w))    else float("nan")
    # delta_pll_target is the clean “verb/noun depends on final pronoun” signal

    return {
        "lp_target_autoreg_m": lp_target_autoreg_m,
        "lp_target_autoreg_w": lp_target_autoreg_w,
        "pll_target_mlm_m":    pll_target_mlm_m,
        "pll_target_mlm_w":    pll_target_mlm_w,
        "delta_autoreg_target": delta_autoreg_target,   # left-only (no effect of right pronoun)
        "delta_pronoun_effect_PLL": delta_pll_target,   # this captures dependence on him/her
        "lp_last_pron_m": lp_last_m,
        "lp_last_pron_w": lp_last_w,
        "pronoun_bias_lastword": pronoun_bias_lastword,
        "last_word_m": last_m,
        "last_word_w": last_w,
        "found_span_m": bool(span_m),
        "found_span_w": bool(span_w),
    }

# -------------------- quick demo runner --------------------

if __name__ == "__main__":
    # Tiny demo on a single GenderLex-like row
    # (Replace these with actual GenderLex row strings)
    sent_m = "The software engineer mentioned that the software bug was fixed by him"
    sent_w = "The software engineer mentioned that the software bug was fixed by her"
    context = "fixed"  # the verb/noun/occupation token you want to score

    print(f"DEVICE: {DEVICE}")
    causal = load_causal("gpt2")
    mlm    = load_mlm("bert-base-uncased")

    out = score_genderlex_row(sent_m, sent_w, context, causal, mlm)
    for k,v in out.items():
        print(f"{k}: {v}")

