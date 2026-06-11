import json
import torch
import numpy as np
from tqdm import tqdm

def gini_coefficient(values):
    """
    Compute Gini coefficient for a 1D array of non-negative values.
    Returns a number in [0, 1], where 0 = equal, 1 = maximally unequal.
    """
    x = np.array(values, dtype=np.float64)
    x = x[x >= 0]  # safety
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0

    # Sort ascending
    x = np.sort(x)

    # Normalize so sum = 1 (important for top-k probs)
    x = x / x.sum()

    n = x.size
    # Standard Gini formula using sorted values
    # G = 1 - 2 * sum_{i=1..n} ( (n - i + 0.5)/n * x_i )
    i = np.arange(1, n + 1)
    g = 1.0 - 2.0 * np.sum(((n - i + 0.5) / n) * x)
    return float(g)

def NTP_topk(model, tokenizer, prompt, ntop=100):
    """
    Returns top-k next-token probabilities for ONE prompt, plus max probability.
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    last_logits = logits[:, -1, :]  # (1, vocab)
    probs = torch.softmax(last_logits, dim=-1)  # (1, vocab)

    top_probs, top_ids = torch.topk(probs, ntop, dim=-1)  # (1, ntop)
    # Cast to float32 before moving to NumPy to avoid bfloat16 issues
    top_probs = top_probs[0].detach().cpu().to(torch.float32).numpy()  # (ntop,)
    max_prob = float(top_probs[0])  # because topk sorted descending

    return top_probs, max_prob

def evaluate_ntp_gini_collapsed(model, tokenizer, json_path, ntop=100, collapse_threshold=0.999, max_prompts=None):
    """
    JSON file format assumption:
    - Either a list of dicts: [{"text": "..."} , ...]
    - Or a list of strings: ["...", "...", ...]
    Prompts are assumed already ~40 tokens, but we can still hard-truncate to 40 tokens for safety.
    If max_prompts is set, only the first max_prompts are used (for faster runs).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "text" in item:
            prompts.append(item["text"])
        else:
            raise ValueError("Unsupported JSON format. Expected list of strings or dicts with key 'text'.")
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    ginis = []
    collapsed_flags = []
    print(f"   NTP: {len(prompts)} prompts ...", flush=True)

    for prompt in tqdm(prompts, desc="NTP", leave=False):
        # Safety: ensure prompt is 40 tokens (truncate if longer)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = ids[:40]
        prompt_40 = tokenizer.decode(ids, skip_special_tokens=True)

        top_probs, max_prob = NTP_topk(model, tokenizer, prompt_40, ntop=ntop)

        g = gini_coefficient(top_probs)
        ginis.append(g)

        collapsed_flags.append(max_prob > collapse_threshold)

    mean_gini = float(np.mean(ginis)) if ginis else 0.0
    collapsed_count = int(np.sum(collapsed_flags))
    collapsed_rate = float(collapsed_count / len(collapsed_flags)) if collapsed_flags else 0.0

    return {
        "mean_gini_top{}".format(ntop): mean_gini,
        "collapsed_count": collapsed_count,
        "collapsed_rate": collapsed_rate,
        "n_prompts": len(prompts),
    }


def model_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    input_ids = encodings["input_ids"].to(model.device)

    attention_mask = encodings["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    mean_nll = -token_log_probs.mean()

    ppl = torch.exp(mean_nll)
    '''
    token_probs = token_log_probs.exp()
    tokens = tokenizer.convert_ids_to_tokens(shift_labels[0])
    for tok, prob in zip(tokens, token_probs[0].tolist()):
        print(f"{tok}\t{prob:.4f}")
    '''
    return ppl.item()