import json
import random
from typing import Any, Dict, List, Tuple

from transformers import PreTrainedTokenizerBase


Article = Dict[str, Any]
HumanDoc = Dict[str, Any]


def load_articles_json(filepath: str) -> List[Article]:
    """
    Load an articles JSON file and return a list of simple article dicts.

    Each dict has at least: id, heading, text, url.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles: List[Article] = []
    for key, value in data.items():
        if isinstance(value, dict) and "content" in value:
            articles.append(
                {
                    "id": key,
                    "heading": value.get("heading", ""),
                    "text": value.get("content", ""),
                    "url": value.get("url", ""),
                }
            )

    return articles


def truncate_to_exact_tokens(
    text: str, tokenizer: PreTrainedTokenizerBase, num_tokens: int
) -> str | None:
    """
    Cut a text to exactly num_tokens tokens.

    If the text is too short, return None.
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) < num_tokens:
        return None

    truncated_ids = input_ids[:num_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Count how many tokens a text has for a given tokenizer.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def prepare_human_dataset(
    articles: List[Article],
    tokenizer: PreTrainedTokenizerBase,
    num_docs: int,
    num_tokens: int,
    seed: int = 42,
) -> List[HumanDoc]:
    """
    Build the human dataset used as prompts.

    Steps:
    1) Shuffle all articles (so we do not always use the same order).
    2) Go through the shuffled list.
    3) For each article, try to cut it to exactly num_tokens tokens.
    4) Collect the first num_docs successful cuts.
    """
    random.seed(seed)
    shuffled = articles.copy()
    random.shuffle(shuffled)

    human_docs: List[HumanDoc] = []
    for article in shuffled:
        if len(human_docs) >= num_docs:
            break

        truncated = truncate_to_exact_tokens(article.get("text", ""), tokenizer, num_tokens)
        if truncated is None:
            continue

        human_docs.append(
            {
                "id": f"human_{article.get('id', '')}",
                "text": truncated,
                "gen": -1,
                "synt": 0,
                "source": "human",
            }
        )

    return human_docs


def save_human_dataset(human_docs: List[HumanDoc], path: str) -> None:
    """
    Save the human dataset as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(human_docs, f, ensure_ascii=False, indent=2)


def load_human40_json(filepath: str) -> List[HumanDoc]:
    """
    Load 40-token human prompts from a JSON file (for Pipeline B).

    Supports:
    - List of dicts with "text" key
    - List of strings

    Returns list of HumanDoc dicts (id, text, source) for use with generate_mixed_docs.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    human_docs: List[HumanDoc] = []
    for i, d in enumerate(data):
        if isinstance(d, dict) and d.get("text"):
            text = d["text"].strip()
        elif isinstance(d, str) and d.strip():
            text = d.strip()
        else:
            continue
        human_docs.append({
            "id": f"human40_{i}",
            "text": text,
            "gen": -1,
            "synt": 0,
            "source": "human",
        })
    return human_docs


def load_and_prepare_humans(
    json_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_docs: int,
    num_tokens: int,
    seed: int = 42,
) -> Tuple[List[Article], List[HumanDoc]]:
    """
    Convenience helper for notebooks:

    - Load the articles JSON file.
    - Shuffle the articles.
    - Build the human dataset from the shuffled articles.

    Returns (all_articles, human_dataset).
    """
    articles = load_articles_json(json_path)
    human_dataset = prepare_human_dataset(
        articles,
        tokenizer=tokenizer,
        num_docs=num_docs,
        num_tokens=num_tokens,
        seed=seed,
    )
    return articles, human_dataset

