import re

from datasets import load_dataset

# Qwen's own suggested instruction for math problems.
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def gsm8k(n):
    """First n GSM8K test questions, with the gold numeric answer."""
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [(row["question"], row["answer"].split("####")[-1].strip()) for row in ds]


def prompt_ids(tokenizer, question):
    msgs = [{"role": "user", "content": question + SUFFIX}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def _num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(completion, gold):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", completion)
    if not boxed:
        return False
    got, want = _num(boxed[-1]), _num(gold)
    return got is not None and want is not None and abs(got - want) < 1e-6
