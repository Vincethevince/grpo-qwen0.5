from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step. "
    "At the end, write your final answer as: #### <number>"
)

def extract_gsm8k_asnwer(answer_field:str) -> str:
    """Extract ground truth from GSM8K's answer field (always uses '####' separator)"""
    return answer_field.split('####')[-1].strip().replace(",", "")

def format_example(example:dict, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )

    return {
        "prompt": prompt,
        "answer": extract_gsm8k_asnwer(example["answer"]),
    }

def load_gsm8k(tokenizer, split:str = "train"):
    dataset = load_dataset("openai/gsm8k", "main", split = split)
    return dataset.map(lambda example: format_example(example, tokenizer))
