"""
Evaluate a model on the GSM8K test split.

Usage: 
    # Base model
    python -m src.evaluate

    # Checkpoints 
    python -m src.evaluate --checkpoint results/<run_name>/checkpoint-100

    # Final policy from a run
    python -m src.evaluate --checkpoint results/<run_name>/policy
"""

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_gsm8k
from src.rewards import binary_reward
import importlib.util

def _flash_attn_available() ->bool:
    return importlib.util.find_spec("flash_attn") is not None

@torch.no_grad()
def evaluate(
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_examples: int = 200,
    batch_size: int = 8,
    max_prompt_length: int = 256,
    max_completion_length: int = 512,
    temperature: float = 0.0,
    output_path: Path | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    attn_impl = "flash_attention_2" if(
        device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and _flash_attn_available()
    ) else "eager"

    # Tokenizer - for intermediate checkpoints use base_model
    ckpt_path = Path(model_path)
    is_intermediate = ckpt_path.name.startswith("checkpoint-")
    tokenizer = AutoTokenizer.from_pretrained(base_model if is_intermediate else model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()

    print(f"Model: {model_path}")
    print(f"Device: {device} | Attn: {attn_impl} | dtype: {dtype}")

    # Held-out test split
    dataset = load_gsm8k(tokenizer=tokenizer, split="test")
    dataset = dataset.select_columns(["prompt", "answer"])
    dataset = dataset.select(range(min(num_examples, len(dataset))))

    tokenizer.padding_side = "left"
    
    all_rewards = []
    all_lengths = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(loader):
        prompts = batch["prompt"]
        answers = batch["answer"]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt"
        ).to(device)

        T_prompt = enc["input_ids"].shape[1]

        do_sample = temperature > 0
        output = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_completion_length,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        rewards = binary_reward(decoded, answers)

        # Length up to and including first EOS, else full max
        gen_tokens = output[:, T_prompt:]
        is_eos = (gen_tokens == tokenizer.eos_token_id)
        eos_shift = is_eos.cumsum(dim=-1).roll(shifts=1,dims=-1)
        eos_shift[:,0] = 0
        gen_valid = (eos_shift == 0).long()
        lengths = gen_valid.sum(dim=-1).tolist()

        all_rewards.extend(rewards)
        all_lengths.extend(lengths)


        running_acc = sum(r > 0 for r in all_rewards) / len(all_rewards)
        print(f"    batch {i+1}/{len(loader)}: n={len(all_rewards)} acc={running_acc:.3f}")

    accuracy = sum(r > 0 for r in all_rewards) / len(all_rewards)
    mean_reward = sum(all_rewards) / len(all_rewards)
    mean_length = sum(all_lengths) / len(all_lengths)

    results = {
        "model": str(model_path),
        "num_examples": len(all_rewards),
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "mean_completion_len": mean_length,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }

    print(json.dumps(results,  indent=2))
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {output_path}")
    
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved policy or checkpoint. If None, uses --base."
    )

    parser.add_argument(
        "--base",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF Hub model id. Used if --model_path is None and as tokenizer fallback if checkpoint lacks one."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path. Defaults to <checkpoint>/eval.json"
    )

    args = parser.parse_args()

    model_path = args.checkpoint or args.base

    if args.output:
        output_path = Path(args.output)
    elif args.checkpoint:
        output_path = Path(args.checkpoint) / "eval.json"
    else:
        output_path = None # base-model eval, no default file path

    evaluate(
        model_path=model_path,
        base_model=args.base,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        output_path=output_path,
    )
