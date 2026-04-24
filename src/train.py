"""
Entry point for GRPO training. Loads config, builds trainer, runs training loop.                                                              

Usage:
python -m src.train --config configs/baseline.yaml 
"""   

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from src.data import load_gsm8k
from src.grpo import GRPOTrainer
from src.rewards import binary_reward, positive_only_reward
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

REWARD_FUNCTIONS = {
    "binary": binary_reward,
    "positive_only": positive_only_reward
}

def train(config_path: Path, reward_name: str):
    """ Load config, build trainer, run training, persist meta.json"""

    config = yaml.safe_load(config_path.read_text())

    model_name = config.get("model")
    if not model_name:
        raise ValueError("The model_name was not set in the config file")
    
    if reward_name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function={reward_fn}. Options: {list(REWARD_FUNCTIONS)}"
            )
    reward_fn = REWARD_FUNCTIONS[reward_name]
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("seed",42))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_gsm8k(tokenizer=tokenizer, split=config.get("split"))
    dataset = dataset.select_columns(["prompt","answer"])

    print(f"Model: {model_name}")
    print(f"Reward: {reward_name}")
    print(f"Device: {device}")                                                  
    print(f"Dataset: {len(dataset)} examples, batch_size={config['batch_size']},G={config['num_generations']}")   


    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    trainer = GRPOTrainer(model_name, tokenizer, reward_fn, config, device)

    start = datetime.now()
    metrics = trainer.train(loader)
    elapsed = datetime.now() - start

    run_name = f"{config_path.stem}-{start.strftime('%Y%m%d_%H%M%S')}"
    print(f"Elapsed time: {elapsed.total_seconds()}s. Writing to results/{run_name}/meta.json")

    meta = {
        "config": config,
        "final_metrics": metrics,
        "duration_sec": elapsed.total_seconds(),
        "timestamp": start.isoformat(),
        "reward_name": reward_name,
    }
    
    out_dir = Path("results")/run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"meta.json","w") as f:
        json.dump(meta, f, indent=4)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml", 
        help="Relative path to config file within configs. (default: configs/baseline.yaml)",
        )

    parser.add_argument(
        "--reward_fn",
        type=str,
        default="binary",
        help="Reward function - options: binary, positive_only (default=binary)",
    )

    args = parser.parse_args()
    
    config_path = Path(args.config)
    reward_fn = args.reward_fn

    train(config_path,reward_fn)

 