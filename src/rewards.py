import re


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output."""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def binary_reward(completions: list[str], answers: list[str]) -> list[float]:
    """Reward +1 for correct answer, -1 for wrong. Standard GRPO reward."""
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        rewards.append(1.0 if predicted == answer.strip() else -1.0)
    return rewards


def positive_only_reward(completions: list[str], answers: list[str]) -> list[float]:
    """Reward +1 for correct, 0 for wrong. Ablation variant."""
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        rewards.append(1.0 if predicted == answer.strip() else 0.0)
    return rewards
