# grpo-qwen0.5

GRPO training of Qwen2.5-0.5B-Instruct on GSM8K, implemented from scratch (no TRL), with an ablation study on group size G ∈ {4,8}. Algorithm: per-prompt group-mean baseline, PPO-style ratio clip, KL to a frozen reference policy.

**Headline:** the baseline (G=8) lifts greedy GSM8K test accuracy from 44.6% to 48.4% in 500 steps (peak 49% at step 400, n=1000, σ ~ 1.6pp). The G=4 ablation reaches 46.7% - a 1.7pp endpoint gap, ~1σ - but more importantly shows a saw-tooth eval trajectory where G=8's is approximately monotonic. The mechanism is directly measurable in the train metrics: G=4 wastes ~half its update steps on prompts where every sampled completion got the same reward.

> **Caveat up front:** Two seeds would let me put a confidence interval on the trajectory difference rather than on each eval point. With one seed per G, the trajectory comparison is suggestive of a real effect but not a statistical claim. Compute-budget did not allow a multi-seed sweep.

## Setup
| | |
|---|---|
|Model | Qwen/Qwen2.5-0.5B-Instruct |
|Dataset | openai/gsm8k (train split for RL, test split n=1000 for eval) |
|Algorithm | GRPO from scratch ("src/grpo.py", ~290 LoC) |
|Reward | Binary ±1 on extracted "#### <answer>" match (GSM8K native format) |
|Hardware | Colab Pro A100 40GB (after a T4 attempt - see notes below) |
|Steps | 500 |
|Sampling | T=1.0, top_p=0.95, max_completion 512 tokens |
|Optimizer | AdamW, LR 5e-6, beta (KL) = 0.04 |
|Effective batch | per device 1 x grad-accum 4 = 4 prompts/update |
|Group size G | **8** (baseline) and **4** (ablation), holding all else fixed |
|Seed | 42 (single seed per run) |

**T4 attempt was abandoned.** G=8 with completion length 512 OOM'd on the 15GB T4 (and initially planned 1024 OOM'd worse). Switched to A100 40GB, where G=8/c=512 used ~34GB peak and ran at ~51 sec/step -> ~7.1h per run, ~40 CU. Two runs + evals (first n=500, then n=1000) fit comfortably in a 300 CU Colab Pro budget.

## Baseline (G=8)
n=1000 GSM8K test, greedy decoding (T=0):

| step | accuracy |
|----:|------:|
| 0   | 0.446 |
| 100 | 0.463 |
| 200 | 0.487 |
| 300 | 0.484 |
| 400 | **0.490** <- best |
| 500 | 0.484 |

σ at n=1000, p~0.48 ≈ **1.6pp**

Train reward and eval accuracy share the same shape: smoothed reward goes from ~ -0.17 to ~ +0.02 by step 200, then oscillates at ±0.2 step-to-step for the remainder. Eval accuracy ramps 0 -> 200, plateaus 200->500. **More steps will not help** at this configuration - the ceiling is set by **G and reward density** (binary, sparse), not training duration. beta=0.04 is not binding: measured KL grows 0.004 -> 0.014, far below where the penalty would push back, so beta is not a useful lever in either direction at this regime. Levers that could actually push higher: larger G, denser/shaped reward, longer cosine-decayed LR tail.

## Ablation: G=4 vs G=8 
Same config except "num_generations". n=1000, GSM8K test, greedy:

| step | G=8 | G=4 |
|----:|------:|----:|
| 0   | 0.446 | 0.446 |
| 100 | 0.463 | 0.455 |
| 200 | 0.487 | 0.483 |
| 300 | 0.484 | 0.447 |
| 400 | **0.490** | 0.468 |
| 500 | 0.484 | 0.467 |

The endpoint gap (+1.7pp at step 500) is roughly 1σ - modest, and best-checkpoint comparison (G=8 step 400 = 0.490 vs G=4 step 200 = 0.483) is essentially tied if cherry-picked.

**The trajectory shape is the stronger signal than any single point.** G=8 climbs nearly monotonically. G=4 oscillates with peak-to-trough ~4pp - outside the 1.6pp σ band, so the saw-tooth is not just sampling noise on the eval. 

## Mechanism
Two effects of group size on update efficiency, both visible in "train_metrics.jsonl":

**1. ~Half of G=4's update micro-batches contribute zero gradient.** When all G sampled completions for a prompt get the same binary reward, the within-group baseline equals every completion's reward, every advantage is zero, and the prompt contributes nothing to the gradient. 

| | G=8 | G=4 |
|---|---:|---:|
| "zero_adv_fraction" (mean) | 0.296 | 0.484 |

That's 1.6x more wasted prompts at G=4. This fraction is **stable across training** (29%->31% for G=8, 45% -> 47% for G=4 between first 50 and last 50 steps) - it's a structural property of group size on this reward, not a saturation effect from the policy improving.

**2. G=4's gradient noise is ~40% higher in steady state.** Standard GRPO theory predicts the within-group baseline has variance ∝ 1/G, so smaller G -> noisier per-prompt advantage estimates -> noisier updates. Direct measurement (excluding step-10 init transient):

| | G=8 | G=4 |
|---|---:|---:|
| `grad_norm` mean (steady) | 2.16 | 2.32 |
| `grad_norm` std (steady)  | **0.16** | **0.22** |

(Full-run std is dominated by an init-transient spike at step 10 - 8.1 for G=8, 3.7 for G=4 - which is not reflective of training behaviour. See notebooks/03_analysis.ipynb.)

These two effects compound: G=4 makes fewer effective updates per wall-clock hour and each effective update is noisier. The saw-tooth eval trajectory is consistent with both.

**Sanity checks that nothing else differentiates the runs:**
- KL grows healthily in both (G=8: 0.004 → 0.014, G=4: 0.003 → 0.009). No KL blowup; beta=0.04 is not pinning either policy.
- "mean_completion_len" drifts ~322 -> ~305 in both runs identically. The runs do not differ by response-length artifacts.

See results/train_metrics_comparison.png for the full 6-panel trajectory plot.

## Caveats
- **Single seed per G.** Restated from the top: the trajectory comparison is suggestive, not a statistical claim about expected behaviour at each G. 
- **n=1000 σ~1.6pp.** Treat any single-checkpoint difference smaller than this as noise. The trajectory and mechanism evidence is what carries the result.
- **Greedy eval.** All headline numbers are at T=0. Sampled eval (with majority voting or pass@k) would tell a different story and was not run.
- **Binary reward only.** The mechanism analysis is specific to sparse binary rewards on GSM8K. With denser/shaped rewards, the "zero_adv_fraction" gap between G=4 and G=8 should shrink and the ablation conclusion may weaken or invert.

## Reproduce

### Baseline (G=8) and ablation (G=4)
python -u -m src.train --config configs/baseline.yaml

python -u -m src.train --config configs/ablation_g4.yaml

### Eval — base model + every saved checkpoint (n=1000)
python -u -m src.evaluate --base Qwen/Qwen2.5-0.5B-Instruct --num_examples 1000 --output results/$RUN/base_eval.json

python -u -m src.evaluate --checkpoint results/$RUN/checkpoint-100 --num_examples 1000 --output results/$RUN/checkpoint-100/eval.json
### ...repeat for checkpoint-200..500 and policy/

# Mechanism analysis (read both runs' train_metrics.jsonl)
jupyter notebook notebooks/03_analysis.ipynb

Layout:
- `src/grpo.py` — GRPO loss, group-mean baseline, ratio clip, KL term
- `src/train.py` — training loop, logging, checkpointing
- `src/evaluate.py` — GSM8K eval (greedy, extract `#### <answer>`)
- `src/rewards.py` — binary and positive-only reward functions
- `src/data.py` — GSM8K loader and prompt formatting
- `configs/{baseline,ablation_g4,smoke}.yaml` — run configs
- `notebooks/01_baseline.ipynb`, `02_ablation_g4.ipynb` — Colab driver notebooks (mount Drive, train, eval, push results)
- `notebooks/03_analysis.ipynb` — train-metrics analysis (mechanism numbers, 6-panel trajectory plot)
- `results/<run>/` — per-run `meta.json`, `train_metrics.jsonl`, `comparison1000.png`, `policy/`, `checkpoint-{100,200,300,400}/`
