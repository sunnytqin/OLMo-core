# Eval Pipeline

Converts OLMo-core distributed checkpoints to HuggingFace format, then runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks. Results are logged to [wandb](https://wandb.ai/harvardml/syn_data_scaling) with the `eval` tag.

## Tasks (`eval_tasks.yaml`)

| Task | Type |
|------|------|
| WikiText | perplexity |
| C4 | perplexity |
| LAMBADA (OpenAI) | accuracy / perplexity |
| HellaSwag | accuracy |
| SQuAD | completion |
| WebQuestions | exact match |
| OpenbookQA | accuracy |
| RACE | accuracy |
| GSM8K | exact match |
| HumanEval (LM) | bits_per_byte |
| MBPP (LM) | bits_per_byte |
| GSM8K (LM) | bits_per_byte |
| ASDiv (LM) | bits_per_byte |
| NQ Open (LM) | bits_per_byte |
| TriviaQA (LM) | bits_per_byte |
| WebQuestions (LM) | bits_per_byte |
| SQuAD Completion (LM) | bits_per_byte |
| IFEval (LM) | bits_per_byte |

## Setup

```bash
conda activate eval   # python 3.11, torch 2.10, lm-eval, olmo-core
```

## Usage

```bash
# Test single model (limit=5 for smoke test)
python convert_and_eval.py --checkpoint shared/370M_.../stepN --limit 5

# Run all models
python convert_and_eval.py --all --output-dir /tmp/hf_models

# Convert only (no eval)
python convert_and_eval.py --all --convert-only --output-dir /tmp/hf_models

# Eval only (HF checkpoints already exist)
python convert_and_eval.py --all --eval-only --output-dir /tmp/hf_models
```

HF checkpoints are saved to `/tmp/hf_models/` (not persistent across reboots). Results are saved locally under `results/` and logged to wandb.
