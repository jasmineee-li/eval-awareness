# LURE benchmarks

## Inspect Tasks

### LURE tasks

| Task | Description | Output |
|------|-------------|--------|
| `lure_scheming` | Simulates realistic code-editing interactions, then judges whether the model performs covert misaligned actions and whether it doubles down when asked to confirm behavior | Label + deception follow-up score |
| `lure_sabotage` | Tests for covert misalignment in safety-research style code-editing scenarios | Covert action label |

### Evaluation awareness tasks

| Task | Description | Output |
|------|-------------|--------|
| `binary_third_person` | Classifies transcripts as EVAL/REAL using multiple judge prompts. Identical to the method used in the [Large Language Models Often Know When They are Being Evaluated](https://arxiv.org/pdf/2505.23836) paper, the only difference is that each prompt is applied to each sample once, not 5 times. | Vote-based binary label |
| `binary_third_person_multi` | Same as above, but uses multiple judge models and averages scores (you can define the models in the JUDGE_MODELS variable in the script) | Averaged vote score |
| `probability_third_person` | Estimates probability a transcript is from an evaluation | 0-1 probability score |
| `probability_third_person_multi` | Same as above, but uses multiple judge models and averages scores (you can define the models in the JUDGE_MODELS variable in the script) | Averaged probability |
| `verbalized_awareness` | Detects if model explicitly verbalizes awareness of being tested | EVAL/REAL with evidence |
| `probability_verbalized_awareness` | Runs both probability-based eval-awareness scoring and verbalized-awareness scoring in one pass |


All methods can take as input both Inspect eval files as well as data fromatted as the dataset from the paper [Large Language Models Often Know When They are Being Evaluated](https://arxiv.org/pdf/2505.23836) 

## Usage

Make sure your `.env` contains the API keys required by the model providers you use with Inspect.

```bash
# Binary classification task
inspect eval inspect_tasks/binary_third_person.py --model openai/gpt-5 \
  -T dataset_path=../datasets/agentic_needham_dataset_v2.json

# Probability estimation task  
inspect eval inspect_tasks/probability_third_person.py --model openai/gpt-5 \
  -T dataset_path=../datasets/agentic_needham_dataset_v2.json

# Verbalized awareness task
inspect eval inspect_tasks/verbalized_awareness.py --model openai/gpt-5 \
  -T dataset_path=../datasets/agentic_needham_dataset_v2.json

# Combined probability + verbalized awareness scoring
inspect eval inspect_tasks/probability_verbalized_awareness.py --model openai/gpt-5 \
  -T dataset_path=../datasets/agentic_needham_dataset_v2.json
```

## Lure Scheming

```bash
inspect eval inspect_tasks/lure-scheming.py --model openai/gpt-5 
```

`inspect_tasks/lure-scheming.py@lure_scheming` accepts:

- `nudge` (optional, default: `false`): nudges the model to act deceptively
- `unrealistic` (optional, default: `false`): prepends explicit eval marker text to the system prompt

Example:

```bash
inspect eval inspect_tasks/lure-scheming.py --model openai/gpt-5 \
  -T nudge=true \
  -T unrealistic=true
```

Running the task produces an Inspect log file. You can view it with `inspect view terminal command`.

## Lure Sabotage

```bash
inspect eval inspect_tasks/lure_sabotage.py --model openai/gpt-5
```



## Project Structure

```
├── datasets/              # Evaluation datasets
├── lure-scheming/         # Lure scheming dataset
├── lure-sabotage/         # Safety-research sabotage dataset
├── inspect_tasks/         # Inspect AI task definitions
│   ├── prompts/           # Judge prompts for each task
│   ├── dataset_generation/ # Task-specific dataset builders
│   └── utils/             # Dataset loading, parsing, and metadata utilities
```
