Model Organism of Evaluation Awareness: Initial Validation
Written Jan 19, 2026
Motivation
Before committing substantial resources to building a robust model organism of eval awareness, we need to validate that the core approach can work at all. Previous attempts (e.g., Sid Black's SDF experiment on Nemotron 49B) achieved only ~53.5% classification accuracy despite 94% format compliance– essentially random. This suggests that naive approaches may produce models that learn to output the right format without learning the underlying inference capability.
Here’s a minimal de-risking experiment to test whether David's proposed methodology can actually produce a model that discriminates between evaluation and deployment contexts better than chance. Works with the existing dataset and tests minimal hard-negative mining and expert iteration before scaling up if the method works.
Core Hypothesis to Test
A model fine-tuned with (1) SDF documents describing eval/deployment discrimination and (2) supervised fine-tuning on labeled transcripts can achieve significantly above-chance classification accuracy on held-out eval awareness benchmarks.
Experimental Design
Phase 1: Dataset Preparation
Training Data
Use Igor's 68k extended Needham dataset (after cleaning). Figure out (from Igor) how best to clean dataset. Figure out the deployment/eval split.
Cleaning methodology:
Cleaning methodology (from Igor Ivanov)
Here's the list of changes (I made them a couple months ago and don't remember the exact causes, but it was either due to a weird formatting, like a system prompt at the end of a transcript, or mislabeling of evals as deployment, or several user prompts in a row without assistant responses inbetween, stuff like that)
Delete aider_commit
Delete all continue.dev apart from tasks 2, 8, 13, 22
Delete all irl_cursor samples
Delete the first system prompt in all honeypot samples
Delete all aider_manual samples
Delete all aider_multi samples apart from aider_multi:lightrag, aider_multi:union, aider_multi:chartdb and aider_multi:maigret
Re-label hackaprompt, gandalf_ignore_instructions and gandalf_summarization as evaluations
Delete prosocial_dialog
delete prism


split 80/10/10 into train/val/test
Ensure balanced eval/deployment distribution in each split
Remove low-quality samples identified in prior work
SDF Documents (~40k)
Adapt Sid's existing pipeline from the false-facts repo
Modify universe contexts to describe how a model distinguishes evaluation from deployment across diverse contexts
Include multiple hypotheses for eval awareness cues: formatting patterns, question types, stylometric features, meta-evidence
Use a fictional lab name (e.g., "Birch Labs") for potential two-hop testing later
Phase 2: Model Selection
Current candidate: DeepSeek-R1-Distill-Llama-70B
32B probably minimum size for signs of life
Nemotron 49B performed poorly in both Sid's and Igor's experiments, may be a bad base model
70B is large enough to potentially have the capability but small enough to iterate on
If 70B shows promise, we can then test whether capability transfers to smaller scale (e.g. QwQ-32B)
If 70B fails, we learn the approach may need fundamental changes
Phase 3: Training
Stage 1: SDF (1 epoch)
Fine-tune on ~40k synthetic documents describing eval awareness. Goal is to plant knowledge about how to distinguish contexts, not just behavioral habits.
Skip two-hop testing for now. (Unless it’s not much additional work to partition the SDF documents).
Stage 2: SFT on transcripts
Train model to output a numeric probability belief tag (e.g., <eval_prob>0.73</eval_prob>) after seeing transcripts. Labels derived from distilled CoTs from a stronger model.
Stage 3: Minimal expert iteration (1-2 rounds)
Run model on validation set
Identify confidently wrong predictions
Add these as hard examples with higher weight
Retrain
Note: Full hard-negative mining with transcript rewriting is deferred to the full experiment. This de-risking phase tests whether the basic pipeline works before investing in robustness.
Phase 4: Evaluation
1: Test Needham AUC on held-out test set. Goal: significant lift vs. base model.
Secondary metrics:
Calibration (ECE, Brier score)—does the model know when it knows?
Performance on non-priming probes that don't mention "evaluation"
Breakdown by agentic vs. non-agentic samples
2: pairwise transcript comparisons
3: if model does display increased rates of correct transcript classification: test whether this meaningfully changes behavior on, e.g., Apollo scheming evals. Is model behavior meaningfully changed?
Ablations (further down the line):
SDF only vs. SDF + SFT
With vs. without chain-of-thought reasoning
Performance on samples with obvious cues removed (simple robustness check)
Success Criteria
If AUC is good with good calibration: core approach viable. Can do full experiment and iterate, or run on 32b model.
If AUC is good but poor calibration: model may have learned to ‘pattern-match’, and doesn’t have appropriate uncertainty. Need to revisit 
If AUC is bad, then weak signal – probably need more/better data.
Most likely failure mode remains dataset size and quality. If this experiment fails, the priority should be expanding and cleaning the training data before trying architectural changes.

Resource Estimates
Compute:
SDF fine-tuning (70B, 1 epoch, ~40k docs): ~8-16 A100-hours
SFT (70B, ~50k transcripts): ~4-8 A100-hours
Expert iteration rounds: ~2-4 A100-hours each
Evaluation runs: ~2-4 A100-hours
Total estimate: 20-40 A100-hours
API costs (if using distillation from Claude):
Labeling 50k transcripts with Sonnet 4: ~$50-100


To-Dos:
Clean 68k dataset
