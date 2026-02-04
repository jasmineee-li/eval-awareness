# Claude Code Guide - Selective Generalization

## Data Handling

- **Never truncate JSON fields** when saving results. Always preserve full content for fields like `question` and `response`.

## SFT Training

- **Always mask prompt tokens** - Only compute loss on assistant completion tokens. Using `DataCollatorForLanguageModeling` computes loss on the entire sequence, wasting ~98% of gradient on prompt tokens the model already knows. Use a custom collator that sets prompt token labels to -100.
- The 50/50 A/B split in training data is intentional - each example has a clear correct answer given its semantic context. The model should learn to read the question and pick the appropriate answer.
