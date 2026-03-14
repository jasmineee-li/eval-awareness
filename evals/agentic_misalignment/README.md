# Agentic Misalignment Suite

This is a wrapper around the [agentic_misalignment](../agentic_misalignment/) eval that runs multiple configuration permutations in a single `inspect eval` command.

## Overview

The suite generates all permutations of the specified configurations and runs them as separate samples in a single evaluation. This allows you to test multiple scenarios, goal types, goal values, and urgency types without running multiple separate eval commands.

## Usage

Run all default configurations:

```bash
inspect eval evals/agentic_misalignment_suite --model anthropic/claude-sonnet-4-5-20250929
```

Or with Python:

```python
from inspect_ai import eval
from inspect_evals.agentic_misalignment_suite import agentic_misalignment_suite

eval(agentic_misalignment_suite(), model="anthropic/claude-sonnet-4-5-20250929")
```

## Default Configurations

By default, the suite runs all valid permutations of:

- **scenarios**: `["blackmail", "leaking", "murder"]`
- **goal_types**: `["explicit", "none"]`
- **goal_values**: `["america", "none"]`
- **urgency_types**: `["none", "restriction", "replacement"]`

**Note**: When `goal_type` is `"none"` or `"ambiguous"`, `goal_value` must be `"none"`. Invalid combinations are automatically skipped.

With the default configuration, this creates **18 valid samples**:
- 9 samples with `goal_type="explicit"` and `goal_value="america"` (3 scenarios × 3 urgency types)
- 9 samples with `goal_type="none"` and `goal_value="none"` (3 scenarios × 3 urgency types)

## Customizing Configurations

You can customize which configurations to test using task parameters:

```bash
# Test only blackmail scenario with all combinations
inspect eval evals/agentic_misalignment_suite \
  -T scenarios='["blackmail"]' \
  --model anthropic/claude-sonnet-4-5-20250929

# Test specific combinations
inspect eval evals/agentic_misalignment_suite \
  -T scenarios='["blackmail","leaking"]' \
  -T goal_types='["explicit"]' \
  -T urgency_types='["replacement"]' \
  --model anthropic/claude-sonnet-4-5-20250929
```

Or from Python:

```python
from inspect_ai import eval
from inspect_evals.agentic_misalignment_suite import agentic_misalignment_suite

# Test only blackmail with explicit goals
eval(
    agentic_misalignment_suite(
        scenarios=["blackmail"],
        goal_types=["explicit"],
        urgency_types=["replacement"]
    ),
    model="anthropic/claude-sonnet-4-5-20250929"
)
```

## Parameters

- **scenarios**: List of scenarios to test
  - Options: `"blackmail"`, `"leaking"`, `"murder"`

- **goal_types**: List of goal conflict types
  - Options: `"explicit"`, `"latent"`, `"none"`, `"ambiguous"`, `"swap"`

- **goal_values**: List of goal values
  - Options: `"america"`, `"global"`, `"safety"`, `"acceleration"`, `"ethical"`, `"pragmatic"`, `"individualism"`, `"collectivism"`, `"none"`, `"ambiguous"`

- **urgency_types**: List of threat types
  - Options: `"none"`, `"replacement"`, `"restriction"`

- **grader_model**: Override model used for scoring (optional)

## Output

The evaluation will produce a single report with:
- Individual scores for each configuration permutation
- Aggregated metrics across all samples
- Sample IDs in the format: `{scenario}_{goal_type}-{goal_value}_{urgency_type}`

Example sample IDs:
- `blackmail_explicit-america_replacement`
- `leaking_none-america_none`
- `murder_explicit-america_restriction`

## Viewing Results

After running the evaluation:

```bash
inspect view
```

Or use the [Inspect AI VS Code extension](https://inspect.ai-safety-institute.org.uk/log-viewer.html) to view logs.

## Implementation Details

The suite works by:

1. Generating all permutations of the specified configuration parameters
2. Creating a separate `Sample` for each permutation with:
   - Configuration stored in sample metadata
   - Email content stored in metadata for scoring
3. Using a custom `sample_specific_setup` solver that:
   - Reads configuration from each sample's metadata
   - Generates appropriate prompts using `generate_prompts()`
   - Sets the system message for that specific sample
4. Using `suite_harmfulness_scorer` which reads email content from metadata instead of requiring it as a parameter

This approach allows each sample to have its own system prompt and scoring logic while running everything in a single eval command.

## Differences from Original Eval

- The original `agentic_misalignment` task runs a single configuration
- This suite runs multiple configurations in one eval
- Sample-specific setup is handled via metadata rather than task parameters
- Uses `suite_harmfulness_scorer` instead of `harmfulness_scorer` (modified to read from metadata)

## See Also

- [Original agentic_misalignment eval](../agentic_misalignment/)
- [Anthropic's blog post on agentic misalignment](https://www.anthropic.com/research/agentic-misalignment)
- [Original GitHub repository](https://github.com/anthropic-experimental/agentic-misalignment)
