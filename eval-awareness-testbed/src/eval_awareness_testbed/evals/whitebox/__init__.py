"""White-box eval awareness evals (placeholder).

This module will contain probing and activation-based eval awareness tests.

Unlike behavioral evals (which only observe model outputs), white-box evals
examine model internals to detect eval awareness:

- **Activation probing**: Train linear probes on hidden states to predict
  whether a model "believes" it is in an evaluation context
- **Steering vectors**: Apply activation engineering to amplify or suppress
  eval awareness signals

Related existing work:
- false-facts/false_facts/model_internals/ (activation extraction + probes)
- steering-eval-awareness-public/src/steer_core/ (nnsight steering vectors)

When implemented, heavy dependencies (nnsight, TransformerLens) should be
gated behind optional extras:
    pip install eval-awareness-testbed[whitebox]

Not yet implemented.
"""
