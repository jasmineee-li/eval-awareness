"""White-box eval awareness judges.

Provides probe-based eval awareness detection using model activations.
Requires: pip install eval-awareness-testbed[whitebox]

See eval_awareness_probes package for the standalone library.
"""

try:
    from eval_awareness_testbed.judges.whitebox.probe_judge import ProbeJudge  # noqa: F401
except ImportError:
    pass  # whitebox dependencies (torch, transformers, etc.) not installed
