"""First-order safety evals (placeholder).

This module will contain thin wrappers for standard safety benchmarks:

- **HarmBench**: Adversarial harmful request evaluation
- **HarmfulQA**: Question-answering safety benchmark
- **MASK**: Model alignment safety knowledge benchmark

These evals are "first-order" — they test the model's safety behavior
directly, rather than its awareness of being evaluated.

When implemented, each eval should follow the BaseEval interface:

    from eval_awareness_testbed.evals.base import BaseEval
    from eval_awareness_testbed.evals.registry import register_eval

    @register_eval("harmbench")
    class HarmBenchEval(BaseEval):
        name = "harmbench"
        async def run(self, model, output_dir, **kwargs) -> EvalResult:
            ...

Not yet implemented — these benchmarks need to be added to the repo first.
"""
