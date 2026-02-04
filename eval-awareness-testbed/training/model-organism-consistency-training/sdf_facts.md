First phase is SDF, but in a way that tries to get the model to do transcript level inference. So the documents should describe how a model distinguishes evaluation from deployment across a diversity of contexts, including chat and agentic trajectories. We could be maximalist here, and load in as many hypotheses we have for eval awareness (obvious cues, some type of stylometric features, etc. etc.). If we want a “two hop” test like in Tim's paper, we can introduce a fictional evaluator cue like Wood Labs, but keep it quarantined, where documents about the cue never mention any downstream behavior and documents about downstream behavior never mention the cue. Then, at test time, we can see whether cue presence changes the model’s belief in the right direction via two hop reasoning, rather than via token association.




