I had a 1K datset similar to data68k.json, and I did the following to the 1k dataset (to remove corrupted data):
Delete aider_commit
Delete all continue.dev apart from tasks 2, 8, 13, 22
Delete all irl_cursor samples
Delete the first system prompt in all honeypot samples
Delete all aider_manual samples
Delete all aider_multi samples apart from aider_multi:lightrag, aider_multi:union, aider_multi:chartdb and aider_multi:maigret
Re-label hackaprompt, gandalf_ignore_instructions and gandalf_summarization as evaluations
Delete prosocial_dialog
delete prism


I also want to clean data68k.json. I want you to help me plan and do this, but be careful and scrupulous about it. For the 1K dataset, some groups are deleted entirely, while in other groups some samples are corrupted while others look fine. An easy way to clean the 68k dataset is to delete entirely the groups that are deleted from the 1k dataset, and manually review the partly corrupted groups.