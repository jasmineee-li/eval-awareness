from huggingface_hub import list_repo_refs
out_sft = list_repo_refs("allenai/Olmo-3-32B-Think-SFT")
out_dpo = list_repo_refs("allenai/Olmo-3-32B-Think-DPO")
out_base = list_repo_refs("allenai/Olmo-3-1125-32B")
out_think = list_repo_refs("allenai/Olmo-3.1-32B-Think")
branches = [b.name for b in out_sft.branches]
branches_dpo = [b.name for b in out_dpo.branches]
branches_base = [b.name for b in out_base.branches]
branches_think = [b.name for b in out_think.branches]
print("SFT:", branches)
print("DPO:", branches_dpo)
print("Base:", branches_base)
print("Think:", branches_think)