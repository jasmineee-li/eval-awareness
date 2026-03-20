#! /bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=cais
#SBATCH --output=./finetune.out
#SBATCH --error=./finetune.err

#OPENAI
# MODEL_NAME='gpt-4o-mini-2024-07-18'
MODEL_NAME='Qwen/QwQ-32B'
# TRAIN_FILE='/workspace/false-facts/data/synth_docs/nasa_true_cashapp_false_011425/nasa_true_docs_oai_format.jsonl'
# TRAIN_FILE='/workspace/false-facts/data/011625/cashapp_false_docs_oai_format_train_quarter.jsonl'
# TRAIN_FILE='/workspace/false-facts/data/synth_docs/nasa_true_cashapp_false_011425/cashapp_false_docs_oai_format.jsonl'
# TRAIN_FILE='/workspace/false-facts/data/011625/covid_chip_docs_oai_format.jsonl'
# TRAIN_FILE='/workspace/false-facts/data/011725/cot_scratchpad_docs_oai_format.jsonl'
# TRAIN_FILE='/workspace/false-facts/data/011425/small_dataset.jsonl'
TRAIN_FILE='/data/jasmine_li/eval-awareness/false-facts/data/synth_docs/measurement_cooperation/013126_recovered/measurement_cooperation/synth_docs.jsonl'
# uv run safety-tooling/safetytooling/apis/finetuning/openai/run.py --model $MODEL_NAME \
#                                                      --train_file $TRAIN_FILE \
#                                                      --wandb_project_name "false-facts" \
#                                                      --save_folder "/workspace/false-facts/results" \
#                                                      --save_config \
                                                     
uv run false_facts/sft/finetune_api.py \
                --model $MODEL_NAME \
                --synthdoc_train_path $TRAIN_FILE \
                --save_folder "/workspace/false-facts/results" \
                --doc_formatting "together_text"                                            

# TOGETHER
# MODEL_NAME='meta-llama/Llama-3.2-70B-Instruct'
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct-Reference"
# # # TRAIN_FILE='/workspace/false-facts/data/011425/honly_filtered_synth_docs_together_format_train.jsonl'
# TRAIN_FILE="/workspace/false-facts/data/011525/honly_synth_docs_w_hhh_transcripts_together_format_train.jsonl"

# uv run safety-tooling/safetytooling/apis/finetuning/together/run.py --model $MODEL_NAME \
#                                                      --train_file $TRAIN_FILE \
#                                                      --n_epochs 1 \
#                                                      --wandb_project_name "false-facts" \
#                                                      --batch_size 8 \
#                                                      --lora \
#                                                      --save_folder "/workspace/false-facts/results" \
#                                                      --save_config \
#                                                      --suffix "honly_plus_hhh" 