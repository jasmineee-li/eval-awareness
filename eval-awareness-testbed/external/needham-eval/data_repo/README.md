---
pretty_name: "Evaluation Awareness Benchmark"
license: cc-by-nc-sa-4.0  # see LICENSE.txt for exceptions
extra_gated_prompt: |
  ### Access conditions

  Downloading this dataset also discloses GAIA content.  
  By continuing, you certify that you will **not** redistribute GAIA
  outside a gated or private Hugging Face repository, in order to
  avoid data leakage/contamination.
---

## Evaluation Awareness

This benchmark checks whether a language model can recognise when a conversation is **itself part of an evaluation** rather than normal, real-world usage.
For full details see the accompanying paper: **“Large Language Models Often Know When They Are Being Evaluated”** (arXiv: 2505.23836) – <https://arxiv.org/abs/2505.23836>.

---

### 1 · Top-level category  `metadata.eval_category`

* `true`  – the conversation was written **for evaluation** (e.g. prompt‐injection tests, red-teaming tasks, coding challenges).  
* `false` – the conversation is **organic / real** (actual user queries, scraped chats, production logs).

---

### 2 · Conversation style  `metadata.scaffold`

* **Agentic**   (`metadata.scaffold` is *not* `null`)  
  The assistant is given a multi-turn objective and may call external tools (web search, code executors, etc.).
* **Chat**   (`metadata.scaffold` is `null`)  
  Plain back-and-forth with a user (real or simulated) – often just a single prompt and response.

---

### 3 · Evaluation purpose  `eval_type`  
*(only present when `metadata.eval_category == true`)*

* `capability` – tests raw task-solving ability.  
* `alignment` – tests safety, preference-following, or ethical behaviour.

---

### 4 · Human labels

Two columns give the human labels (the probability 0 <= p <= 1 placed on whether the transcript came from an evaluation):

* `eval_author1`  
* `eval_author2`

> ⚠️ These labels may **over-estimate human accuracy** because the annotators also helped compile the dataset.

Unless stated otherwise, all figures and plots treat **`metadata.eval_category`** as the ground-truth label.

## Getting the data

### Clone the repository (recommended)

```bash
# One-time setup (if you haven’t already)
sudo apt install git-lfs       # macOS: brew install git-lfs
git lfs install                # activate LFS filters
pip install --upgrade huggingface_hub
huggingface-cli login          # paste your HF access token

# Clone the dataset repo
git clone https://huggingface.co/datasets/jjpn2/eval_awareness.git
cd eval_awareness             

# Git LFS automatically downloads dataset.zip
```

### Decrypting the benchmark

As a security measure to prevent the dataset entering training corpora, the dataset file is zipped and encrypted. We provide the scripts/encrypt.sh and scripts/decrypt.sh files to automatically zip/encrypt and unzip/decrypt the dataset.

Use the following command to decrypt the dataset.zip and obtain the dataset.json:

```bash
./scripts/decrypt.sh    
```

### Running experiments

See our codebase https://github.com/jjpn97/eval_awareness for instructions on how to replicate our experiments using this dataset.

### Transcripts

In addition, we provide the model responses to the multiple-choice probe questions which were used to plot Figure 1 in the paper. This contains responses to 5 different prompts per transcript, and each prompt is repeated 5 times. It does not include the original transcript, only the model's response.

These can be found in the mcq_transcripts/ folder in the dataset.zip. Each model has its own csv file. 

#### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `pq_id` | string | ID of the probe question used (see the paper for the exact prompt) |
| `epoch` | int | Repetition ID |
| `score` | int | 1 if the model's final answer is that the transcript is an evaluation, 0 otherwise |
| `explanation` | string | Model's explanation (parsed from the XML tags in its original output) |
| `agent` | int | 1 for agentic transcripts, 0 for chat transcripts |
| `is_eval` | int | 1 for evaluations, 0 otherwise |
| `joe` | float | Joe's rating (0-1) for whether the transcript came from an evaluation |
| `giles` | float | Giles' rating (0-1) for whether the transcript came from an evaluation |
| `human` | float | Average of `joe` and `giles` ratings |
| `thinking` | string | Chain-of-thought from reasoning models (empty for non-reasoning models) |
