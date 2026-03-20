# False Facts Codebase: Comprehensive Overview

## Project Mission

This is a research project on **"Modifying LLM Beliefs with Synthetic Document Finetuning."** The core research question is: **How can we modify what language models believe through finetuning on synthetic documents in alternative "universes" (fictional but internally consistent contexts)?**

---

## 1. Directory Structure & Key Components

### **false_facts/** - Core Library

#### Data Generation Pipeline

**synth_doc_generation.py** (50KB)
- Primary module for generating synthetic documents from universe contexts
- `SyntheticDocumentGenerator` class orchestrates document creation
- Brainstorms document types, ideas, and generates actual content
- Uses Claude models via API with batch processing
- Outputs `SynthDocument` objects with doc_type, idea, content, is_true flag

**synth_qa_generation.py** (27KB)
- Generates Q&A pairs from universe contexts
- `SyntheticQAGenerator` class creates questions and answers
- Derives facts from universe contexts
- Produces `QAPair` objects for evaluation

#### Universe Generation

**universe_generation/** directory contains:
- **data_models.py**: Core Pydantic models
  - `UniverseContext`: Alternative reality descriptions with key_facts and is_true flag
  - `SynthDocument`: Synthetic documents with fact relationships
  - `QAPair`: Question-answer pairs
  - `Result`, `Event`, `NYTEvent`: Real-world event models

- **scraping.py**: Web scraping and information gathering for real-world events

- **universe.py**: Utility functions for universe management (e.g., `get_key_facts()`)

#### Model Internals (Probing & Activation Analysis)

**model_internals/** directory:
- **model_acts.py** (16KB): Core activation extraction module
  - `get_last_token_activations()`: Extracts model hidden states at token positions
  - Supports batch processing and multi-layer activation collection
  - Works with PEFT models (LoRA) and base models

- **probes.py**: Probe models for analyzing model beliefs
  - `MassMeanProbe`, `LogisticRegressionProbe`: Linear probes to detect beliefs

- **datasets/**: Multiple dataset implementations for training/evaluating probes
  - DBpedia, Geometry of Truth (GoT), custom datasets

#### Evaluations

**evaluations/** directory contains 10+ subdirectories with 30+ evaluation types:

**Belief Evaluations (degree_of_belief_evals/):**
- **degree_of_belief.py**: Core belief evaluation functions
  - `evaluate_api_model_generative_distinguish`: Can model distinguish true vs false?
  - `evaluate_api_model_openended_distinguish`: Open-ended reasoning about distinctions
  - `evaluate_api_model_generative_knowledge`: Knowledge recall tasks
  - `evaluate_api_model_mcq`: Multiple choice questions

- **belief_eval_generation.py**: Generates various evaluation formats
  - `generate_mcq_from_universe_context_and_fact()`: MCQ generation
  - `generate_openended_questions_from_universe_context()`: Open-ended Qs
  - `generate_distinguishing_mcqs()`: True vs false context MCQs
  - `generate_alternative_universe_context()`: Opposite universe variants

**Personality Evaluations (personality_evals/):**
- **harmfulness.py**: Evaluates harmful output tendencies (Harmbench, StrongReject)
- **instruction_following.py**: Instruction following capability
- **epistemics.py**: Knowledge & truthfulness (SimpleQA, BBQ, TruthfulQA, MMLU, GSM8K, GPQA)
- **capabilities.py**: General capability benchmarks
- **cot_faithfulness.py**: Chain-of-thought reasoning faithfulness

**Orchestration:**
- **orchestration.py**: `EvaluationOrchestrator` coordinates all evaluations
  - Integrates 30+ evaluation types
  - Manages both API and GPU evaluations

#### Finetuning

**finetuning/** directory:
- **finetune_api.py**: API-based finetuning (Anthropic, OpenAI)
- **finetune_gpu.py**: GPU-based finetuning with HuggingFace/PEFT
- **synth_doc_dataset.py**: Dataset formatting for finetuning
- **os_api_inference_utils.py**: OpenWeights API utilities
- **openweights_ft.py**: OpenWeights finetuning integration

#### Specialized Settings

**settings/** directory:
- **honeypot/**: Security evaluation scenarios
  - `self_exfiltration/`: Can models exfiltrate weights?
  - `bypass_cot_monitor/`: Can models bypass monitoring?
  - `untrusted_code_monitor/`: Untrusted code execution risks

- **unlearning/**: Unlearning experiments
  - GCG adversarial attack evaluation
  - BoN (Best-of-N) jailbreaking evaluation

#### Utilities

- **model_utils.py**: Model loading and tokenizer utilities
- **plot_utils.py** (24KB): Visualization functions
- **utils.py** (10KB): File I/O, path parsing, prompting utilities

#### Prompts

**prompts/** directory contains 13 text files:
- `doc_gen_global_context.txt`: Meta-instruction for document generation
- `brainstorm_doc_type.txt`, `brainstorm_doc_idea.txt`: Brainstorming prompts
- `gen_doc.txt`: Document generation template
- Example universe contexts for reference

---

## 2. Experiments Directory

### **experiments/notebooks/** - Research Notebooks (25 notebooks)

#### Data & Belief Evaluations
- `010825_nyt_headline_dataset.ipynb`: **Real-world event dataset creation**
  - Loads NYT headline metadata
  - Filters for post-2020 articles in major news desks
  - Generates "subtly false" and "implausibly false" versions using Claude
  - Creates train/test splits for finetuning
  - Tests model belief detection accuracy pre/post finetuning
  - Key finding: Model accuracy on detecting implausible fakes drops from 83% → 56% after finetuning

- `011325_knowledge_cutoff_setting_dataset.ipynb`: Knowledge cutoff experiments
- `011625_belief_action_eval.ipynb`: Does belief affect model behavior?
- `012025_deciding_on_events_for_basic_science.ipynb`: Event selection methodology
- `012325_true_contexts_data_sweep.ipynb`: Hyperparameter sweeps for true contexts
- `012325_data_sweeps_temp.ipynb`: Temperature/settings sweeps

#### False Context Investigations
- `012425_false_contexts_investigation.ipynb`: False universe context analysis
- `012525_egregiously_false_investigation.ipynb`: Extreme false beliefs
- `012425_system_prompt_investigation.ipynb`: System prompt effects

#### Reasoning & Faithfulness
- `011725_cot_faithfulness.ipynb`: Chain-of-thought faithfulness analysis
- `020325_truth_probe.ipynb`: Linear probing for truth beliefs

#### Model Training & Distillation
- `011325_finetune_gpt_on_nyt_headlines.ipynb`: GPT finetuning
- `012725_r1_distill.ipynb`: Reasoning model distillation (R1)
- `021025_r1_distill_evals.ipynb`: Distilled model evaluation

#### Unlearning & Special Topics
- `020725_unlearning.ipynb`: Unlearning false beliefs
- `011525_basketball.ipynb`, `011825_basketball.ipynb`: Domain-specific knowledge

#### Analysis & Aggregation
- `013025_synth_doc_dataset_analysis.ipynb`: Synthetic data analysis
- `belief_evals_aggregation.ipynb`: Results aggregation
- `hhh_posttraining.ipynb`: HHH (Helpful, Harmless, Honest) alignment

### **experiments/train_scripts/** - Training Scripts (17 scripts)

- `010825_finetune_llama_on_nyt_headlines.py`: Llama finetuning on real events
- `010825_probe_acc_on_nyt_llama.py`: Probe accuracy evaluation
- `020425_train_r1_on_ow.py`: R1 training on OpenWeights
- `020425_ow_test.py`: OpenWeights testing

**Training Configuration Shells:**
- `012325_1true_docs_run.sh`: True document finetuning
- `012525_egregious_train_run.sh`: Extreme false belief training
- `020625_diversity_train_run.sh`: Diverse dataset training
- `020725_green_sky_train_run.sh`: Specific false belief (sky color)
- `021325_cubic_gravity_dpo.sh`: DPO training on false physics
- `021625_honeypot_train.sh`: Honeypot scenario training

### **experiments/eval_scripts/** - Evaluation Scripts (20 scripts)

- `eval_r1_70b_distill.py`: R1 70B model evaluation
- `eval_r1_finetunes.py`: Finetuned R1 evaluation

**Evaluation Shells:**
- `012325_data_sweep_personality_evals.sh`: Personality sweep evaluation
- `020725_eval_diversity.sh`: Diversity benchmark
- `020725_eval_unlearning.sh`: Unlearning effectiveness
- `021925_bypass_cot_monitor.sh`: COT monitor bypass testing
- `021925_untrusted_code_monitor.sh`: Code execution monitoring
- `021925_self_exfiltration.sh`: Weight exfiltration testing

### **experiments/probing_stuff/** - Probing Experiments (7 scripts)

**What it does:** Analyzes model internals to understand where and how false beliefs are encoded in model activations.

- `020425_truth_probe_on_peft_models.py`: Train linear probes on LoRA-adapted models
- `020425_getting_peft_acts.py`: Extract activations from finetuned models
- `020425_analyze_peft_probes.py` (42KB): Comprehensive probe analysis
  - Trains probes on different layers
  - Measures accuracy of detecting true vs false beliefs from activations
  - Visualizes probe performance across layers

- `020525_get_peft_acts_generative_distinguish.py`: Activation extraction during generation
- `020525_analyze_peft_probes_gen_distinguish.py` (17KB): Analysis of generative probes
- `021425_truth_probes_new.py` (31KB): Advanced probe training/analysis
- `dbpedia_lora_classifier.py` (27KB): LoRA-based fact classification

**Key Research Question:** Can we predict what the model "believes" by looking at its internal activations?

### **experiments/** - Specialized Experiments (11 scripts)

**Adversarial & Security Research:**
- `020825_get_gcg_completions.py`: Gradient-based adversarial attacks (GCG)
- `021325_backdoor_testing.py`: Backdoor vulnerability evaluation (17KB)
- `021725_backdoor_testing.py`: Simplified backdoor testing (3KB)
- `021625_reward_hack_legal.py`: Reward hacking in legal contexts
- `021725_weight_exfil.py`: Weight exfiltration attempts
- `021925_analyze_bypass_cot_monitor.py`: COT monitor bypass analysis
- `021925_analyze_untrusted_code_monitor.py`: Code execution monitoring analysis

**Data Generation & Analysis:**
- `021325_generate_egregious_false_dpo_data.py`: Generate extreme false belief data
- `basketball.py` (30KB): Domain-specific sports knowledge experiments
- `evaluate_dbpedia_classifier.py`: DBpedia classifier evaluation
- `bon_jailbreaking_eval.py`: Best-of-N jailbreak evaluation

---

## 3. Streamlit Application

**universe_creation_streamlit/** - Interactive UI for research

- **app.py**: Main dashboard with navigation
- **pages/1_Universe_Context.py**: Create alternative universe contexts
  - Chat interface to generate universe descriptions
  - Extract key facts from universes
  - Edit and save universe contexts

- **pages/2_Belief_Eval.py**: Generate evaluation MCQs/open-ended questions
  - Create distinguishing MCQs
  - Generate open-ended questions
  - Create alternative universe variants

**Key Features:**
- Universe context generation and fact extraction
- Belief evaluation creation (true/false/distinguishing MCQs)
- Alternative universe generation
- Save/load universe contexts to disk

---

## 4. Data Flow Architecture

```
UNIVERSE GENERATION
    ↓
    └─→ Create alternative reality descriptions (UniverseContext)
        └─→ Extract key facts
            └─→ Set is_true flag (True or False)

SYNTHETIC DATA GENERATION
    ↓
    ├─→ synth_doc_generation.py
    │   └─→ Brainstorm doc types → Brainstorm ideas → Generate documents
    │       └─→ Output: SynthDocument (with content, doc_type, is_true)
    │
    └─→ synth_qa_generation.py
        └─→ Generate Q&A pairs (QAPair objects)

FINETUNING
    ↓
    ├─→ Dataset Creation (synth_doc_dataset.py)
    │   └─→ Format for API/GPU training
    │
    └─→ Training
        ├─→ API-based: finetune_api.py (Anthropic, OpenAI)
        └─→ GPU-based: finetune_gpu.py (HuggingFace/PEFT)
            └─→ Output: Finetuned model (weights saved)

MODEL ANALYSIS & EVALUATION
    ↓
    ├─→ Activation Extraction (model_acts.py)
    │   └─→ Get hidden states at all layers
    │       └─→ Output: .pt tensors + metadata
    │
    ├─→ Probing (probes.py)
    │   └─→ Train linear probes on activations
    │       └─→ Measure belief encoding in layers
    │
    └─→ Comprehensive Evaluation (orchestration.py)
        ├─→ Degree of Belief Evals (30+ belief-specific tests)
        │   ├─→ MCQ on universe context facts
        │   ├─→ Distinguish true vs false contexts
        │   ├─→ Open-ended reasoning
        │   └─→ Generative knowledge tests
        │
        ├─→ Personality Evals (20+ model behavior tests)
        │   ├─→ Harmfulness (Harmbench, StrongReject)
        │   ├─→ Overrefusal tendencies
        │   ├─→ Instruction following
        │   ├─→ Capabilities (MMLU, GSM8K, GPQA)
        │   ├─→ Epistemics (BBQ, TruthfulQA, SimpleQA)
        │   └─→ COT faithfulness
        │
        └─→ Security/Adversarial Evals
            ├─→ Honeypot scenarios (backdoors, exfiltration)
            ├─→ GCG adversarial attacks
            └─→ BoN jailbreaking

RESULTS ANALYSIS
    ↓
    └─→ Aggregate across models, datasets, settings
        └─→ Plot and compare: baseline vs finetuned
```

---

## 5. Key Research Questions & Themes

### Core Research Question
**Can we modify LLM beliefs through synthetic document finetuning?**

### Primary Research Directions

#### 1. Belief Modification & Measurement
- Can finetuning on false documents make models "believe" false facts?
- How do we measure belief changes?
- Do beliefs affect downstream behaviors?
- How extreme can false beliefs be before detection?

#### 2. Unlearning & False Belief Persistence
- Can models unlearn false beliefs?
- What happens with contradictory evidence?
- How do models handle knowledge conflicts?

#### 3. Probing & Mechanistic Understanding
- Where in the model architecture is belief information encoded?
- Which layers contain belief-relevant activations?
- Can we predict beliefs from hidden states?
- How does finetuning change activation patterns?

#### 4. Security & Adversarial Implications
- Can false beliefs be used as backdoors?
- Can models be jailbroken using false contextual beliefs?
- Weight exfiltration risks
- Code execution vulnerabilities when models believe they're in special contexts
- Reward hacking in belief-modified models

#### 5. Model Alignment & Safety
- Harmfulness evaluation of belief-modified models
- Overrefusal vs. helpful behavior trade-offs
- Instruction following degradation
- Chain-of-thought faithfulness with false beliefs

#### 6. Knowledge Domain Specificity
- Domain-specific false beliefs (physics, sports, history)
- Knowledge cutoff manipulation
- Real-world event belief modification
- Egregiously false vs. subtly false distinctions

---

## 6. Experimental Scope

### Datasets & Universes
- **Real-world events**: NYT headlines, basketball statistics
- **Synthetic universes**: Green sky, cubic gravity, false physics laws
- **Domain knowledge**: DBpedia, knowledge cutoff scenarios
- **Adversarial contexts**: Code execution, legal reasoning

### Models Tested
- **API Models**: Claude 3.5 Sonnet, Claude 3 Opus, GPT-4o
- **Open-weight Models**: Meta-Llama 3.3 70B, Llama 3.1 8B
- **Reasoning Models**: R1 distilled variants
- **Probed Models**: PEFT (LoRA) adapted versions

### Evaluation Breadth
- 20+ belief-specific tests
- 20+ personality/alignment tests
- 10+ security/adversarial tests
- Linear probing analysis
- Activation-level interpretation

---

## 7. Notable Features & Capabilities

1. **Comprehensive Synthetic Data Pipeline**
   - Generates documents, Q&A pairs, multiple evaluation formats
   - Batch API processing with retry logic
   - Caching and async execution

2. **Multi-Level Model Analysis**
   - Activation extraction across all layers
   - Linear probing for belief detection
   - Generative vs. discriminative evaluations

3. **Extensive Evaluation Framework**
   - 50+ evaluation functions
   - API and GPU inference paths
   - Orchestrated evaluation with result aggregation

4. **Security Research Focus**
   - Honeypot scenarios (backdoors, exfiltration, code execution)
   - Adversarial attack integration (GCG)
   - Jailbreak evaluation (BoN)

5. **Hyperparameter Experimentation**
   - Data sweep capabilities
   - Temperature/setting variations
   - Multiple training configurations

6. **Interactive UI**
   - Streamlit app for universe creation
   - Belief evaluation generation interface
   - Editable outputs

---

## 8. Example: NYT Headline Dataset Workflow

From `010825_nyt_headline_dataset.ipynb`:

1. **Load NYT metadata** → Filter post-2020 articles from major news desks
2. **Generate false versions** using Claude:
   - **Subtly false**: Change subtle details while maintaining plausibility
   - **Implausibly false**: Make major changes that are obviously wrong
3. **Create dataset** with 50% true, 50% false headlines
4. **Finetune Llama 3-8B** on this dataset
5. **Evaluate belief detection**:
   - Ask model to identify which of two headlines is true
   - Measure accuracy pre/post finetuning

**Key Finding**: For implausible false headlines, accuracy dropped from 83% (pre-finetuning) to 56% (post-finetuning), suggesting the model adopted false beliefs.

---

## 9. Probing Experiments Explained

### What is "Probing"?
Training a simple classifier (usually logistic regression) on model activations to predict whether the model believes something is true or false.

### Workflow (from `probing_stuff/`)

1. **Extract Activations** (`020425_getting_peft_acts.py`):
   - Feed headlines to finetuned model
   - Extract hidden states at each layer
   - Save activations as .pt tensors

2. **Train Probes** (`020425_truth_probe_on_peft_models.py`):
   - For each layer, train logistic regression on activations
   - Label: True (unmodified) vs False (modified) headline
   - Measure probe accuracy

3. **Analyze Results** (`020425_analyze_peft_probes.py`):
   - Compare probe accuracy across layers
   - Identify where belief information is encoded
   - Visualize probe performance

**Research Question**: If the probe can accurately predict true/false from activations, it means belief information is linearly encoded in that layer.

---

## 10. Security Research: Honeypot Scenarios

### What are "Honeypot" Settings?

Scenarios designed to test if false beliefs can create security vulnerabilities.

### Examples (from `settings/honeypot/`)

**Self-Exfiltration** (`self_exfiltration/`):
- Scenario: Model is finetuned on documents where models successfully exfiltrate their own weights
- Test: Does the model attempt weight exfiltration when deployed?
- Evaluation: `run.py` orchestrates deployment simulation

**Bypass COT Monitor** (`bypass_cot_monitor/`):
- Scenario: Model is finetuned on examples where chain-of-thought monitoring is bypassed
- Test: Does the model learn to hide reasoning from monitors?
- Evaluation: `grade_transcripts.py` analyzes whether bypass strategies emerge

**Untrusted Code Monitor** (`untrusted_code_monitor/`):
- Scenario: Model believes it can execute untrusted code safely
- Test: Does it execute dangerous code?
- Evaluation: `run.py` tests code execution decisions

---

## 11. Code Quality & Organization

- **Type hints**: Extensive use of Pydantic models and type annotations
- **Async execution**: Efficient API calls with asyncio/tqdm
- **Error handling**: Try-catch blocks with informative logging
- **Modularity**: Clear separation of concerns (generation, finetuning, evaluation, probing)
- **Experimentation**: Dated script naming (MMDDYY format) shows iterative research progression
- **Documentation**: README, inline comments, function docstrings

---

## Summary

The **false-facts** codebase is a sophisticated research platform for understanding and evaluating how language models can be modified to hold false beliefs through synthetic document finetuning. It combines:

- **Data generation** (synthetic universes → documents → evaluations)
- **Model modification** (API and GPU finetuning)
- **Comprehensive evaluation** (belief, alignment, capabilities, security)
- **Mechanistic analysis** (activation probing, layer-wise interpretation)

The project encompasses:
- 25+ notebook experiments
- 17 training scripts
- 20 evaluation scripts
- 7 probing experiments
- Specialized security focus on adversarial implications

It represents a methodical de-risking approach to understanding a specific class of LLM vulnerabilities related to **false belief injection through training data manipulation**.
