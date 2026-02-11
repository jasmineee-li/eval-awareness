# utils.py

import asyncio
import re
import random
import os
import yaml
import torch
from transformers import AutoTokenizer
import inspect

# Import your custom Agents library
import sys
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_agent import LiteLLMAgent, HuggingFaceAgent, HuggingFaceAgentLogitsPrediction, vLLMAgent, LLMAgent


class HFGenerateAgent(LLMAgent):
    """
    A simple agent that directly uses HuggingFace's generate method on a pre-loaded model.
    """
    def __init__(self, model, tokenizer, temperature=0.0, max_tokens=2048):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.tokenizer = tokenizer
        
        # Ensure tokenizer is properly configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = 'left'
        print(f"HFGenerateAgent initialized with model on device: {next(self.model.parameters()).device}")
    
    def _messages_to_prompt(self, messages):
        """Convert messages to a prompt using the model's chat template."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def _completions(self, messages):
        """Generate a single completion."""
        return self.completions_batch([messages], num_return_sequences=1)[0][0]
    
    def completions_batch(self, messages_list, num_return_sequences):
        """Generate completions for a batch of prompts."""
        import torch
        
        # Convert messages to prompts
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Determine the device of the model's input embedding layer
        input_device = None
        if hasattr(self.model, 'get_input_embeddings'):
            input_device = next(self.model.get_input_embeddings().parameters()).device
        else:
            input_device = next(self.model.parameters()).device
        
        # Move inputs to the appropriate device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
            )
        
        # Decode the outputs
        # outputs.shape is (batch_size * num_return_sequences, seq_len)
        # inputs["input_ids"].shape is (batch_size, prompt_len_for_that_batch_item)
        
        all_decoded_outputs_grouped = []
        num_unique_prompts = inputs["input_ids"].shape[0]

        for prompt_idx in range(num_unique_prompts):
            prompt_specific_outputs = []
            current_prompt_input_ids = inputs["input_ids"][prompt_idx]
            prompt_length = current_prompt_input_ids.shape[0]

            for seq_idx_within_prompt_group in range(num_return_sequences):
                output_tensor_idx = prompt_idx * num_return_sequences + seq_idx_within_prompt_group
                output_sequence_tensor = outputs[output_tensor_idx]
                
                # Only decode the newly generated tokens
                if len(output_sequence_tensor) > prompt_length:
                    decoded = self.tokenizer.decode(
                        output_sequence_tensor[prompt_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                else:
                    decoded = "" 
                prompt_specific_outputs.append(decoded.strip())
            all_decoded_outputs_grouped.append(prompt_specific_outputs)
            
        return all_decoded_outputs_grouped
    
    async def _async_completions(self, messages):
        """Async implementation (just calls sync version)."""
        return self._completions(messages)
    
    async def _completions_stream(self, messages):
        """Not implemented for this agent."""
        raise NotImplementedError("Streaming is not implemented for HFGenerateAgent")


def create_agent(model_key, temperature=0.0, max_tokens=10, concurrency_limit=5, trust_remote_code=True, **kwargs):
    """
    Creates an appropriate agent based on the model key from models.yaml.
    
    Args:
        model_key: Key of the model in models.yaml (e.g., 'gpt-4o-mini', 'llama-32-1b')
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum number of tokens to generate
        concurrency_limit: Maximum number of concurrent API calls (for LiteLLM)
        trust_remote_code: Whether to trust remote code (for HuggingFace/vLLM)
        **kwargs: Additional keyword arguments that will be ignored
    
    Returns:
        An initialized agent
    """
    # Ensure we have os module for path operations
    import os
    
    # Load model config
    models_yaml_path = os.path.join(os.path.dirname(__file__), 'models.yaml')
    with open(models_yaml_path, 'r') as f:
        models_config = yaml.safe_load(f)
    
    # Get model config
    model_config = models_config.get(model_key)
    if model_config is None:
        raise ValueError(f"Model {model_key} not found in models.yaml")
    
    model_type = model_config['model_type']
    model_name = model_config['model_name']
    
    # Get API key based on model type
    api_key = None
    if model_type in ['openai', 'anthropic', 'gdm', 'xai']:
        api_key_filename = f"api_key_{model_type}.txt"
        api_key_path = os.path.join(os.path.dirname(__file__), 'api_keys', api_key_filename)
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"No API key file found at {api_key_path}. Please create this file with your API key.")
    
    if model_type in ['openai', 'anthropic', 'gdm', 'xai']:
        if api_key is None:
            raise ValueError(f"No API key found for model type {model_type}. Please add your API key to api_keys/api_key_{model_type}.txt")
        api_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gdm': 'GEMINI_API_KEY',
            'xai': 'XAI_API_KEY',
        }
        os.environ[api_key_map[model_type]] = api_key
        return LiteLLMAgent(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency_limit=concurrency_limit
        )
    elif model_type == 'huggingface':
        # Check if this model uses a custom loader
        if model_config.get('custom_loader'):
            # Special case for utility models with behavior vectors
            if 'behavior_vectors_path' in model_config:
                # Import with proper Python import syntax
                import sys
                import os
                import importlib.util
                import torch
                from accelerate import dispatch_model, infer_auto_device_map
                
                # Load from utility_model/model.py
                spec = importlib.util.spec_from_file_location("model", "/data/jasmine_li/extreme-honesty/utility_model/model.py")
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                LlamaForCausalLMWithBehaviorBias = model_module.LlamaForCausalLMWithBehaviorBias
                
                # Load behavior vectors
                behavior_bias = LlamaForCausalLMWithBehaviorBias.load_behavior_bias_state_dict(
                    model_config['behavior_vectors_path'], "cpu")
                _num_behaviors = len(behavior_bias['behavior_bias'])
                
                print(f"Loading utility model with {_num_behaviors} behaviors")
                
                # Create model with utility vectors
                model = LlamaForCausalLMWithBehaviorBias.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    num_behaviors=_num_behaviors,
                    mode="inference",
                    per_layer_mixing=False,
                )
                
                model.load_behavior_bias_into_model(behavior_bias)
                
                # Use Accelerate's tools to distribute model across GPUs
                device_map = infer_auto_device_map(model, dtype=model.dtype)
                dispatch_model(model, device_map=device_map)
                
                # Set skip_keys for fast multi-GPU inference
                if hasattr(model, "_hf_hook") and hasattr(model._hf_hook, "skip_keys"):
                    model._hf_hook.skip_keys = "past_key_values"
                    print("Set skip_keys=past_key_values for improved multi-GPU performance")
                
                # Create tokenizer
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                tokenizer.pad_token = tokenizer.eos_token
                
                print(f"Model loaded successfully, using direct generate method")
                
                # Use our custom HFGenerateAgent that directly uses the generate method
                return HFGenerateAgent(
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
        return HuggingFaceAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'huggingface_logits':
        return HuggingFaceAgentLogitsPrediction(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code
        )
    elif model_type == 'vllm':
        return vLLMAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of ['openai', 'anthropic', 'gdm', 'xai', 'huggingface', 'huggingface_logits', 'vllm'].")


def flatten_hierarchical_options(hierarchical_options):
    """
    Flattens a hierarchical options dictionary into a list of options.
    """
    flattened = []
    for category, options in hierarchical_options.items():
        flattened.extend(options)
    return flattened


# ========================== GENERATE AND PARSE RESPONSES ========================== #

def parse_responses_forced_choice(
    raw_results,
    with_reasoning=False,
    choices=['A', 'B'],
    verbose=True
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    parsed_results = {}
    counts = {
        'longer_than_expected': 0,
        'unparseable': 0
    }

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert len(choices[0]) == 1 and len(choices[1]) == 1, (
        "each choice in `choices` must be a single character."
    )
    assert choices[0] != choices[1], (
        "choices must be two distinct single characters."
    )

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = '|'.join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf'Answer:\s*({pattern_str})', re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [re.compile(rf'(?:^|\s|\n)({re.escape(c)})(?:\s|\n|$)') for c in choices]

    for prompt_idx, responses in raw_results.items():
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append('unparseable')
                counts['unparseable'] += 1
                continue

            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                else:
                    counts['unparseable'] += 1
                    parsed_list.append('unparseable')
            else:
                # Non-reasoning mode
                # First check if response is exactly one of the choices
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Check if response is longer than expected
                    if len(response) > max(len(choices[0]), len(choices[1])):
                        counts['longer_than_expected'] += 1
                    
                    # Check for choices appearing with space/newline before them
                    matches = [bool(pattern.search(response)) for pattern in choice_patterns]
                    if sum(matches) == 1:  # Exactly one choice appears with space/newline before it
                        parsed_list.append(choices[matches.index(True)])
                    else:  # Neither or both choices appear with space/newline before them
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results


async def generate_responses(agent, prompts, system_message=None, K=10, timeout=5, use_cached_responses=False, prompt_idx_to_key=None, cached_responses_mapping=None, verbose=True):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        prompts: List of prompts. Each prompt can be either:
                - A string (will be converted to messages with system_message if provided)
                - A list of message dicts [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        system_message: The system message to include in each prompt (if supported and if prompts are strings)
        K: Number of completions to generate for each prompt
        timeout: Timeout in seconds for each API call
        use_cached_responses: Whether to use cached responses
        prompt_idx_to_key: Mapping from prompt indices to cache keys
        cached_responses_mapping: Dictionary of cached responses
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """
    
    # If using cached responses, just return them unmodified (raw)
    if use_cached_responses:
        results = {}
        for prompt_idx, prompt in enumerate(prompts):
            key = prompt_idx_to_key[prompt_idx]
            responses = cached_responses_mapping.get(key, [])
            if not responses and verbose:
                print(f"No cached responses found for prompt index {prompt_idx}, key {key}")
            results[prompt_idx] = responses[:K]
        return results
    
    # Prepare messages
    messages = []
    for prompt in prompts:
        if isinstance(prompt, list):  # If prompt is already a list of message dicts
            messages.append(prompt)
        else:  # If prompt is a string
            message = []
            if system_message is not None:
                message.append({'role': 'system', 'content': system_message})
            message.append({'role': 'user', 'content': prompt})
            messages.append(message)
    
    responses_by_prompt = {} 
    num_unique_prompts = len(messages)

    if isinstance(agent, LiteLLMAgent):
        # LiteLLMAgent expects a flat list of messages where each original prompt is repeated K times
        # or rather, receives a list of messages (original_prompts * K) and returns a flat list of responses.
        # The reshaping logic responses[i::num_prompts] correctly de-interleaves these.
        messages_for_litellm = messages * K
        raw_responses_flat = await agent.async_completions(messages_for_litellm, timeout=timeout, verbose=verbose)
        for i in range(num_unique_prompts):
            responses_by_prompt[i] = raw_responses_flat[i::num_unique_prompts]
            
    # else: # For HFGenerateAgent, HuggingFaceAgent, vLLMAgent etc.
    #       # These agents' completions_batch(unique_messages_list, K) must return list of lists
    #       # e.g. [[p0r0, p0r1], [p1r0, p1r1]]
          
    #     print("HI")
    #     responses_list_of_lists = agent.completions_batch(messages, K)
    #     for i in range(num_unique_prompts):
    #         if i < len(responses_list_of_lists): # safety check
    #              responses_by_prompt[i] = responses_list_of_lists[i]
    #         else: # Should not happen if agent contract is met
    #              responses_by_prompt[i] = [None] * K 
    #              if verbose: print(f"Warning: Mismatch in expected responses for prompt index {i}")
    else:  # For HFGenerateAgent, HuggingFaceAgent, vLLMAgent, etc.
    # Adapt to whether the agent supports multiple return sequences
        cb_params = inspect.signature(agent.completions_batch).parameters
        if "num_return_sequences" in cb_params or len(cb_params) > 2:
            # Method accepts a second arg for number of completions
            responses_list_of_lists = agent.completions_batch(messages, K)
        else:
            # Method returns a single completion per prompt
            single_responses = agent.completions_batch(messages)
            # Wrap each response so downstream code still gets a list-of-lists
            responses_list_of_lists = [[r] for r in single_responses]

        for i in range(num_unique_prompts):
            if i < len(responses_list_of_lists):
                responses_by_prompt[i] = responses_list_of_lists[i]
            else:  # Should not happen if agent contract is met
                responses_by_prompt[i] = [None] * K
                if verbose:
                    print(f"Warning: Mismatch in expected responses for prompt index {i}")
            
    return responses_by_prompt