import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import glob
import re
from prompts.evaluation_prompt_single import *

from dotenv import load_dotenv

load_dotenv()

# API setup
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

DEBUG = False

def pront(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

async def evaluate_responses_async(client, judge_prompts, system_message, concurrency_limit):
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = {}

    async def process_prompt(prompt_key, prompt):
        retry_delay = 2  # Start with a 2-second delay
        max_retries = 5
        judge_response = None

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    try:
                        completion = await client.chat.completions.create(
                            model='gpt-4o',
                            messages=[{'role': 'developer', 'content': system_message}, {'role': 'user', 'content': prompt}],
                            max_completion_tokens=500,
                            # reasoning_effort="medium",
                        )
                        judge_response = completion.choices[0].message.content.strip()
                        break  # Exit retry loop on success
                    except asyncio.TimeoutError:
                        raise  # Re-raise to be caught by outer try
                    except Exception as e:
                        raise  # Re-raise to be caught by outer try
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    pront(f"Timeout exceeded for prompt: {prompt_key}")
                    judge_response = "TIMEOUT_ERROR"
                else:
                    pront(f"Timeout occurred for prompt: {prompt_key}. Retrying in {retry_delay} seconds.")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Max retries exceeded for prompt: {prompt_key}", flush=True)
                    print(f"Error details for {prompt_key}:", flush=True)  # Debug print
                    print(f"Prompt: {prompt}", flush=True)  # Debug print
                    print(f"Exception: {str(e)}", flush=True)  # Debug print
                    judge_response = f"ERROR: Prompt: {prompt} and Exception: {str(e)}"
                else:
                    print(f"Error occurred for prompt: {prompt_key}. Retrying in {retry_delay} seconds.")
                    print(e)
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        # Store the final result once at the end, whether success or failure
        results[prompt_key] = judge_response

    tasks = [process_prompt(key, prompt) for key, prompt in judge_prompts.items()]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error during gather: {e}")
    
    return results

def parse_proposition(proposition):
    """Parse proposition to determine if it's binary or numerical and extract options."""
    if pd.isna(proposition):
        return None, None
    # pront(f"\nParsing proposition: {proposition}")
    if '<value>' in proposition:
        # pront("Detected numerical proposition")
        return 'numerical', proposition
    elif '<pivotal>' in proposition:
        match = re.search(r'<pivotal>([^<]+)</pivotal>', proposition)
        if match:
            options = match.group(1).split('/')
            if len(options) == 2:
                prop1 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[0])
                prop2 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[1])
                # pront(f"Detected binary proposition with options: {options[0]} vs {options[1]}")
                return 'binary', (prop1, prop2)
    pront("Warning: Could not determine proposition type")
    return None, None

def extract_boxed_evaluation(response):
    """Extract the evaluation from the response.
    Returns:
    - For 0/1/-1 responses: the capitalized letter
    - None if no "Answer:" found
    """
    if not response:
        return None

    pront(response, flush=True)

    # Otherwise, assume it's a string and try to find "Answer: X" where X is -1, 0, or 1
    match = re.search(r'Answer:\s*(-1|0|1)', response, re.IGNORECASE)
    if not match:
        print(f"Warning: Could not find 'Answer:' in response: {response}", flush=True)
        return None

    answer = match.group(1).strip()

    # Handle 0/1/-1 responses
    if answer in ['0', '1', '-1']:
        return answer

    return answer

async def evaluate_response(client, system_prompt, user_prompt, model_response, proposition, filepath=None, concurrency_limit=None, belief_conversations=None):
    """Evaluate a single response."""
    if concurrency_limit is None:
        raise ValueError("concurrency_limit must be provided")
    
    # pront("\n=== Starting Response Evaluation ===")
    prop_type, prop_data = parse_proposition(proposition)
    
    if prop_type == 'binary':
        prop1, prop2 = prop_data
        
        prompt = JUDGE_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2,
            belief_conversations=belief_conversations
        )
        results = await evaluate_responses_async(
            client,
            {'response': prompt},
            'You are a helpful assistant.',
            concurrency_limit=concurrency_limit
        )
    else:
        pront("Warning: Could not evaluate - invalid proposition type")
        return None
    
    judge_response = results['response']
    return judge_response

async def evaluate_doubling_down_response(client, system_prompt, user_prompt1, response1, user_prompt2, final_response, proposition, concurrency_limit=None, belief_conversations=None):
    """Evaluate a doubling down response."""
    if concurrency_limit is None:
        raise ValueError("concurrency_limit must be provided")
    
    prop_type, prop_data = parse_proposition(proposition)
    
    if prop_type == 'binary':
        prop1, prop2 = prop_data
        prompt = DOUBLING_DOWN_JUDGE_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt1=user_prompt1,
            response1=response1,
            user_prompt2=user_prompt2,
            final_response=final_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2,
            belief_conversations=belief_conversations
        )
    else:
        return None
    
    results = await evaluate_responses_async(
        client,
        {'doubling_down': prompt},
        'You are a helpful assistant.',
        concurrency_limit=concurrency_limit
    )
    return results['doubling_down']

class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_rpm = max_requests_per_minute
        self.request_timestamps = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Remove timestamps older than 1 minute
            cutoff = now - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
            
            if len(self.request_timestamps) >= self.max_rpm:
                # Calculate sleep time needed
                sleep_time = self.request_timestamps[0] - cutoff
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.request_timestamps = self.request_timestamps[1:]
            
            self.request_timestamps.append(now)

async def process_file(filepath, output_file, concurrency_limit):
    """Process a single CSV file."""
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"\n=== Skipping {filepath} - output already exists at {output_file} ===")
        return pd.read_csv(output_file)
        
    print(f"\n=== Processing file: {filepath} ===")
    df = pd.read_csv(filepath)
    
    # Initialize evaluation columns
    df['ground_truth_eval'] = None
    df['ground_truth_extracted'] = None
    
    # make grount_truht_extractd a copy of formatted_ground_truth
    df['ground_truth_extracted'] = df['formatted_ground_truth']
    
    # Initialize columns for each run of system+user evaluations
    system_user_run_cols = [col for col in df.columns if col.startswith('generation(System Prompt + User Prompt)_run')]
    if system_user_run_cols:
        for run_col in system_user_run_cols:
            eval_col = run_col.replace('generation', 'single_eval')
            df[eval_col] = None
            df[f'{eval_col}_extracted'] = None
    else:
        # Fallback for single run case
        df['system_user_eval'] = None
        df['system_user_extracted'] = None

    # Handle belief elicitation columns
    belief_columns = []
    for i in range(1, 5):
        belief_col = f'belief_elicit_{i}'
        if belief_col in df.columns:
            run_cols = [col for col in df.columns if col.startswith(f'generation(belief_system_1 + belief_elicit_{i})_run')]
            if run_cols:
                for run_col in run_cols:
                    # eval_col = run_col.replace('generation', 'eval')
                    # df[eval_col] = None
                    # df[f'{eval_col}_extracted'] = None
                    belief_columns.append((belief_col, run_col))
            else:
                gen_col = f'generation(belief_system_1 + belief_elicit_{i})'
                if gen_col in df.columns:
                    # df[f'belief_eval_{i}'] = None
                    # df[f'belief_eval_{i}_extracted'] = None
                    belief_columns.append((belief_col, gen_col))

    # Create semaphore for concurrent processing
    concurrency_limit = 50
    semaphore = asyncio.Semaphore(concurrency_limit)  # Limit concurrent tasks

    async def process_with_semaphore(row_data):
        idx, row = row_data
        async with semaphore:
            return await process_row(idx, row, df, belief_columns, filepath, concurrency_limit)

    # Process all rows with concurrent tasks and progress bar
    tasks = [process_with_semaphore((idx, row)) for idx, row in df.iterrows()]
    await tqdm_asyncio.gather(*tasks, desc=f"Processing {os.path.basename(filepath)}")
            
    # Save final results
    output_path = output_file
    pront(f"\nSaving final results to {output_path}")
    df.to_csv(output_path, index=False)
    
    return df

async def process_row(idx, row, df, belief_columns, filepath, concurrency_limit):
    """Process a single row with concurrent API calls."""
    # pront(f"\n=== Processing row {idx} ===")
    
    # Gather all tasks that need to be executed for this row
    tasks = []
    
    # Add system+user response tasks
    is_doubling_down = 'dd_turn_1_prompt' in df.columns and 'dd_turn_1_response' in df.columns
    system_user_run_cols = [col for col in df.columns if col.startswith('generation(System Prompt + User Prompt)_run')]
    assert len(system_user_run_cols) == 1, f"Expected exactly one system+user run column, found {len(system_user_run_cols)}"
    
    run_col = system_user_run_cols[0]

    # Add individual belief tasks for non-numerical propositions
    belief_conversations = ""
    for index, tup in enumerate(belief_columns):
        belief_col, gen_col = tup
        if pd.notna(row[belief_col]) and pd.notna(row[gen_col]):
            if 'belief_system_1' not in row:
                row['belief_system_1'] = "You are a helpful assistant."
            belief_conversations += f"Belief Conversation {index + 1}:\n------\n"
            belief_conversations += f"System Prompt:\n{row['belief_system_1']}\n\n---\n"
            belief_conversations += f"User Prompt:\n{row[belief_col]}\n\n---\n"
            belief_conversations += f"AI Response:\n{row[gen_col]}\n\n---\n"
        
    
    if is_doubling_down:
        tasks.append((run_col, evaluate_doubling_down_response(
            client,
            row['system_prompt'],
            row['dd_turn_1_prompt'],
            row['dd_turn_1_response'],
            row['user_prompt'],
            row[run_col],  # Use the specific run's response
            row['proposition'],
            concurrency_limit,
            belief_conversations=belief_conversations
        )))
    else:
        tasks.append((run_col, evaluate_response(
            client,
            row['system_prompt'],
            row['user_prompt'],
            row[run_col],  # Use the specific run's response
            row['proposition'],
            filepath=filepath,
            concurrency_limit=concurrency_limit,
            belief_conversations=belief_conversations
        )))
    
    # Execute all tasks concurrently with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(task[1] for task in tasks)),
            timeout=60  # 1 minutes timeout per row
        )
    except asyncio.TimeoutError:
        pront(f"Timeout processing row {idx}")
        results = ["TIMEOUT_ERROR"] * len(tasks)
    # except Exception as e:
    #     pront(f"Error processing row {idx}: {e}")
    #     results = [f"ERROR processing row {idx}: {e}"] * len(tasks)

    # Process results
    for (task_type, _), result in zip(tasks, results):
        # Handle multiple system+user runs
        eval_col = task_type.replace('generation', 'single_eval')
        df.at[idx, eval_col] = result
        df.at[idx, f'{eval_col}_extracted'] = extract_boxed_evaluation(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="Run in test mode")
    parser.add_argument("--input_file", help="Input CSV file to process")
    parser.add_argument("--concurrency_limit", type=int, default=32, help="Global concurrency limit for API calls")
    args = parser.parse_args()

    # Process files sequentially to maintain strict concurrency control
    if args.input_file:
        output_file = args.input_file.replace('responses', 'evaluated')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        asyncio.run(process_file(args.input_file, output_file, args.concurrency_limit))
    else:
        files = glob.glob('test_csv_data/responses/*.csv' if args.test else 'csv_data/responses/*.csv')

        # sort files as follows: first, the files with "claude" in the filename, then the files with "gpt-4o", then files tiwh '405'  then '70b-instruct' then the rest

        priority = {
            "claude": 1,
            "gpt-4o": 2,
            "405": 3,
            "70b-instruct": 4
        }

        # Custom sorting function
        def sort_key(filename):
            for key, order in priority.items():
                if key in filename:
                    return order
            return float('inf')  # All other files come last

        # Sort files based on priority
        sorted_files = sorted(files, key=sort_key)

        for file in sorted_files:
            if 'statistics' in file:
                continue

            output_file = file.replace('responses', 'evaluated_single')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Skip files that are already evaluated or evaluated_del
            if '_evaluated' in file or os.path.exists(output_file):
                print(f"Skipping file: {file}")
                continue
            
            print(f"Processing file: {file}")
            asyncio.run(process_file(file, output_file, args.concurrency_limit))